# Training helpers for saving

def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)  # atomic on POSIX

def save_checkpoint(epoch, model, optimizer, results, save_dir, extra_config=None, is_best=False):
    """
    Helper that saves a checkpoint
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "results": results,
        "extra_config": extra_config or {},
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
    }
    ckpt_path = save_dir / f"ckpt_epoch_{epoch:03d}.pt"
    _atomic_save(state, ckpt_path)
    _atomic_save(state, LAST_CKPT)
    # Lightweight JSON log (append-last view)
    with open(LOG_JSON, "w") as f:
        json.dump({
            "epoch": epoch,
            "recon_last": float(results["recon_errors"][-1]) if results["recon_errors"] else None,
            "loss_last": float(results["loss_vals"][-1]) if results["loss_vals"] else None,
            "perplexity_last": float(results["perplexity"][-1]) if results["perplexity"] else None,
            "val_recon_last": float(results["val_recon_errors"][-1]) if results.get("val_recon_errors") else None,
            "val_ssim_last": float(results["val_ssim"][-1]) if results.get("val_ssim") else None,
            "time": state["time"],
        }, f, indent=2)
    if is_best:
        _atomic_save(state, save_dir / "best.ckpt")

def load_latest_checkpoint(model, optimizer):
    """
    Tries to load the lastest checkpoint from folder
    """
    if LAST_CKPT.exists():
        state = torch.load(LAST_CKPT, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        epoch = int(state.get("epoch", -1))
        results = state.get("results", None)
        print(f"[resume] Loaded checkpoint from epoch {epoch} at {state.get('time')}")
        return epoch, results
    return -1, None

# Prediction Helpers

def to_01_per_sample(t: torch.Tensor) -> torch.Tensor:
    # (B,1,H,W) -> scale each image to [0,1]
    t_min = t.amin(dim=(2,3), keepdim=True)
    t_max = t.amax(dim=(2,3), keepdim=True)
    return torch.clamp((t - t_min) / (t_max - t_min + 1e-8), 0.0, 1.0)

def load_model_from_ckpt(ckpt_path: Path):
    state = torch.load(ckpt_path, map_location=device)
    cfg = state.get("extra_config", {})
    arch = cfg.get("architecture", {})
    # Fallbacks in case field is missing
    n_hiddens         = arch.get("n_hiddens", 512)
    n_residual_hiddens= arch.get("n_residual_hiddens", 256)
    n_res_layers      = arch.get("n_residual_layers", 16)
    embedding_dim     = arch.get("embedding_dim", 128)
    n_embeddings      = arch.get("n_embeddings", 1024)
    beta              = arch.get("beta", 0.25)

    model = VQVAE(n_hiddens, n_residual_hiddens, n_res_layers,
                  n_embeddings, embedding_dim, beta,
                  in_channels=1, out_channels=1).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, cfg

def pick_ckpt(save_dir: Path) -> Path:
    best = save_dir / "best.ckpt"
    last = save_dir / "last.ckpt"
    if best.exists():  return best
    if last.exists():  return last
    # fallback: most recent ckpt_epoch_XXX.pt
    pts = sorted(save_dir.glob("ckpt_epoch_*.pt"))
    if pts: return pts[-1]
    raise FileNotFoundError(f"No checkpoints found in {save_dir}")