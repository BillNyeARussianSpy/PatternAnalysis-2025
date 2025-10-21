import os
from pathlib import Path
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import nibabel as nib
from tqdm import tqdm

# for SSIM (optional; only used in quick val pass below)
from skimage.metrics import structural_similarity as ssim

# modules from other files
from dataset import *
from modules import *

# Hyperparams
epochs = 100
learning_rate = 1e-4
batch_size = 16
weight_decay = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hidden arch
n_hiddens = 512
n_residual_hiddens = 256
n_residual_layers = 16
embedding_dim = 128
n_embeddings = 1024
beta = 0.25
categorical = False
normal_image = True
early_stop = False
target_size = (256, 144)
fit_mode = "pad_or_crop"

# Data paths
train_path = '../../../keras_slices_data/keras_slices_train'
validate_path = '../../../keras_slices_data/keras_slices_validate'

# Checkpointing
SAVE_DIR = Path(os.environ.get("CKPT_DIR", "checkpoints2"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LAST_CKPT = SAVE_DIR / "last.ckpt"
LOG_JSON  = SAVE_DIR / "training_log.json"

def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)  # atomic on POSIX

def save_checkpoint(epoch, model, optimizer, results, extra_config=None, is_best=False):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "results": results,
        "extra_config": extra_config or {},
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
    }
    ckpt_path = SAVE_DIR / f"ckpt_epoch_{epoch:03d}.pt"
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
        _atomic_save(state, SAVE_DIR / "best.ckpt")

def load_latest_checkpoint(model, optimizer):
    if LAST_CKPT.exists():
        state = torch.load(LAST_CKPT, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        epoch = int(state.get("epoch", -1))
        results = state.get("results", None)
        print(f"[resume] Loaded checkpoint from epoch {epoch} at {state.get('time')}")
        return epoch, results
    return -1, None

def _to_01_per_sample(t: torch.Tensor) -> torch.Tensor:
    t_min = t.amin(dim=(2,3), keepdim=True)
    t_max = t.amax(dim=(2,3), keepdim=True)
    return torch.clamp((t - t_min) / (t_max - t_min + 1e-8), 0.0, 1.0)

# Dataload
nii_train = [os.path.join(train_path, img) for img in os.listdir(train_path) if img.endswith(('.nii', '.nii.gz'))]
nii_val   = [os.path.join(validate_path, img) for img in os.listdir(validate_path) if img.endswith(('.nii', '.nii.gz'))]

x_train = load_data_2D(nii_train, normImage=normal_image, categorical=categorical,
                       early_stop=early_stop, target_size=target_size, fit_mode=fit_mode)
x_val   = load_data_2D(nii_val,   normImage=normal_image, categorical=categorical,
                       early_stop=early_stop, target_size=target_size, fit_mode=fit_mode)

# Data into tensors -> add channel
x_train_tensor = torch.from_numpy(x_train).float().unsqueeze(1)
x_val_tensor   = torch.from_numpy(x_val).float().unsqueeze(1)

# Dataloaders
train_loader = DataLoader(x_train_tensor, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(x_val_tensor, batch_size=batch_size)

# Model + Optimiser
model = VQVAE(n_hiddens, n_residual_hiddens,
              n_residual_layers, n_embeddings, embedding_dim, beta).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)
model.train()

# Results DICT
results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexity': [],
    'val_recon_errors': [],
    'val_ssim': [],
    'best_val_recon': float('inf'),
}

# Auto-resume
start_epoch, loaded_results = load_latest_checkpoint(model, optimizer)
if loaded_results is not None:
    # merge (keep history, ensure keys exist)
    for k, v in loaded_results.items():
        if k not in results or not isinstance(v, list):
            results[k] = v
        else:
            results[k].extend(v)

# Extra config to store inside checkpoints for reproducibility
extra_config = {
    "epochs": epochs,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "weight_decay": weight_decay,
    "architecture": {
        "n_hiddens": n_hiddens,
        "n_residual_hiddens": n_residual_hiddens,
        "n_residual_layers": n_residual_layers,
        "embedding_dim": embedding_dim,
        "n_embeddings": n_embeddings,
        "beta": beta,
    },
    "data": {
        "categorical": categorical,
        "normal_image": normal_image,
        "early_stop": early_stop,
        "target_size": target_size,
        "fit_mode": fit_mode,
        "train_path": train_path,
        "validate_path": validate_path,
    },
}

# Validation Helper
@torch.no_grad()
def validate_epoch():
    model.eval()
    recon_losses, ssim_scores = [], []
    for xb in val_loader:
        xb = xb.to(device)
        emb_loss, x_hat, ppx = model(xb)
        recon_losses.append(torch.mean((x_hat - xb) ** 2).item())

        # SSIM on [0,1] views (per-sample, per-channel)
        x_01 = _to_01_per_sample(xb)
        y_01 = _to_01_per_sample(x_hat)

        # compute skimage SSIM per item, then average
        x_np = x_01[:, 0].detach().cpu().numpy()
        y_np = y_01[:, 0].detach().cpu().numpy()
        for i in range(x_np.shape[0]):
            ssim_scores.append(
                ssim(x_np[i], y_np[i], data_range=1.0)
            )
        

    model.train()
    return float(np.mean(recon_losses)), (float(np.mean(ssim_scores)) if ssim_scores else None)

# Training
def train():
    for epoch in range(start_epoch + 1, epochs):
        # TQDM Metrics
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for i, x in enumerate(pbar):
            # data to device + zero grad
            x = x.to(device)
            optimizer.zero_grad()

            embedding_loss, x_hat, perplexity = model(x)

            recon_loss = torch.mean((x_hat - x) ** 2)
            loss = recon_loss + embedding_loss

            loss.backward()
            optimizer.step()

            # logging
            results["recon_errors"].append(float(recon_loss.detach().cpu().numpy()))
            results["perplexity"].append(float(perplexity.detach().cpu().numpy()))
            results["loss_vals"].append(float(loss.detach().cpu().numpy()))
            results["n_updates"] += 1

            pbar.set_postfix({
                "recon": f"{results['recon_errors'][-1]:.4f}",
                "loss": f"{results['loss_vals'][-1]:.4f}",
                "ppx": f"{results['perplexity'][-1]:.2f}",
            })

        #end of epoch
        
        # metrics
        val_recon, val_ssim = validate_epoch()
        results["val_recon_errors"].append(val_recon)
        results["val_ssim"].append(val_ssim)

        # save best model w/recon
        is_best = False
        if val_recon is not None and val_recon < results.get("best_val_recon", float('inf')):
            results["best_val_recon"] = val_recon
            is_best = True

        save_checkpoint(epoch, model, optimizer, results, extra_config=extra_config, is_best=is_best)

        print(f"[epoch {epoch+1}] saved. "
              f"val_recon={val_recon:.6f} | val_ssim={val_ssim:.4f} | best={results['best_val_recon']:.6f}")

if __name__ == "__main__":
    train()



