import os, argparse, random, json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Local Modules
from modules import VQVAE
from dataset import load_data_2D
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Arguments for visualisation
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", default="checkpoints2", type=str,
                   help="Folder containing best.ckpt/last.ckpt")
    p.add_argument("--test_dir", required=False, default="../../../keras_slices_data/keras_slices_test",
                   help="Folder with test .nii/.nii.gz")
    p.add_argument("--num", type=int, default=8, help="How many examples to preview")
    p.add_argument("--out", type=str, default="preview_test_better.png", help="Where to save the panel")
    args = p.parse_args()

    # load model
    ckpt_path = pick_ckpt(Path(args.save_dir))
    model, cfg = load_model_from_ckpt(ckpt_path)
    
    # get data from config
    data_cfg   = cfg.get("data", {})
    target_sz  = tuple(data_cfg.get("target_size", (256, 144)))
    norm_image = bool(data_cfg.get("normal_image", True))
    fit_mode   = data_cfg.get("fit_mode", "pad_or_crop")

    # Collect test data
    test_dir = Path(args.test_dir)
    nii_files = [str(test_dir / f) for f in os.listdir(test_dir)
                 if f.endswith((".nii", ".nii.gz"))]
    if not nii_files:
        raise FileNotFoundError(f"No NIfTI files found in {test_dir}")

    # Load and size-match the data exactly like training
    print(f"[info] Loading {len(nii_files)} test images… (target_size={target_sz}, norm={norm_image}, mode={fit_mode})")
    X = load_data_2D(nii_files, normImage=norm_image, categorical=False,
                     target_size=target_sz, fit_mode=fit_mode)
    X = torch.from_numpy(X).float().unsqueeze(1)  # (N,1,H,W)

    # Choose N samples
    idxs = list(range(len(X)))
    random.shuffle(idxs)
    idxs = idxs[:args.num]
    xb = X[idxs].to(device)

    with torch.no_grad():
        emb_loss, x_hat, ppx = model(xb)

    # Compute SSIM per-sample on [0,1]
    x01 = to_01_per_sample(xb)
    y01 = to_01_per_sample(x_hat)
    x_np = x01[:, 0].cpu().numpy()
    y_np = y01[:, 0].cpu().numpy()
    ssim_list = [ssim(x_np[i], y_np[i], data_range=1.0) for i in range(len(x_np))]
    
    # Prints SSIM values
    print("[info] Per-image SSIM:", [f"{s:.3f}" for s in ssim_list])
    print(f"[info] Mean SSIM: {np.mean(ssim_list):.3f}")

    # Make a panel of images: originals (row 1) and reconstructions (row 2)
    n = len(idxs)
    fig = plt.figure(figsize=(2.8*n, 5))
    for i in range(n):
        # originals
        ax = plt.subplot(2, n, i+1)
        ax.imshow(x_np[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Orig #{idxs[i]}")
        ax.axis("off")
        # recons
        ax = plt.subplot(2, n, n+i+1)
        ax.imshow(y_np[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Recon | SSIM {ssim_list[i]:.3f}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[ok] Saved panel -> {args.out}")

if __name__ == "__main__":
    main()
