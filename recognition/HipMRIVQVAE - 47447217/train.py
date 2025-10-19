import os
from pathlib import Path
import numpy as numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import nibabel as nib

#for SSIM
from skimage.metrics import structural_similarity as ssim

#modules from other files
from dataset import *
from modules import *

# Actual ingestion
epochs = 10
learning_rate = 1e-3
batch_size = 16
weight_decay = 1e-5


# hidden arch
n_hiddens = 512
n_residual_hiddens = 256
n_residual_layers = 16
embedding_dim = 512
n_embeddings = 1024
beta = 0.1
categorical = False
normal_image = False
early_stop = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


# Newer, cleaner dataloader
train_path = '../keras_slices_data/keras_slices_train'
validate_path = '../keras_slices_data/keras_slices_validate'


nii_files_train = [os.path.join(train_path, img) for img in os.listdir(train_path) if img.endswith(('.nii', '.nii.gz'))]
nii_files_validate = [os.path.join(validate_path, img) for img in os.listdir(validate_path) if img.endswith(('.nii', '.nii.gz'))]

x_train = load_data_2D(nii_files_train, normImage=normal_image, categorical=categorical, early_stop=early_stop)
x_val = load_data_2D(nii_files_validate, normImage=normal_image, categorical=categorical,early_stop=early_stop)

# Data into tensors -> add channel
x_train_tensor = torch.from_numpy(x_train).float().unsqueeze(1) 
x_val_tensor = torch.from_numpy(x_val).float().unsqueeze(1)

# Dataloaders
train_loader = torch.utils.data.DataLoader(x_train_tensor, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(x_val_tensor, batch_size=batch_size)

# initialise model
model = VQVAE(n_hiddens, n_residual_hiddens,
              n_residual_layers, n_embeddings, embedding_dim, beta).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

model.train()

# results dictionary to store model data
results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}

def train():

    for epoch in range(epochs): 
        # use of tqdm to create a progress bar of training batches for each epoch
        for i, (x) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')):

            x = x.to(device) # move batch to device (preferably gpu)
            optimizer.zero_grad() # zero out gradients

            # forward pass
            embedding_loss, x_hat, perplexity = model(x)

            # compute reconstruction loss
            recon_loss = torch.mean((x_hat - x)**2)
            loss = recon_loss + embedding_loss

            # backwards pass and optimization
            loss.backward()
            optimizer.step()

            # store results for logging and saving
            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["perplexities"].append(perplexity.cpu().detach().numpy())
            results["loss_vals"].append(loss.cpu().detach().numpy())
            results["n_updates"] = i

        # save the model and data
        if args.save:
            hyperparameters = args.__dict__
            utils.save_model_and_results(
                model, results, hyperparameters)

if __name__ == "__main__":
    train()



"""
OLD Training Model
ROOT = Path("../keras_slices_data")

#SSIM Hyper
ssim_weight = 0.2   # set >0.0 to include (1-SSIM)*ssim_weight in the loss, e.g., 0.1
ssim_range  = 1.0   # dynamic range of normalized Images

def _to_01_per_sample(t: torch.Tensor) -> torch.Tensor:
    # t: (N,C,H,W)
    t_min = t.amin(dim=(2,3), keepdim=True)           # per-sample, per-channel
    t_max = t.amax(dim=(2,3), keepdim=True)
    return torch.clamp((t - t_min) / (t_max - t_min + 1e-8), 0.0, 1.0)

# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# gather file paths
splits = list_paths(ROOT)

# segmentation: aligning pairs per split
train_imgs, train_segs = align_by_name(splits["train"]["imgs"], splits["train"]["segs"])
test_imgs, test_segs = align_by_name(splits["test"]["imgs"], splits["test"]["segs"])
validate_imgs, validate_segs = align_by_name(splits["validate"]["imgs"], splits["validate"]["segs"])


# preprocess with shakes function
# X Values
X_train, train_aff = load_data_2D([str(p) for p in train_imgs], normImage=True, categorical=False, getAffines=True, early_stop= True, target_size=(256, 144), fit_mode="pad_or_crop")
X_test, test_aff = load_data_2D([str(p) for p in test_imgs], normImage=True, categorical=False, getAffines=True, early_stop= True, target_size=(256, 144), fit_mode="pad_or_crop")
X_val, val_aff = load_data_2D([str(p) for p in validate_imgs], normImage=True, categorical=False, getAffines=True, early_stop= True, target_size=(256, 144), fit_mode="pad_or_crop")

Y_train = load_data_2D([str(p) for p in train_segs], normImage=False, categorical=False, early_stop= True, target_size=(256, 144), fit_mode="pad_or_crop")
Y_train = load_data_2D([str(p) for p in train_segs], normImage=False, categorical=False, early_stop= True, target_size=(256, 144), fit_mode="pad_or_crop")
Y_val = load_data_2D([str(p) for p in validate_segs], normImage=False, categorical=False, early_stop= True, target_size=(256, 144), fit_mode="pad_or_crop")

#TODO: Play around with what I need (check spec sheet)



def _to_tensor_nchw(x_np: np.ndarray, in_channel=1) -> torch.Tensor:
    
    (N,H,W) or (N,H,W,C) -> (N,C,H,W) float32
    
    if x_np.ndim == 3:
        x_np = x_np[:, None, :, :]
    elif x_np.ndim == 4:
        x_np = np.transpose(x_np, (0, 3, 1, 2))
    if in_channel == 1 and x_np.shape[1] == 3:
        x_np = x_np[:, :1]
    elif in_channel == 3 and x_np.shape[1] == 1:
        x_np = np.repeat(x_np, 3, axis=1)
    return torch.from_numpy(x_np.astype(np.float32))

def make_loader(x_np, batch_size=16, shuffle=False, num_workers=0) -> DataLoader:
    
    Function for making a dataloader
    
    # x, y arrays
    x = _to_tensor_nchw(x_np, in_channel=1)
    y = torch.zeros(len(x), dtype=torch.long) # dummy labels
    # dataloader
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size = batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)
    
    
def main():
    # output directories
    out_dir = Path("runs/vqvae")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ensure all splits share same HxW
    assert X_train.shape[1:] == X_val.shape[1:] == X_test.shape[1:], \
        f"Shapes differ: {X_train.shape} vs {X_val.shape} vs {X_test.shape}. " \
        # If wrong, use load_data_2D with target_size/fitting"
    
    
    # train + vak dataloaders
    train_loader = make_loader(X_train, batch_size=32, shuffle = True, num_workers=0)
    val_loader = make_loader(X_val, batch_size=32, shuffle = True, num_workers=0)

    # Variance scale for normalized recon loss
    x_train_var = 1.0 #change later
    
    # Model!
    model = VQVAE(
        h_dim=128, res_h_dim=32, n_res_layers=2,
        n_embeddings=512, embedding_dim=64, beta=0.25,
        in_channels=1, out_channels=1
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, amsgrad=True)
    mse = nn.MSELoss(reduction="mean")
    
    # TRAINING!
    
    # hyperparams
    epochs = 10          # start small; bump later
    log_every = 1        # print every epoch
    save_every = 2
    best_val = float("inf")

    print(f"Device: {device} | train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    for epoch in range(1, epochs + 1):
        # training
        model.train()
        tr_loss = tr_recon = tr_pp = 0.0
        tr_ssim = 0.0  # ---- SSIM metric (train) ----
        for bi, (xb, _) in enumerate(train_loader, 1):
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            emb_loss, x_hat, ppx = model(xb)
            
            recon = torch.mean((x_hat - xb) ** 2) / max(x_train_var, 1e-8)

            xh_ssim = _to_01_per_sample(x_hat)
            xb_ssim = _to_01_per_sample(xb)
            ssim_val = piq.ssim(xh_ssim, xb_ssim, data_range=1.0, reduction='mean')
            loss = recon + emb_loss + (1.0 - ssim_val) * ssim_weight
            # ------------------------

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # loss
            tr_loss  += loss.item()
            tr_recon += recon.item()
            tr_pp    += float(ppx.item())
            tr_ssim  += float(ssim_val.item())  # track mean SSIM

            if bi % 50 == 0 or bi == len(train_loader):
                print(f"  epoch {epoch} | batch {bi}/{len(train_loader)} "
                      f"loss {loss.item():.4f} recon {recon.item():.4f} ssim {float(ssim_val):.4f} ppx {float(ppx):.2f}", flush=True)

        tr_loss  /= len(train_loader)
        tr_recon /= len(train_loader)
        tr_pp    /= len(train_loader)
        tr_ssim  /= len(train_loader)  # ---- average train SSIM ----

        # validation
        model.eval()
        va_loss = va_recon = va_pp = 0.0
        va_ssim = 0.0  # ---- SSIM metric (val) ----
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device, non_blocking=True)
                emb_loss, x_hat, ppx = model(xb)
                recon = torch.mean((x_hat - xb) ** 2) / max(x_train_var, 1e-8)
                xh_ssim = _to_01_per_sample(x_hat)
                xb_ssim = _to_01_per_sample(xb)
                ssim_val = piq.ssim(xh_ssim, xb_ssim, data_range=1.0, reduction='mean')
                loss = recon + emb_loss + (1.0 - ssim_val) * ssim_weight  # optional SSIM term
                va_loss  += loss.item()
                va_recon += recon.item()
                va_pp    += float(ppx.item())
                va_ssim  += float(ssim_val.item())
        # validation loss
        va_loss  /= len(val_loader)
        va_recon /= len(val_loader)
        va_pp    /= len(val_loader)
        va_ssim  /= len(val_loader)  # ---- average val SSIM ----

        # debug printing
        if epoch % log_every == 0:
            print(f"[epoch {epoch:3d}] "
                  f"train loss {tr_loss:.4f} | recon {tr_recon:.4f} | ssim {tr_ssim:.4f} | ppx {tr_pp:.2f}  "
                  f"|| val loss {va_loss:.4f} | recon {va_recon:.4f} | ssim {va_ssim:.4f} | ppx {va_pp:.2f}",
                  flush=True)

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict()}, out_dir / "best.pt")

        # write reconstructions for reference 
        if epoch % save_every == 0:
            xb, _ = next(iter(val_loader))
            xb = xb.to(device)
            with torch.no_grad():
                _, xh, _ = model(xb)
            xh = xh.cpu().numpy()
            xb = xb.cpu().numpy()
            N = min(len(xh), 4)
            for i in range(N):
                recon_2d = xh[i, 0]
                input_2d = xb[i, 0]
                affine = val_aff[i] if i < len(val_aff) else np.eye(4)
                nib.save(nib.Nifti1Image(recon_2d.astype(np.float32), affine), str(out_dir / f"recon_e{epoch}_i{i}.nii.gz"))
                nib.save(nib.Nifti1Image(input_2d.astype(np.float32), affine), str(out_dir / f"input_e{epoch}_i{i}.nii.gz"))


if __name__ == "__main__":
    main()
    
    
"""