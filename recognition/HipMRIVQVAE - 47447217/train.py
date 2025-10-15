import os
from pathlib import Path
import numpy as numpy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import nibabel as nib

#modules from other files
from dataset import *
from modules import *



# Actual ingestion


#Launch from folder C:\Users\james\Desktop\COMP3710\PatternAnalysis-2025>
#py '.\recognition\HipMRIVQVAE - 47447217\train.py'


ROOT = Path("../keras_slices_data")

# gather file paths
splits = list_paths(ROOT)

# segmentation: aligning pairs per split
train_imgs, train_segs = align_by_name(splits["train"]["imgs"], splits["train"]["segs"])
test_imgs, test_segs = align_by_name(splits["test"]["imgs"], splits["test"]["segs"])
validate_imgs, validate_segs = align_by_name(splits["validate"]["imgs"], splits["validate"]["segs"])


# preprocess with shakes function
# images
X_train, train_aff = load_data_2D([str(p) for p in train_imgs], normImage=True, categorical=False, getAffines=True, early_stop= True, target_size=(256, 144), fit_mode="pad_or_crop")
X_test, test_aff = load_data_2D([str(p) for p in test_imgs], normImage=True, categorical=False, getAffines=True, early_stop= True, target_size=(256, 144), fit_mode="pad_or_crop")
X_val, val_aff = load_data_2D([str(p) for p in validate_imgs], normImage=True, categorical=False, getAffines=True, early_stop= True, target_size=(256, 144), fit_mode="pad_or_crop")

Y_train = load_data_2D([str(p) for p in train_segs], normImage=False, categorical=False, early_stop= True, target_size=(256, 144), fit_mode="pad_or_crop")
Y_train = load_data_2D([str(p) for p in train_segs], normImage=False, categorical=False, early_stop= True, target_size=(256, 144), fit_mode="pad_or_crop")
Y_val = load_data_2D([str(p) for p in validate_segs], normImage=False, categorical=False, early_stop= True, target_size=(256, 144), fit_mode="pad_or_crop")

#TODO: Play around with what I need (check spec sheet)


def _to_tensor_nchw(x_np: np.ndarray, in_channel=1) -> torch.Tensor:
    """
    (N,H,W) or (N,H,W,C) -> (N,C,H,W) float32
    """
    if x_np.ndim == 3:
        x_np = x_np[:, None, :, :]
    elif x_np.ndim == 4:
        x_np = np.transpose(x_np, (0, 3, 1, 2))
    if in_channels == 1 and x_np.shape[1] == 3:
        x_np = x_np[:, :1]
    elif in_channels == 3 and x_np.shape[1] == 1:
        x_np = np.repeat(x_np, 3, axis=1)
    return torch.from_numpy(x_np.astype(np.float32))

def make_loader(x_np, batch_size=16, shuffle=False, num_workers=0):
    """
    Function for making a dataloader
    """
    x = _to_tensor_nchw(x_np, in_channels=1)
    y = torch.zeros(len(x), dtype=torch.long) # dummy labels
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size = batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)
    
    
def main():
    pass


        



