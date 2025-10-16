import os, re, glob
import numpy as np
import nibabel as nib
from tqdm import tqdm


def _ensure_2d(arr):
    # if shape is (H,W,*) keep first slice; if (H,W) return as-is
    if arr.ndim == 3:
        return arr[:, :, 0]
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"Unexpected image ndim={arr.ndim}, shape={arr.shape}")

def _center_crop(img, out_h, out_w):
    """
    Crop image H & W from center
    """
    h, w = img.shape
    y0 = max((h - out_h) // 2, 0)
    x0 = max((w - out_w) // 2, 0)
    return img[y0:y0 + out_h, x0:x0 + out_w]

def _pad_to(img, out_h, out_w, pad_value=0):
    """
    Padding with specified width/height
    """
    h, w = img.shape
    pad_y = max(out_h - h, 0)
    pad_x = max(out_w - w, 0)
    top  = pad_y // 2
    bot  = pad_y - top
    left = pad_x // 2
    right= pad_x - left
    return np.pad(img, ((top, bot), (left, right)), mode='constant', constant_values=pad_value)

def _fit_to_canvas(img, out_h, out_w, mode="pad_or_crop", pad_value=0):
    h, w = img.shape
    res = img
    if mode in ("pad_or_crop", "crop"):
        if h > out_h or w > out_w:
            res = _center_crop(res, min(h, out_h), min(w, out_w))
    if mode in ("pad_or_crop", "pad"):
        if res.shape[0] < out_h or res.shape[1] < out_w:
            res = _pad_to(res, out_h, out_w, pad_value=pad_value)
    # final safety: exact target
    if res.shape != (out_h, out_w):
        res = _center_crop(res, out_h, out_w)
        if res.shape != (out_h, out_w):
            res = _pad_to(res, out_h, out_w, pad_value=pad_value)
    return res

def _to_channels(x2d, dtype):
    try:
        from utils import to_channels
        return to_channels(x2d, dtype=dtype)
    except Exception:
        return x2d[..., None]  # (H,W,1)

def load_data_2D(
    imageNames,
    normImage=False,
    categorical=False,
    dtype=np.float32,
    getAffines=False,
    early_stop=False,
    target_size=None,        # (rows, cols) or None to auto
    fit_mode="pad_or_crop"   # "pad_or_crop" | "pad" | "crop"
):
    """
    Loads 2D medical images to a fixed canvas size.

    - If target_size is None: scans all files first to choose a target (max H,W).
    - fit_mode controls how each slice is adapted to the canvas.
    - categorical=True: keeps integer labels; no normalization; expands to channels-last.
    """
    affines = []
    # --- PASS 1: determine target size if not provided ---
    shapes = []
    if target_size is None:
        for p in imageNames:
            # use header+first slice size with minimal IO
            img = nib.load(p).get_fdata(caching='unchanged')
            img2d = _ensure_2d(img)
            shapes.append(img2d.shape)
            if early_stop and len(shapes) > 20:
                break
        if not shapes:
            raise ValueError("No images found to infer target_size.")
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        target_size = (max_h, max_w)

    out_h, out_w = target_size
    num = len(imageNames)

    # --- allocate ---
    if categorical:
        # masks as integer labels (single channel unless your to_channels expands)
        images = np.zeros((num, out_h, out_w), dtype=dtype)
    else:
        images = np.zeros((num, out_h, out_w), dtype=dtype)

    # --- PASS 2: load, normalise, fit to canvas, assign ---
    for i, inName in enumerate(tqdm(imageNames)):
        nifti = nib.load(inName)
        arr = _ensure_2d(nifti.get_fdata(caching='unchanged')).astype(dtype)
        affine = nifti.affine

        if categorical:
            # do NOT normalize labels; just fit to canvas, pad with 0 label
            arr_fit = _fit_to_canvas(arr, out_h, out_w, mode=fit_mode, pad_value=0)
            # expand to channels if your downstream expects channels-last one-hot:
            # leave as (H,W) here; let training pipeline one-hot if needed
            images[i, :, :] = arr_fit
        else:
            # normalize BEFORE padding so zeros remain near 0 mean afterwards
            if normImage:
                mu, sigma = arr.mean(), arr.std()
                arr = (arr - mu) / (sigma + 1e-8)
            arr_fit = _fit_to_canvas(arr, out_h, out_w, mode=fit_mode, pad_value=0.0)
            images[i, :, :] = arr_fit

        affines.append(affine)
        if i > 20 and early_stop:
            break

    # If you really want channels-last for categorical, convert later via one-hot.
    if getAffines:
        return images, affines
    return images


# Data ingestion pipeline

from pathlib import Path

# ---------------- Helpers ----------------
def list_paths(root: Path) -> dict:
    """
    Collect nii.gz file paths for each split
    args:
        root::Path
            parent folder location
    returns
        dict
            dictionary with imgs and segmentation
    """
    splits = {
        "train" : {
            "imgs" : sorted((root / "keras_slices_train").rglob("*.nii.gz")),
            "segs" : sorted((root / "keras_slices_seg_train").rglob("*nii.gz")),
        },
        "test" : {
            "imgs" : sorted((root / "keras_slices_test").rglob("*.nii.gz")),
            "segs" : sorted((root / "keras_slices_seg_test").rglob("*nii.gz")),
        },
        "validate" : {
            "imgs" : sorted((root / "keras_slices_validate").rglob("*.nii.gz")),
            "segs" : sorted((root / "keras_slices_seg_validate").rglob("*nii.gz")),
        },
    }
    return splits

def normalise_name(p: Path) -> str:
    """
    Extract comparable key for matching images and segmentations
    args:
        p:Path
            path for file
    returns:
        base:str
            normalised filename
    """
    base = p.stem.replace(".nii", "")  # remove .nii if double suffix
    # remove prefix -> either 'case_' or 'seg_'
    base = base.replace("case_", "").replace("seg_", "")
    return base

def align_by_name(imgs, segs) -> list:
    """
    Images and segmentations are paired by filename (w/o extensions)
    args:
        imgs
            2D scan image files from HipMRI
        segs
            Preprocessed segmentations from HipMRI
    returns
        img::list
            list of images
        segs:list
            list of segmentations matched with images
    """
    # build a map w/ key
    img_map = {normalise_name(p): p for p in imgs}
    seg_map = {normalise_name(p): p for p in segs}
    
    # sort
    common = sorted(set(img_map) & set(seg_map))
    if not common:
        raise ValueError("No matching image/segmentation pairs found!")
    # debug: in case there are missing imgs/segs
    missing_imgs = sorted(set(seg_map) - set(img_map))
    missing_segs = sorted(set(img_map) - set(seg_map))
    
    if missing_imgs or missing_segs:
        raise FileNotFoundError(
            f"Unpaired files: missing_imgs = {missing_imgs[:5]} missing_segs = {missing_segs[:5]}"
        )
    return [img_map[k] for k in common], [seg_map[k] for k in common]

