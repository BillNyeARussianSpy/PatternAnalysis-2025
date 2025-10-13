import os, re, glob
import numpy as np
import nibabel as nib
from tqdm import tqdm


def to_channels ( arr : np . ndarray , dtype = np . uint8 ) -> np . ndarray :
    channels = np . unique ( arr )
    res = np . zeros ( arr . shape + ( len ( channels ) ,) , dtype = dtype )
    for c in channels :
        c = int ( c )
        res [... , c : c +1][ arr == c ] = 1

    return res


# load medical image functions
def load_data_2D (imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage : bool (normalise the image 0.0–1.0)
    early_stop : Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
            # inImage = inImage / np.linalg.norm(inImage)
            # inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = utils.to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
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
    print(root / "keras_slice_train")
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

# Actual ingestion

ROOT = Path("../../../keras_slices_data")

# gather file paths
splits = list_paths(ROOT)

# segmentation: aligning pairs per split
train_imgs, train_segs = align_by_name(splits["train"]["imgs"], splits["train"]["segs"])
test_imgs, test_segs = align_by_name(splits["test"]["imgs"], splits["test"]["segs"])
validate_imgs, validate_segs = align_by_name(splits["validate"]["imgs"], splits["validate"]["segs"])
#DEBUG: WORKS!


# preprocess with shakes function
# images
X_train, train_aff = load_data_2D([str(p) for p in train_imgs], normImage=True, categorical=False, getAffines=True)
X_test, test_aff = load_data_2D([str(p) for p in test_imgs], normImage=True, categorical=False, getAffines=True)
X_val, val_aff = load_data_2D([str(p) for p in val_imgs], normImage=True, categorical=False, getAffines=True)

Y_train = load_data_2D([str(p) for p in train_segs], normImage=False, categorical=False)
Y_train = load_data_2D([str(p) for p in train_segs], normImage=False, categorical=False)
Y_val = load_data_2D([str(p) for p in val_segs], normImage=False, categorical=False)

#TODO: Play around with what I need (check spec sheet)