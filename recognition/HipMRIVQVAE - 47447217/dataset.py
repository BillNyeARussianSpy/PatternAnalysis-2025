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


# 3D image loader (from Shakes)

def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, orient=False, early_stop=False):
    """
    Load medical image data from a list of file names into a NumPy array.

    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

    Parameters:
        imageNames : list
            List of image file paths (.nii or .nii.gz).
        normImage : bool
            Normalize the image to mean 0 and std 1 if True.
        categorical : bool
            Convert images to one-hot channel format if True.
        dtype : type
            Data type of the output array (e.g., np.float32 or np.uint8).
        getAffines : bool
            Return affine matrices for each image if True.
        orient : bool
            Apply orientation and resampling for anisotropic images.
        early_stop : bool
            Stop loading early for quick testing.

    Returns:
        images : np.ndarray
            Loaded image data in 4D or 5D format (num, rows, cols, depth[, channels]).
        affines : list (optional)
            List of affine matrices, if getAffines=True.
    """
    affines = []
    
    interp = 'linear'
    if dtype == np.uint8:
        interp = 'nearest'  # assume labels

    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])

    if orient:
        niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)

    first_case = niftiImage.get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # remove extra dimension

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if orient:
            niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)

        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine

        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]

        inImage = inImage[:, :, :depth]
        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = utils.to_channels(inImage, dtype=dtype)
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2], :inImage.shape[3]] = inImage
        else:
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2]] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


# load our dataset in:
filepath