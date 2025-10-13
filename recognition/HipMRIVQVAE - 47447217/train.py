from dataset import *
from modules import *


# Data Ingestion


ROOT = Path("../../../keras_slices_data")

# gather file paths
splits = list_paths(ROOT)

# segmentation: aligning pairs per split
train_imgs, train_segs = align_by_name(splits["train"]["imgs"], splits["train"]["segs"])
test_imgs, test_segs = align_by_name(splits["test"]["imgs"], splits["test"]["segs"])
validate_imgs, validate_segs = align_by_name(splits["validate"]["imgs"], splits["validate"]["segs"])


# preprocess with shakes function

# images
X_train, train_aff = load_data_2D([str(p) for p in train_imgs], normImage=True, categorical=False, getAffines=True, early_stop=True)
X_test, test_aff = load_data_2D([str(p) for p in test_imgs], normImage=True, categorical=False, getAffines=True, early_stop=True)
X_val, val_aff = load_data_2D([str(p) for p in val_imgs], normImage=True, categorical=False, getAffines=True, early_stop=True)

Y_train = load_data_2D([str(p) for p in train_segs], normImage=False, categorical=False, early_stop=True)
Y_train = load_data_2D([str(p) for p in train_segs], normImage=False, categorical=False, early_stop=True)
Y_val = load_data_2D([str(p) for p in val_segs], normImage=False, categorical=False, early_stop=True)

#TODO: Play around with what I need (check spec sheet)