import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml

# --- Configuration ---
SOURCE_DATA_DIR = "images/"
DATASET_OUTPUT_DIR = "dataset/"

# Define split ratios - should sum to 1.0, recommend train = 0.7, valid = 0.2, test = 0.1
TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO = 0.10

# Class configuration
NUM_CLASSES = 1
CLASS_NAMES = ['Target']


# --- Input paths ---
images_input_dir = os.path.join(SOURCE_DATA_DIR, "images")
labels_input_dir = os.path.join(SOURCE_DATA_DIR, "labels")
masks_input_dir = os.path.join(SOURCE_DATA_DIR, "masks")

# --- Output paths ---
train_img_dir = os.path.join(DATASET_OUTPUT_DIR, "train", "images")
train_lbl_dir = os.path.join(DATASET_OUTPUT_DIR, "train", "labels")
train_msk_dir = os.path.join(DATASET_OUTPUT_DIR, "train", "masks")

valid_img_dir = os.path.join(DATASET_OUTPUT_DIR, "valid", "images")
valid_lbl_dir = os.path.join(DATASET_OUTPUT_DIR, "valid", "labels")
valid_msk_dir = os.path.join(DATASET_OUTPUT_DIR, "valid", "masks")

test_img_dir = os.path.join(DATASET_OUTPUT_DIR, "test", "images")
test_lbl_dir = os.path.join(DATASET_OUTPUT_DIR, "test", "labels")
test_msk_dir = os.path.join(DATASET_OUTPUT_DIR, "test", "masks")

# --- Create output directories ---
for path in [train_img_dir, train_lbl_dir, train_msk_dir,
             valid_img_dir, valid_lbl_dir, valid_msk_dir,
             test_img_dir, test_lbl_dir, test_msk_dir]:
    os.makedirs(path, exist_ok=True)

# --- Get List of Image Files ---
print("Scanning source directory")
image_files = list(Path(images_input_dir).glob("*.jpg")) # Change from .jpg if necessary
if not image_files:
    print(f"Error: No images found in {images_input_dir}")
    exit()

print(f"Found {len(image_files)} images.")

# --- Shuffle and split ---
random.shuffle(image_files)
total_images = len(image_files)
train_end_idx = int(total_images * TRAIN_RATIO)
valid_end_idx = train_end_idx + int(total_images * VALID_RATIO)

train_files = image_files[:train_end_idx]
valid_files = image_files[train_end_idx:valid_end_idx]
test_files = image_files[valid_end_idx:]

print(f"Splitting into: Train={len(train_files)}, Valid={len(valid_files)}, Test={len(test_files)}")

# --- Function to Copy Files ---
def copy_files_for_split(file_list, img_dest, lbl_dest, msk_dest):
    """
    Copies image, label, and masks to destination.
    """
    copied_count = 0
    for img_path in tqdm(file_list, desc=f"Copying to {Path(img_dest).parent.name}"):
        stem = img_path.stem
        lbl_path = Path(labels_input_dir) / f"{stem}.txt"
        mask_paths = list(Path(masks_input_dir).glob(f"{stem}_*.png"))  # Glob is used to find all mask files associated with the image stem

        # Check if all corresponding files exist
        if not lbl_path.exists():
            print(f"Warning: Label file missing for {img_path.name}, skipping.")
            continue
        if not mask_paths:
             if lbl_path.exists() and len(list(lbl_path.read_text().strip())) > 0 : # Check if label file is not empty
                 print(f"Warning: Mask file(s) missing for {img_path.name} but label exists, skipping.")
             continue

        try:
            shutil.copy2(str(img_path), img_dest)   # Copy image
            shutil.copy2(str(lbl_path), lbl_dest)   # Copy label
            for msk_path in mask_paths:             # Copy all masks
                shutil.copy2(str(msk_path), msk_dest)
            copied_count += 1
        except Exception as e:
            print(f"Error copying files for {img_path.name}: {e}")
    return copied_count

# --- Perform Copying ---
print("\nCopying training files")
train_copied = copy_files_for_split(train_files, train_img_dir, train_lbl_dir, train_msk_dir)

print("\nCopying validation files")
valid_copied = copy_files_for_split(valid_files, valid_img_dir, valid_lbl_dir, valid_msk_dir)

print("\nCopying testing files")
test_copied = copy_files_for_split(test_files, test_img_dir, test_lbl_dir, test_msk_dir)

print(f"\nFinished copying. Total copied: Train={train_copied}, Valid={valid_copied}, Test={test_copied}")

# --- Create data.yaml ---
yaml_path = os.path.join(DATASET_OUTPUT_DIR, "data.yaml")
data_yaml = {
    'path': Path(DATASET_OUTPUT_DIR).resolve().as_posix(), # Absolute path set to dataset root
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'nc': NUM_CLASSES,
    'names': CLASS_NAMES
}

try:
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False, default_flow_style=None)
    print(f"\nCreated data.yaml at: {yaml_path}")
except Exception as e:
    print(f"\nError creating data.yaml: {e}")

print("\nDataset splitting complete")