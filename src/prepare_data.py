import os
import shutil
import random
from pathlib import Path
import stat  # <--- Add this import
# --- CONFIGURATION ---
SOURCE_DIR = "../bottle"  # The folder you extracted
TARGET_DIR = "../dataset"  # Where we will put the ready-to-train data
SPLIT_RATIO = 0.8  # 80% Training, 20% Validation


# ---------------------
def copy_and_overwrite(src, dst_folder):
    # Get the filename
    filename = os.path.basename(src)
    dst = os.path.join(dst_folder, filename)

    # If file exists, check if it's read-only and fix it
    if os.path.exists(dst):
        # Make it writable so we can delete it
        os.chmod(dst, stat.S_IWRITE)
        os.remove(dst)

    # Copy the new file
    shutil.copy(src, dst)
    # Ensure the NEW file is writable (fixes the issue for next time)
    os.chmod(dst, stat.S_IWRITE)
def setup_directories():
    for split in ['train', 'val']:
        for category in ['good', 'defective']:
            os.makedirs(os.path.join(TARGET_DIR, split, category), exist_ok=True)


def get_image_paths():
    good_images = []
    defective_images = []

    # 1. Collect all 'Good' images (from original train and test/good)
    for root, dirs, files in os.walk(SOURCE_DIR):
        if "good" in root:
            for file in files:
                if file.endswith(".png"):
                    good_images.append(os.path.join(root, file))

        # 2. Collect all 'Defective' images (anything in test that isn't 'good')
        elif "test" in root and "good" not in root:
            for file in files:
                if file.endswith(".png"):
                    defective_images.append(os.path.join(root, file))

    return good_images, defective_images


def split_and_copy(images, label):
    random.shuffle(images)
    split_point = int(len(images) * SPLIT_RATIO)

    train_imgs = images[:split_point]
    val_imgs = images[split_point:]

    print(f"Processing '{label}': {len(train_imgs)} training, {len(val_imgs)} validation")

    # Use the new function here instead of shutil.copy directly
    for img_path in train_imgs:
        copy_and_overwrite(img_path, os.path.join(TARGET_DIR, 'train', label))

    for img_path in val_imgs:
        copy_and_overwrite(img_path, os.path.join(TARGET_DIR, 'val', label))

if __name__ == "__main__":
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Could not find folder '{SOURCE_DIR}'. Did you extract the MVTec data?")
    else:
        print("Creating dataset structure...")
        setup_directories()

        print("Gathering file paths...")
        goods, defects = get_image_paths()

        print(f"Found {len(goods)} Good images and {len(defects)} Defective images.")

        # Note: MVTec usually has way more 'Good' images.
        # In a real scenario, you might want to limit the 'goods' to balance the classes.
        # For now, we use all of them.

        split_and_copy(goods, "good")
        split_and_copy(defects, "defective")

        print(f"\nSuccess! Data is ready in '{TARGET_DIR}/'")