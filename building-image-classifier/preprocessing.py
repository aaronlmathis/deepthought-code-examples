import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, UnidentifiedImageError

# Define paths
BASE_DIR = Path("cifar-10")
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
LABELS_CSV = BASE_DIR / "trainLabels.csv"
OUTPUT_TRAIN_FILE = Path("processed","prepared_train_dataset.pt")

def load_label_mapping(labels_csv=LABELS_CSV):
    """Loads the label mapping from the CSV file."""
    df = pd.read_csv(labels_csv)

    # Create a label map
    label_map = {row['id']: row['label'] for _, row in df.iterrows()}

    # Get sorted unique class names and assign integer IDs
    class_names = sorted(set(label_map.values()))
    class_to_idx = {label: idx for idx, label in enumerate(class_names)}


    print(f"Loaded {len(label_map)} labels across {len(class_names)} classes.")
    return label_map, class_to_idx, class_names

def is_image_valid(path: str) -> bool:
    try:
        img = Image.open(path)
        img.verify()  # Verifies without loading full image into memory
        return True
    except (UnidentifiedImageError, IOError):
        return False
    
def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold

def pad_and_resize(image: np.ndarray, target_size: int = 32) -> np.ndarray:
    h, w, _ = image.shape
    diff = abs(h - w)

    # Compute padding
    pad1, pad2 = diff // 2, diff - diff // 2
    if h > w:
        # Image is taller than wide: pad width
        padded = cv2.copyMakeBorder(image, 0, 0, pad1, pad2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        # Image is wider than tall: pad height
        padded = cv2.copyMakeBorder(image, pad1, pad2, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Resize to target size
    resized = cv2.resize(padded, (target_size, target_size))
    return resized

def preprocess_train_set():
    print("Loading label map...")
    label_map, class_to_idx, class_names = load_label_mapping()

    X, y = [], []
    total = len(label_map)
    print(f"Found {total} labeled training images.")

    for i, (img_id, label) in enumerate(label_map.items(), start=1):
        img_path = os.path.join(TRAIN_DIR, f"{img_id}.png")

        if not os.path.exists(img_path):
            print(f"[{i}/{total}] Skipping missing: {img_path}")
            continue
        if not is_image_valid(img_path):
            print(f"[{i}/{total}] Skipping corrupted: {img_path}")
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        img = pad_and_resize(img)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # Convert HWC to CHW
        X.append(torch.tensor(img))
        y.append(class_to_idx[label])

        if i % 500 == 0:
            print(f"Processed {i} images...")

    X_tensor = torch.stack(X)
    y_tensor = torch.tensor(y, dtype=torch.long)

    print(f"Saving torch dataset to {OUTPUT_TRAIN_FILE}...")
    torch.save((X_tensor, y_tensor), OUTPUT_TRAIN_FILE)

    print(f"Done. Total usable images: {X_tensor.shape[0]}")
    return X_tensor, y_tensor, class_names



if __name__ == "__main__":

    preprocess_train_set()