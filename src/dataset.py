import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class CellTrackingDataset(Dataset):
    def __init__(self, image_dir, target_dir):
        self.image_dir = Path(image_dir)
        self.target_dir = Path(target_dir)

        self.images = sorted(self.image_dir.glob("*.npy"))

        self.targets = [
            self.target_dir / image.name
            for image in self.images
        ]

        valid_pairs = [
            (image, target)
            for image, target in zip(self.images, self.targets)
            if target.exists()
        ]

        self.images = [pair[0] for pair in valid_pairs]
        self.targets = [pair[1] for pair in valid_pairs]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.load(self.images[index]).astype(np.float32)
        target = np.load(self.targets[index]).astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)
        target = torch.from_numpy(target).unsqueeze(0)

        return image, target

TRAIN_IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "images"
TRAIN_TARGET_DIR = PROJECT_ROOT / "data" / "processed" / "targets"

VAL_IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "val_images"
VAL_TARGET_DIR = PROJECT_ROOT / "data" / "processed" / "val_targets"

train_dataset = CellTrackingDataset(
    TRAIN_IMAGE_DIR,
    TRAIN_TARGET_DIR
)

val_dataset = CellTrackingDataset(
    VAL_IMAGE_DIR,
    VAL_TARGET_DIR
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

train_images, train_targets = next(iter(train_loader))
val_images, val_targets = next(iter(val_loader))

print("=" * 60)
print("TRAINING DATASET")
print("=" * 60)
print(f"Dataset size: {len(train_dataset)}")
print(f"Image batch shape: {train_images.shape}")
print(f"Target batch shape: {train_targets.shape}")
print(f"Image range: {train_images.min():.4f} - {train_images.max():.4f}")
print(f"Target range: {train_targets.min():.4f} - {train_targets.max():.4f}")

print()
print("=" * 60)
print("VALIDATION DATASET")
print("=" * 60)
print(f"Dataset size: {len(val_dataset)}")
print(f"Image batch shape: {val_images.shape}")
print(f"Target batch shape: {val_targets.shape}")
print(f"Image range: {val_images.min():.4f} - {val_images.max():.4f}")
print(f"Target range: {val_targets.min():.4f} - {val_targets.max():.4f}")