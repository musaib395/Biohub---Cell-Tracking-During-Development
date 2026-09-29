import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import CellTrackingDataset
from model import UNet3D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = "data/processed/val_images"
TARGET_DIR = "data/processed/val_targets"

# Possible paths where your trained model might be stored
POSSIBLE_PATHS = [
    "data/processed/best_model_weighted.pth",
    "data/processed/best_model.pth",
    "best_model_weighted.pth",
    "best_model.pth",
    "models/best_model_weighted.pth",
]

# Auto-locate model file if exact path isn't found
MODEL_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    # Find any .pth file in data/ or root directory
    found_pth = glob.glob("**/*.pth", recursive=True)
    if found_pth:
        MODEL_PATH = found_pth[0]
    else:
        raise FileNotFoundError(
            "No .pth model weight file found in project! Check your training script output path."
        )

print(f"[INFO] Loading model weights from: {MODEL_PATH}")

dataset = CellTrackingDataset(IMAGE_DIR, TARGET_DIR)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

model = UNet3D().to(DEVICE)

# Safely load weights whether saved as full dict or raw state_dict
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

distances_voxels = []
distances_um = []
scale = np.array([1.625, 0.40625, 0.40625])  # [Z, Y, X] anisotropy in µm

with torch.no_grad():
    for images, targets in loader:
        images = images.to(DEVICE)
        predictions = model(images)

        predictions = predictions.cpu().numpy()
        targets = targets.numpy()

        for prediction, target in zip(predictions, targets):
            prediction = prediction[0]
            target = target[0]

            pred_center = np.array(np.unravel_index(np.argmax(prediction), prediction.shape))
            target_center = np.array(np.unravel_index(np.argmax(target), target.shape))

            distance_voxel = np.linalg.norm(pred_center - target_center)
            distance_um = np.linalg.norm((pred_center - target_center) * scale)

            distances_voxels.append(distance_voxel)
            distances_um.append(distance_um)

        if len(distances_voxels) >= 100:
            break

distances_voxels = np.array(distances_voxels[:100])
distances_um = np.array(distances_um[:100])

print("\n" + "=" * 55)
print("       WEIGHTED MODEL DETECTION PERFORMANCE")
print("=" * 55)
print(f"Samples evaluated:        {len(distances_um)}")
print(f"Mean center distance:     {distances_voxels.mean():.2f} voxels")
print(f"Median center distance:   {np.median(distances_voxels):.2f} voxels")
print(f"Within 1 voxel:          {(distances_voxels <= 1).mean() * 100:.1f}%")
print(f"Within 2 voxels:         {(distances_voxels <= 2).mean() * 100:.1f}%")
print(f"Within 5 voxels:         {(distances_voxels <= 5).mean() * 100:.1f}%")
print("-" * 55)
print(f"Mean physical distance:   {distances_um.mean():.2f} µm")
print(f"Median physical distance: {np.median(distances_um):.2f} µm")
print(f"Within 2 µm:             {(distances_um <= 2).mean() * 100:.1f}%")
print(f"Within 5 µm:             {(distances_um <= 5).mean() * 100:.1f}%")
print(f"Within 7 µm:             {(distances_um <= 7).mean() * 100:.1f}%")
print(f"Maximum physical distance:{distances_um.max():.2f} µm")
print("=" * 55)