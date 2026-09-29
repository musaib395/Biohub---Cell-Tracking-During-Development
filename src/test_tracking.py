import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import CellTrackingDataset
from model import UNet3D
from tracking import extract_cell_centers_3d, link_two_frames

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = "data/processed/val_images"
TARGET_DIR = "data/processed/val_targets"
MODEL_PATH = "data/processed/best_model.pth"

print("=" * 60)
print("       RUNNING 3D CENTROID EXTRACTION & TRACKING TEST")
print("=" * 60)

# Check directory contents
val_files = glob.glob(os.path.join(IMAGE_DIR, "*"))
print(f"[INFO] Found {len(val_files)} files in validation directory: {IMAGE_DIR}")

if len(val_files) == 0:
    raise FileNotFoundError(f"No validation image files found in {IMAGE_DIR}!")

# Load Dataset
dataset = CellTrackingDataset(IMAGE_DIR, TARGET_DIR)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
print(f"[INFO] DataLoader initialized with {len(dataset)} samples.")

# Load Model
model = UNet3D().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

prev_coords_voxels = None
scale = np.array([1.625, 0.40625, 0.40625])  # [Z, Y, X] anisotropy in µm

with torch.no_grad():
    for step, batch in enumerate(loader):
        if step >= 5:  # Evaluate across 5 sequential validation volumes
            break

        # Handle various PyTorch Dataset return formats
        if isinstance(batch, (list, tuple)):
            image = batch[0]
        else:
            image = batch

        image = image.to(DEVICE)
        pred = model(image)[0, 0].cpu().numpy()

        # Step 1: Extract 3D Cell Centers via Local Maxima
        coords_voxels, coords_um = extract_cell_centers_3d(
            pred, threshold=0.25, min_distance=3
        )
        print(f"\n[Timepoint t={step}] Detected {len(coords_voxels)} cell centers.")

        # Step 2: Link with previous frame if available
        if prev_coords_voxels is not None:
            matches, unmatched_t0, unmatched_t1 = link_two_frames(
                prev_coords_voxels,
                coords_voxels,
                max_distance_um=10.0,
                scale=scale,
            )
            print(f"  └── Tracked Links (t={step-1} -> t={step}): {len(matches)}")
            print(f"  └── Lost Tracks (Disappearances):     {len(unmatched_t0)}")
            print(f"  └── New Tracks (Births/Detections):   {len(unmatched_t1)}")

            if len(matches) > 0:
                dists = [
                    np.linalg.norm(
                        (prev_coords_voxels[m[0]] - coords_voxels[m[1]]) * scale
                    )
                    for m in matches
                ]
                print(f"  └── Mean Displacement:               {np.mean(dists):.2f} µm")

        prev_coords_voxels = coords_voxels

print("\n" + "=" * 60)
print("Tracking test complete.")
print("=" * 60)