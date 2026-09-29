import zarr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZARR_PATH = PROJECT_ROOT / "data" / "raw" / "train" / "44b6_12dfb391.zarr"
GEFF_PATH = PROJECT_ROOT / "data" / "raw" / "train" / "44b6_12dfb391.geff"

zarr_root = zarr.open_group(ZARR_PATH, mode="r")
data = zarr_root["0"]

geff_root = zarr.open_group(GEFF_PATH, mode="r")
nodes = geff_root["nodes"]

cells = pd.DataFrame({
    "cell_id": nodes["ids"][:],
    "t": nodes["props"]["t"]["values"][:],
    "x": nodes["props"]["x"]["values"][:],
    "y": nodes["props"]["y"]["values"][:],
    "z": nodes["props"]["z"]["values"][:]
})

timepoint = 0
annotations = cells[cells["t"] == timepoint].copy()

patch_z = 32
patch_y = 64
patch_x = 64

center = annotations.iloc[0]

cz = int(center["z"])
cy = int(center["y"])
cx = int(center["x"])

z1 = max(0, cz - patch_z // 2)
y1 = max(0, cy - patch_y // 2)
x1 = max(0, cx - patch_x // 2)

z2 = min(data.shape[1], z1 + patch_z)
y2 = min(data.shape[2], y1 + patch_y)
x2 = min(data.shape[3], x1 + patch_x)

patch = data[timepoint, z1:z2, y1:y2, x1:x2].astype(np.float32)

low = np.percentile(patch, 1)
high = np.percentile(patch, 99)

patch = np.clip((patch - low) / (high - low), 0, 1)

local_annotations = annotations[
    (annotations["x"] >= x1) &
    (annotations["x"] < x2) &
    (annotations["y"] >= y1) &
    (annotations["y"] < y2) &
    (annotations["z"] >= z1) &
    (annotations["z"] < z2)
]

target = np.zeros(patch.shape, dtype=np.float32)

for _, cell in local_annotations.iterrows():
    local_z = int(cell["z"]) - z1
    local_y = int(cell["y"]) - y1
    local_x = int(cell["x"]) - x1

    target[local_z, local_y, local_x] = 1.0

target = gaussian_filter(
    target,
    sigma=(2, 3, 3)
)

if target.max() > 0:
    target = target / target.max()

center_z = int(center["z"]) - z1

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(patch[center_z], cmap="gray")

for _, cell in local_annotations.iterrows():
    axes[0].scatter(
        int(cell["x"]) - x1,
        int(cell["y"]) - y1,
        s=80
    )

axes[0].set_title("Input Patch")
axes[0].axis("off")

axes[1].imshow(target[center_z], cmap="hot")
axes[1].set_title("Target Heatmap")
axes[1].axis("off")

plt.tight_layout()
plt.show()

print("=" * 60)
print("TRAINING PAIR")
print("=" * 60)
print(f"Timepoint: {timepoint}")
print(f"Input shape: {patch.shape}")
print(f"Target shape: {target.shape}")
print(f"Annotations in full frame: {len(annotations)}")
print(f"Annotations in patch: {len(local_annotations)}")

print()
print("Local cell coordinates:")

for _, cell in local_annotations.iterrows():
    print(
        f"X={int(cell['x']) - x1}, "
        f"Y={int(cell['y']) - y1}, "
        f"Z={int(cell['z']) - z1}"
    )
    