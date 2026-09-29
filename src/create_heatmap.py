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
frame = data[timepoint]

annotations = cells[cells["t"] == timepoint]

target = np.zeros(frame.shape, dtype=np.float32)

for _, cell in annotations.iterrows():
    z = int(cell["z"])
    y = int(cell["y"])
    x = int(cell["x"])

    if 0 <= z < target.shape[0] and 0 <= y < target.shape[1] and 0 <= x < target.shape[2]:
        target[z, y, x] = 1.0

heatmap = gaussian_filter(
    target,
    sigma=(2, 3, 3)
)

if heatmap.max() > 0:
    heatmap = heatmap / heatmap.max()

center_z = int(annotations.iloc[0]["z"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(frame[center_z], cmap="gray")
axes[0].set_title(f"Original Image - Z={center_z}")
axes[0].axis("off")

axes[1].imshow(heatmap[center_z], cmap="hot")
axes[1].set_title(f"Cell Center Heatmap - Z={center_z}")
axes[1].axis("off")

plt.tight_layout()
plt.show()

print("=" * 60)
print("3D CELL CENTER HEATMAP")
print("=" * 60)
print(f"Timepoint: {timepoint}")
print(f"Annotated cells: {len(annotations)}")
print(f"Image shape: {frame.shape}")
print(f"Heatmap shape: {heatmap.shape}")
print(f"Heatmap minimum: {heatmap.min():.6f}")
print(f"Heatmap maximum: {heatmap.max():.6f}")