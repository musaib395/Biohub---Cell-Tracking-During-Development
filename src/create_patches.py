import zarr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

cell = cells.iloc[0]

t = int(cell["t"])
x = int(cell["x"])
y = int(cell["y"])
z = int(cell["z"])

patch_z = 32
patch_y = 64
patch_x = 64

z1 = max(0, z - patch_z // 2)
z2 = min(data.shape[1], z1 + patch_z)

y1 = max(0, y - patch_y // 2)
y2 = min(data.shape[2], y1 + patch_y)

x1 = max(0, x - patch_x // 2)
x2 = min(data.shape[3], x1 + patch_x)

patch = data[t, z1:z2, y1:y2, x1:x2]

local_z = z - z1
local_y = y - y1
local_x = x - x1

print("=" * 60)
print("3D PATCH EXTRACTION")
print("=" * 60)
print(f"Cell ID: {cell['cell_id']}")
print(f"Timepoint: {t}")
print(f"Original coordinate: X={x}, Y={y}, Z={z}")
print(f"Patch shape: {patch.shape}")
print(f"Local coordinate: X={local_x}, Y={local_y}, Z={local_z}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(patch[local_z], cmap="gray")
axes[0].scatter(local_x, local_y, s=80)
axes[0].set_title(f"Center Z={local_z}")
axes[0].axis("off")

axes[1].imshow(patch[:, local_y, :], cmap="gray", aspect="auto")
axes[1].axvline(local_x)
axes[1].axhline(local_z)
axes[1].set_title("XZ View")
axes[1].axis("off")

axes[2].imshow(patch[:, :, local_x], cmap="gray", aspect="auto")
axes[2].axvline(local_y)
axes[2].axhline(local_z)
axes[2].set_title("YZ View")
axes[2].axis("off")

plt.tight_layout()
plt.show()