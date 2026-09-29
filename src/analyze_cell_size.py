import zarr
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

radius_xy = 15
z_offsets = [-10, -5, 0, 5, 10]

x1 = max(0, x - radius_xy)
x2 = min(256, x + radius_xy + 1)
y1 = max(0, y - radius_xy)
y2 = min(256, y + radius_xy + 1)

fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for ax, offset in zip(axes, z_offsets):
    current_z = z + offset
    image = data[t, current_z, y1:y2, x1:x2]

    ax.imshow(image, cmap="gray")
    ax.scatter(x - x1, y - y1, s=60)
    ax.set_title(f"Z = {current_z}")
    ax.axis("off")

plt.tight_layout()
plt.show()

print("=" * 60)
print("3D CELL LOCAL ANALYSIS")
print("=" * 60)
print(f"Cell ID: {cell['cell_id']}")
print(f"Time: {t}")
print(f"Center: X={x}, Y={y}, Z={z}")