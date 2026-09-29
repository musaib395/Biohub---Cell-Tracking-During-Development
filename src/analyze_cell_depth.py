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

z_start = max(0, z - 10)
z_end = min(64, z + 11)

values = []

for current_z in range(z_start, z_end):
    patch = data[t, current_z, max(0, y - 5):min(256, y + 6), max(0, x - 5):min(256, x + 6)]
    values.append(patch.mean())

zs = list(range(z_start, z_end))

print("=" * 60)
print("CELL DEPTH INTENSITY PROFILE")
print("=" * 60)
print(f"Cell ID: {cell['cell_id']}")
print(f"Time: {t}")
print(f"Center: X={x}, Y={y}, Z={z}")
print()

for current_z, value in zip(zs, values):
    print(f"Z={current_z:2d}  Mean intensity={value:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(zs, values, marker="o")
plt.axvline(z, linestyle="--")
plt.xlabel("Z Slice")
plt.ylabel("Mean Local Intensity")
plt.title("Cell Intensity Across Z Slices")
plt.grid()
plt.show()