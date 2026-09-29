import zarr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZARR_PATH = PROJECT_ROOT / "data" / "raw" / "train" / "44b6_0113de3b.zarr"
GEFF_PATH = PROJECT_ROOT / "data" / "raw" / "train" / "44b6_0113de3b.geff"

root = zarr.open_group(ZARR_PATH, mode="r")
data = root["0"]

geff = zarr.open_group(GEFF_PATH, mode="r")
nodes = geff["nodes"]

cells = pd.DataFrame({
    "cell_id": nodes["ids"][:],
    "t": nodes["props"]["t"]["values"][:],
    "x": nodes["props"]["x"]["values"][:],
    "y": nodes["props"]["y"]["values"][:],
    "z": nodes["props"]["z"]["values"][:]
})

timepoint = 0
cells_t = cells[cells["t"] == timepoint]

fig, axes = plt.subplots(1, len(cells_t), figsize=(15, 5))

if len(cells_t) == 1:
    axes = [axes]

for ax, (_, cell) in zip(axes, cells_t.iterrows()):
    z = int(cell["z"])

    ax.imshow(data[timepoint, z], cmap="gray")
    ax.scatter(cell["x"], cell["y"], s=80)
    ax.text(cell["x"], cell["y"], str(cell["cell_id"]), fontsize=8)

    ax.set_title(f"Z={z}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

plt.tight_layout()
plt.show()