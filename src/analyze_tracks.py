import zarr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEFF_PATH = PROJECT_ROOT / "data" / "raw" / "train" / "44b6_0113de3b.geff"

root = zarr.open_group(GEFF_PATH, mode="r")
nodes = root["nodes"]

cells = pd.DataFrame({
    "cell_id": nodes["ids"][:],
    "t": nodes["props"]["t"]["values"][:],
    "x": nodes["props"]["x"]["values"][:],
    "y": nodes["props"]["y"]["values"][:],
    "z": nodes["props"]["z"]["values"][:]
})

counts = cells.groupby("t").size()

print("=" * 60)
print("TEMPORAL CELL DISTRIBUTION")
print("=" * 60)
print(f"Total annotated cells: {len(cells)}")
print(f"Annotated timepoints: {len(counts)}")
print(f"Minimum cells in a timepoint: {counts.min()}")
print(f"Maximum cells in a timepoint: {counts.max()}")
print(f"Average cells per annotated timepoint: {counts.mean():.2f}")

print()
print(counts)

plt.figure(figsize=(12, 5))
plt.plot(counts.index, counts.values, marker="o")
plt.xlabel("Timepoint")
plt.ylabel("Number of annotated cells")
plt.title("Cells per Timepoint")
plt.grid()
plt.show()