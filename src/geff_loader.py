import zarr
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEFF_PATH = PROJECT_ROOT / "data" / "raw" / "train" / "44b6_0113de3b.geff"

root = zarr.open_group(GEFF_PATH, mode="r")

nodes = root["nodes"]

node_ids = nodes["ids"][:]
t = nodes["props"]["t"]["values"][:]
x = nodes["props"]["x"]["values"][:]
y = nodes["props"]["y"]["values"][:]
z = nodes["props"]["z"]["values"][:]

cells = pd.DataFrame({
    "cell_id": node_ids,
    "t": t,
    "x": x,
    "y": y,
    "z": z
})

edge_data = root["edges"]["ids"][:]

edges = pd.DataFrame(
    edge_data,
    columns=["source", "target"]
)

print("=" * 60)
print("CELL ANNOTATIONS")
print("=" * 60)
print(f"Number of cells: {len(cells)}")
print(f"Timepoints: {cells['t'].min()} to {cells['t'].max()}")
print()
print(cells.head(10))

print()
print("=" * 60)
print("CELL CONNECTIONS")
print("=" * 60)
print(f"Number of connections: {len(edges)}")
print()
print(edges.head(10))
cell_lookup = cells.set_index("cell_id")

edges["source_t"] = edges["source"].map(cell_lookup["t"])
edges["target_t"] = edges["target"].map(cell_lookup["t"])

edges["time_difference"] = edges["target_t"] - edges["source_t"]

print()
print("=" * 60)
print("EDGE ANALYSIS")
print("=" * 60)
print(edges["time_difference"].value_counts().sort_index())

outgoing = edges.groupby("source").size()
incoming = edges.groupby("target").size()

print()
print(f"Maximum outgoing connections: {outgoing.max()}")
print(f"Maximum incoming connections: {incoming.max()}")

print()
print("Possible division events:")

divisions = outgoing[outgoing > 1]

print(divisions)