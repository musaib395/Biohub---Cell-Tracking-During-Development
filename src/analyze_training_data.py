import zarr
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train"

def analyze_geff(path):
    root = zarr.open_group(path, mode="r")
    nodes = root["nodes"]

    node_ids = nodes["ids"][:]
    times = nodes["props"]["t"]["values"][:]
    edges = root["edges"]["ids"][:]

    cells = pd.DataFrame({
        "cell_id": node_ids,
        "t": times
    })

    edge_df = pd.DataFrame(edges, columns=["source", "target"])

    outgoing = edge_df.groupby("source").size()
    incoming = edge_df.groupby("target").size()

    return {
        "sample": path.stem,
        "nodes": len(cells),
        "edges": len(edges),
        "timepoints": cells["t"].nunique(),
        "max_cells": cells.groupby("t").size().max(),
        "avg_cells": cells.groupby("t").size().mean(),
        "divisions": int((outgoing > 1).sum()),
        "merges": int((incoming > 1).sum())
    }

geff_files = sorted(TRAIN_DIR.glob("*.geff"))

results = []

for path in geff_files:
    try:
        results.append(analyze_geff(path))
    except Exception as e:
        print(f"Error: {path.name} - {e}")

df = pd.DataFrame(results)

print("=" * 70)
print("BIOHUB TRAINING DATASET OVERVIEW")
print("=" * 70)

print(f"Training samples: {len(df)}")
print(f"Total nodes: {df['nodes'].sum():,}")
print(f"Total edges: {df['edges'].sum():,}")
print(f"Total divisions: {df['divisions'].sum():,}")
print(f"Total merges: {df['merges'].sum():,}")

print()
print("=" * 70)
print("NODES PER SAMPLE")
print("=" * 70)
print(f"Minimum: {df['nodes'].min():,}")
print(f"Maximum: {df['nodes'].max():,}")
print(f"Mean: {df['nodes'].mean():.2f}")
print(f"Median: {df['nodes'].median():.2f}")

print()
print("=" * 70)
print("CELLS PER TIMEPOINT")
print("=" * 70)
print(f"Minimum maximum: {df['max_cells'].min()}")
print(f"Maximum maximum: {df['max_cells']}")
print(f"Average maximum: {df['max_cells'].mean():.2f}")

print()
print("=" * 70)
print("DIVISION ANALYSIS")
print("=" * 70)
print(f"Samples with divisions: {(df['divisions'] > 0).sum()}")
print(f"Samples without divisions: {(df['divisions'] == 0).sum()}")

print()
print("Samples with most divisions:")
print(df.nlargest(10, "divisions")[["sample", "nodes", "max_cells", "divisions"]].to_string(index=False))

output_path = PROJECT_ROOT / "outputs" / "training_data_summary.csv"
df.to_csv(output_path, index=False)

print()
print(f"Saved: {output_path}")