import zarr
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

ZARR_PATH = DATA_DIR / "train" / "44b6_0113de3b.zarr"

def inspect_zarr(group, level=0):
    indent = "    " * level

    for key in group.keys():
        item = group[key]

        if isinstance(item, zarr.Array):
            print(f"{indent}[ARRAY] {key}")
            print(f"{indent}Shape: {item.shape}")
            print(f"{indent}Dtype: {item.dtype}")
        else:
            print(f"{indent}[GROUP] {key}")
            inspect_zarr(item, level + 1)

if __name__ == "__main__":
    print("=" * 60)
    print("ZARR DATASET INSPECTION")
    print("=" * 60)
    print(f"Path: {ZARR_PATH}")

    root = zarr.open(ZARR_PATH, mode="r")

    print("\nZARR STRUCTURE")
    print("-" * 60)

    inspect_zarr(root)