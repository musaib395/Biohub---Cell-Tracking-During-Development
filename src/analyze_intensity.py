import zarr
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZARR_PATH = PROJECT_ROOT / "data" / "raw" / "train" / "44b6_0113de3b.zarr"

root = zarr.open_group(ZARR_PATH, mode="r")
data = root["0"]

volume = data[0]

print("=" * 60)
print("IMAGE INTENSITY ANALYSIS")
print("=" * 60)
print(f"Shape: {volume.shape}")
print(f"Dtype: {volume.dtype}")
print(f"Minimum: {volume.min()}")
print(f"Maximum: {volume.max()}")
print(f"Mean: {volume.mean():.2f}")
print(f"Median: {np.median(volume):.2f}")
print(f"Standard deviation: {volume.std():.2f}")

percentiles = np.percentile(volume, [1, 5, 25, 50, 75, 95, 99])

print()
print("Percentiles:")
for p, value in zip([1, 5, 25, 50, 75, 95, 99], percentiles):
    print(f"{p}%: {value:.2f}")