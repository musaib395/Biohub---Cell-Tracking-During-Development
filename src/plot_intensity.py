import zarr
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZARR_PATH = PROJECT_ROOT / "data" / "raw" / "train" / "44b6_0113de3b.zarr"

root = zarr.open_group(ZARR_PATH, mode="r")
data = root["0"]

volume = data[0]

plt.figure(figsize=(10, 5))
plt.hist(volume.ravel(), bins=100)
plt.xlabel("Intensity")
plt.ylabel("Pixel Count")
plt.title("Intensity Distribution")
plt.show()