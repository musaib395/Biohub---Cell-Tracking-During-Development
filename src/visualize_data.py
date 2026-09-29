import zarr
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZARR_PATH = PROJECT_ROOT / "data" / "raw" / "train" / "44b6_0113de3b.zarr"

root = zarr.open_group(ZARR_PATH, mode="r")
data = root["0"]

volume = data[0]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(volume[10], cmap="gray")
axes[0].set_title("Z Slice 10")
axes[0].axis("off")

axes[1].imshow(volume[30], cmap="gray")
axes[1].set_title("Z Slice 30")
axes[1].axis("off")

axes[2].imshow(volume[50], cmap="gray")
axes[2].set_title("Z Slice 50")
axes[2].axis("off")

plt.tight_layout()
plt.show()