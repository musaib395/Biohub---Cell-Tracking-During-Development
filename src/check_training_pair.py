import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "val_images"
TARGET_DIR = PROJECT_ROOT / "data" / "processed" / "val_targets"

image_files = sorted(IMAGE_DIR.glob("*.npy"))

index = 0

image_path = image_files[index]
target_path = TARGET_DIR / image_path.name

image = np.load(image_path)
target = np.load(target_path)

target_center = np.unravel_index(
    np.argmax(target),
    target.shape
)

image_max = np.unravel_index(
    np.argmax(image),
    image.shape
)

z_target, y_target, x_target = target_center
z_image, y_image, x_image = image_max

print("=" * 60)
print("TRAINING PAIR DIAGNOSTIC")
print("=" * 60)
print(f"File: {image_path.name}")
print(f"Image shape: {image.shape}")
print(f"Target shape: {target.shape}")
print(f"Ground-truth center: {target_center}")
print(f"Maximum intensity location: {image_max}")
print(f"Maximum intensity value: {image.max():.4f}")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes[0, 0].imshow(image[z_target], cmap="gray")
axes[0, 0].scatter(x_target, y_target, marker="x", s=100)
axes[0, 0].set_title("Image + Ground Truth")

axes[0, 1].imshow(target[z_target])
axes[0, 1].set_title("Ground Truth Heatmap")

axes[0, 2].imshow(image[z_target], cmap="gray")
axes[0, 2].scatter(x_target, y_target, marker="x", s=100)
axes[0, 2].scatter(x_image, y_image, marker="o", s=80)
axes[0, 2].set_title("Ground Truth + Brightest Pixel")

axes[1, 0].imshow(image[z_image], cmap="gray")
axes[1, 0].scatter(x_image, y_image, marker="x", s=100)
axes[1, 0].set_title("Brightest Image Slice")

axes[1, 1].imshow(image[z_target])
axes[1, 1].set_title("Image at Target Z")

axes[1, 2].imshow(target[z_target])
axes[1, 2].set_title("Target at Target Z")

plt.tight_layout()
plt.show()