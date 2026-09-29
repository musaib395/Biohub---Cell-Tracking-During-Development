import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "images"
TARGET_DIR = PROJECT_ROOT / "data" / "processed" / "targets"

image_files = sorted(IMAGE_DIR.glob("*.npy"))
target_files = sorted(TARGET_DIR.glob("*.npy"))

image_names = {f.name for f in image_files}
target_names = {f.name for f in target_files}

print("=" * 60)
print("PROCESSED DATASET VALIDATION")
print("=" * 60)

print(f"Image files: {len(image_files)}")
print(f"Target files: {len(target_files)}")
print(f"Matching pairs: {len(image_names & target_names)}")

missing_targets = image_names - target_names
missing_images = target_names - image_names

print(f"Missing targets: {len(missing_targets)}")
print(f"Missing images: {len(missing_images)}")

sample_count = min(100, len(image_files))

valid_images = 0
valid_targets = 0
non_empty_targets = 0
shape_errors = 0
value_errors = 0

for image_file in image_files[:sample_count]:
    target_file = TARGET_DIR / image_file.name

    if not target_file.exists():
        continue

    try:
        image = np.load(image_file, mmap_mode="r")
        target = np.load(target_file, mmap_mode="r")

        if image.shape == (32, 64, 64) and target.shape == (32, 64, 64):
            valid_images += 1
            valid_targets += 1
        else:
            shape_errors += 1

        if image.min() >= 0 and image.max() <= 1:
            pass
        else:
            value_errors += 1

        if target.max() > 0:
            non_empty_targets += 1

    except Exception as e:
        print(f"Error: {image_file.name}")
        print(e)

print()
print("=" * 60)
print("SAMPLE VALIDATION")
print("=" * 60)
print(f"Files checked: {sample_count}")
print(f"Valid image shapes: {valid_images}")
print(f"Valid target shapes: {valid_targets}")
print(f"Non-empty targets: {non_empty_targets}")
print(f"Shape errors: {shape_errors}")
print(f"Image value errors: {value_errors}")

if image_files:
    image = np.load(image_files[0], mmap_mode="r")
    target = np.load(TARGET_DIR / image_files[0].name, mmap_mode="r")

    print()
    print("=" * 60)
    print("FIRST SAMPLE")
    print("=" * 60)
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image range: {image.min():.4f} - {image.max():.4f}")
    print(f"Target shape: {target.shape}")
    print(f"Target dtype: {target.dtype}")
    print(f"Target range: {target.min():.4f} - {target.max():.4f}")

print()
print("=" * 60)
print("VALIDATION FINISHED")
print("=" * 60)