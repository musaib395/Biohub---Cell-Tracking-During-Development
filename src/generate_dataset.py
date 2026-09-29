import zarr
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "train"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_SPLIT = PROCESSED_DIR / "train_samples.csv"

IMAGE_DIR = PROCESSED_DIR / "images"
TARGET_DIR = PROCESSED_DIR / "targets"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
TARGET_DIR.mkdir(parents=True, exist_ok=True)

samples = pd.read_csv(TRAIN_SPLIT)

PATCH_Z = 32
PATCH_Y = 64
PATCH_X = 64

patch_count = 0

for sample_index, row in samples.iterrows():

    sample = row["sample"]

    zarr_path = RAW_DIR / f"{sample}.zarr"
    geff_path = RAW_DIR / f"{sample}.geff"

    print(f"Processing {sample_index + 1}/{len(samples)}: {sample}")

    zarr_root = zarr.open_group(zarr_path, mode="r")
    data = zarr_root["0"]

    geff_root = zarr.open_group(geff_path, mode="r")
    nodes = geff_root["nodes"]

    cells = pd.DataFrame({
        "cell_id": nodes["ids"][:],
        "t": nodes["props"]["t"]["values"][:],
        "x": nodes["props"]["x"]["values"][:],
        "y": nodes["props"]["y"]["values"][:],
        "z": nodes["props"]["z"]["values"][:]
    })

    for timepoint in sorted(cells["t"].unique()):

        annotations = cells[cells["t"] == timepoint].copy()

        if len(annotations) == 0:
            continue

        center = annotations.iloc[0]

        cz = int(center["z"])
        cy = int(center["y"])
        cx = int(center["x"])

        z1 = max(0, cz - PATCH_Z // 2)
        y1 = max(0, cy - PATCH_Y // 2)
        x1 = max(0, cx - PATCH_X // 2)

        z2 = min(data.shape[1], z1 + PATCH_Z)
        y2 = min(data.shape[2], y1 + PATCH_Y)
        x2 = min(data.shape[3], x1 + PATCH_X)

        patch = data[timepoint, z1:z2, y1:y2, x1:x2].astype(np.float32)

        if patch.shape != (PATCH_Z, PATCH_Y, PATCH_X):
            continue

        low = np.percentile(patch, 1)
        high = np.percentile(patch, 99)

        if high <= low:
            continue

        patch = np.clip((patch - low) / (high - low), 0, 1)

        local_annotations = annotations[
            (annotations["x"] >= x1) &
            (annotations["x"] < x2) &
            (annotations["y"] >= y1) &
            (annotations["y"] < y2) &
            (annotations["z"] >= z1) &
            (annotations["z"] < z2)
        ]

        target = np.zeros(patch.shape, dtype=np.float32)

        for _, cell in local_annotations.iterrows():

            local_z = int(cell["z"]) - z1
            local_y = int(cell["y"]) - y1
            local_x = int(cell["x"]) - x1

            target[local_z, local_y, local_x] = 1.0

        target = gaussian_filter(
            target,
            sigma=(2, 3, 3)
        )

        if target.max() > 0:
            target = target / target.max()

        sample_id = f"{sample}_t{int(timepoint):03d}"

        np.save(
            IMAGE_DIR / f"{sample_id}.npy",
            patch.astype(np.float32)
        )

        np.save(
            TARGET_DIR / f"{sample_id}.npy",
            target.astype(np.float32)
        )

        patch_count += 1

print()
print("=" * 60)
print("DATASET GENERATION COMPLETE")
print("=" * 60)
print(f"Training samples processed: {len(samples)}")
print(f"Training patches generated: {patch_count}")
print(f"Image directory: {IMAGE_DIR}")
print(f"Target directory: {TARGET_DIR}")