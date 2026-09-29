import os
import numpy as np
import pandas as pd
import zarr
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "train"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "val_samples.csv"
IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "val_images"
TARGET_DIR = PROJECT_ROOT / "data" / "processed" / "val_targets"

PATCH_SIZE = (32, 64, 64)
SIGMA = 2.0

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
TARGET_DIR.mkdir(parents=True, exist_ok=True)

val_df = pd.read_csv(VAL_CSV)

def normalize_patch(patch):
    p1 = np.percentile(patch, 1)
    p99 = np.percentile(patch, 99)
    patch = np.clip(patch, p1, p99)
    if p99 > p1:
        patch = (patch - p1) / (p99 - p1)
    else:
        patch = np.zeros_like(patch, dtype=np.float32)
    return patch.astype(np.float32)

def create_heatmap(shape, center, sigma):
    z, y, x = np.indices(shape)
    cz, cy, cx = center
    heatmap = np.exp(
        -((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) /
        (2 * sigma ** 2)
    )
    return heatmap.astype(np.float32)

def extract_patch(volume, center, patch_size):
    pz, py, px = patch_size
    cz, cy, cx = center

    z0 = cz - pz // 2
    y0 = cy - py // 2
    x0 = cx - px // 2

    z0 = max(0, min(z0, volume.shape[0] - pz))
    y0 = max(0, min(y0, volume.shape[1] - py))
    x0 = max(0, min(x0, volume.shape[2] - px))

    patch = volume[z0:z0+pz, y0:y0+py, x0:x0+px]

    return patch, (z0, y0, x0)

total = 0

for _, row in val_df.iterrows():
    sample_id = row["sample"]

    zarr_path = RAW_DIR / f"{sample_id}.zarr"
    geff_path = RAW_DIR / f"{sample_id}.geff"

    zarr_root = zarr.open_group(zarr_path, mode="r")
    volume = zarr_root["0"]

    geff_root = zarr.open_group(geff_path, mode="r")

    nodes = geff_root["nodes"]

    t_values = nodes["props"]["t"]["values"][:]
    x_values = nodes["props"]["x"]["values"][:]
    y_values = nodes["props"]["y"]["values"][:]
    z_values = nodes["props"]["z"]["values"][:]

    cells = pd.DataFrame({
        "t": t_values,
        "x": x_values,
        "y": y_values,
        "z": z_values
    })

    for t in sorted(cells["t"].unique()):
        frame_nodes = cells[cells["t"] == t]

        if len(frame_nodes) == 0:
            continue

        node = frame_nodes.iloc[0]

        center = (
            int(round(node["z"])),
            int(round(node["y"])),
            int(round(node["x"]))
        )

        frame = np.asarray(volume[int(t)])

        patch, origin = extract_patch(
            frame,
            center,
            PATCH_SIZE
        )

        normalized = normalize_patch(patch)

        local_center = (
            center[0] - origin[0],
            center[1] - origin[1],
            center[2] - origin[2]
        )

        target = create_heatmap(
            PATCH_SIZE,
            local_center,
            SIGMA
        )

        filename = f"{sample_id}_{int(t):04d}.npy"

        np.save(IMAGE_DIR / filename, normalized)
        np.save(TARGET_DIR / filename, target)

        total += 1

    print(f"Processed: {sample_id}")

print()
print(f"Validation samples: {len(val_df)}")
print(f"Validation patches generated: {total}")