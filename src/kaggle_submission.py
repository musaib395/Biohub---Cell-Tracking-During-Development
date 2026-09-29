import os
import glob
import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment

# Kaggle input paths
KAGGLE_TEST_DIR = "/kaggle/input/competitions/biohub-cell-tracking-during-development/test"
LOCAL_TEST_DIR = "data/test"
SUBMISSION_PATH = "submission.csv"

TEST_DIR = KAGGLE_TEST_DIR if os.path.exists(KAGGLE_TEST_DIR) else LOCAL_TEST_DIR
SCALE_ZYX = np.array([1.625, 0.40625, 0.40625])  # um per voxel

print("=" * 65)
print("       KAGGLE SUBMISSION GENERATOR")
print("=" * 65)


def extract_voxels_3d(vol, thresh=0.18, min_dist=2):
    """Extract 3D local maximum peaks from raw Zarr volume."""
    smoothed = gaussian_filter(vol.astype(np.float32), sigma=1.0)
    max_filtered = maximum_filter(smoothed, size=min_dist * 2 + 1)
    mask = (smoothed == max_filtered) & (smoothed > (thresh * smoothed.max()))
    coords_voxels = np.array(np.where(mask)).T  # (N, 3) -> Z, Y, X
    return coords_voxels


def process_dataset(zarr_path, dataset_name):
    """Process single 3D+t movie and generate node/edge rows."""
    z_group = zarr.open(zarr_path, mode="r")
    vol_4d = z_group["0"]  # Shape: (T, Z, Y, X)
    num_frames = vol_4d.shape[0]

    nodes = []
    edges = []

    global_node_id = 1
    prev_nodes = None  # [(node_id, voxel_coord), ...]

    for t in range(num_frames):
        vol_3d = np.array(vol_4d[t])
        curr_voxels = extract_voxels_3d(vol_3d)

        curr_nodes = []
        for v in curr_voxels:
            node_info = {
                "dataset": dataset_name,
                "row_type": "node",
                "node_id": global_node_id,
                "t": int(t),
                "z": int(v[0]),
                "y": int(v[1]),
                "x": int(v[2]),
                "source_id": -1,
                "target_id": -1,
            }
            nodes.append(node_info)
            curr_nodes.append((global_node_id, v))
            global_node_id += 1

        # Frame-to-frame linking via Hungarian Matching
        if prev_nodes and curr_nodes:
            prev_coords = np.array([p[1] for p in prev_nodes]) * SCALE_ZYX
            curr_coords = np.array([c[1] for c in curr_nodes]) * SCALE_ZYX

            # Distance matrix in physical micrometers
            dist_matrix = np.linalg.norm(
                prev_coords[:, None, :] - curr_coords[None, :, :], axis=2
            )
            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            for r, c in zip(row_ind, col_ind):
                if dist_matrix[r, c] <= 8.0:  # Distance cutoff gating (um)
                    src_id = prev_nodes[r][0]
                    tgt_id = curr_nodes[c][0]
                    edges.append({
                        "dataset": dataset_name,
                        "row_type": "edge",
                        "node_id": -1,
                        "t": -1,
                        "z": -1,
                        "y": -1,
                        "x": -1,
                        "source_id": src_id,
                        "target_id": tgt_id,
                    })

        prev_nodes = curr_nodes

    return nodes, edges


all_rows = []
row_idx = 0

zarr_files = glob.glob(os.path.join(TEST_DIR, "*.zarr"))
print(f"Found {len(zarr_files)} test samples in '{TEST_DIR}'")

for zpath in zarr_files:
    dset_name = os.path.basename(zpath).replace(".zarr", "")
    print(f"Processing sample: {dset_name}...")

    nodes, edges = process_dataset(zpath, dset_name)

    for n in nodes:
        all_rows.append({"id": row_idx, **n})
        row_idx += 1

    for e in edges:
        all_rows.append({"id": row_idx, **e})
        row_idx += 1

sub_df = pd.DataFrame(all_rows)
sub_df.to_csv(SUBMISSION_PATH, index=False)

print("\n" + "=" * 65)
print(f"SUCCESS! Submission saved to '{SUBMISSION_PATH}' with {len(sub_df)} rows.")
print("=" * 65)