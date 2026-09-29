import os
import csv
import torch
import numpy as np
import networkx as nx
from tracking import extract_cell_centers_3d, link_two_frames, detect_cell_divisions

SYNTHETIC_DIR = "data/processed/synthetic_series"
CSV_OUTPUT_PATH = "data/processed/lineage_tracks.csv"

scale = np.array([1.625, 0.40625, 0.40625])

print("=" * 65)
print("       END-TO-END 3D CELL TRACKING & LINEAGE EXPORT")
print("=" * 65)

lineage_graph = nx.DiGraph()
next_global_track_id = 0
prev_coords = None
prev_track_ids = []
csv_rows = []

files = sorted([f for f in os.listdir(SYNTHETIC_DIR) if f.endswith(".pt")])

for step, file_name in enumerate(files):
    file_path = os.path.join(SYNTHETIC_DIR, file_name)
    pred = torch.load(file_path)[0, 0].numpy()

    coords_voxels, coords_um = extract_cell_centers_3d(
        pred, threshold=0.20, min_distance=3
    )

    current_track_ids = [-1] * len(coords_voxels)

    if step == 0:
        for idx in range(len(coords_voxels)):
            current_track_ids[idx] = next_global_track_id
            lineage_graph.add_node(
                next_global_track_id, time=step, pos=coords_um[idx].tolist()
            )
            csv_rows.append([next_global_track_id, step, *coords_um[idx], -1])
            next_global_track_id += 1
    else:
        matches, unmatched_t0, unmatched_t1 = link_two_frames(
            prev_coords, coords_voxels, max_distance_um=12.0, scale=scale
        )

        divisions, updated_matches, final_unmatched_t1 = detect_cell_divisions(
            prev_coords,
            coords_voxels,
            matches,
            unmatched_t0,
            unmatched_t1,
            max_mother_daughter_dist_um=12.0,
            max_sister_dist_um=18.0,
            scale=scale,
        )

        # 1-to-1 track continuity
        for m_idx, d_idx in updated_matches:
            global_id = prev_track_ids[m_idx]
            current_track_ids[d_idx] = global_id
            lineage_graph.add_node(
                global_id, time=step, pos=coords_um[d_idx].tolist()
            )
            csv_rows.append([global_id, step, *coords_um[d_idx], -1])

        # Division handling
        for div in divisions:
            mother_global_id = prev_track_ids[div["mother_t0"]]

            d1_global_id = next_global_track_id
            next_global_track_id += 1

            d2_global_id = next_global_track_id
            next_global_track_id += 1

            current_track_ids[div["daughter1_t1"]] = d1_global_id
            current_track_ids[div["daughter2_t1"]] = d2_global_id

            lineage_graph.add_edge(mother_global_id, d1_global_id)
            lineage_graph.add_edge(mother_global_id, d2_global_id)

            csv_rows.append([
                d1_global_id,
                step,
                *coords_um[div["daughter1_t1"]],
                mother_global_id,
            ])
            csv_rows.append([
                d2_global_id,
                step,
                *coords_um[div["daughter2_t1"]],
                mother_global_id,
            ])

        # Unmatched births
        for b_idx in final_unmatched_t1:
            if current_track_ids[b_idx] == -1:
                current_track_ids[b_idx] = next_global_track_id
                csv_rows.append([
                    next_global_track_id,
                    step,
                    *coords_um[b_idx],
                    -1,
                ])
                next_global_track_id += 1

        print(f"\n[Timepoint t={step-1} -> t={step}]")
        print(
            f"  ├── Active Track Links:   {len(updated_matches)} |"
            f" Divisions: {len(divisions)}"
        )
        if len(divisions) > 0:
            for div in divisions:
                m_gid = prev_track_ids[div["mother_t0"]]
                d1_gid = current_track_ids[div["daughter1_t1"]]
                d2_gid = current_track_ids[div["daughter2_t1"]]
                print(
                    f"      └── [DIVISION] Mother Cell #{m_gid} ->"
                    f" Daughters (#{d1_gid}, #{d2_gid})"
                )

    prev_coords = coords_voxels
    prev_track_ids = current_track_ids

# Export CSV
with open(CSV_OUTPUT_PATH, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["TrackID", "Timepoint", "Z_um", "Y_um", "X_um", "ParentID"])
    writer.writerows(csv_rows)

print("\n" + "=" * 65)
print(f"TRACKING COMPLETE! Exported lineage file: {CSV_OUTPUT_PATH}")
print(f"Total Unique Tracks Created: {next_global_track_id}")
print("=" * 65)