import pandas as pd
import networkx as nx
import numpy as np

CSV_PATH = "data/processed/lineage_tracks.csv"

df = pd.read_csv(CSV_PATH)

print("=" * 65)
print("       LINEAGE TREE METRICS & GRAPH ANALYSIS")
print("=" * 65)

# 1. Build Directed Lineage Graph
G = nx.DiGraph()

for _, row in df.iterrows():
    node_id = (int(row["TrackID"]), int(row["Timepoint"]))
    G.add_node(
        node_id,
        track_id=int(row["TrackID"]),
        timepoint=int(row["Timepoint"]),
        pos=(row["Z_um"], row["Y_um"], row["X_um"]),
    )

# Connect consecutive frame tracks
for track_id, group in df.groupby("TrackID"):
    sorted_group = group.sort_values("Timepoint")
    timepoints = sorted_group["Timepoint"].values
    for i in range(len(timepoints) - 1):
        u = (int(track_id), int(timepoints[i]))
        v = (int(track_id), int(timepoints[i + 1]))
        G.add_edge(u, v, type="link")

# Connect division parent -> child linkages
division_rows = df[df["ParentID"] != -1]
for _, row in division_rows.iterrows():
    parent_id = int(row["ParentID"])
    child_id = int(row["TrackID"])
    t = int(row["Timepoint"])

    parent_node = (parent_id, t - 1)
    child_node = (child_id, t)

    if G.has_node(parent_node) and G.has_node(child_node):
        G.add_edge(parent_node, child_node, type="division")

# 2. Compute Trajectory Metrics
metrics = []
for track_id, group in df.groupby("TrackID"):
    sorted_group = group.sort_values("Timepoint")
    coords = sorted_group[["Z_um", "Y_um", "X_um"]].values

    if len(coords) > 1:
        step_dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        total_dist = np.sum(step_dists)
        mean_speed = np.mean(step_dists)  # um per frame
    else:
        total_dist = 0.0
        mean_speed = 0.0

    metrics.append({
        "TrackID": track_id,
        "Start_t": sorted_group["Timepoint"].min(),
        "End_t": sorted_group["Timepoint"].max(),
        "Duration_frames": len(sorted_group),
        "Total_Distance_um": round(total_dist, 3),
        "Mean_Speed_um_per_frame": round(mean_speed, 3),
    })

metrics_df = pd.DataFrame(metrics)
print("\n[Track Motility Metrics]")
print(metrics_df.to_string(index=False))

print("\n" + "=" * 65)
print(f"Graph Construction Complete: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")
print("=" * 65)