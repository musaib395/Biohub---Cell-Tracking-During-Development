import os
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

CSV_PATH = "data/processed/lineage_tracks.csv"
PNG_OUTPUT_PATH = "data/processed/3d_cell_tracks.png"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Lineage CSV not found at {CSV_PATH}. Run run_tracking_pipeline.py first."
    )

df = pd.read_csv(CSV_PATH)

print("=" * 65)
print("       GENERATING 3D CELL LINEAGE PLOT (MATPLOTLIB)")
print("=" * 65)
print(
    f"Loaded {len(df)} coordinate points across"
    f" {df['Timepoint'].nunique()} timepoints."
)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

unique_tracks = df["TrackID"].unique()
cmap = plt.get_cmap("tab10")

# 1. Plot trajectories for each track
for idx, track_id in enumerate(unique_tracks):
    track_df = df[df["TrackID"] == track_id].sort_values("Timepoint")

    ax.plot(
        track_df["X_um"],
        track_df["Y_um"],
        track_df["Z_um"],
        marker="o",
        linewidth=2,
        markersize=6,
        label=f"Track #{track_id}",
        color=cmap(idx % 10),
    )

    # Annotate timepoints along trajectory
    for _, row in track_df.iterrows():
        ax.text(
            row["X_um"],
            row["Y_um"],
            row["Z_um"],
            f"t={int(row['Timepoint'])}",
            fontsize=8,
        )

# 2. Draw parent-daughter linkage lines for division events
division_events = df[df["ParentID"] != -1]
for _, row in division_events.iterrows():
    parent_id = int(row["ParentID"])
    parent_rows = df[
        (df["TrackID"] == parent_id) & (df["Timepoint"] == row["Timepoint"] - 1)
    ]

    if not parent_rows.empty:
        parent_row = parent_rows.iloc[0]
        ax.plot(
            [parent_row["X_um"], row["X_um"]],
            [parent_row["Y_um"], row["Y_um"]],
            [parent_row["Z_um"], row["Z_um"]],
            color="red",
            linestyle="--",
            linewidth=2,
            label="Division Event"
            if "Division Event" not in ax.get_legend_handles_labels()[1]
            else "",
        )

ax.set_xlabel("X (µm)")
ax.set_ylabel("Y (µm)")
ax.set_zlabel("Z (µm)")
ax.set_title("3D Spatio-Temporal Cell Trajectories & Division Lineage")
ax.legend()
plt.tight_layout()

plt.savefig(PNG_OUTPUT_PATH, dpi=300)
print(f"\n3D Visualization saved to: {PNG_OUTPUT_PATH}")
plt.show()
print("=" * 65)