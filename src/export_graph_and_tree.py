import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

CSV_PATH = "data/processed/lineage_tracks.csv"
GRAPHML_OUTPUT_PATH = "data/processed/lineage_graph.graphml"
TREE_PLOT_PATH = "data/processed/2d_lineage_tree.png"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

print("=" * 65)
print("       LINEAGE TREE DIAGRAM & GRAPHML EXPORT")
print("=" * 65)

# 1. Build Directed Lineage Graph
G = nx.DiGraph()

for _, row in df.iterrows():
    node_id = f"T{int(row['TrackID'])}_t{int(row['Timepoint'])}"
    G.add_node(
        node_id,
        track_id=int(row["TrackID"]),
        timepoint=int(row["Timepoint"]),
        z_um=float(row["Z_um"]),
        y_um=float(row["Y_um"]),
        x_um=float(row["X_um"]),
    )

# Frame-to-frame temporal links
for track_id, group in df.groupby("TrackID"):
    sorted_group = group.sort_values("Timepoint")
    t_vals = sorted_group["Timepoint"].values
    for i in range(len(t_vals) - 1):
        u = f"T{int(track_id)}_t{int(t_vals[i])}"
        v = f"T{int(track_id)}_t{int(t_vals[i+1])}"
        G.add_edge(u, v, edge_type="continuity")

# Parent-daughter division links
division_rows = df[df["ParentID"] != -1]
for _, row in division_rows.iterrows():
    parent_id = int(row["ParentID"])
    child_id = int(row["TrackID"])
    t = int(row["Timepoint"])

    u = f"T{parent_id}_t{t-1}"
    v = f"T{child_id}_t{t}"
    if G.has_node(u) and G.has_node(v):
        G.add_edge(u, v, edge_type="division")

# Export to GraphML
nx.write_graphml(G, GRAPHML_OUTPUT_PATH)
print(f"Exported GraphML file to: {GRAPHML_OUTPUT_PATH}")

# 2. Render 2D Hierarchical Lineage Diagram
fig, ax = plt.subplots(figsize=(10, 6))

# Custom 2D layout: X-axis = Timepoint, Y-axis = Track ID offset
pos = {}
for node, data in G.nodes(data=True):
    pos[node] = (data["timepoint"], -data["track_id"])

# Draw temporal edges
continuity_edges = [
    (u, v)
    for u, v, d in G.edges(data=True)
    if d.get("edge_type") == "continuity"
]
division_edges = [
    (u, v)
    for u, v, d in G.edges(data=True)
    if d.get("edge_type") == "division"
]

nx.draw_networkx_nodes(
    G, pos, node_size=300, node_color="skyblue", edgecolors="black", ax=ax
)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=continuity_edges,
    width=2,
    edge_color="gray",
    arrows=True,
    ax=ax,
)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=division_edges,
    width=2.5,
    edge_color="red",
    style="dashed",
    arrows=True,
    ax=ax,
)

# Label nodes with Track ID
node_labels = {node: f"#{data['track_id']}" for node, data in G.nodes(data=True)}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)

ax.set_title("2D Cell Lineage Tree (Red Dashed = Division Event)")
ax.set_xlabel("Timepoint (t)")
ax.set_ylabel("Track Lineage Branch")
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig(TREE_PLOT_PATH, dpi=300)
print(f"Saved 2D Lineage Tree plot to: {TREE_PLOT_PATH}")
plt.show()

print("=" * 65)