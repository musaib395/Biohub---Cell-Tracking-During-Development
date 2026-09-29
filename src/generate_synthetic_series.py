import os
import numpy as np
import torch

OUTPUT_DIR = "data/processed/synthetic_series"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_FRAMES = 15
SHAPE = (32, 64, 64)
SCALE = np.array([1.625, 0.40625, 0.40625])  # Z, Y, X physical resolution in um

print("=" * 65)
print("       GENERATING SYNTHETIC CONTIGUOUS 3D TIME-SERIES")
print("=" * 65)

# Initial cell positions (in voxel indices: Z, Y, X)
cells = [
    {"id": 0, "pos": np.array([16.0, 32.0, 32.0]), "velocity": np.array([0.2, 0.5, 0.3])}
]

division_frame = 6  # Trigger division at t=6

for t in range(NUM_FRAMES):
    prob_map = np.zeros(SHAPE, dtype=np.float32)
    current_cells = []

    for cell in cells:
        pos = cell["pos"]

        # Add 3D Gaussian spot at cell location
        z_grid, y_grid, x_grid = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
        dist_sq = (
            ((z_grid - pos[0]) * SCALE[0]) ** 2 +
            ((y_grid - pos[1]) * SCALE[1]) ** 2 +
            ((x_grid - pos[2]) * SCALE[2]) ** 2
        )
        gaussian = np.exp(-dist_sq / (2 * (1.5 ** 2)))
        prob_map = np.maximum(prob_map, gaussian)

        # Division logic at t=6
        if t == division_frame and cell["id"] == 0:
            print(f"[Synthetic Event] Cell #{cell['id']} divides at t={t}")
            daughter1 = {
                "id": 0,
                "pos": pos + np.array([0.5, 1.5, 1.5]),
                "velocity": np.array([0.1, 0.4, 0.4])
            }
            daughter2 = {
                "id": 1,
                "pos": pos + np.array([-0.5, -1.5, -1.5]),
                "velocity": np.array([-0.1, -0.4, -0.4])
            }
            current_cells.extend([daughter1, daughter2])
        else:
            # Random walk motion
            jitter = np.random.uniform(-0.3, 0.3, size=3)
            new_pos = pos + cell["velocity"] + jitter
            new_pos = np.clip(new_pos, [2, 2, 2], [SHAPE[0]-3, SHAPE[1]-3, SHAPE[2]-3])
            current_cells.append({"id": cell["id"], "pos": new_pos, "velocity": cell["velocity"]})

    cells = current_cells

    # Save probability map tensor
    file_path = os.path.join(OUTPUT_DIR, f"prob_map_t{t:02d}.pt")
    torch.save(torch.from_numpy(prob_map).unsqueeze(0).unsqueeze(0), file_path)

print(f"Generated {NUM_FRAMES} contiguous 3D probability maps in '{OUTPUT_DIR}'.")
print("=" * 65)