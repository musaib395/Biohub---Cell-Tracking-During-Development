import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CellTrackingDataset
from model import CellUNet3D
from losses import WeightedHeatmapLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IMAGE_DIR = "data/processed/images"
TRAIN_TARGET_DIR = "data/processed/targets"
VAL_IMAGE_DIR = "data/processed/val_images"
VAL_TARGET_DIR = "data/processed/val_targets"

BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-4

train_dataset = CellTrackingDataset(TRAIN_IMAGE_DIR, TRAIN_TARGET_DIR)
val_dataset = CellTrackingDataset(VAL_IMAGE_DIR, VAL_TARGET_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = CellUNet3D().to(DEVICE)

criterion = WeightedHeatmapLoss(alpha=20.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, targets in train_loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(images)
        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            predictions = model(images)
            loss = criterion(predictions, targets)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "data/processed/best_model_weighted.pth")
        print("Saved best weighted model")

print("Training complete")
def detect_cell_divisions(
    coords_t0: np.ndarray,
    coords_t1: np.ndarray,
    matches: list[tuple[int, int]],
    unmatched_t0: list[int],
    unmatched_t1: list[int],
    max_mother_daughter_dist_um: float = 12.0,
    max_sister_dist_um: float = 10.0,
    scale: np.ndarray = np.array([1.625, 0.40625, 0.40625]),
) -> tuple[list[dict], list[tuple[int, int]], list[int]]:
    """Detects 1-to-2 cell division events where 1 mother cell at t0 splits into

    2 daughter cells at t1.

    Returns:
        divisions: List of dicts describing mother -> (daughter1, daughter2) splits.
        updated_matches: Adjusted list of 1-to-1 tracking matches.
        remaining_unmatched_t1: Updated list of unmatched detections in t1.
    """
    divisions = []
    claimed_t1 = set()
    matches_to_remove = set()

    # Candidate mothers can be either unmatched cells at t0 or existing match sources
    candidate_mothers = list(range(len(coords_t0)))

    for m_idx in candidate_mothers:
        mother_pos = coords_t0[m_idx]

        # Find potential daughter candidates in t1 within physical distance
        potential_daughters = []
        for d_idx in unmatched_t1:
            if d_idx in claimed_t1:
                continue

            daughter_pos = coords_t1[d_idx]
            dist_um = np.linalg.norm((mother_pos - daughter_pos) * scale)

            if dist_um <= max_mother_daughter_dist_um:
                potential_daughters.append((d_idx, dist_um))

        # Sort candidate daughters by proximity to mother
        potential_daughters.sort(key=lambda x: x[1])

        # If at least 2 candidates exist, test sister-sister proximity constraint
        if len(potential_daughters) >= 2:
            d1_idx, d1_dist = potential_daughters[0]
            d2_idx, d2_dist = potential_daughters[1]

            d1_pos = coords_t1[d1_idx]
            d2_pos = coords_t1[d2_idx]
            sister_dist_um = np.linalg.norm((d1_pos - d2_pos) * scale)

            if sister_dist_um <= max_sister_dist_um:
                divisions.append({
                    "mother_t0": m_idx,
                    "daughter1_t1": d1_idx,
                    "daughter2_t1": d2_idx,
                    "mother_d1_dist_um": d1_dist,
                    "mother_d2_dist_um": d2_dist,
                    "sister_dist_um": sister_dist_um,
                })

                claimed_t1.add(d1_idx)
                claimed_t1.add(d2_idx)

                # Flag existing 1-to-1 match for removal if mother was previously matched
                for match_idx, (r, c) in enumerate(matches):
                    if r == m_idx:
                        matches_to_remove.add(match_idx)

    updated_matches = [
        m for idx, m in enumerate(matches) if idx not in matches_to_remove
    ]
    remaining_unmatched_t1 = [
        idx for idx in unmatched_t1 if idx not in claimed_t1
    ]

    return divisions, updated_matches, remaining_unmatched_t1