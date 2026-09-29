import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max


def extract_cell_centers_3d(
    prob_map: np.ndarray,
    threshold: float = 0.3,
    min_distance: int = 3,
    scale: np.ndarray = np.array([1.625, 0.40625, 0.40625]),
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts 3D physical coordinates [Z, Y, X] from 3D probability maps

    using local maxima detection.
    """
    coords = peak_local_max(
        prob_map,
        min_distance=min_distance,
        threshold_abs=threshold,
        exclude_border=False,
    )

    if len(coords) == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    coords_um = coords * scale
    return coords, coords_um


def link_two_frames(
    coords_t0: np.ndarray,
    coords_t1: np.ndarray,
    max_distance_um: float = 12.0,
    scale: np.ndarray = np.array([1.625, 0.40625, 0.40625]),
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Connects cell detections between frame t and frame t+1 using the Hungarian Algorithm.

    Returns:
        matches: List of tuples (index_t0, index_t1)
        unmatched_t0: List of cell indices in t0 with no match
        unmatched_t1: List of cell indices in t1 with no match
    """
    num_t0 = len(coords_t0)
    num_t1 = len(coords_t1)

    if num_t0 == 0 or num_t1 == 0:
        return [], list(range(num_t0)), list(range(num_t1))

    diff = coords_t0[:, None, :] - coords_t1[None, :, :]
    diff_um = diff * scale
    cost_matrix = np.linalg.norm(diff_um, axis=2)

    gating_penalty = 1e6
    gated_cost = np.where(
        cost_matrix <= max_distance_um, cost_matrix, gating_penalty
    )

    row_ind, col_ind = linear_sum_assignment(gated_cost)

    matches = []
    unmatched_t0 = set(range(num_t0))
    unmatched_t1 = set(range(num_t1))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= max_distance_um:
            matches.append((r, c))
            unmatched_t0.discard(r)
            unmatched_t1.discard(c)

    return matches, list(unmatched_t0), list(unmatched_t1)


def detect_cell_divisions(
    coords_t0: np.ndarray,
    coords_t1: np.ndarray,
    matches: list[tuple[int, int]],
    unmatched_t0: list[int],
    unmatched_t1: list[int],
    max_mother_daughter_dist_um: float = 12.0,
    max_sister_dist_um: float = 18.0,
    scale: np.ndarray = np.array([1.625, 0.40625, 0.40625]),
) -> tuple[list[dict], list[tuple[int, int]], list[int]]:
    """Detects true 1-to-2 cell division events where 1 mother cell splits into

    2 daughter cells (requires unmatched birth at t1).
    """
    divisions = []
    claimed_t1 = set()
    matched_mothers_removed = set()

    match_dict = {m[0]: m[1] for m in matches}

    # Division ONLY triggers if there is at least one unmatched cell in t1
    if len(unmatched_t1) == 0:
        return [], matches, unmatched_t1

    for m_idx in range(len(coords_t0)):
        mother_pos = coords_t0[m_idx]
        d1_idx = match_dict.get(m_idx, None)

        for d2_idx in unmatched_t1:
            if d2_idx in claimed_t1:
                continue

            d2_pos = coords_t1[d2_idx]
            dist_m_d2 = np.linalg.norm((mother_pos - d2_pos) * scale)

            if dist_m_d2 <= max_mother_daughter_dist_um:
                if d1_idx is not None:
                    d1_pos = coords_t1[d1_idx]
                    dist_m_d1 = np.linalg.norm((mother_pos - d1_pos) * scale)
                    sister_dist = np.linalg.norm((d1_pos - d2_pos) * scale)

                    if (
                        dist_m_d1 <= max_mother_daughter_dist_um
                        and sister_dist <= max_sister_dist_um
                    ):
                        divisions.append({
                            "mother_t0": m_idx,
                            "daughter1_t1": d1_idx,
                            "daughter2_t1": d2_idx,
                            "mother_d1_dist_um": dist_m_d1,
                            "mother_d2_dist_um": dist_m_d2,
                            "sister_dist_um": sister_dist,
                        })
                        claimed_t1.add(d1_idx)
                        claimed_t1.add(d2_idx)
                        matched_mothers_removed.add(m_idx)
                        break

    updated_matches = [
        m for m in matches if m[0] not in matched_mothers_removed
    ]
    remaining_unmatched_t1 = [
        idx for idx in range(len(coords_t1)) if idx not in claimed_t1
    ]

    return divisions, updated_matches, remaining_unmatched_t1