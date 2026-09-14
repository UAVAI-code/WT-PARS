"""Clean-response alignment helpers; legacy search is not included."""
from __future__ import annotations
import numpy as np
from .cross_unmixer import match_to_anchors
EPS = 1e-8


def _endmember_set_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Permutation-invariant mean SAD between two returned endmember sets."""
    from scipy.optimize import linear_sum_assignment

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    numerator = first.T @ second
    denominator = (
        np.linalg.norm(first, axis=0)[:, None]
        * np.linalg.norm(second, axis=0)[None, :]
        + EPS
    )
    cost = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))
    rows, columns = linear_sum_assignment(cost)
    return float(np.mean(cost[rows, columns]))


def _canonical_clean_outputs(
    outputs: dict[str, object],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Choose a clean-output medoid and align every response to it."""
    names = list(outputs)
    if not names:
        raise ValueError("At least one query service is required.")
    distances = np.zeros((len(names), len(names)), dtype=np.float64)
    for row, first in enumerate(names):
        for column in range(row + 1, len(names)):
            second = names[column]
            value = _endmember_set_distance(
                outputs[first].endmembers, outputs[second].endmembers
            )
            distances[row, column] = distances[column, row] = value
    medoid_name = names[int(np.argmin(np.sum(distances, axis=1)))]
    canonical = np.asarray(outputs[medoid_name].endmembers, dtype=np.float32).copy()
    abundance: dict[str, np.ndarray] = {}
    endmembers: dict[str, np.ndarray] = {}
    for name, output in outputs.items():
        aligned_abundance, aligned_endmembers, _ = match_to_anchors(canonical, output)
        abundance[name] = aligned_abundance.astype(np.float32)
        endmembers[name] = aligned_endmembers.astype(np.float32)
    return canonical, abundance, endmembers
