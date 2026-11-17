from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def active_feature_indices(raw_std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(raw_std, dtype=float).reshape(-1)
    return np.flatnonzero(np.isfinite(values) & (values >= eps))


def endpoint_rows(sequence, end_rows: np.ndarray) -> np.ndarray:
    local = np.asarray(end_rows, dtype=np.int64) - int(sequence.original_row_offset)
    if local.ndim != 1 or len(local) == 0 or local.min() < 0 or local.max() >= len(sequence.x):
        raise ValueError(f"Endpoint rows do not align with {sequence.file_name}/{sequence.split}")
    return sequence.x[local]


def pointwise_state_alarm(scores: np.ndarray, states: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    states = np.asarray(states, dtype=np.int64)
    thresholds = np.asarray(thresholds, dtype=float)
    if scores.ndim != 1 or states.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("scores/states must be aligned finite one-dimensional arrays")
    if len(thresholds) == 0 or states.min(initial=0) < 0 or states.max(initial=0) >= len(thresholds):
        raise ValueError("state ids exceed threshold table")
    return (scores > thresholds[states]).astype(np.int8)


def robust_location_scale(values: np.ndarray) -> tuple[float, float, str, float, float, float]:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values)); mad = float(np.median(np.abs(values - median)))
    iqr = float(np.quantile(values, .75) - np.quantile(values, .25)); std = float(values.std())
    candidates = ((1.4826 * mad, "mad"), (iqr / 1.349, "iqr"), (std, "std"), (1.0, "unit"))
    scale, source = next((value, name) for value, name in candidates if np.isfinite(value) and value > 1e-12)
    return median, float(scale), source, mad, iqr, std


def robust_z(scores: np.ndarray, states: np.ndarray, medians: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.asarray(scores, float) - medians[states]) / scales[states]


def run_lengths(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if len(values) == 0:
        return np.asarray([], dtype=np.int64)
    changes = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return np.diff(changes)


def normalized_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(counts))) if len(counts) > 1 else 0.0


def centroid_matching_distance(a: np.ndarray, b: np.ndarray) -> float:
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(distances)
    return float(distances[rows, cols].mean())


def clustering_agreement(labels_a: np.ndarray, centers_a: np.ndarray,
                         labels_b: np.ndarray, centers_b: np.ndarray) -> dict[str, float]:
    return {
        "adjusted_rand_index": float(adjusted_rand_score(labels_a, labels_b)),
        "normalized_mutual_information": float(normalized_mutual_info_score(labels_a, labels_b)),
        "centroid_matching_distance": centroid_matching_distance(centers_a, centers_b),
        "cluster_size_variation": float(np.std(np.bincount(labels_b, minlength=len(centers_b)) /
                                                max(len(labels_b), 1))),
    }
