from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FISHER_CLIP = 0.999999


def fisher_transform(correlation: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(np.asarray(correlation, dtype=np.float64), -FISHER_CLIP, FISHER_CLIP))


def window_pair_fisher(windows: np.ndarray, pairs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return Fisher-z Pearson features for [N,T,C] windows and ordered pairs."""
    values = np.asarray(windows, dtype=np.float64)
    pair_index = np.asarray(pairs, dtype=np.int64)
    left = values[:, :, pair_index[:, 0]]
    right = values[:, :, pair_index[:, 1]]
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    numerator = np.einsum("ntp,ntp->np", left, right)
    denominator = np.sqrt(
        np.einsum("ntp,ntp->np", left, left) * np.einsum("ntp,ntp->np", right, right)
    )
    correlation = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > eps,
    )
    return fisher_transform(correlation)


@dataclass(frozen=True)
class RobustReference:
    median: np.ndarray
    scale: np.ndarray
    fallback: np.ndarray
    sequence_median: np.ndarray
    sequence_scale: np.ndarray
    sequence_fallback: np.ndarray


def _fit_columns(values: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    medians = np.nanmedian(x, axis=0)
    scales = np.empty(x.shape[1], dtype=np.float64)
    fallback = np.empty(x.shape[1], dtype="U8")
    for column in range(x.shape[1]):
        valid = x[np.isfinite(x[:, column]), column]
        if not len(valid):
            medians[column], scales[column], fallback[column] = 0.0, epsilon, "epsilon"
            continue
        median = medians[column]
        mad_scale = 1.4826 * np.median(np.abs(valid - median))
        iqr_scale = (np.quantile(valid, 0.75) - np.quantile(valid, 0.25)) / 1.349
        std_scale = np.std(valid)
        if np.isfinite(mad_scale) and mad_scale > epsilon:
            scales[column], fallback[column] = mad_scale, "mad"
        elif np.isfinite(iqr_scale) and iqr_scale > epsilon:
            scales[column], fallback[column] = iqr_scale, "iqr"
        elif np.isfinite(std_scale) and std_scale > epsilon:
            scales[column], fallback[column] = std_scale, "std"
        else:
            scales[column], fallback[column] = epsilon, "epsilon"
    return medians, scales, fallback


def fit_robust_reference(
    train_z: np.ndarray, sequence_id: np.ndarray, sequence_count: int = 6, epsilon: float = 1e-6
) -> RobustReference:
    median, scale, fallback = _fit_columns(train_z, epsilon)
    seq_medians, seq_scales, seq_fallback = [], [], []
    for sequence in range(sequence_count):
        current = np.asarray(train_z)[np.asarray(sequence_id) == sequence]
        m, s, f = _fit_columns(current, epsilon)
        seq_medians.append(m)
        seq_scales.append(s)
        seq_fallback.append(f)
    return RobustReference(
        median=median,
        scale=scale,
        fallback=fallback,
        sequence_median=np.asarray(seq_medians),
        sequence_scale=np.asarray(seq_scales),
        sequence_fallback=np.asarray(seq_fallback),
    )


def relation_distances(
    z: np.ndarray,
    reference: RobustReference,
    component_mask: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(z, dtype=np.float64)
    clean = np.where(np.isfinite(values), values, reference.median)
    distances = np.abs(clean - reference.median) / reference.scale
    if component_mask is not None and np.asarray(component_mask, bool).any():
        mask = np.asarray(component_mask, bool)
        component = np.abs(clean[:, None, mask] - reference.sequence_median[None, :, mask])
        component = component / reference.sequence_scale[None, :, mask]
        distances[:, mask] = np.min(component, axis=1)
    return distances


def aggregate_distances(distances: np.ndarray, method: str) -> np.ndarray:
    values = np.asarray(distances, dtype=np.float64)
    if method == "max":
        return np.max(values, axis=1)
    if method == "top3_mean":
        count = min(3, values.shape[1])
        return np.partition(values, values.shape[1] - count, axis=1)[:, -count:].mean(axis=1)
    raise ValueError(f"Unsupported aggregation: {method}")
