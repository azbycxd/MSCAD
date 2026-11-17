from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


FISHER_CLIP = 0.999999


def fisher_z(correlation: np.ndarray | float) -> np.ndarray:
    return np.arctanh(np.clip(np.asarray(correlation, dtype=np.float64), -FISHER_CLIP, FISHER_CLIP))


def stable_pairs(feature_names: list[str], excluded_features: set[str]) -> pd.DataFrame:
    allowed = [(index, name) for index, name in enumerate(feature_names) if name not in excluded_features]
    rows = []
    for (left_index, left), (right_index, right) in combinations(allowed, 2):
        rows.append({
            "pair_name": f"{left}__{right}", "channel_i": left, "channel_j": right,
            "channel_i_index": left_index, "channel_j_index": right_index,
        })
    return pd.DataFrame(rows)


def rolling_pair_correlations(
    values: np.ndarray,
    end_indices: np.ndarray,
    pairs: pd.DataFrame,
    window_size: int,
    point_validity: np.ndarray | None = None,
    variance_epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlations and per-channel activity for windows ending at end_indices."""
    x = np.asarray(values, dtype=np.float64)
    ends = np.asarray(end_indices, dtype=np.int64)
    starts = ends - int(window_size) + 1
    if x.ndim != 2 or (starts < 0).any() or (ends >= len(x)).any():
        raise ValueError("Invalid values/end_indices/window_size contract")
    finite_points = np.isfinite(x).all(axis=1)
    valid_points = finite_points if point_validity is None else finite_points & np.asarray(point_validity, dtype=bool)
    clean = np.where(np.isfinite(x), x, 0.0)

    def rolling_sum(array: np.ndarray) -> np.ndarray:
        cumulative = np.concatenate([np.zeros((1,) + array.shape[1:], dtype=np.float64), np.cumsum(array, axis=0)], axis=0)
        return cumulative[ends + 1] - cumulative[starts]

    sums = rolling_sum(clean)
    sums_sq = rolling_sum(clean * clean)
    variance_terms = np.maximum(sums_sq - sums * sums / window_size, 0.0)
    activity = variance_terms > variance_epsilon
    invalid_cumulative = np.concatenate([[0], np.cumsum(~valid_points)])
    input_valid = (invalid_cumulative[ends + 1] - invalid_cumulative[starts]) == 0
    pair_values = np.full((len(ends), len(pairs)), np.nan, dtype=np.float64)
    for column, row in enumerate(pairs.itertuples(index=False)):
        left, right = int(row.channel_i_index), int(row.channel_j_index)
        product_sum = rolling_sum((clean[:, left] * clean[:, right])[:, None])[:, 0]
        numerator = product_sum - sums[:, left] * sums[:, right] / window_size
        denominator = np.sqrt(variance_terms[:, left] * variance_terms[:, right])
        valid = input_valid & (denominator > variance_epsilon)
        pair_values[valid, column] = np.clip(numerator[valid] / denominator[valid], -1.0, 1.0)
    return pair_values, activity & input_valid[:, None]


def window_batch_pair_correlations(windows: np.ndarray, pairs: pd.DataFrame,
                                   variance_epsilon: float = 1e-12) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("windows must have shape [N,T,C]")
    result = np.full((len(values), len(pairs)), np.nan, dtype=np.float64)
    for column, row in enumerate(pairs.itertuples(index=False)):
        left = values[:, :, int(row.channel_i_index)]
        right = values[:, :, int(row.channel_j_index)]
        left = left - left.mean(axis=1, keepdims=True)
        right = right - right.mean(axis=1, keepdims=True)
        numerator = np.sum(left * right, axis=1)
        denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
        valid = denominator > variance_epsilon
        result[valid, column] = np.clip(numerator[valid] / denominator[valid], -1.0, 1.0)
    return result


def _sign_consistency(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 0.0
    return float(max((finite >= 0).mean(), (finite <= 0).mean()))


def screen_train_relations(
    train_values: np.ndarray,
    train_validity: np.ndarray,
    feature_names: list[str],
    zero_variance_features: list[str],
    window_size: int,
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    excluded = set(zero_variance_features)
    pairs = stable_pairs(feature_names, excluded)
    ends = np.arange(window_size - 1, len(train_values), dtype=np.int64)
    correlations, activity = rolling_pair_correlations(train_values, ends, pairs, window_size, train_validity)
    valid_input_windows = np.array([
        bool(train_validity[end - window_size + 1 : end + 1].all()) for end in ends
    ])
    valid_input_count = int(valid_input_windows.sum())
    activity_ratios = np.divide(
        (activity & valid_input_windows[:, None]).sum(axis=0), max(valid_input_count, 1), dtype=np.float64
    )
    diagnostics: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs.itertuples(index=False)):
        current = correlations[:, pair_index]
        finite = current[np.isfinite(current) & valid_input_windows]
        valid_count = len(finite)
        median = float(np.median(finite)) if valid_count else float("nan")
        abs_values = np.abs(finite)
        iqr = float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25)) if valid_count else float("nan")
        ratio_i, ratio_j = activity_ratios[int(pair.channel_i_index)], activity_ratios[int(pair.channel_j_index)]
        local_strength = bool(valid_count and np.median(abs_values) >= thresholds["strict_median_abs_corr"])
        stability = bool(valid_count and iqr <= thresholds["strict_corr_iqr"] and
                         _sign_consistency(finite) >= thresholds["strict_sign_consistency"])
        ratio = valid_count / max(valid_input_count, 1)
        strict = bool(ratio >= thresholds["strict_valid_window_ratio"] and local_strength and stability and
                      ratio_i >= thresholds["activity_ratio_min"] and ratio_j >= thresholds["activity_ratio_min"])
        expanded = bool(ratio >= thresholds["expanded_valid_window_ratio"] and valid_count and
                        np.median(abs_values) >= thresholds["expanded_median_abs_corr"] and
                        iqr <= thresholds["expanded_corr_iqr"] and ratio_i >= thresholds["activity_ratio_min"] and
                        ratio_j >= thresholds["activity_ratio_min"])
        diagnostics.append({
            **pair._asdict(), "valid_window_count": valid_count, "valid_window_ratio": ratio,
            "median_abs_corr": float(np.median(abs_values)) if valid_count else float("nan"),
            "mean_abs_corr": float(np.mean(abs_values)) if valid_count else float("nan"),
            "corr_median": median,
            "corr_mad": float(np.median(np.abs(finite - median))) if valid_count else float("nan"),
            "corr_iqr": iqr, "sign_consistency": _sign_consistency(finite),
            "activity_ratio_i": float(ratio_i), "activity_ratio_j": float(ratio_j),
            "local_strength_pass": local_strength, "stability_pass": stability,
            "strict_pass": strict, "expanded_pass": expanded,
        })
    channel_rows = []
    for index, name in enumerate(feature_names):
        channel_rows.append({
            "channel": name, "channel_index": index, "zero_variance_train": name in excluded,
            "excluded_from_candidates": name in excluded,
            "activity_ratio_128": float(activity_ratios[index]),
            "candidate_pair_count": int(((pairs.channel_i_index == index) | (pairs.channel_j_index == index)).sum()),
        })
    return pd.DataFrame(diagnostics), pd.DataFrame(channel_rows), correlations


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["valid_window_ratio", "median_abs_corr", "corr_iqr", "pair_name"],
        ascending=[False, False, True, True], kind="mergesort",
    ).reset_index(drop=True)


def select_relation_sets(diagnostics: pd.DataFrame, strict_limit: int = 20, diverse_limit: int = 10,
                         max_degree: int = 2) -> dict[str, pd.DataFrame]:
    strict = _ordered(diagnostics[diagnostics.strict_pass]).head(strict_limit).copy()
    expanded = _ordered(diagnostics[diagnostics.expanded_pass]).copy()
    degree: Counter[str] = Counter()
    diverse_rows = []
    for _, row in strict.iterrows():
        left, right = str(row.channel_i), str(row.channel_j)
        if degree[left] >= max_degree or degree[right] >= max_degree:
            continue
        diverse_rows.append(row.to_dict())
        degree[left] += 1
        degree[right] += 1
        if len(diverse_rows) >= diverse_limit:
            break
    diverse = pd.DataFrame(diverse_rows, columns=strict.columns)
    return {"strict_set": strict, "diverse_set": diverse, "expanded_set": expanded}


@dataclass(frozen=True)
class RelationReference:
    center: np.ndarray
    raw_scale: np.ndarray
    scale: np.ndarray
    scale_floor: float


def fit_robust_reference(correlations: np.ndarray, scale_floor: float) -> RelationReference:
    z = fisher_z(correlations)
    center = np.nanmedian(z, axis=0)
    raw_scale = 1.4826 * np.nanmedian(np.abs(z - center), axis=0)
    scale = np.where(np.isfinite(raw_scale) & (raw_scale >= scale_floor), raw_scale, scale_floor)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("Relation reference contains non-finite values")
    return RelationReference(center, raw_scale, scale, float(scale_floor))


def relation_deviation(correlations: np.ndarray, reference: RelationReference) -> np.ndarray:
    z = fisher_z(correlations)
    result = np.abs(z - reference.center) / reference.scale
    return np.where(np.isfinite(result), result, 0.0)


def aggregate_deviation(deviation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(deviation, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("At least one relation is required")
    maximum = np.max(values, axis=1)
    count = min(3, values.shape[1])
    top3 = np.partition(values, values.shape[1] - count, axis=1)[:, -count:].mean(axis=1)
    return maximum, top3


def circular_relation_injection(window: np.ndarray, channel_index: int, shift: int) -> np.ndarray:
    result = np.asarray(window).copy()
    result[:, channel_index] = np.roll(result[:, channel_index], int(shift))
    return result
