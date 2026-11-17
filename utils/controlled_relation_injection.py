from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np


def pearson_corr(x: np.ndarray, y: np.ndarray, variance_epsilon: float = 1e-8) -> float:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    left_var = float(np.mean(np.square(left_centered)))
    right_var = float(np.mean(np.square(right_centered)))
    if left_var <= variance_epsilon or right_var <= variance_epsilon:
        return float("nan")
    corr = float(np.mean(left_centered * right_centered) / np.sqrt(left_var * right_var))
    return float(np.clip(corr, -0.999, 0.999))


def temporal_shift(window: np.ndarray, target_channel: int, steps: int) -> np.ndarray:
    injected = np.asarray(window).copy()
    injected[:, target_channel] = np.roll(injected[:, target_channel], int(steps))
    return injected


def block_permutation(
    window: np.ndarray,
    target_channel: int,
    rng: np.random.Generator,
    block_count: int = 4,
) -> np.ndarray:
    injected = np.asarray(window).copy()
    values = injected[:, target_channel].copy()
    if len(values) % block_count != 0:
        raise ValueError("Window length must be divisible by block_count")
    blocks = np.split(values, block_count)
    order = rng.permutation(block_count)
    if np.array_equal(order, np.arange(block_count)):
        order = np.roll(order, 1)
    injected[:, target_channel] = np.concatenate([blocks[int(index)] for index in order])
    return injected


def cross_window_swap(
    base_window: np.ndarray,
    donor_window: np.ndarray,
    target_channel: int,
    mean_tolerance: float = 0.10,
    std_relative_tolerance: float = 0.10,
    variance_epsilon: float = 1e-8,
) -> tuple[np.ndarray, bool, str]:
    base = np.asarray(base_window)
    donor = np.asarray(donor_window)
    original = np.asarray(base[:, target_channel], dtype=np.float64)
    replacement = np.asarray(donor[:, target_channel], dtype=np.float64)
    original_mean = float(original.mean())
    replacement_mean = float(replacement.mean())
    original_std = float(original.std())
    replacement_std = float(replacement.std())
    if original_std <= variance_epsilon:
        return base.copy(), False, "base_channel_near_constant"
    if abs(replacement_mean - original_mean) > mean_tolerance * original_std:
        return base.copy(), False, "donor_mean_mismatch"
    relative_std_change = abs(replacement_std - original_std) / max(original_std, variance_epsilon)
    if relative_std_change > std_relative_tolerance:
        return base.copy(), False, "donor_std_mismatch"
    injected = base.copy()
    injected[:, target_channel] = donor[:, target_channel]
    return injected, True, ""


def robust_scale_floor(scales: Iterable[float], absolute_floor: float = 1e-3) -> float:
    finite = np.asarray([float(value) for value in scales if np.isfinite(float(value))])
    if finite.size == 0:
        return float(absolute_floor)
    return float(max(absolute_floor, np.quantile(finite, 0.10)))


def pair_score(
    correlation: float,
    normal_median_fisher: float,
    normal_robust_scale: float,
    scale_floor: float,
) -> float:
    if not np.isfinite(correlation):
        return float("nan")
    fisher = float(np.arctanh(np.clip(correlation, -0.999, 0.999)))
    denominator = max(float(normal_robust_scale), float(scale_floor))
    return abs(fisher - float(normal_median_fisher)) / denominator


def aggregate_pair_scores(scores: np.ndarray) -> dict[str, float]:
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"mean_pair_score": float("nan"), "top3_mean_pair_score": float("nan"), "max_pair_score": float("nan")}
    top = np.sort(finite)[-min(3, finite.size) :]
    return {
        "mean_pair_score": float(finite.mean()),
        "top3_mean_pair_score": float(top.mean()),
        "max_pair_score": float(finite.max()),
    }


def localization_result(scores: np.ndarray, target_index: int) -> dict[str, float | bool | int]:
    values = np.asarray(scores, dtype=np.float64)
    if target_index < 0 or target_index >= len(values) or not np.isfinite(values[target_index]):
        return {"target_pair_rank": -1, "top1_hit": False, "top3_hit": False, "top5_hit": False}
    order = np.argsort(np.where(np.isfinite(values), -values, np.inf), kind="stable")
    rank = int(np.flatnonzero(order == target_index)[0]) + 1
    return {
        "target_pair_rank": rank,
        "top1_hit": rank <= 1,
        "top3_hit": rank <= 3,
        "top5_hit": rank <= 5,
    }


def select_diversity_capped(rows: list[dict[str, Any]], max_channel_degree: int = 2) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["relation_instability"]),
            -float(row["abs_median_corr"]),
            -float(row["valid_ratio"]),
            str(row["pair_id"]),
        ),
    )
    degree: Counter[int] = Counter()
    selected: list[dict[str, Any]] = []
    for row in ordered:
        left = int(row["channel_i_index"])
        right = int(row["channel_j_index"])
        if degree[left] >= max_channel_degree or degree[right] >= max_channel_degree:
            continue
        selected.append(row)
        degree[left] += 1
        degree[right] += 1
    return selected


def relation_set_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    degree: Counter[int] = Counter()
    for row in rows:
        degree[int(row["channel_i_index"])] += 1
        degree[int(row["channel_j_index"])] += 1
    pair_count = len(rows)
    return {
        "pair_count": pair_count,
        "involved_channel_count": len(degree),
        "maximum_channel_degree": max(degree.values(), default=0),
        "channel_occurrence_ratio": max(degree.values(), default=0) / max(1, pair_count),
        "median_strength": float(np.median([float(row["abs_median_corr"]) for row in rows])) if rows else float("nan"),
        "median_instability": float(np.median([float(row["relation_instability"]) for row in rows])) if rows else float("nan"),
        "median_valid_ratio": float(np.median([float(row["valid_ratio"]) for row in rows])) if rows else float("nan"),
    }
