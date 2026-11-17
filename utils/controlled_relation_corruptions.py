from __future__ import annotations

import numpy as np


BLOCK_ORDER = (2, 0, 3, 1)


def circular_shift_one(window: np.ndarray, channel: int, shift: int) -> np.ndarray:
    result = np.array(window, copy=True)
    result[:, channel] = np.roll(result[:, channel], shift)
    return result


def circular_shift_both(window: np.ndarray, channels: tuple[int, int], shift: int) -> np.ndarray:
    result = np.array(window, copy=True)
    for channel in channels:
        result[:, channel] = np.roll(result[:, channel], shift)
    return result


def _permute(values: np.ndarray, order: tuple[int, ...] = BLOCK_ORDER) -> np.ndarray:
    blocks = np.split(values, len(order))
    return np.concatenate([blocks[index] for index in order])


def block_permute_one(window: np.ndarray, channel: int) -> np.ndarray:
    result = np.array(window, copy=True)
    result[:, channel] = _permute(result[:, channel])
    return result


def block_permute_both(window: np.ndarray, channels: tuple[int, int]) -> np.ndarray:
    result = np.array(window, copy=True)
    for channel in channels:
        result[:, channel] = _permute(result[:, channel])
    return result


def replace_one(window: np.ndarray, source: np.ndarray, channel: int) -> np.ndarray:
    result = np.array(window, copy=True)
    result[:, channel] = source[:, channel]
    return result


def same_multiset(before: np.ndarray, after: np.ndarray) -> bool:
    return bool(np.array_equal(np.sort(np.asarray(before)), np.sort(np.asarray(after))))


def marginal_summary(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return np.asarray([x.mean(), x.std(), x.min(), x.max()])


def matched_replacement_indices(
    windows: np.ndarray, channel: int, starts: np.ndarray, window_size: int, quantile_bins: int = 5
) -> np.ndarray:
    """Deterministically match mean/std bins within one validation sequence."""
    x = np.asarray(windows)[:, :, channel]
    means, stds = x.mean(axis=1), x.std(axis=1)
    mean_edges = np.unique(np.quantile(means, np.linspace(0, 1, quantile_bins + 1)))
    std_edges = np.unique(np.quantile(stds, np.linspace(0, 1, quantile_bins + 1)))
    mean_bin = np.searchsorted(mean_edges[1:-1], means)
    std_bin = np.searchsorted(std_edges[1:-1], stds)
    result = np.full(len(x), -1, dtype=np.int64)
    for key in sorted(set(zip(mean_bin.tolist(), std_bin.tolist()))):
        group = np.flatnonzero((mean_bin == key[0]) & (std_bin == key[1]))
        if len(group) < 2:
            continue
        for offset, item in enumerate(group):
            for step in range(1, len(group)):
                candidate = group[(offset + step) % len(group)]
                if abs(int(starts[candidate]) - int(starts[item])) >= window_size:
                    result[item] = candidate
                    break
    return result
