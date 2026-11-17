from __future__ import annotations

import numpy as np

from scripts.stage4_2a_analyze_relation_stability import (
    classify_row,
    fisher_transform,
    summarize_correlation_samples,
    window_pair_correlations,
)


def _summary(windows: np.ndarray) -> dict:
    correlations, _, _ = window_pair_correlations(windows)
    return summarize_correlation_samples(correlations[:, 0])


def test_identical_channels_are_strong_and_stable() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(128, 64))
    summary = _summary(np.stack([x, x], axis=-1))
    assert summary["relation_strength"] > 0.99
    assert summary["relation_instability"] < 0.01


def test_independent_channels_have_low_strength() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(512, 64))
    y = rng.normal(size=(512, 64))
    summary = _summary(np.stack([x, y], axis=-1))
    assert summary["relation_strength"] < 0.15


def test_reversing_relation_direction_is_unstable() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 64))
    y = x.copy()
    y[100:] *= -1.0
    summary = _summary(np.stack([x, y], axis=-1))
    assert summary["relation_instability"] > 1.0


def test_constant_channel_is_invalid_and_not_selected() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(32, 64))
    constant = np.ones_like(x)
    summary = _summary(np.stack([x, constant], axis=-1))
    row = dict(summary)
    classify_row(row, min_strength=0.5, max_instability=0.15, min_valid_ratio=0.95)
    assert summary["valid_ratio"] == 0.0
    assert not row["strong_and_stable"]
    assert row["diagnostic_exclusion_reason"] == "near_constant_channel"


def test_fisher_transform_is_finite() -> None:
    transformed = fisher_transform(np.asarray([-1.0, -0.999, 0.0, 0.999, 1.0, np.nan]))
    assert np.all(np.isfinite(transformed[:-1]))
    assert not np.isinf(transformed).any()
