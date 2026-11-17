from __future__ import annotations

import numpy as np

from scripts.stage4_2b_diagnose_relation_validity import (
    fisher_transform,
    pair_row_from_diagnostics,
    rolling_pair_diagnostics,
)


def _naive_correlations(x: np.ndarray, y: np.ndarray, window_size: int) -> np.ndarray:
    result = []
    for start in range(len(x) - window_size + 1):
        xw = x[start : start + window_size]
        yw = y[start : start + window_size]
        if np.var(xw, ddof=1) <= 1e-8 or np.var(yw, ddof=1) <= 1e-8:
            result.append(np.nan)
        else:
            result.append(np.corrcoef(xw, yw)[0, 1])
    return np.asarray(result)


def test_rolling_pearson_matches_naive_corrcoef() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    y = 0.4 * x + rng.normal(scale=0.7, size=300)
    actual = rolling_pair_diagnostics(x, y, 32)["correlations"]
    expected = _naive_correlations(x, y, 32)
    assert np.allclose(actual, expected, atol=1e-10, equal_nan=True)


def test_one_constant_channel_has_exclusive_i_reason() -> None:
    x = np.ones(100)
    y = np.linspace(0.0, 1.0, 100)
    result = rolling_pair_diagnostics(x, y, 20)
    assert result["invalid_i_constant_count"] == result["total_window_count"]
    assert result["invalid_j_constant_count"] == 0
    assert result["invalid_both_constant_count"] == 0


def test_both_constant_channels_are_not_double_counted() -> None:
    result = rolling_pair_diagnostics(np.ones(80), np.ones(80), 16)
    assert result["invalid_both_constant_count"] == result["total_window_count"]
    assert result["invalid_i_constant_count"] == 0
    assert result["invalid_j_constant_count"] == 0


def test_longer_window_improves_intermittent_channel_valid_ratio() -> None:
    x = np.zeros(600)
    x[::80] = 1.0
    y = x.copy()
    short = rolling_pair_diagnostics(x, y, 32)
    long = rolling_pair_diagnostics(x, y, 128)
    short_ratio = short["valid_window_count"] / short["total_window_count"]
    long_ratio = long["valid_window_count"] / long["total_window_count"]
    assert long_ratio > short_ratio


def test_linear_continuous_channels_are_strong() -> None:
    x = np.linspace(-2.0, 2.0, 400)
    result = rolling_pair_diagnostics(x, 3.0 * x + 2.0, 64)
    row = pair_row_from_diagnostics(result, (0, 1), 1, ["x", "y"], 64, 0.95, 0.5, 0.15)
    assert row["abs_median_corr"] > 0.99


def test_fisher_transform_has_no_inf_or_nan_for_finite_inputs() -> None:
    transformed = fisher_transform(np.asarray([-1.0, -0.999, 0.0, 0.999, 1.0]))
    assert np.all(np.isfinite(transformed))


def test_validity_counts_are_conserved() -> None:
    x = np.r_[np.zeros(30), np.linspace(0, 1, 70)]
    y = np.r_[np.zeros(50), np.linspace(0, 2, 50)]
    result = rolling_pair_diagnostics(x, y, 20)
    counted = sum(
        result[key]
        for key in (
            "valid_window_count",
            "invalid_i_constant_count",
            "invalid_j_constant_count",
            "invalid_both_constant_count",
            "invalid_numeric_count",
        )
    )
    assert counted == result["total_window_count"]
