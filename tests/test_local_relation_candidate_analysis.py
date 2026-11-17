from __future__ import annotations

import math

from scripts.stage4_2c_analyze_local_relation_candidates import (
    PAIR_FIELDS,
    candidate_channel_concentration,
    classify_activity,
    generate_unordered_pairs,
    pair_activity_group,
    pareto_candidate_flags,
    threshold_sensitivity_rows,
)


def _row(pair_id: str, window: int, valid: float, strength: float, instability: float, group: str = "dense_dense") -> dict:
    return {
        "pair_id": pair_id,
        "window_size": window,
        "valid_ratio": valid,
        "abs_median_corr": strength,
        "relation_instability": instability,
        "q05_q95_width": 0.2,
        "pair_activity_group": group,
        "channel_i_index": 0,
        "channel_j_index": 1,
        "channel_i": "a",
        "channel_j": "b",
    }


def test_activity_class_boundaries() -> None:
    assert classify_activity(0.95) == "dense_active"
    assert classify_activity(0.9499) == "moderately_active"
    assert classify_activity(0.20) == "moderately_active"
    assert classify_activity(0.1999) == "sparse_switching"


def test_pair_activity_groups() -> None:
    assert pair_activity_group("dense_active", "dense_active") == "dense_dense"
    assert pair_activity_group("sparse_switching", "dense_active") == "dense_sparse"
    assert pair_activity_group("moderately_active", "dense_active") == "dense_moderate"
    assert pair_activity_group("sparse_switching", "moderately_active") == "moderate_sparse"


def test_threshold_grid_counts_candidates_and_excludes_nan() -> None:
    rows = [
        _row("a-b", 64, 0.96, 0.55, 0.20),
        _row("a-c", 64, 0.94, 0.70, 0.10),
        _row("a-d", 64, 0.99, float("nan"), 0.10),
        _row("a-b", 128, 0.98, 0.60, 0.20),
    ]
    sensitivity = threshold_sensitivity_rows(rows)
    selected = next(
        row for row in sensitivity
        if row["window_size"] == 64
        and row["min_valid_ratio"] == 0.95
        and row["min_strength"] == 0.50
        and row["max_instability"] == 0.25
    )
    assert selected["candidate_pair_count"] == 1


def test_pareto_detection() -> None:
    rows = [
        _row("dominated", 64, 0.90, 0.40, 0.30),
        _row("front-a", 64, 0.95, 0.50, 0.20),
        _row("front-b", 64, 0.99, 0.45, 0.10),
    ]
    assert pareto_candidate_flags(rows) == [False, True, True]


def test_candidate_channel_concentration() -> None:
    rows = [
        {**_row("a-b", 64, 1.0, 0.8, 0.1), "channel_i_index": 0, "channel_j_index": 1},
        {**_row("a-c", 64, 1.0, 0.8, 0.1), "channel_i_index": 0, "channel_j_index": 2, "channel_j": "c"},
    ]
    concentration = candidate_channel_concentration(rows, "test")
    a = next(row for row in concentration if row["channel"] == "a")
    assert a["candidate_occurrence_count"] == 2
    assert a["candidate_occurrence_ratio"] == 1.0
    assert a["concentration_warning"]


def test_unordered_pairs_have_no_self_or_duplicates() -> None:
    pairs = generate_unordered_pairs(8)
    assert all(i < j for i, j in pairs)
    assert len(pairs) == len(set(pairs))


def test_51_channels_create_1275_pairs() -> None:
    assert len(generate_unordered_pairs(51)) == 1275


def test_window_outputs_use_same_pair_fields() -> None:
    assert "window_size" in PAIR_FIELDS
    assert "valid_ratio" in PAIR_FIELDS
    assert "relation_instability" in PAIR_FIELDS


def test_nonfinite_values_do_not_pollute_pareto() -> None:
    rows = [
        _row("valid", 64, 0.95, 0.50, 0.20),
        _row("nan", 64, 0.99, 0.90, math.nan),
        _row("inf", 64, 0.99, 0.90, math.inf),
    ]
    assert pareto_candidate_flags(rows) == [True, False, False]
