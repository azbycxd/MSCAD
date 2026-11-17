from __future__ import annotations

import inspect

import numpy as np

from utils.controlled_relation_injection import (
    block_permutation,
    cross_window_swap,
    localization_result,
    pair_score,
    pearson_corr,
    robust_scale_floor,
    select_diversity_capped,
    temporal_shift,
)


def _window() -> np.ndarray:
    x = np.linspace(-2.0, 2.0, 128)
    return np.column_stack([x, x + 0.05 * np.sin(np.arange(128)), np.cos(np.arange(128) / 7)])


def test_temporal_shift_preserves_values_and_changes_relation() -> None:
    window = _window()
    injected = temporal_shift(window, 1, 32)
    assert np.array_equal(np.sort(window[:, 1]), np.sort(injected[:, 1]))
    assert abs(pearson_corr(window[:, 0], window[:, 1]) - pearson_corr(injected[:, 0], injected[:, 1])) >= 0.20


def test_block_permutation_preserves_values_and_is_reproducible() -> None:
    window = _window()
    first = block_permutation(window, 1, np.random.default_rng(3))
    second = block_permutation(window, 1, np.random.default_rng(3))
    assert np.array_equal(np.sort(window[:, 1]), np.sort(first[:, 1]))
    assert np.array_equal(first, second)


def test_cross_window_swap_enforces_marginal_constraints() -> None:
    base = _window()
    donor = base.copy()
    donor[:, 1] = np.roll(base[:, 1], 48)
    injected, valid, reason = cross_window_swap(base, donor, 1)
    assert valid and reason == ""
    assert np.isclose(injected[:, 1].mean(), base[:, 1].mean())
    assert np.isclose(injected[:, 1].std(), base[:, 1].std())


def test_injection_changes_target_relation() -> None:
    window = _window()
    injected = block_permutation(window, 1, np.random.default_rng(1))
    assert abs(pearson_corr(window[:, 0], window[:, 1]) - pearson_corr(injected[:, 0], injected[:, 1])) >= 0.20


def test_diversity_capped_degree_is_at_most_two() -> None:
    rows = []
    for index, (left, right) in enumerate([(0, 1), (0, 2), (0, 3), (1, 2), (3, 4)]):
        rows.append({
            "pair_id": f"{left}-{right}", "channel_i_index": left, "channel_j_index": right,
            "relation_instability": 0.1 + index * 0.01, "abs_median_corr": 0.8,
            "valid_ratio": 1.0,
        })
    selected = select_diversity_capped(rows)
    degree = {channel: 0 for channel in range(5)}
    for row in selected:
        degree[row["channel_i_index"]] += 1
        degree[row["channel_j_index"]] += 1
    assert max(degree.values()) <= 2


def test_candidate_selector_has_no_anomaly_argument() -> None:
    from scripts.stage4_3a_eval_candidate_relation_sensitivity import build_candidate_sets

    parameters = inspect.signature(select_diversity_capped).parameters
    assert set(parameters) == {"rows", "max_channel_degree"}
    assert set(inspect.signature(build_candidate_sets).parameters) == {"pair_rows"}


def test_pair_score_is_independent_of_reconstruction() -> None:
    parameters = inspect.signature(pair_score).parameters
    assert "x_hat" not in parameters
    assert pair_score(0.8, np.arctanh(0.5), 0.2, 0.1) > 0


def test_scale_floor_prevents_division_by_zero() -> None:
    floor = robust_scale_floor([0.0, 0.0, 0.01])
    assert floor >= 1e-3
    assert np.isfinite(pair_score(0.5, 0.0, 0.0, floor))


def test_topk_localization() -> None:
    result = localization_result(np.array([0.5, 3.0, 2.0, 1.0, 0.1]), 2)
    assert result["target_pair_rank"] == 2
    assert not result["top1_hit"] and result["top3_hit"] and result["top5_hit"]


def test_seed_reproducibility() -> None:
    first = block_permutation(_window(), 1, np.random.default_rng(42))
    second = block_permutation(_window(), 1, np.random.default_rng(42))
    assert np.array_equal(first, second)


def test_stage_script_does_not_reference_test_data_or_labels() -> None:
    from scripts import stage4_3a_eval_candidate_relation_sensitivity as stage

    source = inspect.getsource(stage.load_stage_data)
    assert "test_csv" not in source
    assert "test_label" not in source
    assert "load_normal_training_data" in source


def test_aupr_uses_fixed_class_prior() -> None:
    from scripts.stage4_3a_eval_candidate_relation_sensitivity import metric_values

    normal = np.array([0.1, 0.2, 0.3, 0.4])
    anomaly = np.array([0.5, 0.6, 0.7])
    base = metric_values(normal, anomaly, normal)
    duplicated = metric_values(normal, np.tile(anomaly, 4), normal)
    assert base["aupr_positive_class_prior"] == 0.5
    assert np.isclose(base["aupr"], duplicated["aupr"])
