from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.swat2026_contextual_structure import (
    LAG_SECONDS, audit_state_channels, binary_metrics, circular_shift_window,
    discovery_selected_lags, lagged_pair_correlations, parse_binary_state,
    select_contextual_set, select_primary, stable_state_window_mask,
)
from data.swat2026_structure import stable_pairs
from data.swat2026_structure_recovery import train_relation_split, validation_usage_split
from scripts.swat2026 import analyze_contextual_structure as contextual


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "swat2026" / "contextual_structure_seed1.yaml"


def cfg() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def pairs() -> pd.DataFrame:
    return stable_pairs(["A.Pv", "B.Pv", "C.Pv"], set())


def timestamps() -> pd.DatetimeIndex:
    return pd.date_range("2026-03-11 09:00:00", periods=28_860, freq="s")


def test_lag_set_is_prespecified_and_fixed():
    assert LAG_SECONDS == (-30, -20, -10, -5, 0, 5, 10, 20, 30)
    assert cfg()["lag_seconds"] == list(LAG_SECONDS)


def test_best_lag_is_selected_from_discovery_only():
    signature = inspect.signature(discovery_selected_lags)
    assert set(signature.parameters) == {"pairs", "discovery_by_lag"}
    values = {lag: np.full((20, 3), .3) for lag in LAG_SECONDS}
    values[10][:, 0] = .9
    assert discovery_selected_lags(pairs(), values).iloc[0].selected_lag == 10


def test_confirmation_does_not_reselect_lag():
    source = inspect.getsource(contextual.discover_lagged)
    assert "discovery_selected_lags(all_pairs, discovery_by_lag)" in source
    assert "discovery_selected_lags(all_pairs, confirmation_by_lag)" not in source


def test_positive_lag_means_first_channel_leads_second():
    rng = np.random.default_rng(1); n = 300
    first = rng.normal(size=n); second = np.r_[np.zeros(5), first[:-5]]
    values = np.column_stack([first, second]); valid = np.ones(n, dtype=bool)
    result = lagged_pair_correlations(values, valid, np.array([299]), pairs().head(1), 5, "lagged_level")
    assert result[0, 0] > .99


def test_lag_outside_fixed_set_is_rejected():
    with pytest.raises(ValueError, match="prespecified"):
        lagged_pair_correlations(np.ones((200, 3)), np.ones(200, bool), np.array([199]), pairs(), 7, "lagged_level")


def test_delta_lagged_representation_uses_differences():
    x = np.arange(300.0) ** 2; values = np.column_stack([x, 2 * x, x + 1])
    result = lagged_pair_correlations(values, np.ones(300, bool), np.array([299]), pairs().head(1), 0, "lagged_delta")
    assert result[0, 0] == pytest.approx(1.0)


def test_state_mapping_is_explicit():
    values, known = parse_binary_state(pd.Series(["Open", "Closed"]), {"Open": "open", "Closed": "closed"})
    assert values.tolist() == [1, 0] and known.all()


def test_unknown_state_is_not_silently_mapped_to_zero():
    values, known = parse_binary_state(pd.Series(["Open", "Mystery"]), {"Open": "open"})
    assert values.tolist() == [1, -1] and known.tolist() == [True, False]


def test_alarm_fields_are_not_state_candidates():
    frame = pd.DataFrame({"P1_STATE": [0] * 30, "LS401.Alarm": [0] * 30, "X.Al": [0] * 30})
    state_cfg = {**cfg()["states"], "explicit_binary_mappings": {"P1_STATE": {"0": "off"}}, "minimum_state_points": 1}
    audit, _ = audit_state_channels(frame, np.arange(15), np.arange(15, 30), state_cfg)
    assert audit.state_channel.tolist() == ["P1_STATE"]


def test_state_qualification_uses_train_sections_only():
    assert set(inspect.signature(audit_state_channels).parameters) == {"frame", "discovery", "confirmation", "state_cfg"}
    assert "test" not in inspect.signature(audit_state_channels).parameters


def test_state_switching_windows_are_not_applicable():
    state = np.r_[np.zeros(127, int), np.ones(129, int)]
    mask = stable_state_window_mask(state, np.array([127, 255]), 1)
    assert mask.tolist() == [False, True]


def test_discovery_and_confirmation_are_disjoint_with_guard():
    manifest, split = train_relation_split(timestamps())
    assert manifest["raw_timestamp_overlap"] is False
    assert not set(split["discovery"]) & set(split["confirmation"])
    assert len(split["guard"]) == 127


def _diagnostics(count: int) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append({"pair_name": f"A{index}__B{index}", "channel_i": f"A{index}", "channel_j": f"B{index}",
                     "channel_i_index": index, "channel_j_index": index + count, "reproducible": True,
                     "confirmation_valid_window_ratio": 1.0, "discovery_median_abs_corr": .8,
                     "confirmation_median_abs_corr": .8, "median_gap": .01,
                     "discovery_corr_iqr": .1, "confirmation_corr_iqr": .1, "selected_lag": 5})
    return pd.DataFrame(rows)


def test_at_least_three_relations_are_required_for_general_set():
    assert select_contextual_set(_diagnostics(2))["status"] == "narrow"
    assert select_contextual_set(_diagnostics(3))["status"] == "general"


def test_narrow_legacy_relation_is_not_automatically_promoted():
    source = inspect.getsource(contextual.run)
    assert '"automatic_promotion": False' in source


def test_validation_threshold_and_controlled_do_not_overlap():
    manifest, split = validation_usage_split(timestamps())
    assert manifest["raw_timestamp_overlap"] is False
    assert not set(split["threshold_raw"]) & set(split["controlled_raw"])


def test_injection_preserves_univariate_value_multiset():
    window = np.arange(384.0).reshape(128, 3)
    changed = circular_shift_window(window, 1, 16)
    assert np.array_equal(np.sort(window[:, 1]), np.sort(changed[:, 1]))
    assert np.array_equal(window[:, 0], changed[:, 0])


def test_conditioned_injection_does_not_change_state_values():
    source = inspect.getsource(contextual.controlled_validation)
    assert '"state_value_changed": False' in source
    assert "circular_shift_window" in source


def test_threshold_is_computed_from_separate_normal_values():
    metric = binary_metrics(np.array([.1, .2]), np.array([.8, .9]), np.array([.05, .15]))
    assert metric["threshold"] == pytest.approx(np.quantile([.05, .15], .99))


def test_primary_selection_has_no_test_argument():
    assert "test" not in inspect.signature(select_primary).parameters


def test_primary_requires_relation_count_coverage_and_controlled_auroc():
    row = pd.DataFrame([{"representation": "lagged_level", "score_name": "structure_max", "auroc": .9,
                         "marginal_statistics_preserved": True, "reference_valid": True}])
    assert not select_primary(row, {"lagged_level": 2}, {"lagged_level": 1.0})["primary_frozen"]
    assert not select_primary(row, {"lagged_level": 3}, {"lagged_level": .7})["primary_frozen"]
    assert select_primary(row, {"lagged_level": 3}, {"lagged_level": .9})["primary_frozen"]


def test_primary_tie_prefers_simpler_representation():
    rows = pd.DataFrame([
        {"representation": "conditioned_delta", "score_name": "structure_max", "auroc": .90, "marginal_statistics_preserved": True, "reference_valid": True},
        {"representation": "lagged_level", "score_name": "structure_max", "auroc": .895, "marginal_statistics_preserved": True, "reference_valid": True},
    ])
    selected = select_primary(rows, {name: 3 for name in cfg()["representations"]}, {name: 1.0 for name in cfg()["representations"]})
    assert selected["representation"] == "lagged_level"


def test_test_labels_are_opened_only_in_post_freeze_function():
    run_source = inspect.getsource(contextual.run)
    assert "formal_test_evaluation" in run_source
    formal = inspect.getsource(contextual.formal_test_evaluation)
    assert "if not primary_manifest" in formal and "point_labels_11mar.csv" in formal


def test_coverage_gate_is_enforced():
    assert cfg()["controlled"]["minimum_coverage"] == .80
    assert "minimum_coverage" in inspect.getsource(select_primary)


def test_no_point_adjustment_or_fusion():
    assert cfg()["evaluation"]["point_adjustment"] is False
    assert cfg()["evaluation"]["fusion"] is False
    assert "fusion_score" not in inspect.getsource(contextual)


def test_m1_is_never_retrained():
    source = inspect.getsource(contextual)
    assert "optimizer" not in source.lower() and ".backward(" not in source


def test_checkpoint_and_csv_hashes_are_fixed():
    configuration = cfg()
    assert configuration["data"]["checkpoint_sha256"] == "0f145cc0385bdbb6d8b638dde3c087cf757380c7f845dfdc45bb90d6fc4a5728"
    assert configuration["data"]["source_sha256"] == "d1dd05034b393a6b201f39e86ac3b933000551d26680c5d91c5ff8db0559cf72"
    assert "validate_m1_artifacts" in inspect.getsource(contextual.run)


def test_no_hai_dependency_and_outputs_are_ignored():
    source = inspect.getsource(contextual)
    assert "data.hai" not in source and "hai_" not in source.lower()
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "outputs/" in ignore and "dataset/" in ignore and "*.pt" in ignore
