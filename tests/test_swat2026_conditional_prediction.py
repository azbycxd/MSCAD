from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.swat2026_conditional_prediction import (
    ALPHAS, CROSS_LAGS, SELF_LAGS, aggregate_target_scores, chronological_discovery_split,
    circular_target_injection, cross_design, eligible_indices, mechanism_status,
    predictor_labels, robust_residual_reference, select_alpha, select_primary,
    self_history_design, validation_split,
)
from scripts.swat2026 import analyze_conditional_prediction as conditional


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "swat2026" / "conditional_prediction_seed1.yaml"


def cfg() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_cross_model_excludes_target_current_value():
    values = np.arange(400 * 4.0).reshape(400, 4)
    x, y, predictors = cross_design(values, 1, np.array([100]))
    assert 1 not in predictors and values[100, 1] not in x[0]


def test_cross_model_excludes_all_target_history():
    values = np.zeros((100, 3)); values[:, 1] = np.arange(100) + 1000
    x, _, _ = cross_design(values, 1, np.array([50]))
    assert not np.any(x >= 1000)


def test_self_model_contains_only_target_history():
    values = np.arange(400 * 3.0).reshape(400, 3)
    x, y = self_history_design(values, 1, np.array([100]))
    assert x.shape == (1, 5)
    assert x[0].tolist() == [values[99, 1], values[95, 1], values[90, 1], values[80, 1], values[70, 1]]


def test_fixed_lag_sets():
    assert CROSS_LAGS == (0, 5, 10, 20, 30)
    assert SELF_LAGS == (1, 5, 10, 20, 30)


def test_fixed_alpha_candidates():
    assert ALPHAS == (.01, .1, 1.0, 10.0, 100.0)


def test_alpha_selection_accepts_discovery_subsets_only():
    assert set(inspect.signature(select_alpha).parameters) == {"values", "target_index", "fit_indices", "selection_indices", "kind", "alphas"}


def test_discovery_split_is_chronological_not_random():
    fit, selection = chronological_discovery_split(np.arange(100), .8)
    assert fit.tolist() == list(range(80)) and selection.tolist() == list(range(80, 100))


def test_confirmation_does_not_reselect_alpha():
    source = inspect.getsource(conditional.discover_targets)
    assert "fit_target_discovery" in source
    assert "select_alpha" not in source


def test_validation_and_test_do_not_select_targets():
    assert set(inspect.signature(conditional.discover_targets).parameters) == {"cfg", "data", "values", "names"}
    source = inspect.getsource(conditional.discover_targets)
    assert "VALIDATION" not in source and "TEST_" not in source


def test_ait401_is_excluded_before_candidate_loop():
    source = inspect.getsource(conditional.run)
    assert 'data.scaler_metadata["zero_variance_features"]' in source
    assert "active_full_indices" in source


def test_bad_input_rows_and_histories_are_excluded():
    valid = np.ones(100, bool); valid[75] = False
    indices = eligible_indices(valid, 0, 100, CROSS_LAGS)
    assert 75 not in indices and 80 not in indices and 85 not in indices and 95 not in indices


def test_m1_train_only_scaler_is_reused():
    source = inspect.getsource(conditional.run)
    assert "load_recovery_data" in source
    assert "StandardScaler(" not in inspect.getsource(conditional)


def test_target_qualification_thresholds_are_fixed():
    q = cfg()["qualification"]
    assert q == {"minimum_confirmation_samples": 2048, "minimum_cross_r2": .20,
                 "minimum_median_improvement": .20, "minimum_prediction_correlation": .45,
                 "maximum_confirmation_selection_mae_ratio": 1.50, "general_target_count": 3}


def test_fewer_than_three_targets_is_narrow():
    assert mechanism_status(0) == "no_conditional_mechanism"
    assert mechanism_status(1) == mechanism_status(2) == "narrow_conditional_diagnostic"
    assert mechanism_status(3) == "general_conditional_candidate"


def test_complete_train_refit_has_no_validation_argument():
    assert "validation" not in inspect.signature(conditional.freeze_qualified_models).parameters
    assert "test" not in inspect.signature(conditional.freeze_qualified_models).parameters


def test_robust_residual_reference_uses_median_and_mad():
    residual = np.array([-1., 0., 1., 2.])
    reference = robust_residual_reference(residual, .05)
    absolute = np.abs(residual); center = np.median(absolute)
    assert reference["center"] == center
    assert reference["raw_scale"] == pytest.approx(1.4826 * np.median(np.abs(absolute - center)))


def test_validation_threshold_and_controlled_histories_do_not_overlap():
    manifest, split = validation_split(np.ones(3000, bool), 0, 3000, 30)
    assert manifest["raw_history_overlap"] is False
    assert not set(split["threshold"]) & set(split["controlled"])


def test_injection_preserves_target_value_multiset():
    window = np.arange(128 * 3.0).reshape(128, 3)
    changed = circular_target_injection(window, 1, 16)
    assert np.array_equal(np.sort(window[:, 1]), np.sort(changed[:, 1]))


def test_self_history_uses_injected_target_sequence():
    source = inspect.getsource(conditional.controlled_validation)
    assert "score_models(injected" in source
    assert '"self_history_uses_injected_target": True' in source


def test_cross_inputs_leave_other_channels_unchanged():
    source = inspect.getsource(conditional.controlled_validation)
    assert '"other_cross_channels_unchanged"' in source


def test_primary_selection_has_no_test_argument():
    assert "test" not in inspect.signature(select_primary).parameters


def test_primary_requires_three_targets_and_advantages():
    rows = pd.DataFrame([
        {"score_name": "cross_max", "auroc": .9, "coverage": 1., "marginal_statistics_preserved": True},
        {"score_name": "cross_top3_mean", "auroc": .85, "coverage": 1., "marginal_statistics_preserved": True},
        {"score_name": "self_max", "auroc": .7, "coverage": 1., "marginal_statistics_preserved": True},
        {"score_name": "self_top3_mean", "auroc": .7, "coverage": 1., "marginal_statistics_preserved": True},
        {"score_name": "rec_score", "auroc": .7, "coverage": 1., "marginal_statistics_preserved": True},
    ])
    assert not select_primary(rows, 2)["primary_frozen"]
    assert select_primary(rows, 3)["primary_frozen"]


def test_test_labels_are_read_only_after_primary_gate():
    source = inspect.getsource(conditional.formal_test_evaluation)
    assert source.index("if not primary") < source.index("point_labels_11mar.csv")


def test_test_first_point_can_use_context_history():
    assert conditional.TEST_CONTEXT_START < conditional.TEST_START
    source = inspect.getsource(conditional.formal_test_evaluation)
    assert "_score_available" in source


def test_formal_test_has_14400_points():
    assert cfg()["evaluation"]["test_points"] == 14400
    assert "exactly 14,400" in inspect.getsource(conditional.formal_test_evaluation)


def test_no_point_adjustment_or_fusion():
    assert cfg()["evaluation"]["point_adjustment"] is False and cfg()["evaluation"]["fusion"] is False
    assert "fusion_score" not in inspect.getsource(conditional)


def test_m1_is_not_retrained():
    source = inspect.getsource(conditional)
    assert "optimizer" not in source.lower() and ".backward(" not in source


def test_checkpoint_and_source_hashes_are_fixed():
    data = cfg()["data"]
    assert data["checkpoint_sha256"] == "0f145cc0385bdbb6d8b638dde3c087cf757380c7f845dfdc45bb90d6fc4a5728"
    assert data["source_sha256"] == "d1dd05034b393a6b201f39e86ac3b933000551d26680c5d91c5ff8db0559cf72"
    assert "validate_m1_artifacts" in inspect.getsource(conditional.run)


def test_no_hai_dependency_and_outputs_ignored():
    source = inspect.getsource(conditional)
    assert "data.hai" not in source and "hai_" not in source.lower()
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "outputs/" in ignore and "dataset/" in ignore and "*.pt" in ignore
