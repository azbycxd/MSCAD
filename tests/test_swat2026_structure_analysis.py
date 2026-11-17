from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.swat2026_structure import (
    FISHER_CLIP,
    aggregate_deviation,
    circular_relation_injection,
    fisher_z,
    fit_robust_reference,
    relation_deviation,
    rolling_pair_correlations,
    screen_train_relations,
    select_relation_sets,
    stable_pairs,
    window_batch_pair_correlations,
)
from scripts.swat2026 import analyze_rec_and_structure as analysis


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "swat2026" / "independent_structure_seed1.yaml"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def relation_rows(count: int, strict_count: int | None = None) -> pd.DataFrame:
    strict_count = count if strict_count is None else strict_count
    rows = []
    for index in range(count):
        rows.append({
            "pair_name": f"F{index:02d}.Pv__F{index+1:02d}.Pv", "channel_i": f"F{index:02d}.Pv",
            "channel_j": f"F{index+1:02d}.Pv", "channel_i_index": index, "channel_j_index": index + 1,
            "valid_window_count": 100, "valid_window_ratio": 1.0 - index * .001,
            "median_abs_corr": .9 - index * .01, "mean_abs_corr": .9,
            "corr_median": .8, "corr_mad": .01, "corr_iqr": .1 + index * .001,
            "sign_consistency": 1.0, "activity_ratio_i": 1.0, "activity_ratio_j": 1.0,
            "local_strength_pass": True, "stability_pass": True,
            "strict_pass": index < strict_count, "expanded_pass": True,
        })
    return pd.DataFrame(rows)


def test_relation_screening_api_has_train_inputs_only():
    parameters = set(inspect.signature(screen_train_relations).parameters)
    assert parameters == {"train_values", "train_validity", "feature_names", "zero_variance_features", "window_size", "thresholds"}


def test_relation_screening_has_no_validation_or_test_labels():
    source = inspect.getsource(screen_train_relations)
    assert "validation" not in source.lower()
    assert "test" not in source.lower()
    assert "label" not in source.lower()


def test_ait401_is_excluded_from_candidates():
    names = ["LIT101.Pv", "AIT401.Pv", "FIT101.Pv"]
    pairs = stable_pairs(names, {"AIT401.Pv"})
    assert len(pairs) == 1
    assert not pairs.pair_name.str.contains("AIT401").any()


def test_relation_set_sorting_is_stable():
    frame = relation_rows(5).sample(frac=1, random_state=3)
    first = select_relation_sets(frame)["strict_set"].pair_name.tolist()
    second = select_relation_sets(frame.sample(frac=1, random_state=7))["strict_set"].pair_name.tolist()
    assert first == second


def test_fewer_than_three_strict_relations_does_not_expand_strict():
    sets = select_relation_sets(relation_rows(8, strict_count=2))
    assert len(sets["strict_set"]) == 2
    assert len(sets["expanded_set"]) == 8


def test_fisher_z_clips_correctly():
    result = fisher_z(np.array([-2.0, 0.0, 2.0]))
    assert np.isclose(result[0], np.arctanh(-FISHER_CLIP))
    assert result[1] == 0.0
    assert np.isclose(result[2], np.arctanh(FISHER_CLIP))


def test_robust_reference_uses_median_and_mad():
    correlations = np.tanh(np.array([[0.0], [1.0], [2.0], [100.0]]))
    reference = fit_robust_reference(correlations, .01)
    z = fisher_z(correlations)[:, 0]
    expected_center = np.median(z)
    expected_scale = 1.4826 * np.median(np.abs(z - expected_center))
    assert np.isclose(reference.center[0], expected_center)
    assert np.isclose(reference.raw_scale[0], expected_scale)


def test_scale_floor_is_applied():
    reference = fit_robust_reference(np.full((10, 1), .5), .05)
    assert reference.scale[0] == .05
    assert np.isfinite(relation_deviation(np.array([[.6]]), reference)).all()


def test_primary_is_strict_max_128():
    assert config()["relations"]["primary_score"] == "strict_max_128"
    maximum, _ = aggregate_deviation(np.array([[1.0, 3.0, 2.0]]))
    assert maximum[0] == 3.0


def test_window64_is_diagnostic_and_cannot_replace_primary():
    cfg = config()
    assert cfg["relations"]["diagnostic_window"] == 64
    assert "strict_max_window64" in cfg["relations"]["secondary_scores"]
    assert cfg["relations"]["primary_window"] == 128


def test_rolling_score_uses_supplied_window_end():
    values = np.column_stack([np.arange(10.0), np.arange(10.0)])
    pairs = stable_pairs(["A.Pv", "B.Pv"], set())
    correlations, _ = rolling_pair_correlations(values, np.array([4, 9]), pairs, 5)
    assert correlations.shape == (2, 1)
    assert np.allclose(correlations, 1.0)


def test_validation_threshold_api_has_no_test_argument():
    from scripts.swat2026.train_eval_rec_baseline import threshold_from_validation
    assert list(inspect.signature(threshold_from_validation).parameters) == ["validation_scores", "quantile"]


def test_circular_injection_preserves_value_multiset_and_marginals():
    window = np.column_stack([np.arange(128), np.sin(np.arange(128))])
    injected = circular_relation_injection(window, 1, 16)
    assert np.array_equal(np.sort(window[:, 1]), np.sort(injected[:, 1]))
    assert np.isclose(window[:, 1].mean(), injected[:, 1].mean())
    assert np.isclose(window[:, 1].std(), injected[:, 1].std())


def test_injection_changes_only_one_channel_and_time_relation():
    window = np.column_stack([np.arange(128.0), np.arange(128.0), np.ones(128)])
    injected = circular_relation_injection(window, 1, 32)
    assert np.array_equal(window[:, 0], injected[:, 0])
    assert np.array_equal(window[:, 2], injected[:, 2])
    assert not np.array_equal(window[:, 1], injected[:, 1])


def test_controlled_injection_function_has_no_test_input():
    assert list(inspect.signature(circular_relation_injection).parameters) == ["window", "channel_index", "shift"]


def test_controlled_auroc_is_correct_for_separable_scores():
    metrics = analysis.metrics_for_control(np.array([0.0, .1, .2]), np.array([.8, .9, 1.0]), .99)
    assert metrics["auroc"] == 1.0
    assert metrics["aupr"] == 1.0


def test_window_batch_correlations_are_independent_of_reconstruction():
    windows = np.stack([np.column_stack([np.arange(8.0), np.arange(8.0)])])
    pairs = stable_pairs(["A.Pv", "B.Pv"], set())
    assert np.allclose(window_batch_pair_correlations(windows, pairs), 1.0)
    assert "x_hat" not in inspect.signature(window_batch_pair_correlations).parameters


def test_formal_test_contract_is_14400_points():
    source = inspect.getsource(analysis.run)
    assert 'f"structure_scores_{split}.csv"' in source
    assert "M1 reconstruction and M2 structure timestamps do not align exactly" in source
    assert "14_400" in (ROOT / "data" / "swat2026_training.py").read_text(encoding="utf-8")


def test_no_point_adjustment():
    assert config()["evaluation"]["point_adjustment"] is False
    assert '"point_adjustment": False' in inspect.getsource(analysis.run)


def test_preexisting_alarm_is_recognized():
    test = pd.DataFrame({"timestamp": pd.date_range("2026-03-11 12:59:59", periods=4, freq="s"),
                         "score": [2.0, 3.0, 0.0, 0.0], "point_label": [0, 1, 1, 0]})
    schedule = pd.DataFrame([{"attack_id": 1, "attack_name": "a", "target": "x", "date": "2026-03-11",
                              "start_time": "13:00:00", "end_time": "13:00:02"}])
    result = analysis.event_diagnostics(test, schedule, "score", 1.0).iloc[0]
    assert bool(result.preexisting_alarm)
    assert result.first_detection_delay_seconds == 0.0


def test_point_and_event_complementarity_outputs_are_separate():
    source = inspect.getsource(analysis.run)
    assert "point_level_complementarity.csv" in source
    assert "event_level_complementarity.csv" in source


def test_low_correlation_alone_never_claims_complementarity():
    source = inspect.getsource(analysis.run)
    assert '"low_correlation_alone_claims_complementarity": False' in source


def test_no_fusion_score_is_generated():
    assert config()["evaluation"]["fusion"] is False
    source = inspect.getsource(analysis)
    assert "fuse_scores(" not in source


def test_m2_never_retrains_model():
    source = inspect.getsource(analysis)
    assert "optimizer" not in source.lower()
    assert ".backward(" not in source
    assert "train_epoch(" not in source
    assert "requires_grad_(False)" in source


def test_m2_never_writes_raw_csv():
    source = inspect.getsource(analysis)
    assert "to_csv(resolve(cfg[\"data\"][\"data_path\"]" not in source
    assert "write_text(resolve(cfg[\"data\"][\"data_path\"]" not in source


def test_m2_does_not_import_hai_or_old_relation_sets():
    source = inspect.getsource(analysis)
    assert "data.hai" not in source
    assert "hai_" not in source.lower()
    assert '"old_relation_sets_used": False' in source


def test_checkpoint_hash_mismatch_blocks(tmp_path: Path):
    m1 = tmp_path / "m1"; (m1 / "checkpoints").mkdir(parents=True)
    for name in analysis.REQUIRED_M1:
        path = m1 / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (m1 / "batch_m1_readiness.json").write_text(
        json.dumps({"classification": "A. single_seed_rec_baseline_complete"}), encoding="utf-8")
    cfg = {"data": {"m1_dir": str(m1), "checkpoint": str(m1 / "checkpoints" / "best_seed1.pt"),
                    "checkpoint_sha256": "not-the-real-hash"}}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        analysis.validate_m1_artifacts(cfg)


def test_outputs_checkpoints_and_data_are_git_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "outputs/" in ignore and "*.pt" in ignore and "dataset/" in ignore
