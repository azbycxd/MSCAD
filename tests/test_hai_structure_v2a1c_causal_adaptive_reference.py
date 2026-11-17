from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.hai_structure_v2a1c_causal_adaptive_reference as stage


def tiny_stream(values=None):
    values = np.asarray(values if values is not None else [[1.0, 2.0], [1.2, 2.2], [8.0, 2.1]], float)
    return stage.run_adaptive_stream(
        values,
        np.asarray(["s1"] * len(values)),
        np.asarray([1.0, 2.0]),
        np.asarray([1.0, 1.0]),
        np.asarray([0, 1]),
        stage.alpha_from_half_life(512),
        99.0,
        np.asarray([99.0, 99.0]),
        True,
    )


def test_01_only_two_half_lives():
    assert stage.HALF_LIVES == (512, 2048)


def test_02_alpha_512_exact():
    assert stage.alpha_from_half_life(512) == pytest.approx(1 - np.exp(np.log(0.5) / 512))


def test_03_alpha_2048_exact():
    assert stage.alpha_from_half_life(2048) == pytest.approx(1 - np.exp(np.log(0.5) / 2048))


def test_04_rejects_third_half_life():
    with pytest.raises(ValueError):
        stage.alpha_from_half_life(1024)


def test_05_reference_initialization():
    result = tiny_stream()
    np.testing.assert_allclose(result["location_before"][0], [1.0, 2.0])
    np.testing.assert_allclose(result["scale_before"][0], [1.0, 1.0])


def test_06_score_is_computed_before_update():
    result = tiny_stream([[2.0, 2.0]])
    assert result["edge_score"][0, 0] == pytest.approx(1.0)


def test_07_huber_clip():
    loc, _, _ = stage.adaptive_update(
        np.asarray([100.0]), np.asarray([0.0]), np.asarray([1.0]), np.asarray([1.0]),
        np.asarray([1.0]), 1.0, np.asarray([True]),
    )
    assert loc[0] == pytest.approx(stage.LOCATION_STEP_CAP)


def test_08_location_step_cap():
    loc, _, _ = stage.adaptive_update(
        np.asarray([-100.0]), np.asarray([0.0]), np.asarray([1.0]), np.asarray([2.0]),
        np.asarray([2.0]), 1.0, np.asarray([True]),
    )
    assert loc[0] == pytest.approx(-0.1)


def test_09_scale_ratio_upper_cap():
    _, _, scale = stage.adaptive_update(
        np.asarray([100.0]), np.asarray([0.0]), np.asarray([100.0]), np.asarray([1.0]),
        np.asarray([1.0]), 1.0, np.asarray([True]),
    )
    assert scale[0] == pytest.approx(1.02)


def test_10_scale_ratio_lower_cap():
    _, _, scale = stage.adaptive_update(
        np.asarray([0.0]), np.asarray([0.0]), np.asarray([0.0]), np.asarray([1.0]),
        np.asarray([1.0]), 1.0, np.asarray([True]),
    )
    assert scale[0] == pytest.approx(0.98)


def test_11_scale_floor():
    _, _, scale = stage.adaptive_update(
        np.asarray([0.0]), np.asarray([0.0]), np.asarray([0.0]), np.asarray([1.0]),
        np.asarray([4.0]), 1.0, np.asarray([True]),
    )
    assert scale[0] >= 1.0


def test_12_update_mask_freezes_state():
    loc, dev, scale = stage.adaptive_update(
        np.asarray([5.0]), np.asarray([1.0]), np.asarray([2.0]), np.asarray([3.0]),
        np.asarray([3.0]), 0.5, np.asarray([False]),
    )
    np.testing.assert_allclose([loc[0], dev[0], scale[0]], [1.0, 2.0, 3.0])


def test_13_global_freeze_blocks_all_edges():
    result = stage.run_adaptive_stream(
        np.asarray([[10.0, 10.0]]), np.asarray(["s1"]), np.zeros(2), np.ones(2),
        np.asarray([0, 1]), stage.alpha_from_half_life(512), 1.0, np.full(2, 99.0), True,
    )
    assert result["global_frozen"][0]
    assert not result["updated"][0].any()


def test_14_edge_freeze_is_local():
    result = stage.run_adaptive_stream(
        np.asarray([[10.0, 0.1]]), np.asarray(["s1"]), np.zeros(2), np.ones(2),
        np.asarray([0, 1]), stage.alpha_from_half_life(512), 99.0, np.asarray([1.0, 1.0]), True,
    )
    assert result["updated"][0].tolist() == [False, True]


def test_15_new_sequence_resets_reference():
    result = stage.run_adaptive_stream(
        np.asarray([[2.0, 2.0], [2.0, 2.0]]), np.asarray(["s1", "s2"]),
        np.zeros(2), np.ones(2), np.asarray([0, 1]), stage.alpha_from_half_life(512),
        99.0, np.full(2, 99.0), True,
    )
    np.testing.assert_allclose(result["location_before"][1], 0.0)


def test_16_sequence_slices_preserve_boundaries():
    slices = stage.sequence_slices(np.asarray(["a", "a", "b", "b", "b"]))
    assert [(name, part.start, part.stop) for name, part in slices] == [("a", 0, 2), ("b", 2, 5)]


def test_17_update_gates_are_train_quantiles():
    values = np.arange(400, dtype=float).reshape(200, 2)
    global_gate, edge_gates, _ = stage.fit_update_gates(values, np.zeros(2), np.ones(2), np.asarray([0, 1]))
    assert edge_gates[0] == pytest.approx(np.quantile(values[:, 0], 0.995))
    assert np.isfinite(global_gate)


def test_18_candidate_selection_prefers_2048_on_tie():
    frame = pd.DataFrame({"half_life": [512, 2048], "train_evaluation_worst_sequence_fpr": [0.02, 0.021], "valid": [True, True]})
    assert stage.select_candidate(frame) == 2048


def test_19_candidate_selection_excludes_invalid():
    frame = pd.DataFrame({"half_life": [512, 2048], "train_evaluation_worst_sequence_fpr": [0.01, 0.02], "valid": [False, True]})
    assert stage.select_candidate(frame) == 2048


def test_20_no_valid_candidate_stops_selection():
    frame = pd.DataFrame({"half_life": [512, 2048], "train_evaluation_worst_sequence_fpr": [0.01, 0.02], "valid": [False, False]})
    assert stage.select_candidate(frame) is None


def test_21_source_window_mapping():
    ids = stage.source_window_ids(np.asarray(["validation1", "validation6"]), np.asarray([47, 99]))
    assert ids.tolist() == ["train1_val_row_47", "train6_val_row_99"]


def test_22_controlled_scoring_does_not_update_reference():
    source = inspect.getsource(stage.controlled_scores)
    assert "adaptive_update" not in source


def test_23_persistent_test_runs_update_gate():
    source = inspect.getsource(stage.persistent_absorption)
    assert "update_mask" in source and "freeze_all" in source


def test_24_score_retention_formula():
    source = inspect.getsource(stage.persistent_absorption)
    assert "global_scores / np.maximum(initial_score" in source


def test_25_relation_pool_guard_is_12():
    source = inspect.getsource(stage.run)
    assert "len(pool) != 12" in source


def test_26_grouped_aggregation_is_frozen():
    assert "grouped_scores" in inspect.getsource(stage.run_adaptive_stream)
    assert "target_group_max_then_top2_mean" in inspect.getsource(stage.run)


def test_27_no_ridge_fit_or_model_load():
    source = Path(stage.__file__).read_text(encoding="utf-8").lower()
    assert ".fit(" not in source
    assert "joblib" not in source
    assert "torch" not in source


def test_28_no_raw_csv_loader_or_hai_test_path():
    source = inspect.getsource(stage.run)
    assert "data_root" not in source
    assert "pd.read_csv(path" not in source
    assert "test1" not in source and "test2" not in source


def test_29_no_attack_label_ae_or_rec_score_load():
    source = inspect.getsource(stage.run).lower()
    assert "attack_label" not in source.split("selected_config =", 1)[0]
    assert "torch.load" not in source and "joblib.load" not in source
    assert "rec_score" not in source


def test_30_resume_precedes_cache_loading():
    source = inspect.getsource(stage.run)
    assert source.index("REUSE completed V2-A1C") < source.index("cache_payload")


def test_31_output_cache_validation_detects_missing():
    missing, finite = stage.validate_cache({"a": np.asarray([1.0])}, ["a", "b"])
    assert missing == ["b"] and finite


def test_32_output_cache_validation_detects_nonfinite():
    missing, finite = stage.validate_cache({"a": np.asarray([np.inf])}, ["a"])
    assert not missing and not finite


def test_33_absorption_gate_thresholds():
    rows = []
    for family in ("R1", "R2", "R3"):
        rows.extend([
            {"family": family, "step": 256, "median_score_retention": 0.8, "detection_retention_change": -0.01},
            {"family": family, "step": 1024, "median_score_retention": 0.6, "detection_retention_change": -0.02},
        ])
    assert all(stage.absorption_gates(pd.DataFrame(rows)).values())


def test_34_historical_hash_function_is_read_only():
    source = inspect.getsource(stage.historical_hashes)
    assert "write" not in source and "unlink" not in source


def test_35_frozen_cache_files_exist():
    for name in ("train_raw_dynamic_dependency_cache.npz", "validation_raw_dynamic_dependency_cache.npz", "controlled_raw_dynamic_dependency_cache.npz"):
        assert (stage.A1M / name).exists()


def test_36_frozen_relation_count_is_12():
    assert len(pd.read_csv(stage.V2A / "selected_dynamic_relation_pool.csv")) == 12


def test_37_historical_cache_hashes_are_stable_during_read():
    path = stage.A1M / "validation_raw_dynamic_dependency_cache.npz"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as archive:
        assert archive["rolling_dependency_loss"].shape[1] == 12
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after


def test_38_all_checkpoint_steps_are_preregistered():
    assert stage.CHECKPOINT_STEPS == (1, 16, 64, 256, 1024)


def test_39_no_sequence_specific_threshold():
    source = Path(stage.__file__).read_text(encoding="utf-8")
    assert "train_calibration_q99_threshold" in source
    assert "sequence_specific_threshold" not in source


def test_40_non_a_classification_is_not_frozen():
    source = inspect.getsource(stage.run)
    assert 'else "not_frozen"' in source
