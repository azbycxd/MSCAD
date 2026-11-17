from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hai_step3b_aggregate_rec_multiseed.py"
SPEC = importlib.util.spec_from_file_location("hai_multiseed", SCRIPT)
aggregate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(aggregate)


def configs():
    return {seed: json.loads((ROOT / "configs" / f"hai_rec_only_seed{seed}.yaml").read_text(encoding="utf-8"))
            for seed in (1, 2, 3)}


def synthetic_arrays():
    result = {}
    for seed in (1, 2, 3):
        values = {}
        for file_index, file_name in enumerate(aggregate.FILES):
            score = np.linspace(0, 1, 8) + seed * 0.01 + file_index * 0.001
            values[f"{file_name}_score"] = score
            values[f"{file_name}_label"] = np.array([0, 0, 0, 1, 1, 0, 1, 0])
            values[f"{file_name}_timestamp"] = np.array([f"{file_name}_t{i}" for i in range(8)])
            values[f"{file_name}_sequence_id"] = np.repeat(file_name, 8)
            values[f"{file_name}_end_row"] = np.arange(63, 71)
            for method in aggregate.METHODS:
                values[f"{file_name}_pred_{method}"] = (score > 0.5).astype(int)
        result[seed] = values
    return result


def test_configs_only_allow_seed_run_and_output_differences():
    frame, valid = aggregate.config_diff_frame(configs())
    assert valid
    assert set(frame.loc[~frame.all_equal, "config_key"]).issubset(aggregate.ALLOWED_CONFIG_DIFFERENCES)


def test_three_seeds_use_identical_train_validation_window_settings():
    cfg = configs()
    keys = ("data_root", "window_size", "stride_train", "stride_val", "val_ratio", "label_mode")
    assert all(tuple(cfg[seed]["data"][key] for key in keys) == tuple(cfg[1]["data"][key] for key in keys)
               for seed in (2, 3))


def test_timestamp_and_label_alignment_is_exact():
    arrays = synthetic_arrays()
    aggregate.validate_alignment(arrays)
    arrays[2]["test2_timestamp"] = arrays[2]["test2_timestamp"][::-1]
    try:
        aggregate.validate_alignment(arrays)
    except ValueError as exc:
        assert "Alignment mismatch" in str(exc)
    else:
        raise AssertionError("Reordered timestamps must fail")


def test_seed1_formal_result_is_read_only_input():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "RUN_DIRS" in source
    assert "ensure_fresh(output)" in source
    assert "to_csv(RUN_DIRS" not in source and "write_text(RUN_DIRS" not in source


def test_seed_outputs_are_isolated():
    paths = [configs()[seed]["output"]["dir"] for seed in (1, 2, 3)]
    assert len(set(paths)) == 3


def test_seed1_aupr_auroc_do_not_depend_on_threshold():
    frame = pd.read_csv(ROOT / "outputs" / "hai_step3a_rec_baseline_seed1" / "test_point_metrics_summary.csv")
    for scope, group in frame.groupby("summary_scope"):
        assert group.aupr.nunique() == 1, scope
        assert group.auroc.nunique() == 1, scope


def test_score_correlation_rejects_length_mismatch():
    arrays = synthetic_arrays()
    arrays[3]["test1_score"] = arrays[3]["test1_score"][:-1]
    try:
        aggregate.validate_alignment(arrays)
    except ValueError as exc:
        assert "Length mismatch" in str(exc)
    else:
        raise AssertionError("Length mismatch must fail")


def test_pearson_and_spearman_are_correct():
    frame = aggregate.score_correlations(synthetic_arrays())
    assert np.allclose(frame.pearson, 1.0)
    assert np.allclose(frame.spearman, 1.0)


def test_alarm_jaccard_is_correct():
    left = np.array([1, 1, 0, 0])
    right = np.array([1, 0, 1, 0])
    assert np.isclose(aggregate.alarm_jaccard(left, right), 1 / 3)


def test_events_align_by_sequence_and_event_id():
    source = inspect.getsource(aggregate.event_consistency)
    assert '"sequence_id"' in source and '"event_id"' in source and '"global_event_id"' in source
    assert "base_keys.equals" in source


def test_events_never_cross_files():
    source = inspect.getsource(aggregate.event_consistency)
    assert 'groupby(["threshold_method", "file_name"])' in source


def test_event_classification_is_correct():
    assert aggregate.event_classification(3) == "robust_detected"
    assert aggregate.event_classification(2) == "unstable_detected"
    assert aggregate.event_classification(1) == "unstable_detected"
    assert aggregate.event_classification(0) == "consistently_missed"


def test_mean_and_population_std_are_correct():
    values = pd.Series([1.0, 2.0, 3.0])
    assert values.mean() == 2.0
    assert np.isclose(values.std(ddof=0), np.sqrt(2 / 3))
    source = SCRIPT.read_text(encoding="utf-8")
    assert "std(ddof=0)" in source


def test_vote_is_not_generated_as_formal_metric():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"multiseed_vote_used_as_detection_method": False' in source
    assert "diagnostic_only_not_a_detection_method" in source


def test_no_structure_model_is_loaded_or_trained():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "torch.load" not in source
    assert "optimizer" not in source
    assert "short_score" not in source and "long_score" not in source


def test_aggregation_does_not_use_test_labels_for_tuning():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "best threshold" not in source
    assert "argmax" not in source
    assert "test_score_arrays" in source
    assert "threshold_summary.json" in source
