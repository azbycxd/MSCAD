from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import hai_step3d_evaluate_rec_channel_calibration as step3d
from utils.reconstruction_scores import (
    CHANNEL_SCORE_VARIANTS,
    aggregate_endpoint_channel_scores,
    calibrate_endpoint_errors,
    endpoint_channel_errors,
    fit_channel_calibration,
)


ROOT = Path(__file__).resolve().parents[1]


def test_only_existing_three_checkpoints_are_registered():
    cfg = step3d.load_config(ROOT / "configs/hai_rec_channel_calibration.yaml")
    assert cfg["seeds"] == [1, 2, 3]
    assert all((ROOT / cfg["checkpoint_pattern"].format(seed=seed)).exists() for seed in cfg["seeds"])


def test_script_has_no_training_or_optimizer_step():
    source = inspect.getsource(step3d)
    assert "optimizer.step(" not in source
    assert ".train(" not in source


def test_checkpoint_audit_is_read_only():
    source = inspect.getsource(step3d.audit_checkpoints)
    assert "torch.load" in source and "torch.save" not in source


def test_endpoint_channel_error_is_exact():
    x = torch.tensor([[[1., 2.], [3., 5.]]])
    x_hat = torch.tensor([[[9., 9.], [1., 8.]]])
    assert torch.equal(endpoint_channel_errors(x, x_hat), torch.tensor([[2., 3.]]))


def test_endpoint_error_shape_validation():
    with pytest.raises(ValueError):
        endpoint_channel_errors(torch.zeros(2, 3), torch.zeros(2, 3))


def test_calibration_median_and_mad():
    values = np.array([[1., 2.], [2., 4.], [3., 6.]])
    fitted = fit_channel_calibration(values)
    np.testing.assert_allclose(fitted["residual_median"], [2., 4.])
    np.testing.assert_allclose(fitted["residual_mad"], [1., 2.])
    np.testing.assert_allclose(fitted["robust_scale"], [1.4826, 2.9652])


def test_calibration_fallback_order_iqr_std_unit():
    values = np.array([[0., 0., 1.], [0., 1., 1.], [0., 0., 1.], [0., 1., 1.]])
    fitted = fit_channel_calibration(values)
    assert fitted["fallback_method"][0] == "unit"
    assert fitted["effective_scale"][0] == 1.0
    assert fitted["fallback_method"][1] in {"mad", "iqr"}


def test_calibrated_error_is_nonnegative_and_finite():
    fitted = fit_channel_calibration(np.array([[1., 1.], [2., 1.], [3., 1.]]))
    result = calibrate_endpoint_errors(np.array([[0., 1.], [5., 2.]]), fitted)
    assert np.isfinite(result).all() and (result >= 0).all()


def test_calibration_uses_train_argument_only():
    signature = inspect.signature(fit_channel_calibration)
    assert "validation" not in signature.parameters and "test" not in signature.parameters


def test_all_seven_fixed_scores_are_generated():
    raw = np.arange(24, dtype=float).reshape(2, 12)
    scores, _ = aggregate_endpoint_channel_scores(raw, raw)
    assert tuple(scores) == CHANNEL_SCORE_VARIANTS


def test_raw_mean_and_fixed_topk_are_correct():
    raw = np.arange(12, dtype=float).reshape(1, 12)
    scores, details = aggregate_endpoint_channel_scores(raw, raw)
    assert scores["endpoint_raw_mean"][0] == pytest.approx(5.5)
    assert scores["endpoint_raw_top5"][0] == pytest.approx(np.mean([7, 8, 9, 10, 11]))
    assert scores["endpoint_calibrated_top3"][0] == pytest.approx(10.)
    assert scores["endpoint_calibrated_top10"][0] == pytest.approx(6.5)
    assert scores["endpoint_calibrated_max"][0] == 11
    assert details["calibrated_top5_indices"].shape == (1, 5)


def test_topk_is_row_local_and_label_free():
    signature = inspect.signature(aggregate_endpoint_channel_scores)
    assert "label" not in signature.parameters
    raw = np.zeros((2, 12)); raw[0, :2] = [9., 1.]; raw[1, 3:5] = [1., 9.]
    _, details = aggregate_endpoint_channel_scores(raw, raw)
    assert details["calibrated_top3_indices"][0, 0] == 0
    assert details["calibrated_top3_indices"][1, 0] == 4


def test_score_registry_marks_primary_before_results():
    registry = step3d.score_registry()
    row = registry[registry.score_variant == "endpoint_calibrated_top5"].iloc[0]
    assert row.role == "primary_precision_candidate"


def test_four_independent_thresholds_are_fitted():
    cfg = step3d.load_config(ROOT / "configs/hai_rec_channel_calibration.yaml")
    rows = step3d.fit_score_thresholds(np.linspace(0, 1, 1000),
                                      {f"train{i}": np.linspace(0, 1 + i / 10, 200) for i in range(1, 7)}, cfg)
    assert {row["threshold_method"] for row in rows} == set(step3d.THRESHOLD_METHODS)
    assert all(row["status"] == "ok" for row in rows)


def test_q99_q995_and_sequence_max_do_not_accept_test():
    signature = inspect.signature(step3d.fit_score_thresholds)
    assert "test" not in signature.parameters and "labels" not in signature.parameters


def test_ranking_metrics_are_threshold_independent():
    labels = np.array([0, 0, 1, 1]); scores = np.array([0.1, 0.4, 0.3, 0.8])
    low = step3d.metric_record(labels, scores, .2); high = step3d.metric_record(labels, scores, .7)
    assert low["aupr"] == high["aupr"] and low["auroc"] == high["auroc"]


def test_events_are_matched_within_file():
    source = inspect.getsource(step3d.build_event_diagnostics)
    assert 'splits["test"][event["file_name"]]' in source


def test_contribution_format_contains_indices_names_values():
    indices, names, values = step3d._format_top(np.array([2, 0]), np.array([3.5, 1.]), ["a", "b", "c"])
    assert indices == "2;0" and names == "c;a" and values.startswith("3.5;1")


def test_metadata_alignment_fields_are_checked():
    source = inspect.getsource(step3d.run)
    assert all(f'"{field}"' in source for field in ("sequence_id", "timestamp", "end_row", "label"))
    assert "step3c_score_pattern" in source


def test_no_structure_score_or_fusion_is_computed():
    registry = step3d.score_registry()
    assert not registry.score_variant.str.contains("struct|fusion", case=False, regex=True).any()


def test_smoke_and_formal_outputs_are_isolated():
    cfg = step3d.load_config(ROOT / "configs/hai_rec_channel_calibration.yaml")
    assert cfg["output_dir"] != cfg["smoke_output_dir"]


def test_formal_score_functions_reject_nonfinite_training_data():
    with pytest.raises(ValueError):
        fit_channel_calibration(np.array([[1., np.nan]]))


def test_required_output_registry_is_complete():
    assert len(step3d.OUTPUT_FILES) >= 18
    assert "rec_channel_summary.json" in step3d.OUTPUT_FILES


def test_config_has_fixed_k_values_only():
    cfg = step3d.load_config(ROOT / "configs/hai_rec_channel_calibration.yaml")
    assert cfg["score_variants"] == list(CHANNEL_SCORE_VARIANTS)
    assert not any("top" in key.lower() and key not in {"score_variants"} for key in cfg)
