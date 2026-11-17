"""Integrity tests for the frozen Innovation-2D2 real-SWaT diagnostic."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "thesis" / "02_collaboration" / "run_innovation2d2_real_swat.py"
OUT = ROOT / "outputs" / "thesis_innovation2d2_real_swat"
SPEC = importlib.util.spec_from_file_location("innovation2d2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
MANIFEST = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
SUMMARY = MANIFEST["summary"]


def test_data_protocol_and_attack_scope() -> None:
    assert SUMMARY["tests"]["train_validation_test_protocol"]
    assert SUMMARY["tests"]["attack_evaluation_only"]
    assert MANIFEST["shared_endpoint_count"] == 449793


def test_frozen_targets_and_views() -> None:
    assert tuple(MANIFEST["targets"]) == MODULE.EXPECTED_TARGETS
    assert SUMMARY["tests"]["six_frozen_targets"]
    assert SUMMARY["tests"]["target_excluded_cross"]
    assert all(item["target_excluded_from_cross"] for item in MANIFEST["targets"].values())


def test_train_only_scaler_and_fixed_models() -> None:
    assert SUMMARY["tests"]["scaler_train_only"]
    assert SUMMARY["tests"]["self_lags_fixed"]
    assert SUMMARY["tests"]["cross_lags_fixed"]
    assert SUMMARY["tests"]["alpha_one"]
    for item in MANIFEST["targets"].values():
        assert item["model_fit_raw_interval"] == [0, 396000]
        assert item["self_lags"] == [1, 5, 10, 20, 30]
        assert item["cross_lags"] == [0, 1, 5, 10, 20, 30]
        assert item["alpha"] == 1.0


def test_disagreement_score_and_calibration() -> None:
    assert SUMMARY["tests"]["prediction_disagreement_only"]
    assert SUMMARY["tests"]["observed_target_residual_absent"]
    assert SUMMARY["tests"]["calibration_normal_validation_front_half"]
    assert SUMMARY["tests"]["threshold_calibration_only"]
    assert "abs(pred_self_j-pred_cross_j)" in MANIFEST["score_formula"]
    for item in MANIFEST["targets"].values():
        assert item["calibration_raw_interval"] == [396000, 445500]
        assert item["calibration_legal_interval"] == [396030, 445500]
        assert not item["observed_target_residual_in_score"]


def test_segment_mask_and_a0_binding() -> None:
    assert MANIFEST["test_segment_raw_bounds"] == [[0, 298837], [298837, 449919]]
    assert SUMMARY["tests"]["gap_not_crossed"]
    assert SUMMARY["tests"]["aligned_valid_mask"]
    assert SUMMARY["tests"]["a0_frozen"]
    assert SUMMARY["a0_threshold"] == 0.18396227061748505


def test_event_identity_and_contextual_ids() -> None:
    events = pd.read_csv(OUT / "event_metrics_36.csv")
    contextual = pd.read_csv(OUT / "contextual_events.csv")
    assert SUMMARY["tests"]["canonical_36"]
    assert len(events) == 36 and events["event_id"].nunique() == 36
    assert SUMMARY["tests"]["contextual_ids_exact"]
    assert set(contextual["event_id"]) == {13, 14, 29}


def test_no_adjustment_search_or_nonfinite_values() -> None:
    assert SUMMARY["tests"]["no_point_adjustment"]
    assert SUMMARY["tests"]["no_threshold_search"]
    assert SUMMARY["tests"]["no_nan_inf"]
    metrics = pd.read_csv(OUT / "global_metrics.csv").select_dtypes(include=[np.number])
    events = pd.read_csv(OUT / "event_metrics_36.csv").select_dtypes(include=[np.number])
    assert np.isfinite(metrics.to_numpy()).all()
    assert np.isfinite(events.to_numpy()).all()
    assert not SUMMARY["point_adjustment"] and not SUMMARY["fusion"]

