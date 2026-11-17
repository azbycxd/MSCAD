"""Integrity checks for Innovation-2D0 dual-view disagreement."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_innovation2d0_dual_view_comut"
SUMMARY = json.loads((OUT / "innovation2d0_summary.json").read_text(encoding="utf-8"))
METRICS = pd.read_csv(OUT / "case_metrics.csv")
SANITY = pd.read_csv(OUT / "target_normal_sanity.csv")


def test_historical_path(): assert Path(SUMMARY["historical_dataset_root"]).name == "CoMuT_historical_v1"
def test_compat_not_read(): assert SUMMARY["tests"]["compat_not_read"]
def test_scaler_train_only(): assert SUMMARY["tests"]["scaler_train_only"]
def test_target_excluded_cross(): assert SUMMARY["tests"]["target_excluded_from_cross_view"]
def test_self_lags(): assert SUMMARY["self_lags"] == [1, 5, 10, 20, 30]
def test_cross_lags(): assert SUMMARY["cross_lags"] == [0, 1, 5, 10, 20, 30]
def test_alpha(): assert SUMMARY["alpha"] == 1.0
def test_chronology(): assert SUMMARY["tests"]["fit_calibration_chronology"]
def test_disagreement_only(): assert SUMMARY["tests"]["score_only_prediction_disagreement"]
def test_no_observed_residual_score(): assert not SUMMARY["observed_target_residual_in_score"]
def test_labels_eval_only(): assert SUMMARY["tests"]["labels_evaluation_only"]
def test_threshold_calibration_only(): assert SUMMARY["tests"]["threshold_train_calibration_only"]
def test_no_pa(): assert not SUMMARY["point_adjustment"]
def test_finite(): assert np.isfinite(METRICS.select_dtypes("number")).all().all() and np.isfinite(SANITY.select_dtypes("number")).all().all()
def test_comparators_read_only(): assert SUMMARY["tests"]["comparators_read_only"]
