"""Integrity checks for Innovation-2D1 SWaT controlled validation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_innovation2d1_swat_controlled"
SUMMARY = json.loads((OUT / "innovation2d1_summary.json").read_text(encoding="utf-8"))
METRICS = pd.read_csv(OUT / "target_metrics.csv")
SANITY = pd.read_csv(OUT / "normal_sanity.csv")


def test_no_attack(): assert not SUMMARY["attack_data_read"]
def test_targets(): assert tuple(METRICS.target) == ("AIT202", "FIT201", "DPIT301", "FIT301", "FIT401", "FIT601")
def test_indices_reused(): assert SUMMARY["tests"]["b2b_sample_indices_reused"]
def test_donors_reused(): assert SUMMARY["tests"]["b2b_donors_reused"]
def test_corruption_unchanged(): assert SUMMARY["tests"]["target_current_history_corruption_unchanged"]
def test_train_scaler(): assert SUMMARY["tests"]["scaler_train_only"]
def test_self_lags(): assert SUMMARY["tests"]["self_lags_fixed"]
def test_cross_lags(): assert SUMMARY["tests"]["cross_lags_fixed"]
def test_target_excluded(): assert SUMMARY["tests"]["target_excluded_cross"]
def test_alpha(): assert SUMMARY["tests"]["alpha_1"]
def test_disagreement_score(): assert SUMMARY["tests"]["score_prediction_disagreement_only"]
def test_no_observed_residual_score(): assert not SUMMARY["observed_target_residual_in_score"]
def test_normal_calibration(): assert SUMMARY["tests"]["calibration_normal_only"]
def test_no_threshold_search(): assert not SUMMARY["threshold_search"]
def test_finite(): assert np.isfinite(METRICS.select_dtypes("number")).all().all() and np.isfinite(SANITY.select_dtypes("number")).all().all()
