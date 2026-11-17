"""Integrity checks for Innovation-2C0-B Historical CoMuT structure test."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_innovation2c0b_comut_structure"
SUMMARY = json.loads((OUT / "innovation2c0b_summary.json").read_text(encoding="utf-8"))
METRICS = pd.read_csv(OUT / "case_metrics.csv")
MANIFEST = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))


def test_historical_path(): assert Path(SUMMARY["historical_dataset_root"]).name == "CoMuT_historical_v1"
def test_compat_not_read(): assert SUMMARY["tests"]["compat_path_not_read"]
def test_thirteen_bindings(): assert len(METRICS) == 13 and len(MANIFEST["binding"]) == 13
def test_scaler_train_only(): assert SUMMARY["tests"]["scaler_train_only"]
def test_delta(): assert SUMMARY["tests"]["delta_definition_correct"]
def test_window(): assert SUMMARY["window"] == 64
def test_lags(): assert SUMMARY["lags"] == [0, 1, 5, 10, 20, 30]
def test_directed_pairs(): assert SUMMARY["tests"]["directed_pairs_exclude_self"]
def test_formula_reused(): assert SUMMARY["tests"]["formula_matches_2c0a"]
def test_train_feature_filter(): assert SUMMARY["tests"]["nondegenerate_train_only"]
def test_pca_fit_scope(): assert SUMMARY["tests"]["pca_train_first_80"]
def test_k95(): assert SUMMARY["tests"]["k95_rule_fixed"]
def test_calibration_train_only(): assert SUMMARY["tests"]["calibration_train_only"]
def test_no_pa(): assert not SUMMARY["point_adjustment"]
def test_finite(): assert np.isfinite(METRICS.select_dtypes("number")).all().all()
