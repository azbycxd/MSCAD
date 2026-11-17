"""Protocol and output checks for Innovation-2C0-A."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "thesis" / "02_collaboration"
sys.path.insert(0, str(HELPER))
import audit_innovation2c0a_dynamic_signature as audit  # noqa: E402

OUT = ROOT / "outputs" / "thesis_innovation2c0a_dependency_signature"
SUMMARY = json.loads((OUT / "innovation2c0a_summary.json").read_text(encoding="utf-8"))
STAGES = pd.read_csv(OUT / "stage_feasibility.csv")
FEATURES = pd.read_csv(OUT / "feature_statistics.csv")


def test_attack_not_read(): assert not SUMMARY["attack_data_read"] and not SUMMARY["attack_labels_read"]
def test_boundaries(): assert SUMMARY["train_raw_interval"] == [0, 396000] and SUMMARY["validation_raw_interval"] == [396000, 495000]
def test_scaler_train_only(): assert SUMMARY["tests"]["scaler_train_only"]
def test_official_stages(): assert tuple(STAGES.stage) == ("P1", "P2", "P3", "P4", "P5", "P6")
def test_constants_excluded(): assert SUMMARY["tests"]["constants_excluded"]
def test_delta_definition(): assert SUMMARY["tests"]["delta_definition"]
def test_window_fixed(): assert SUMMARY["window"] == 64
def test_lags_fixed(): assert SUMMARY["lags"] == [0, 1, 5, 10, 20, 30]
def test_directed_no_self(): assert SUMMARY["tests"]["directed_pairs_exclude_self"]


def test_dependency_formula():
    dz = np.arange(240, dtype=float).reshape(120, 2) / 10
    matrix, names = audit.signature_matrix(dz, np.array([94]), [0, 1], ["A", "B"])
    column = names.index("A->B@lag5")
    tau = np.arange(31, 95)
    assert np.isclose(matrix[0, column], np.mean(dz[tau - 5, 0] * dz[tau, 1]))


def test_pca_fit_scope(): assert SUMMARY["tests"]["pca_fit_train_first_80_only"]
def test_validation_not_feature_filter(): assert SUMMARY["tests"]["validation_not_used_for_feature_filter"]
def test_k95_rule(): assert SUMMARY["tests"]["k_from_95pct_variance_rule"]
def test_finite(): assert np.isfinite(STAGES.select_dtypes("number")).all().all()
def test_no_anomaly_metrics(): assert not SUMMARY["anomaly_metrics"] and SUMMARY["threshold"] is None
