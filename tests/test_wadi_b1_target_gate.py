"""Pre-Test contract checks for the empty WADI-B1 frozen target gate."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_wadi_b1_external"
M = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
S = json.loads((OUT / "wadi_b1_summary.json").read_text(encoding="utf-8"))
D = pd.read_csv(OUT / "normal_target_discovery_all.csv")
F = pd.read_csv(OUT / "dual_targets_frozen.csv")


def test_01_status(): assert S["status"] == "WADI_B1_STOPPED_NO_STABLE_DUAL_TARGETS"
def test_02_a2_only(): assert M["dataset"] == "WADI.A2"
def test_03_schema(): assert M["feature_count"] == 123
def test_04_exclusions(): assert set(M["excluded_features"]) == {"2_LS_001_AL","2_LS_002_AL","2_P_001_STATUS","2_P_002_STATUS"}
def test_05_scaler_train_only(): assert M["scaler_fit_rows"] == sum(x["rows"] for x in M["train_valid_runs"])
def test_06_candidate_count(): assert len(D) == S["candidate_target_count"] == 67
def test_07_stable_rule_exact():
    expected=(D.discovery_gain>=.05)&(D.validation_gain>=.05)&(D.validation_first_gain>=.02)&(D.validation_second_gain>=.02)
    assert np.array_equal(expected.to_numpy(), D.stable.to_numpy())
def test_08_empty_freeze(): assert F.empty and S["frozen_target_count"] == 0
def test_09_freeze_hash(): assert hashlib.sha256((OUT/"dual_targets_frozen.csv").read_bytes()).hexdigest() == M["target_freeze_sha256"]
def test_10_no_test(): assert M["test_opened_after_freeze"] is False and S["test_opened"] is False
def test_11_no_rec_training(): assert M["rec_training_started"] is False and S["rec_training_started"] is False
def test_12_no_search(): assert not any([S["lag_search"],S["threshold_search"],S["fusion_weight_search"],S["attack13_patch"]])
def test_13_no_attack_selection(): assert S["no_attack_driven_selection"] and S["no_test_tuning"]
def test_14_no_imputation_and_nan_breaks(): assert S["tests"]["no_imputation"] and S["tests"]["nan_breaks_preserved"]
def test_15_finite_discovery(): assert np.isfinite(D.select_dtypes(include=[np.number]).to_numpy()).all()
