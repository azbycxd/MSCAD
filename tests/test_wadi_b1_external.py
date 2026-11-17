"""Artifact contract for WADI-B1 frozen external validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_wadi_b1_external"
M = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
S = json.loads((OUT / "wadi_b1_summary.json").read_text(encoding="utf-8"))
if S.get("status") == "WADI_B1_STOPPED_NO_STABLE_DUAL_TARGETS":
    pytest.skip("full WADI-B1 evaluation did not pass the preregistered target gate", allow_module_level=True)
T = json.loads((OUT / "thresholds.json").read_text(encoding="utf-8"))


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_01_wadi_a2_only(): assert M["dataset"] == "WADI.A2"
def test_02_frozen_schema(): assert M["feature_count"] == 123 and len(M["feature_names"]) == 123
def test_03_four_exclusions(): assert set(M["excluded_features"]) == {"2_LS_001_AL","2_LS_002_AL","2_P_001_STATUS","2_P_002_STATUS"}
def test_04_exclusion_train_only(): assert S["tests"]["exclusion_normal_train_only"]
def test_05_no_imputation(): assert S["tests"]["no_imputation"]
def test_06_nan_breaks(): assert S["tests"]["nan_breaks"] and S["tests"]["no_cross_segment_window_lag"]
def test_07_scaler_train_only(): assert M["scaler_fit_rows"] == sum(x["rows"] for x in M["train_valid_runs"])
def test_08_discovery_normal_only(): assert S["tests"]["discovery_normal_only"] and M["test_opened_after_freeze"]
def test_09_stable_rules():
    d=pd.read_csv(OUT/"normal_target_discovery_all.csv")
    expected=(d.discovery_gain>=.05)&(d.validation_gain>=.05)&(d.validation_first_gain>=.02)&(d.validation_second_gain>=.02)
    assert np.array_equal(expected.to_numpy(), d.stable.to_numpy())
def test_10_target_freeze(): assert digest(OUT/"dual_targets_frozen.csv") == M["target_freeze_sha256"] and S["tests"]["targets_written_before_test"]
def test_11_high_ids(): assert set(pd.read_csv(OUT/"high_event_metrics.csv").attack_id)=={2,3,4,7,15}
def test_12_attack13_excluded(): assert 13 not in set(pd.read_csv(OUT/"event_metrics_14.csv").attack_id) and len(pd.read_csv(OUT/"event_metrics_14.csv"))==14
def test_13_self_lags(): assert M["self_lags"] == [1,5,10,20,30]
def test_14_cross_lags(): assert M["cross_lags"] == [0,1,5,10,20,30] and M["joint_lags"] == [1,5,10,20,30]
def test_15_target_excluded_cross(): assert S["tests"]["target_excluded_cross"]
def test_16_alpha(): assert M["alpha"] == 1.0
def test_17_dual_score(): assert M["dual_score"] == "max_j max(0,(abs(pred_self_j-pred_cross_j)-median_j)/scale_j)"
def test_18_rec_frozen():
    assert M["rec_model"]=={"window":64,"hidden":64,"latent":32,"dropout":.1,"loss":"SmoothL1","optimizer":"Adam","lr":.001,"weight_decay":.0001,"batch":256,"max_epochs":15,"patience":4}
def test_19_fusion_max(): assert M["fusion_score"] == "max(Z_rec,Z_dual)" and not S["fusion_weight_search"]
def test_20_thresholds_normal_only(): assert T["quantile"]==.99 and "Normal Validation" in T["source"]
def test_21_common_mask():
    g=pd.read_csv(OUT/"global_metrics.csv"); assert g[["tn","fp","fn","tp"]].sum(axis=1).nunique()==1 and S["tests"]["common_valid_mask"]
def test_22_no_pa(): assert not S["point_adjustment"] and S["tests"]["no_point_adjustment"]
def test_23_labels_eval_only(): assert S["tests"]["labels_evaluation_only"] and S["no_attack_driven_selection"]
def test_24_no_search_or_patch(): assert not any([S["lag_search"],S["threshold_search"],S["fusion_weight_search"],S["attack13_patch"],S["no_test_tuning"] is False])
def test_25_finite():
    for name in ("global_metrics.csv","event_metrics_14.csv","high_event_metrics.csv","point_complementarity.csv","category_summary.csv","cross_dataset_summary.csv"):
        f=pd.read_csv(OUT/name); assert np.isfinite(f.select_dtypes(include=[np.number]).to_numpy()).all(), name
