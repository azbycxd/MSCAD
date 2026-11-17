"""Contracts for WADI-B1-Final frozen external validation."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"/"thesis_wadi_b1_final_external"; B0=ROOT/"outputs"/"thesis_wadi_b0_unified_reliability"
M=json.loads((OUT/"manifest.json").read_text(encoding="utf-8")); S=json.loads((OUT/"wadi_b1_final_summary.json").read_text(encoding="utf-8")); T=json.loads((OUT/"thresholds.json").read_text(encoding="utf-8")); G=pd.read_csv(OUT/"global_metrics.csv"); E=pd.read_csv(OUT/"event_metrics_14.csv"); H=pd.read_csv(OUT/"high_event_metrics.csv")
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def test_01_b0_direct(): assert M["b0_target_file"]=="outputs/thesis_wadi_b0_unified_reliability/reliable_targets.csv"
def test_02_targets_47(): assert len(pd.read_csv(B0/"reliable_targets.csv"))==S["frozen_target_count"]==47
def test_03_target_hash(): assert digest(B0/"reliable_targets.csv")==M["b0_target_sha256"]==S["frozen_target_sha256"]
def test_04_wadi_a2(): assert M["dataset"]=="WADI.A2"
def test_05_features(): assert M["feature_count"]==len(M["features"])==123
def test_06_no_imputation(): assert M["summary"]["tests"]["no_imputation"]
def test_07_breaks(): assert len(M["train_valid_runs"])==6 and len(M["validation_valid_runs"])==3
def test_08_scaler_train_only(): assert M["scaler_fit_rows"]==sum(x["rows"] for x in M["train_valid_runs"])
def test_09_self_lags(): assert M["self_lags"]==[1,5,10,20,30]
def test_10_cross_lags(): assert M["cross_lags"]==[0,1,5,10,20,30]
def test_11_target_excluded(): assert S["tests"]["target_excluded_cross"]
def test_12_alpha(): assert M["alpha"]==1.0
def test_13_dual_formula(): assert M["dual_score"]=="max_j max(0,(abs(pred_self_j-pred_cross_j)-median_j)/scale_j)"
def test_14_rec(): assert M["rec"]=={"window":64,"hidden":64,"latent":32,"dropout":.1,"loss":"SmoothL1","optimizer":"Adam","lr":.001,"weight_decay":.0001,"batch":256,"max_epochs":15,"patience":4}
def test_15_fusion(): assert M["fusion_score"]=="max(Z_rec,Z_dual)"
def test_16_thresholds(): assert T["quantile"]==.99 and M["threshold_source"]=="Normal Validation only"
def test_17_first_test(): assert M["test_read_count"]==1 and json.loads((OUT/"pretest_freeze.json").read_text())["thresholds_written_before_test"]
def test_18_attack13(): assert M["attack13_patch"] is False and 13 not in set(E.attack_id)
def test_19_events(): assert len(E)==14 and set(E.attack_id)=={1,2,3,4,5,6,7,8,9,10,11,12,14,15}
def test_20_high(): assert set(H.attack_id)=={2,3,4,7,15}
def test_21_common_mask(): assert G[["tn","fp","fn","tp"]].sum(axis=1).nunique()==1 and G[["tn","fp","fn","tp"]].sum(axis=1).iloc[0]==S["valid_test_points"]
def test_22_no_pa(): assert S["point_adjustment"] is False
def test_23_no_search(): assert S["lag_search"] is False and S["no_test_tuning"] and S["no_target_change"]
def test_24_finite():
    for name in ("global_metrics.csv","event_metrics_14.csv","high_event_metrics.csv","category_summary.csv","point_complementarity.csv","event_complementarity.csv","target_attribution.csv","normal_domain_shift.csv","cross_dataset_summary.csv"):
        f=pd.read_csv(OUT/name); assert np.isfinite(f.select_dtypes(include=[np.number]).to_numpy()).all(),name
def test_25_labels_eval_only(): assert S["tests"]["labels_eval_only"]
def test_26_test_source_untouched(): assert M["attack_source_sha256_actual"]==M["attack_source_sha256_expected_from_a0"] and S["attack_source_sha256_verified"]
