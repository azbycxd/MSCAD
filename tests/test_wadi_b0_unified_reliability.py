"""Contracts for WADI-B0 Normal-only unified reliability audit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"/"thesis_wadi_b0_unified_reliability"
M=json.loads((OUT/"reliability_manifest.json").read_text(encoding="utf-8")); S=json.loads((OUT/"wadi_b0_summary.json").read_text(encoding="utf-8")); A=pd.read_csv(OUT/"all_targets.csv"); R=pd.read_csv(OUT/"reliable_targets.csv")

def test_01_test_not_read(): assert M["attack_file_path_loaded"] is False and S["tests"]["wadi_test_file_open_attempts"]==0
def test_02_schema(): assert M["feature_count"]==123
def test_03_no_impute(): assert S["tests"]["no_imputation"]
def test_04_segments(): assert S["tests"]["no_segment_crossing"] and len(M["train_valid_runs"])==6 and len(M["validation_valid_runs"])==3
def test_05_candidates(): assert len(A)==67 and A.cross_source_count.gt(0).all()
def test_06_baseline(): assert M["baseline"]=="median(target) fit on matching Normal fit partition only"
def test_07_self_lags(): assert M["self_lags"]==[1,5,10,20,30]
def test_08_cross_lags(): assert M["cross_lags"]==[0,1,5,10,20,30]
def test_09_target_excluded(): assert S["tests"]["target_excluded_cross"]
def test_10_alpha(): assert M["alpha"]==1.0
def test_11_skill_formula():
    assert np.allclose(A.discovery_self_skill,1-A.discovery_mae_self/A.discovery_mae_base)
    assert np.allclose(A.discovery_cross_skill,1-A.discovery_mae_cross/A.discovery_mae_base)
def test_12_validation_isolated(): assert S["tests"]["validation_not_discovery_fit"]
def test_13_ratios():
    med=np.maximum(A.median_disagreement_first/np.maximum(A.median_disagreement_second,1e-12),A.median_disagreement_second/np.maximum(A.median_disagreement_first,1e-12)); assert np.allclose(med,A.median_ratio)
def test_14_old_not_selection():
    expected=(A.discovery_self_skill>0)&(A.discovery_cross_skill>0)&(A.validation_full_self_skill>0)&(A.validation_full_cross_skill>0)&(A.validation_first_self_skill>0)&(A.validation_first_cross_skill>0)&(A.validation_second_self_skill>0)&(A.validation_second_cross_skill>0)&(A.median_ratio<=2)&(A.q95_ratio<=3)
    assert np.array_equal(expected.to_numpy(),A.reliable.to_numpy())
def test_15_finite(): assert np.isfinite(A.select_dtypes(include=[np.number]).to_numpy()).all()
def test_16_source_has_no_attack_csv_access():
    source=(ROOT/"scripts"/"thesis"/"02_collaboration"/"audit_wadi_b0_unified_reliability.py").read_text(encoding="utf-8")
    assert "WADI_attackdataLABLE.csv" not in source
    assert source.count("pd.read_csv(NORMAL") == 1
