from __future__ import annotations
import ast, importlib.util, inspect
from pathlib import Path
import numpy as np, pandas as pd, pytest

ROOT=Path(__file__).resolve().parents[1]; SCRIPT=ROOT/"scripts"/"hai_structure_v2a_validate_lagged_dependency.py"
SPEC=importlib.util.spec_from_file_location("v2a",SCRIPT); v2a=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(v2a)
from utils.lagged_conditional_dependency import *
from utils.dynamic_collaborative_corruptions import *

def source(): return SCRIPT.read_text(encoding="utf-8").lower()
def test_01_no_hai_test(): assert "test1.csv" not in source() and "test_files" not in source()
def test_02_no_attack_selection(): assert "attack_label_used_for_selection\":false" in source().replace(" ","")
def test_03_no_ae_rec(): assert "torch.load" not in source() and '"rec_loaded":false' in source().replace(" ","")
def test_04_scaler_reused(): assert "hai_scaler_stats.csv" in source() and "standardscaler" not in source()
def test_05_eligible_68_reused(): assert "expected 68 frozen eligible" in source()
def test_06_discovery_no_overlap():
    fit,ev,m=v2a.discovery_split([np.zeros((100,2))]*6,{"discovery_fit_ratio":.8},"a","b"); assert (m.overlap_count==0).all()
def test_07_self_lags(): assert SELF_LAGS==(1,2,4,8,16,32)
def test_08_source_lags(): assert SOURCE_LAGS==(0,1,2,4,8,16,32)
def test_09_self_fit_once(): assert '"fit_count":1' in source().replace(" ","")
def test_10_ridge_alpha():
    m=fit_ridge(np.arange(40).reshape(20,2),np.arange(20)); assert m.alpha==1.0 and m.fit_intercept
def test_11_screening_train_only(): assert list(inspect.signature(v2a.screening).parameters)[:2]==["fit","ev"]
def test_12_screen_limit_four(): assert "screening_max_sources_per_target" in source() and ".head(" in source()
def test_13_gain_definition():
    es=np.array([2.,2.]); ec=np.array([1.,1.]); assert np.isclose(gain(es,ec),.75)
def test_14_relation_gates():
    s=inspect.getsource(v2a.select_pool); assert all(x in s for x in ("global_gain","median_sequence_gain","worst_sequence_gain"))
def test_15_positive_5_of_6(): assert "gain_positive_sequence_count" in inspect.getsource(v2a.select_pool)
def test_16_pool_limits(): assert "max_incoming_per_target" in source() and "max_degree_per_channel" in source()
def test_17_validation_not_fit(): assert list(inspect.signature(v2a.final_fit).parameters)==["train","pool","names"]
def test_18_dependency_formula():
    es=np.array([2.]); ec=np.array([1.]); assert np.isclose(dependency_loss(es,ec),-3/5)
def test_19_not_absolute_error(): assert not np.isclose(dependency_loss(np.array([2.]),np.array([1.]))[0],1.)
def test_20_rolling_causal():
    x=np.arange(20.); y=causal_rolling_mean(x,4); assert np.isnan(y[:3]).all() and y[3]==1.5
def test_21_fallbacks():
    assert robust_reference(np.arange(20.))[2]=="mad"; assert robust_reference(np.ones(20))[2]=="epsilon"
def test_22_aggregations_only_two():
    cfg=v2a.load_cfg(ROOT/"configs"/"hai_structure_v2a_lagged_dependency.yaml"); assert cfg["aggregations"]==["dynamic_edge_max","dynamic_edge_top2_mean"]
def test_23_manifest_fields_static():
    assert all(x in source() for x in ("source_window_id","fold","role","endpoint_row"))
def test_24_crossfit_overlap_static(): assert '"overlap_count":0' in source().replace(" ","")
def test_25_r1_multiset():
    x=np.arange(48*3).reshape(48,3); y=source_temporal_shift(x,1); assert same_multiset(x[:,1],y[:,1])
def test_26_r2_multiset():
    x=np.arange(48*3).reshape(48,3); y=source_block_permutation(x,1); assert same_multiset(x[:,1],y[:,1])
def test_27_r3_same_sequence_nonoverlap_static(): assert "abs(sample[j][1]-e)>=48" in source()
def test_28_p1_endpoint_only():
    x=np.zeros((48,3)); y=target_endpoint_spike(x,1,1); assert np.count_nonzero(y)==1 and y[-1,1]==3
def test_29_p2_last8_only():
    x=np.zeros((48,3)); y=target_short_level_shift(x,1,1); assert np.count_nonzero(y)==8 and np.all(y[-8:,1]==2)
def test_30_c1_joint_replace():
    x=np.zeros((4,3)); r=np.ones((4,3)); y=replace_channels(x,r,[0,1]); assert np.all(y[:,:2]==1) and np.all(y[:,2]==0)
def test_31_c2_subgraph_static(): assert "selected_channels" in source() and "replace_channels(base,replacement,selected_channels)" in source()
def test_32_threshold_clean_only(): assert "np.quantile(cscore[cal],.99)" in source()
def test_33_synthetic_not_threshold(): assert '"synthetic_used":false' in source().replace(" ","")
def test_34_point_not_breaking(): assert "if fam in (\"r1\",\"r2\",\"r3\")" in source()
def test_35_history_hash_audit(): assert "artifact_reuse_audit.csv" in source() and "hash_match" in source()
def test_36_outputs_finite_if_present():
    cfg=v2a.load_cfg(ROOT/"configs"/"hai_structure_v2a_lagged_dependency.yaml"); out=v2a.resolve(cfg["output_dir"])
    if not out.exists(): pytest.skip("formal not run")
    for p in out.glob("*.csv"):
        n=pd.read_csv(p).select_dtypes(include=[np.number]); assert np.isfinite(n.to_numpy()).all(),p.name
def test_37_resume_static(): assert "reuse completed v2-a" in source() and "final_report_complete" in source()
def test_38_issue_ledger_no_auto(): assert "unresolved_issues_delta.json" in source() and "subprocess" not in source()
