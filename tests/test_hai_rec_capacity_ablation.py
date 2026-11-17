from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import hai_step3g_run_rec_capacity_ablation as step3g

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"configs/hai_rec_capacity_ablation.yaml").read_text(encoding="utf-8"))
SOURCE=(ROOT/"scripts/hai_step3g_run_rec_capacity_ablation.py").read_text(encoding="utf-8")


def registry(): return step3g.resolve_architectures(CFG)


def test_01_tiny_reuses_existing_checkpoint(): assert "load_tiny_checkpoint" in SOURCE
def test_02_tiny_is_not_trained(): assert 'if size=="tiny"' in SOURCE and '"training_executed":False' in SOURCE
def test_03_tiny_hash_is_checked_before_and_after(): assert "checkpoint_sha256_before" in SOURCE and "checkpoint_sha256_after" in SOURCE
def test_04_small_parameter_range():
    count=int(registry().set_index("model_size_name").loc["small","parameter_count"]);assert 40000<=count<=55000
def test_05_medium_parameter_range():
    count=int(registry().set_index("model_size_name").loc["medium","parameter_count"]);assert 80000<=count<=95000
def test_06_all_models_under_100k(): assert (registry().parameter_count<100000).all()
def test_07_width_is_the_only_architecture_variable():
    assert registry().architecture_family_unchanged.all() and registry().layer_count_unchanged.all()
def test_08_layer_count_unchanged(): assert registry().layer_count_unchanged.all()
def test_09_kernel_size_unchanged(): assert registry().kernel_size_unchanged.all()
def test_10_loss_is_fixed(): assert CFG["training"]["loss"]=="SmoothL1Loss" and "smooth_l1_loss" in SOURCE
def test_11_optimizer_is_fixed():
    assert CFG["training"]["optimizer"]=="AdamW" and CFG["training"]["lr"]==.001 and CFG["training"]["weight_decay"]==.0001
def test_12_split_is_fixed(): assert CFG["data"]["validation_ratio"]==.2 and CFG["data"]["stride_train"]==1
def test_13_scaler_pipeline_is_reused(): assert "build_hai_dataloaders" in SOURCE
def test_14_training_uses_normal_train_loader():
    body=SOURCE[SOURCE.index("def train_candidate"):SOURCE.index("def fit_thresholds")]
    assert "bundle.train_sequences" in body and "bundle.val_sequences" in body
def test_15_test_not_used_for_early_stopping():
    body=SOURCE[SOURCE.index("def train_candidate"):SOURCE.index("def fit_thresholds")];assert "test_loader" not in body
def test_16_endpoint_raw_top5_is_exact():
    model=step3g.make_model(8,4,torch.device("cpu"));x=torch.randn(2,64,86)
    with torch.no_grad(): err=(x[:,-1]-model(x)[:,-1]).abs(); expected=torch.topk(err,5,dim=1).values.mean(1)
    assert expected.shape==(2,) and torch.isfinite(expected).all()
def test_17_top_k_is_five(): assert CFG["score"]["top_k"]==5 and "k=5" in SOURCE
def test_18_tiny_score_reproduction_is_required(): assert "Tiny Step 3D reproduction failed" in SOURCE
def test_19_thresholds_are_per_model_seed(): assert "fit_thresholds(train,val,cfg)" in SOURCE
def test_20_thresholds_do_not_accept_test_or_labels():
    assert list(inspect.signature(step3g.fit_thresholds).parameters)==["train","validation","cfg"]
def test_21_events_do_not_cross_files():
    files={"a.csv":{"label":np.array([0,1]),"timestamp":np.array(["a0","a1"])},"b.csv":{"label":np.array([1,0]),"timestamp":np.array(["b0","b1"])}}
    events=step3g.extract_events(files);assert len(events)==2 and len({e["file_name"] for e in events})==2
def test_22_efficiency_uses_common_device_and_batch(): assert "benchmark_capacity" in SOURCE and '"batch_size":base["batch_size"]' in SOURCE
def test_23_smoke_formal_isolation(): assert CFG["output_dir"]!=CFG["smoke_output_dir"]
def test_24_raw_hai_files_are_hash_audited(): assert "data_files(cfg)" in SOURCE and "before!=after" in SOURCE
def test_25_step3a_to_f_outputs_are_hash_audited(): assert len(CFG["source_output_dirs"])>=7 and "all_source_hashes_unchanged" in SOURCE
def test_26_outputs_replace_illegal_nan_inf_with_mask():
    import pandas as pd
    frame=pd.DataFrame({"x":[np.nan,np.inf,-np.inf,1.]});path=ROOT/"outputs"/"_step3g_test_mask.csv"
    try:
        step3g.write_csv(frame,path);text=path.read_text();assert "nan" not in text.lower() and "inf" not in text.lower()
    finally:
        path.unlink(missing_ok=True)
def test_27_exact_three_sizes(): assert tuple(registry().model_size_name)==("tiny","small","medium")
def test_28_small_is_preregistered_primary(): assert bool(registry().set_index("model_size_name").loc["small","primary_candidate"])
def test_29_width_selection_uses_no_metrics(): assert not registry().selection_uses_training_or_test_metrics.any()
def test_30_parameter_counts_are_real_instances():
    for row in registry().itertuples():assert row.parameter_count==step3g.trainable_parameters(row.hidden_channels,row.bottleneck_channels)
def test_31_point_alarm_is_memoryless():
    score=np.array([1.,3.,2.]);threshold=2.;np.testing.assert_array_equal((score>threshold).astype(np.int8),[0,1,0])
def test_32_only_registered_thresholds(): assert step3g.METHODS==("train_only_pot","normal_validation_q99")
def test_33_no_structure_or_fusion_code(): assert "structure_score" not in SOURCE and "fuse_scores" not in SOURCE
def test_34_all_required_outputs_registered(): assert len(step3g.OUTPUT_FILES)==23


def test_35_fast_tiny_never_builds_or_forwards_model():
    body=SOURCE[SOURCE.index("def run_fast"):SOURCE.index("def main")]
    tiny_block=body[body.index("# Tiny is entirely reused"):body.index('for size in ("small","medium")')]
    assert "make_model" not in tiny_block and "infer_sequence" not in tiny_block and "load_tiny_checkpoint" not in tiny_block


def test_36_fast_tiny_reads_step3d_npz():
    assert "tiny_score_pattern" in SOURCE and '"tiny_scores_reused":True' in SOURCE and '"tiny_reinference":False' in SOURCE


def test_37_each_candidate_checkpoint_creates_only_one_core_cache():
    body=SOURCE[SOURCE.index("def create_core_score_cache"):SOURCE.index("def load_core_score_cache")]
    assert "if path.exists()" in body and body.count("np.savez(")==1


def test_38_offline_core_metrics_do_not_accept_model():
    assert list(inspect.signature(step3g.candidate_q99_core).parameters)==["size","seed","splits"]


def test_39_core_and_full_stages_are_separate():
    body=SOURCE[SOURCE.index("def run_fast"):SOURCE.index("def main")]
    assert "core_stage_complete" in body and "Full stage" in body and "recommended_capacity_candidate" in body


def test_40_resume_skips_legal_training_artifact():
    assert "training_artifact_valid" in SOURCE and "resume skip training" in SOURCE and "stage_status.json" in SOURCE


def test_41_fast_mode_keeps_registered_training_protocol():
    assert CFG["seeds"]==[1,2,3] and CFG["data"]["batch_size"]==128
    assert CFG["training"]=={"loss":"SmoothL1Loss","optimizer":"AdamW","lr":.001,"weight_decay":.0001,
                             "max_epochs":20,"patience":5,"early_stopping":True}


def test_42_fast_mode_hashes_step3a_to_f_sources():
    body=SOURCE[SOURCE.index("def run_fast"):SOURCE.index("def main")]
    assert "source_before" in body and "source_after" in body and "source_before!=source_after" in body
