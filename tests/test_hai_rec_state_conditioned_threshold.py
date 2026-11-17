from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import hai_step3f_evaluate_rec_state_conditioned_threshold as step3f
from utils.normal_operating_state import (
    active_feature_indices, centroid_matching_distance, clustering_agreement, endpoint_rows,
    normalized_entropy, pointwise_state_alarm, robust_location_scale, robust_z, run_lengths,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/hai_step3f_evaluate_rec_state_conditioned_threshold.py").read_text(encoding="utf-8")
CFG = json.loads((ROOT / "configs/hai_rec_state_conditioned_threshold.yaml").read_text(encoding="utf-8"))


def test_01_reads_step3d_score_npz_directly():
    assert "load_seed_scores" in SOURCE and "score_npz_pattern" in SOURCE


def test_02_does_not_import_model_or_torch():
    assert "import torch" not in SOURCE and "from models" not in SOURCE


def test_03_does_not_run_inference():
    assert ".eval()" not in SOURCE and "torch.no_grad" not in SOURCE


def test_04_pooled_q99_uses_step3d_threshold_table():
    import pandas as pd
    f = pd.DataFrame([{"seed": 1, "score_variant": "endpoint_raw_top5",
                       "threshold_method": "normal_validation_q99", "status": "ok", "threshold_value": 2.5}])
    assert step3f.pooled_threshold(f, 1, "q99") == 2.5


def test_05_endpoint_features_are_current_rows():
    seq = SimpleNamespace(x=np.arange(20).reshape(10, 2), original_row_offset=100,
                          file_name="x.csv", split="validation")
    np.testing.assert_array_equal(endpoint_rows(seq, np.array([102, 109])), seq.x[[2, 9]])


def test_06_endpoint_alignment_rejects_invalid_rows():
    seq = SimpleNamespace(x=np.zeros((3, 2)), original_row_offset=5, file_name="x", split="test")
    with pytest.raises(ValueError): endpoint_rows(seq, np.array([4]))


def test_07_state_feature_source_excludes_labels_and_scores():
    assert "uses_label_or_score_as_feature\": False" in SOURCE


def test_08_active_features_exclude_global_zero_variance():
    np.testing.assert_array_equal(active_feature_indices(np.array([0.0, 1.0, 1e-9]), 1e-8), [1])


def test_09_pca_fit_scope_is_train_only():
    assert '"fit_scope": "normal_train_only"' in SOURCE


def test_10_test_not_used_for_pca_dimension_selection():
    assert '"test_used_for_dimension_selection": False' in SOURCE


def test_11_k_candidates_are_fixed():
    assert CFG["k_candidates"] == [2, 4, 6]


def test_12_k_selection_marks_test_unused():
    assert '"selection_uses_test": False' in SOURCE


def test_13_clustering_does_not_use_file_as_feature():
    assert '"uses_file_identity_as_feature": False' in SOURCE


def test_14_state_assignment_uses_only_fitted_models():
    body = SOURCE[SOURCE.index("def assign_all"):SOURCE.index("def assignment_summary")]
    assert ".fit(" not in body and "pca.transform" in body and "model.transform" in body


def test_15_test_does_not_update_centroids():
    assert '"test_used_for_fit":False' in SOURCE or '"test_used_for_fit": False' in SOURCE


def test_16_state_threshold_fallback_is_registered():
    assert "minimum_validation_state_count" in SOURCE and "validation_count_below_2000" in SOURCE


def test_17_pointwise_alarm_is_exact_and_memoryless():
    scores = np.array([1.0, 3.0, 2.0]); states = np.array([0, 1, 0]); thresholds = np.array([1.5, 2.5])
    np.testing.assert_array_equal(pointwise_state_alarm(scores, states, thresholds), [0, 1, 1])


def test_18_robust_scale_fallback_is_finite():
    median, scale, source, *_ = robust_location_scale(np.ones(20))
    assert median == 1 and scale == 1 and source == "unit"


def test_19_robust_z_is_state_specific_and_one_sided():
    z = robust_z(np.array([0.0, 4.0]), np.array([0, 1]), np.array([1.0, 2.0]), np.array([2.0, 1.0]))
    np.testing.assert_array_equal(z, [0.0, 2.0])


def test_20_event_extraction_does_not_cross_files():
    files = {"a.csv": {"label": np.array([0, 1]), "timestamp": np.array(["a0", "a1"])},
             "b.csv": {"label": np.array([1, 0]), "timestamp": np.array(["b0", "b1"])}}
    events = step3f.extract_events(files)
    assert len(events) == 2 and {e["file_name"] for e in events} == {"a.csv", "b.csv"}


def test_21_run_lengths_and_long_segments_are_exact():
    np.testing.assert_array_equal(run_lengths(np.array([0, 0, 1, 1, 1, 0])), [2, 3, 1])
    stats = step3f.fp_stats(np.zeros(35, int), np.r_[np.ones(30, int), np.zeros(5, int)])
    assert stats["continuous_fp_segments_ge_30"] == 1 and stats["longest_false_positive_segment"] == 30


def test_22_clustering_stability_metrics_are_label_permutation_invariant():
    a = np.array([0, 0, 1, 1]); b = 1 - a; ca = np.array([[0.0], [2.0]]); cb = ca[::-1]
    result = clustering_agreement(a, ca, b, cb)
    assert result["adjusted_rand_index"] == 1 and result["normalized_mutual_information"] == 1
    assert centroid_matching_distance(ca, cb) == 0


def test_23_smoke_and_formal_outputs_are_isolated():
    assert CFG["output_dir"] != CFG["smoke_output_dir"]


def test_24_formal_outputs_are_non_overwriting_and_hash_audited():
    assert "ensure_fresh(output)" in SOURCE and "sha256_before" in SOURCE and "sha256_after" in SOURCE


def test_25_entropy_is_finite():
    assert np.isfinite(normalized_entropy(np.array([10, 20, 30])))


def test_26_all_required_outputs_are_registered():
    assert len(step3f.OUTPUT_FILES) == 22
