from __future__ import annotations

import importlib.util
import inspect
import os
from itertools import combinations
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hai_step4a_diagnose_normal_relations.py"
SPEC = importlib.util.spec_from_file_location("hai_step4a", SCRIPT)
step4a = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(step4a)


def test_formal_mode_names_only_six_train_prefixes():
    assert step4a.TRAIN_FILES == tuple(f"train{i}.csv" for i in range(1, 7))
    source = inspect.getsource(step4a.load_train_prefixes)
    assert "nrows=used_rows" in source and "usecols=usecols" in source


def test_validation_and_test_values_are_excluded():
    source = inspect.getsource(step4a.load_train_prefixes)
    assert "row.train_rows" in source
    assert "label_name" in source and "usecols = [timestamp_name" in source
    assert "test" in source and "forbidden test file" in source


def test_global_zero_variance_feature_is_ineligible():
    raw = [np.column_stack([np.arange(256), np.ones(256)]).astype(np.float32) for _ in range(6)]
    scaler = step4a.StandardScaler(eps=step4a.GLOBAL_EPS).fit_many(raw)
    normalized = [scaler.transform(x).astype(np.float32) for x in raw]
    frame = step4a.feature_eligibility(raw, normalized, ["P1_A", "P1_CONST"], 32)
    assert frame.eligible_for_relation_analysis.tolist() == [True, False]


def test_feature_eligibility_does_not_delete_rec_channels():
    source = inspect.getsource(step4a.run)
    assert "eligibility.to_csv" in source
    assert "feature_schema_count" in source
    assert "86 retained" in source


def test_local_windows_do_not_cross_csv_boundaries():
    arrays = [np.zeros((100, 2)), np.zeros((100, 2))]
    count = sum(len(step4a.rolling_moments(x, 64, 32)[2]) for x in arrays)
    assert count == 4
    assert count != len(step4a.rolling_moments(np.concatenate(arrays), 64, 32)[2])


def test_local_constant_pair_is_invalid():
    values = np.column_stack([np.ones(256), np.arange(256)]).astype(np.float32)
    sums, variances, starts = step4a.rolling_moments(values, 64, 32)
    corr, const_left, _, valid = step4a.pair_correlations(values, sums, variances, starts,
                                                          np.array([[0, 1]]), 64)
    assert const_left.all() and not valid.any() and np.isnan(corr).all()


def test_fisher_transform_and_mad_are_correct():
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    transformed = step4a.fisher_transform(values)
    assert np.isfinite(transformed).all()
    stats = step4a.correlation_statistics(np.array([-0.5, 0.0, 0.5]))
    expected = np.median(np.abs(step4a.fisher_transform([-0.5, 0.0, 0.5])))
    assert np.isclose(stats["fisher_z_mad"], expected)
    assert np.isclose(stats["fisher_z_robust_sigma"], 1.4826 * expected)


def test_sign_consistency_is_correct():
    assert np.isclose(step4a.sign_consistency(np.array([1, 2, -1, 3])), 0.75)
    assert step4a.sign_consistency(np.zeros(4)) == 0.0


def test_per_sequence_correlation_statistics_are_correct():
    base = np.arange(256, dtype=np.float32)
    arrays = [np.column_stack([base, base * 2]) for _ in range(6)]
    pairs, metadata = step4a.pair_metadata(["P1_A", "P1_B"], np.array([0, 1]))
    frame, _ = step4a.global_sequence_stats(arrays, pairs, metadata)
    assert np.allclose(frame[[f"sequence{i}_pearson" for i in range(1, 7)]], 1.0)
    assert np.isclose(frame.pooled_pearson.iloc[0], 1.0)


def test_relation_matrix_similarity_is_correct():
    frame = pd.DataFrame({"pair_id": [1, 2, 3], "relation_scope": ["within_P1", "cross_subsystem", "within_P2"],
                          "median_abs_corr": [.5, .4, .3], **{f"sequence{i}_pearson": [.1, .2, .3] for i in range(1, 7)}})
    local = pd.DataFrame({"valid_ratio": [1.0, 1.0, 1.0]})
    similarity, _ = step4a.sequence_similarity(frame, local)
    assert np.allclose(similarity.pearson_similarity, 1.0)
    assert np.allclose(similarity.spearman_similarity, 1.0)
    assert np.allclose(similarity.cosine_similarity, 1.0)


def test_window_counts_for_64_and_128_are_correct():
    values = np.zeros((1000, 3))
    for window in (64, 128):
        starts = step4a.rolling_moments(values, window, 32)[2]
        assert len(starts) == (len(values) - window) // 32 + 1


def test_threshold_sensitivity_count_is_correct():
    rows = []
    for pair_id in (1, 2):
        for seq in range(6):
            rows.append({"pair_id": pair_id, "valid_ratio": 1.0, "median_abs_corr": .8 if pair_id == 1 else .1,
                         "fisher_z_robust_sigma": .1, "sign_consistency": 1.0})
    by = pd.DataFrame(rows)
    aggregate = pd.DataFrame({"pair_id": [1, 2], "valid_ratio": [1, 1], "median_abs_corr": [.8, .1],
                              "fisher_z_robust_sigma": [.1, .1], "sign_consistency": [1, 1]})
    result = step4a.threshold_sensitivity({64: by, 128: by}, {64: aggregate, 128: aggregate})
    selected = result[(result.minimum_valid_ratio == .95) & (result.minimum_median_abs_corr == .7) &
                      (result.maximum_fisher_z_robust_sigma == .2) & (result.minimum_sign_consistency == .95) &
                      (result.minimum_sequence_coverage == 6)]
    assert (selected.candidate_count == 1).all()


def _classification_frames(count: int):
    aggregate = pd.Series({"valid_ratio": 1.0, "median_abs_corr": .7,
                           "fisher_z_robust_sigma": .1, "sign_consistency": 1.0,
                           "feature_i_local_constant_count": 0, "feature_j_local_constant_count": 0})
    rows = [{"valid_ratio": 1.0 if i < count else .5, "median_abs_corr": .7 if i < count else .1,
             "fisher_z_robust_sigma": .1 if i < count else .8, "sign_consistency": 1.0 if i < count else .5}
            for i in range(6)]
    return aggregate, pd.DataFrame(rows)


def test_strict_moderate_and_sequence_specific_classification():
    aggregate, sequence = _classification_frames(6)
    assert step4a.classify_pair(aggregate, sequence, .1)[0] == "strict_cross_sequence"
    aggregate, sequence = _classification_frames(4)
    assert step4a.classify_pair(aggregate, sequence, .4)[0] == "moderate_cross_sequence"
    aggregate, sequence = _classification_frames(2)
    assert step4a.classify_pair(aggregate, sequence, .4)[0] == "sequence_specific_stable"


def test_diversity_pool_degree_is_capped_at_two():
    rows = []
    for pair_id, (left, right) in enumerate(combinations(["P1_A", "P1_B", "P2_A", "P2_B"], 2), 1):
        rows.append({"pair_id": pair_id, "feature_i": left, "feature_j": right,
                     "subsystem_i": step4a.subsystem(left), "subsystem_j": step4a.subsystem(right),
                     "relation_category": "moderate_cross_sequence", "valid_ratio": 1.0,
                     "median_abs_corr": .7, "sign_consistency": 1.0, "fisher_z_robust_sigma": .1,
                     "strict_sequence_count": 4})
    _, diversity = step4a.candidate_pools(pd.DataFrame(rows))
    degree = pd.concat([diversity.feature_i, diversity.feature_j]).value_counts()
    assert (degree <= 2).all()


def test_candidate_pool_has_no_test_label_input():
    signature = inspect.signature(step4a.candidate_pools)
    assert list(signature.parameters) == ["classified128"]
    source = inspect.getsource(step4a.candidate_pools).lower()
    assert "label" not in source and "test" not in source


def test_no_model_or_checkpoint_is_loaded():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "import torch" not in source
    assert "models." not in source
    assert "torch.load" not in source
    assert "load_state_dict" not in source


def test_smoke_and_formal_outputs_are_isolated():
    assert step4a.FORMAL_OUTPUT != step4a.SMOKE_OUTPUT


def test_result_metadata_records_thresholds_and_scope():
    source = inspect.getsource(step4a.run)
    assert '"threshold_sensitivity_grid": THRESHOLDS' in source
    assert '"classification_thresholds": CLASSIFICATION_THRESHOLDS' in source
    assert '"data_ranges": data_ranges' in source
    assert '"test_files_read": False' in source
