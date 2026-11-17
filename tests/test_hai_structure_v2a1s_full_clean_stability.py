from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.hai_structure_v2a1s_full_clean_stability as stage


def source():
    return inspect.getsource(stage).lower()


def test_01_hai_test_is_not_loaded():
    loader = inspect.getsource(stage.read_validation_timestamps_only).lower()
    assert "test1.csv" not in loader and "test2.csv" not in loader


def test_02_attack_label_is_not_used():
    loader = inspect.getsource(stage.read_validation_timestamps_only).lower()
    assert "attack" not in loader and "label" not in loader


def test_03_ae_and_rec_are_not_loaded():
    assert "torch.load" not in source() and "rec_score" not in source()


def test_04_ridge_is_not_fit_or_loaded():
    assert "joblib" not in source() and ".fit(" not in source() and "fit_ridge" not in source()


def test_05_clean_scores_are_loaded_from_existing_npz():
    text = inspect.getsource(stage.load_full_clean_grouped)
    assert "validation_dynamic_edge_score_cache.npz" in text


def test_06_controlled_scores_are_loaded_from_existing_npz():
    text = inspect.getsource(stage.evaluate_controlled)
    assert "controlled_dynamic_edge_scores.npz" in text
    assert "controlled_dynamic_aggregate_scores.npz" in text


def test_07_relation_pool_remains_twelve():
    pool = pd.read_csv(stage.V2A / "selected_dynamic_relation_pool.csv")
    assert len(pool) == 12


def test_08_grouped_definition_is_unchanged():
    scores = np.asarray([[9.0, 8.0, 7.0]])
    grouped, targets, _ = stage.grouped_scores(scores, np.asarray([0, 0, 1]))
    assert grouped[0] == 8.0 and set(targets[0]) == {0, 1}


def test_09_each_sequence_is_split_chronologically():
    grouped = [np.arange(10, dtype=float) for _ in range(6)]
    timestamps = [np.asarray([str(i) for i in range(57)]) for _ in range(6)]
    manifest, scores = stage.build_manifest(grouped, timestamps)
    for _, frame in scores.groupby("sequence_id"):
        assert frame.half.tolist() == ["first"] * 5 + ["second"] * 5


def test_10_calibration_and_evaluation_do_not_overlap_within_fold():
    grouped = [np.arange(10, dtype=float) for _ in range(6)]
    timestamps = [np.asarray([str(i) for i in range(57)]) for _ in range(6)]
    manifest, _ = stage.build_manifest(grouped, timestamps)
    assert manifest.groupby(["source_window_id", "fold"]).role.nunique().max() == 1


def test_11_no_random_sampling_in_formal_path():
    text = inspect.getsource(stage.build_manifest)
    assert "random" not in text and "choice" not in text


def test_12_q99_uses_calibration_half_only():
    frame = pd.DataFrame({"half": ["first"] * 4 + ["second"] * 4, "score": np.arange(8.0)})
    thresholds, _ = stage.fit_full_clean_thresholds(frame)
    assert thresholds["A"] == np.quantile(np.arange(4.0), 0.99)
    assert thresholds["B"] == np.quantile(np.arange(4.0, 8.0), 0.99)


def test_13_controlled_source_inherits_time_fold():
    assert stage.evaluation_fold("first") == "B"
    assert stage.evaluation_fold("second") == "A"


def test_14_p2_is_not_a_hard_gate():
    decision = inspect.getsource(stage.decide)
    assert 'family.loc["p2"' not in decision.lower()


def test_15_c1_is_not_a_hard_gate():
    decision = inspect.getsource(stage.decide)
    assert 'family.loc["c1"' not in decision.lower()


def test_16_c2_excess_fpr_is_one_sided():
    text = inspect.getsource(stage.decide)
    assert "max(0.0, c2_fpr - clean_fpr)" in text


def test_17_absolute_c2_gap_is_diagnostic_only():
    text = inspect.getsource(stage.decide)
    assert "c2_absolute_gap_diagnostic_only" in text
    assert '"c2_excess_fpr": c2_excess <= 0.015' in text


def test_18_historical_hashes_are_audited():
    text = inspect.getsource(stage.run)
    assert "before = frozen_hashes()" in text and "after = frozen_hashes()" in text


def test_19_outputs_require_finite_scores():
    text = inspect.getsource(stage.load_full_clean_grouped)
    assert "np.isfinite" in text


def test_20_resume_skips_valid_results():
    text = inspect.getsource(stage.run)
    assert "final_report_complete" in text and "dependency_hash" in text
