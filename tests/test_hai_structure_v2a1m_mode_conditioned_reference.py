from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

import scripts.hai_structure_v2a1m_mode_conditioned_reference as stage


def source():
    return inspect.getsource(stage).lower()


def test_01_hai_test_not_loaded():
    assert "test1.csv" not in source() and "test2.csv" not in source()


def test_02_attack_label_not_used():
    assert "attack label" not in inspect.getsource(stage.load_frozen_sequences).lower()


def test_03_ae_and_rec_not_loaded():
    assert "torch.load" not in source() and "rec_score" not in source()


def test_04_ridge_is_not_refit():
    assert "fit_ridge" not in source() and "ridge(" not in source()


def test_05_a1s_global_baseline_gate_is_fixed():
    text = inspect.getsource(stage.reproduce_global)
    assert '"overall_fpr": 0.009118' in text and stage.TOLERANCE == 1e-6


def test_06_mode_features_are_strictly_causal():
    assert stage.CONTEXT_LAGS == (8, 12, 16, 24, 32) and min(stage.CONTEXT_LAGS) > 0


def test_07_mode_features_exclude_structure_score():
    text = inspect.getsource(stage.context_for_sequences).lower()
    assert "dependency" not in text and "residual" not in text


def test_08_pca_fit_receives_train_features_only():
    text = inspect.getsource(stage.fit_train_only_modes)
    assert "pca.fit(features)" in text


def test_09_kmeans_fit_receives_train_projection_only():
    text = inspect.getsource(stage.fit_train_only_modes)
    assert "model.fit(projected_fit)" in text


def test_10_k_candidates_are_only_two_and_three():
    assert stage.K_CANDIDATES == (2, 3)


def test_11_validation_is_not_used_for_k_selection():
    assert "validation" not in inspect.signature(stage.fit_train_only_modes).parameters


def test_12_seed_ari_uses_adjusted_rand():
    assert "adjusted_rand_score" in inspect.getsource(stage.fit_train_only_modes)


def test_13_occupancy_and_coverage_gates_are_present():
    text = inspect.getsource(stage.fit_train_only_modes)
    assert "occupancy >= 0.05" in text and "min(coverage) >= 3" in text


def test_14_ood_radius_is_train_q995():
    text = inspect.getsource(stage.run)
    assert "np.quantile(distances[train_labels == mode], 0.995)" in text


def test_15_validation_does_not_update_centroids():
    text = inspect.getsource(stage.assign_modes)
    assert "predict" in text and ".fit" not in text


def test_16_mode_reference_uses_train_labels():
    text = inspect.getsource(stage.run)
    assert 'train_raw["rolling_dependency_loss"], train_labels' in text


def test_17_ood_uses_global_reference():
    values = np.asarray([[1.0, 2.0], [1.0, 2.0]])
    labels = np.asarray([0, 0])
    ood = np.asarray([False, True])
    medians = np.asarray([[0.0, 0.0]])
    scales = np.ones((1, 2))
    global_refs = np.asarray([[1.0, 2.0], [2.0, 2.0]])
    scores = stage.conditioned_edge_scores(values, labels, ood, medians, scales, global_refs)
    assert np.allclose(scores[1], [0.0, 0.0])


def test_18_controlled_uses_source_mode():
    text = inspect.getsource(stage.controlled_conditioned_scores)
    assert "source_ood[window]" in text and "source_labels[window]" in text


def test_19_aggregation_is_target_grouped_top2():
    assert "grouped_scores" in inspect.getsource(stage.controlled_conditioned_scores)


def test_20_threshold_is_clean_crossfit_q99():
    assert inspect.signature(stage.thresholds_from_a1s_manifest).parameters["quantile"].default == 0.99


def test_21_no_mode_specific_threshold():
    text = inspect.getsource(stage.thresholds_from_a1s_manifest)
    assert '"mode_specific_threshold": false' in text.lower()


def test_22_relation_pool_is_twelve():
    assert len(pd.read_csv(stage.V2A / "selected_dynamic_relation_pool.csv")) == 12


def test_23_historical_hashes_are_checked():
    text = inspect.getsource(stage.run)
    assert "before = historical_hashes()" in text and "after = historical_hashes()" in text


def test_24_raw_score_helpers_reject_nonfinite_by_construction():
    text = inspect.getsource(stage.raw_dependency_for_sequences)
    assert "np.isfinite" in text


def test_25_resume_skips_valid_cache():
    text = inspect.getsource(stage.run)
    assert "final_report_complete" in text and "dependency_hash" in text
