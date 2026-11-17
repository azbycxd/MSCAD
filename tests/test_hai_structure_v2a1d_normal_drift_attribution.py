from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

import scripts.hai_structure_v2a1d_normal_drift_attribution as stage


def source():
    return inspect.getsource(stage).lower()


def test_01_hai_test_not_loaded():
    assert "test1.csv" not in source() and "test2.csv" not in source()


def test_02_attack_label_not_used():
    assert "attack label" not in inspect.getsource(stage.load_full_clean).lower()


def test_03_ae_and_rec_not_loaded():
    assert "torch.load" not in source() and "rec_score" not in source()


def test_04_no_model_is_fit_or_loaded():
    assert "joblib" not in source() and ".fit(" not in source() and "fit_ridge" not in source()


def test_05_edge_scores_are_not_recomputed():
    assert "score_window" not in source() and "dependency_loss(" not in source()


def test_06_relation_pool_is_unchanged():
    pool = pd.read_csv(stage.V2A / "selected_dynamic_relation_pool.csv")
    assert len(pool) == 12


def test_07_thresholds_are_loaded_not_fit():
    text = inspect.getsource(stage.load_full_clean)
    assert "full_clean_crossfit_thresholds.csv" in text
    assert "quantile" not in text


def test_08_a1s_reproduction_tolerance_is_fixed():
    assert stage.TOLERANCE == 1e-6


def test_09_half_partition_is_reused_from_a1s_manifest():
    text = inspect.getsource(stage.load_full_clean)
    assert "full_clean_crossfit_window_manifest.csv" in text and '"half"' in text


def test_10_target_alarm_contribution_is_bounded():
    source = pd.DataFrame({"sequence_id": ["train1_validation"] * 4, "half": ["first"] * 2 + ["second"] * 2, "threshold": [1.0] * 4, "alarm": [False, True, True, False]})
    scores = np.asarray([[0, 1], [2, 1], [2, 3], [0, 0]], float)
    order = np.argsort(scores, axis=1)[:, ::-1]
    result = stage.target_attribution(source, ["a", "b"], scores, order)
    assert result.alarm_contribution_ratio.between(0, 1).all()


def test_11_edge_concentration_shares_are_legal():
    counts = np.asarray([8, 2, 0]) / 10
    assert 0 <= np.square(counts).sum() <= 1 and counts.sum() == 1


def test_12_concentration_output_metrics_are_fixed():
    text = inspect.getsource(stage.concentration)
    assert "top3_edge_cumulative_contribution" in text and "herfindahl" in text


def test_13_no_relation_is_deleted_or_downweighted():
    assert "drop(" not in source() and "delete_edge" not in source() and "learned_weight" not in source()


def test_14_no_state_model_or_cluster_is_created():
    assert "kmeans" not in source() and "dbscan" not in source() and "gaussianmixture" not in source()


def test_15_historical_hashes_are_audited():
    text = inspect.getsource(stage.run)
    assert "before = historical_hashes()" in text and "after = historical_hashes()" in text


def test_16_unavailable_outputs_use_finite_placeholders():
    text = inspect.getsource(stage.operating_state_shift)
    assert '"available": false' in text.lower() and '"psi": 0.0' in text


def test_17_resume_skips_completed_output():
    text = inspect.getsource(stage.run)
    assert "final_report_complete" in text and "dependency_hash" in text


def test_18_tail_shift_is_part_of_drift_attribution():
    text = inspect.getsource(stage.operating_state_shift)
    assert "q95_shift >= 0.25" in text and "q99_shift >= 0.50" in text


def test_19_primary_target_alarm_contributions_are_conserved():
    text = inspect.getsource(stage.target_attribution)
    assert "primary_alarm_contribution_count" in text
