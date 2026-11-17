from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from models.mscad import MSCAD
from scripts.stage4_3b_compare_structure_scores import (
    checkpoint_for_seed,
    complementarity_counts,
    event_reduce,
    independent_64_event_scores,
    independent_scores,
    infer_existing_scores,
    macro_average_seed_metrics,
    q99_threshold,
    split_128_window,
    validate_same_sample_ids,
)


def test_split_128_into_two_nonoverlapping_64_windows() -> None:
    window = np.arange(128 * 3).reshape(128, 3)
    first, second = split_128_window(window)
    assert first.shape == second.shape == (64, 3)
    assert np.array_equal(first, window[:64]) and np.array_equal(second, window[64:])


def test_event_max_and_mean() -> None:
    maximum, mean = event_reduce(np.array([1.0, 4.0]), np.array([3.0, 2.0]))
    assert np.array_equal(maximum, [3.0, 4.0])
    assert np.array_equal(mean, [2.0, 3.0])


def test_model_inference_does_not_update_parameters() -> None:
    model = MSCAD(in_channels=3, hidden_channels=4, latent_channels=2, dropout=0.0)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    infer_existing_scores(model, np.random.default_rng(1).normal(size=(2, 128, 3)).astype(np.float32), torch.device("cpu"), 2)
    assert all(torch.equal(left, right) for left, right in zip(before, model.parameters()))
    assert not model.training


def test_rec_reuses_project_score_components() -> None:
    source = inspect.getsource(infer_existing_scores)
    assert "model.score_components" in source
    assert 'components["rec"]' in source


def test_existing_long_reuses_checkpoint_pairs() -> None:
    source = inspect.getsource(infer_existing_scores)
    assert 'components["long"]' in source
    assert "structure_distance" not in source


def test_independent_scores_do_not_accept_x_hat() -> None:
    assert "x_hat" not in inspect.signature(independent_scores).parameters
    assert "x_hat" not in inspect.signature(independent_64_event_scores).parameters


def test_independent_score_uses_zero_when_all_pairs_invalid() -> None:
    windows = np.ones((2, 64, 2), dtype=np.float32)
    models = [{
        "channel_i_index": 0, "channel_j_index": 1,
        "normal_median_fisher": 0.0, "normal_robust_scale": 0.2,
    }]
    assert np.array_equal(independent_scores(windows, models, 0.1, 64), np.zeros(2))


def test_all_scores_share_one_sample_id_row() -> None:
    import pandas as pd

    frame = pd.DataFrame([{
        "sample_id": "one", "rec_event_max": 1.0, "rec_event_mean": 1.0,
        "existing_long_event_max": 1.0, "existing_long_event_mean": 1.0,
        "independent_128_max": 1.0, "independent_64_max": 1.0,
    }])
    validate_same_sample_ids(frame)


def test_q99_uses_only_calibration_argument() -> None:
    assert set(inspect.signature(q99_threshold).parameters) == {"calibration_scores"}
    assert q99_threshold(np.arange(100, dtype=float)) == np.quantile(np.arange(100), 0.99)


def test_relation_summary_macro_averages_seed_metrics() -> None:
    base = {
        "normal_sample_count": 10, "anomaly_sample_count": 5, "calibration_sample_count": 8,
        "threshold_q99": 2.0, "normal_evaluation_fpr": 0.01, "relation_only_recall": 0.5,
        "auroc": 0.8, "aupr": 0.7, "effect_size_cohen_d": 1.0,
        "normal_score_median": 0.2, "anomaly_score_median": 1.2,
    }
    second = {**base, "auroc": 1.0, "relation_only_recall": 0.9}
    summary = macro_average_seed_metrics([base, second])
    assert summary["auroc"] == 0.9 and summary["relation_only_recall"] == 0.7
    assert summary["normal_sample_count"] == 20 and summary["seed_count"] == 2


def test_complementarity_counts_conserve_anomalies() -> None:
    import pandas as pd

    frame = pd.DataFrame({
        "y_true": [1, 1, 1, 1], "rec_event_max": [2, 2, 0, 0],
        "existing_long_event_max": [0, 2, 0, 2], "independent_128_max": [2, 0, 2, 0],
    })
    row = complementarity_counts(frame, {
        "rec_event_max": 1, "existing_long_event_max": 1, "independent_128_max": 1,
    })
    assert row["both_rec_and_independent_detected"] + row["rec_detected_independent_missed"] + row["independent_detected_rec_missed"] + row["neither_detected"] == 4


def test_seed_maps_to_matching_checkpoint() -> None:
    root = Path("D:/workspace/python/mscad")
    assert checkpoint_for_seed(root, 2).as_posix().endswith("seed_2/ft/best.pt")


def test_stage_loader_does_not_read_test_or_attack_labels() -> None:
    from scripts import stage4_3b_compare_structure_scores as stage

    source = inspect.getsource(stage.load_stage_data)
    assert "test_csv" not in source and "attack" not in source.lower()


def test_validation_rejects_nan() -> None:
    import pandas as pd

    frame = pd.DataFrame([{
        "sample_id": "one", "rec_event_max": np.nan, "rec_event_mean": 1.0,
        "existing_long_event_max": 1.0, "existing_long_event_mean": 1.0,
        "independent_128_max": 1.0, "independent_64_max": 1.0,
    }])
    try:
        validate_same_sample_ids(frame)
    except ValueError:
        return
    raise AssertionError("NaN score was accepted")
