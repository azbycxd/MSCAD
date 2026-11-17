from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.stage4_3c_eval_independent_on_real_swat import (
    align_scores,
    checkpoint_for_seed,
    common_alignment_indices,
    contiguous_events,
    event_diagnostics,
    fit_thresholds,
    independent_series_scores,
    load_fixed_strict_set,
    point_complementarity,
    top_relation_attribution,
    validate_finite,
)


def test_64_and_128_scores_align_to_same_end_time() -> None:
    scores64 = np.arange(137, dtype=float)
    scores128 = np.arange(73, dtype=float) + 1000
    labels = np.zeros(200, dtype=int)
    aligned64, aligned128, _, indices = align_scores(scores64, scores128, labels)
    assert indices[0] == 127 and aligned64[0] == scores64[64] and aligned128[0] == scores128[0]


def test_first_127_points_are_excluded() -> None:
    indices, _, _ = common_alignment_indices(300)
    assert indices[0] == 127 and len(indices) == 173


def test_independent_score_has_no_x_hat_argument() -> None:
    assert "x_hat" not in inspect.signature(independent_series_scores).parameters


def test_strict_set_is_loaded_not_selected() -> None:
    source = inspect.getsource(load_fixed_strict_set)
    assert 'candidate_sets["strict_stable"]' in source
    assert "select" not in source.lower()


def test_normal_models_loader_has_no_test_argument() -> None:
    assert set(inspect.signature(load_fixed_strict_set).parameters) == {"source_dir"}


def test_thresholds_do_not_accept_test_labels() -> None:
    assert set(inspect.signature(fit_thresholds).parameters) == {"normal_scores", "eval_cfg"}


def test_point_complementarity_conserves_anomalies() -> None:
    labels = np.array([1, 1, 1, 1, 0])
    row = point_complementarity(labels, np.array([2, 2, 0, 0, 0]), np.array([2, 0, 2, 0, 2]), 1, 1)
    assert row["both_detected"] + row["rec_detected_independent128_missed"] + row["independent128_detected_rec_missed"] + row["neither_detected"] == 4


def test_contiguous_event_detection() -> None:
    labels = np.array([0, 1, 1, 0, 1, 0])
    events = contiguous_events(labels, np.arange(100, 106))
    assert len(events) == 2 and events[0]["start_index"] == 101 and events[0]["duration"] == 2


def test_top_relation_attribution() -> None:
    rows = [{"pair_id": "a"}, {"pair_id": "b"}, {"pair_id": "c"}]
    result = top_relation_attribution(np.array([[1.0, 3.0, 2.0]]), rows, 0)
    assert result["top1_relation"] == "b" and result["top3_relations"] == "b|c|a"


def test_seed_checkpoint_mapping() -> None:
    root = Path("D:/workspace/python/mscad")
    assert checkpoint_for_seed(root, 3).as_posix().endswith("seed_3/ft/best.pt")


def test_stage_does_not_fuse_scores() -> None:
    from scripts import stage4_3c_eval_independent_on_real_swat as stage

    source = inspect.getsource(stage.analyze)
    assert "fuse_scores" not in source and "alpha" not in source and "gamma" not in source


def test_output_validation_rejects_nan() -> None:
    frame = pd.DataFrame({"score": [1.0, np.nan]})
    try:
        validate_finite(frame, ["score"])
    except ValueError:
        return
    raise AssertionError("NaN output was accepted")
