from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

from data.hai import HAISequence, MultiSequenceWindowDataset
from utils.reconstruction_scores import (
    ALL_VARIANTS, CAUSAL_VARIANTS, add_overlap_errors, finalize_overlap_scores, temporal_reconstruction_scores,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hai_step3c_evaluate_rec_score_alignment.py"
CONFIG = ROOT / "configs" / "hai_rec_score_alignment.yaml"
SPEC = importlib.util.spec_from_file_location("hai_step3c", SCRIPT)
step3c = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(step3c)


def config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_only_existing_three_best_checkpoints_are_configured():
    cfg = config()
    assert cfg["seeds"] == [1, 2, 3]
    paths = [ROOT / cfg["checkpoint_pattern"].format(seed=seed) for seed in cfg["seeds"]]
    assert all(path.name == "best.pt" and path.exists() for path in paths)


def test_no_optimizer_step_or_training_is_executed():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "optimizer.step" not in source
    assert ".train()" not in source
    assert "model.eval()" in source


def test_model_input_output_shape_is_unchanged():
    checkpoint = torch.load(ROOT / "outputs/hai_step3a_rec_baseline_seed1/best.pt", map_location="cpu", weights_only=False)
    model = step3c.make_model(checkpoint, torch.device("cpu"))
    x = torch.randn(2, 64, 86)
    assert model(x).shape == x.shape


def test_window_mean_score_is_correct():
    x = torch.arange(2 * 64 * 3, dtype=torch.float32).reshape(2, 64, 3)
    scores, _ = temporal_reconstruction_scores(x, torch.zeros_like(x))
    assert torch.allclose(scores["window_mean_current"], x.abs().mean(dim=(1, 2)))


def test_endpoint_mean_score_is_correct():
    x = torch.arange(64 * 3, dtype=torch.float32).reshape(1, 64, 3)
    scores, _ = temporal_reconstruction_scores(x, torch.zeros_like(x))
    assert torch.allclose(scores["endpoint_mean"], x[:, -1].abs().mean(dim=1))


def test_recent8_and_recent16_scores_are_correct():
    x = torch.arange(64 * 2, dtype=torch.float32).reshape(1, 64, 2)
    scores, _ = temporal_reconstruction_scores(x, torch.zeros_like(x))
    assert torch.allclose(scores["recent8_mean"], x[:, -8:].abs().mean(dim=(1, 2)))
    assert torch.allclose(scores["recent16_mean"], x[:, -16:].abs().mean(dim=(1, 2)))


def test_offline_overlap_point_accumulation_is_correct():
    error_sum, coverage = np.zeros(4), np.zeros(4, dtype=int)
    add_overlap_errors(error_sum, coverage, np.array([0, 1]), np.array([[1., 2., 3.], [4., 5., 6.]]))
    scores, covered = finalize_overlap_scores(error_sum, coverage)
    assert np.allclose(scores, [1., 3., 4., 6.])
    assert covered.all() and coverage.tolist() == [1, 2, 2, 1]


def test_offline_overlap_cannot_cross_csv_boundary():
    try:
        add_overlap_errors(np.zeros(4), np.zeros(4, dtype=int), np.array([2]), np.ones((1, 3)))
    except IndexError as exc:
        assert "boundary" in str(exc)
    else:
        raise AssertionError("Cross-boundary overlap must fail")


def test_offline_overlap_is_reset_between_train_validation_splits():
    source = inspect.getsource(step3c.run)
    assert "bundle.train_sequences" in source and "bundle.val_sequences" in source
    assert "infer_sequence(model, sequence" in source


def test_endpoint_metadata_uses_window_end():
    sequence = HAISequence("s", "s.csv", "test", np.zeros((70, 2), np.float32), np.arange(70) % 2,
                           np.asarray([f"t{i}" for i in range(70)]))
    _, label, metadata = MultiSequenceWindowDataset([sequence], 64, 1, return_metadata=True)[0]
    assert metadata["end_row"] == 63 and metadata["timestamp"] == "t63" and label.item() == sequence.y[63]


def test_current_historical_arrays_have_alignment_fields():
    old = np.load(ROOT / "outputs/hai_step3a_rec_baseline_seed1/test_score_arrays.npz")
    for name in ("score", "sequence_id", "timestamp", "end_row", "label"):
        assert f"test1_{name}" in old.files


def test_each_variant_fits_its_own_threshold():
    signature = inspect.signature(step3c.fit_thresholds)
    assert list(signature.parameters) == ["train", "validation", "cfg"]
    source = inspect.getsource(step3c.run)
    assert 'splits["train_all"][variant]' in source and 'splits["validation_all"][variant]' in source


def test_threshold_fitting_has_no_test_label_parameter():
    source = inspect.getsource(step3c.fit_thresholds).lower()
    assert "test" not in source and "label" not in source


def test_aupr_auroc_are_threshold_independent():
    labels = np.array([0, 0, 1, 1]); scores = np.array([.1, .4, .3, .9])
    left = step3c.metric_record(labels, scores, .2)
    right = step3c.metric_record(labels, scores, .8)
    assert left["aupr"] == right["aupr"] and left["auroc"] == right["auroc"]


def test_events_are_evaluated_by_file():
    source = inspect.getsource(step3c.event_diagnostics)
    assert 'splits["test"][event["file_name"]]' in source


def test_onset_summary_is_correct():
    frame = pd.DataFrame({"detected": [True, True, False], "detection_delay_points": [0, 8, np.nan],
                          "detected_at_first_point": [True, False, False], "detected_within_5_points": [True, False, False],
                          "detected_within_10_points": [True, True, False], "detected_within_30_points": [True, True, False]})
    row = step3c.event_summary_row(1, "v", "m", "all_tests", frame)
    assert np.isclose(row["first_point_hit_rate"], 1 / 3)
    assert np.isclose(row["within10_hit_rate"], 2 / 3)


def test_event_recovered_and_lost_logic_is_present():
    source = inspect.getsource(step3c.event_transitions)
    assert "recovered_by_endpoint" in source and "lost_by_endpoint" in source
    assert "delay_change" in source


def test_three_seed_timestamps_and_labels_are_identical():
    arrays = {seed: np.load(ROOT / f"outputs/hai_step3a_rec_baseline_seed{seed}/test_score_arrays.npz") for seed in (1, 2, 3)}
    for field in ("timestamp", "label", "end_row", "sequence_id"):
        assert np.array_equal(arrays[1][f"test4_{field}"], arrays[2][f"test4_{field}"])
        assert np.array_equal(arrays[1][f"test4_{field}"], arrays[3][f"test4_{field}"])


def test_noncausal_variant_is_diagnostic_only():
    row = step3c.score_registry().set_index("score_variant").loc["offline_overlap_point_mean"]
    assert not row.causal and "diagnostic_upper_bound_only" in row.role


def test_no_structure_score_is_computed():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "score_components(" not in source
    assert "structure_distance" not in source


def test_checkpoint_files_are_opened_read_only():
    source = inspect.getsource(step3c.audit_checkpoints)
    assert "torch.load" in source
    assert "torch.save" not in SCRIPT.read_text(encoding="utf-8")


def test_smoke_and_formal_outputs_are_isolated():
    cfg = config()
    assert cfg["output_dir"] != cfg["smoke_output_dir"]


def test_original_hai_file_hash_matches_step1_inventory():
    inventory = pd.read_csv(ROOT / "outputs/hai_step1_dataset_audit/hai_file_inventory.csv")
    row = inventory.iloc[0]
    digest = hashlib.sha256(Path(row.absolute_path).read_bytes()).hexdigest()
    assert digest == row.sha256


def test_covered_overlap_output_is_finite_and_uncovered_is_masked():
    scores, covered = finalize_overlap_scores(np.array([0., 4., 0.]), np.array([0, 2, 0]))
    assert np.isnan(scores[~covered]).all()
    assert np.isfinite(scores[covered]).all() and scores[1] == 2.0
