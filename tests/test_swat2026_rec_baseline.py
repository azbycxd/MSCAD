from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.swat2026_training import (
    TEST_CONTEXT_START,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    SWaTWindowDataset,
    PreparedSWaTData,
    _window_rows,
    fit_train_only_scaler,
)
from scripts.swat2026 import train_eval_rec_baseline as baseline


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "swat2026" / "rec_baseline_seed1.yaml"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def timeline() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    timestamps = pd.date_range(TRAIN_START, periods=28_860, freq="s")
    names = [f"F{i:02d}.Pv" for i in range(31)]
    seconds = np.arange(len(timestamps), dtype=np.float64)
    raw = np.column_stack([seconds * (index + 1) / 1000.0 for index in range(31)])
    valid = np.ones(len(timestamps), dtype=bool)
    valid[1_000:1_008] = False
    raw[~valid, -1] = np.nan
    labels = pd.DataFrame({
        "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase": np.where((timestamps >= TEST_START) & (timestamps < TEST_END), "attack_experiment_phase", "normal_phase"),
        "point_label": 0, "attack_id": "", "evaluation_eligible": (timestamps >= TEST_START) & (timestamps < TEST_END),
    })
    attack6 = (timestamps >= pd.Timestamp("2026-03-11 15:00:00")) & (timestamps < pd.Timestamp("2026-03-11 15:02:00"))
    attack7 = (timestamps >= pd.Timestamp("2026-03-11 15:02:00")) & (timestamps < pd.Timestamp("2026-03-11 15:07:00"))
    labels.loc[attack6, ["point_label", "attack_id"]] = [1, "6"]
    labels.loc[attack7, ["point_label", "attack_id"]] = [1, "7"]
    return timestamps, raw, valid, names, labels


def test_config_is_fixed_single_seed_rec_only():
    cfg = config()
    assert cfg["seed"] == 1
    assert cfg["data"]["window_size"] == 128
    assert cfg["model"]["input_channels"] == 31
    assert cfg["model"]["rec_only"] is True
    assert cfg["model"]["use_structure_loss"] is False
    assert cfg["model"]["init_checkpoint"] is None


def test_scaler_fits_unique_train_raw_points_only(timeline):
    timestamps, raw, valid, names, _ = timeline
    scaler, metadata, _ = fit_train_only_scaler(raw, timestamps, valid, names, "fixture")
    train = ((timestamps >= TRAIN_START) & (timestamps < TRAIN_END)) & valid
    assert metadata["valid_fit_rows"] == 11_410
    assert metadata["total_train_raw_points"] == 11_418
    assert metadata["excluded_fit_rows"] == 8
    assert np.allclose(scaler.mean_, np.nanmean(raw[train], axis=0))
    assert metadata["overlapping_windows_used_for_fit"] is False


def test_scaler_does_not_read_validation_or_test(timeline):
    timestamps, raw, valid, names, _ = timeline
    changed = raw.copy()
    changed[timestamps >= TRAIN_END] = 1e12
    scaler, metadata, _ = fit_train_only_scaler(changed, timestamps, valid, names, "fixture")
    assert np.all(scaler.mean_ < 1e12)
    assert metadata["validation_used"] is False
    assert metadata["test_used"] is False
    assert metadata["labels_used"] is False


def test_feature_order_and_statistics_are_stable(timeline):
    timestamps, raw, valid, names, _ = timeline
    _, metadata, statistics = fit_train_only_scaler(raw, timestamps, valid, names, "fixture")
    assert metadata["feature_order"] == names
    assert statistics["feature"].tolist() == names
    assert len(names) == 31


def test_window_counts_and_invalid_exclusion(timeline):
    timestamps, _, valid, _, labels = timeline
    rows = _window_rows(timestamps, valid, labels, 128)
    grouped = rows.groupby("split")["valid"].agg(["size", "sum"])
    assert tuple(grouped.loc["train"]) == (11_291, 11_156)
    assert tuple(grouped.loc["validation"]) == (2_728, 2_728)
    assert tuple(grouped.loc["test"]) == (14_400, 14_400)
    assert (rows.query("split == 'train' and not valid")["exclusion_reason"] == "contains_invalid_model_feature").all()


def test_validation_does_not_borrow_train_context(timeline):
    timestamps, _, valid, _, labels = timeline
    rows = _window_rows(timestamps, valid, labels, 128)
    first = rows.query("split == 'validation'").iloc[0]
    assert pd.Timestamp(first.window_start) == VALIDATION_START
    assert pd.Timestamp(first.window_end) == VALIDATION_START + pd.Timedelta(seconds=127)


def test_test_first_window_uses_127_seconds_test_context(timeline):
    timestamps, _, valid, _, labels = timeline
    first = _window_rows(timestamps, valid, labels, 128).query("split == 'test'").iloc[0]
    assert pd.Timestamp(first.window_start) == TEST_CONTEXT_START
    assert pd.Timestamp(first.window_end) == TEST_START
    assert pd.Timestamp(first.score_timestamp) == TEST_START


def test_test_count_and_outside_exclusion(timeline):
    timestamps, _, valid, _, labels = timeline
    test = _window_rows(timestamps, valid, labels, 128).query("split == 'test'")
    assert len(test) == 14_400
    assert pd.Timestamp(test.iloc[-1].score_timestamp) == TEST_END - pd.Timedelta(seconds=1)
    assert (pd.to_datetime(test.score_timestamp) < TEST_END).all()


def test_attack_6_7_left_closed_right_open_boundary(timeline):
    timestamps, _, valid, _, labels = timeline
    test = _window_rows(timestamps, valid, labels, 128).query("split == 'test'").set_index("score_timestamp")
    assert test.loc["2026-03-11T15:01:59", "attack_id"] == "6"
    assert test.loc["2026-03-11T15:02:00", "attack_id"] == "7"
    assert int(test.loc["2026-03-11T15:07:00", "point_label"]) == 0


def test_lazy_dataset_returns_equal_input_target(timeline):
    timestamps, raw, valid, names, labels = timeline
    scaler, metadata, statistics = fit_train_only_scaler(raw, timestamps, valid, names, "fixture")
    scaled = ((raw - scaler.mean_) / scaler.scale_).astype(np.float32)
    prepared = PreparedSWaTData(Path("fixture.csv"), "fixture", names, timestamps, raw, scaled, valid, labels,
                                scaler, metadata, statistics, _window_rows(timestamps, valid, labels, 128))
    x, target = SWaTWindowDataset(prepared, "validation")[0]
    assert x.shape == (128, 31)
    assert torch.equal(x, target)
    assert torch.isfinite(x).all()


def test_existing_model_accepts_31_channels_and_preserves_shape():
    model = baseline.make_model(config(), torch.device("cpu"))
    x = torch.randn(2, 128, 31)
    assert model(x).shape == x.shape


def test_model_backward_is_finite():
    model = baseline.make_model(config(), torch.device("cpu"))
    x = torch.randn(2, 128, 31)
    loss = model.rec_loss_fn(model(x), x)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_no_old_checkpoint_or_hai_loader_is_used():
    source = inspect.getsource(baseline)
    assert 'model_cfg.get("init_checkpoint") is not None' in source
    assert "data.hai" not in source
    assert "build_hai_dataloaders" not in source


def test_early_stopping_uses_only_train_and_validation():
    assert list(inspect.signature(baseline.train_model).parameters) == [
        "model", "train_loader", "validation_loader", "cfg", "device", "checkpoint_path", "epochs_override"
    ]
    source = inspect.getsource(baseline.train_model)
    assert "test_loader" not in source
    assert "test_scores" not in source
    assert "point_label" not in source
    assert "validation_value < best_loss" in source


def test_threshold_uses_validation_only_and_not_test_f1():
    validation = np.arange(100, dtype=float)
    expected = np.quantile(validation, 0.99)
    assert baseline.threshold_from_validation(validation, 0.99) == expected
    assert list(inspect.signature(baseline.threshold_from_validation).parameters) == ["validation_scores", "quantile"]


def test_primary_score_is_full_window_mae_and_end_aligned():
    x = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
    assert torch.equal(baseline.reconstruction_score(x, torch.zeros_like(x)), x.abs().mean(dim=(1, 2)))
    assert config()["scoring"]["primary_score_name"] == "full_window_mean_absolute_reconstruction_error"


def test_metrics_do_not_apply_point_adjustment():
    result = baseline.point_metrics(np.array([0, 1, 1]), np.array([0.1, 0.9, 0.2]), 0.5)
    assert result["point_adjustment"] is False
    assert result["confusion_matrix"] == {"tn": 1, "fp": 0, "fn": 1, "tp": 1}


def test_non_finite_loss_blocks_training():
    class BadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.rec_loss_fn = lambda output, target: output.mean() * torch.tensor(float("nan"))

        def forward(self, value):
            return value * self.weight

    loader = DataLoader(TensorDataset(torch.ones(2, 2, 2), torch.ones(2, 2, 2)), batch_size=1)
    model = BadModel()
    with pytest.raises(FloatingPointError):
        baseline.train_epoch(model, loader, torch.optim.Adam(model.parameters()), torch.device("cpu"))


def test_event_detection_has_no_point_adjustment():
    timestamps = pd.date_range("2026-03-11 13:00:00", periods=5, freq="s")
    scores = pd.DataFrame({"timestamp": timestamps, "rec_score": [0, 0, 1, 0, 0], "point_label": 1})
    schedule = pd.DataFrame([{"attack_id": 1, "attack_name": "a", "date": "2026-03-11",
                              "start_time": "13:00:00", "end_time": "13:00:05"}])
    result = baseline.event_evaluation(scores, schedule, 0.5).iloc[0]
    assert result.alarm_points == 1
    assert result.point_recall == 0.2
    assert result.first_detection_delay_seconds == 2.0


def test_no_structure_or_fusion_configuration():
    cfg = config()
    assert cfg["model"]["use_structure_loss"] is False
    assert cfg["model"]["use_short_score"] is False
    assert cfg["model"]["use_long_score"] is False
    source = inspect.getsource(baseline.run)
    assert "fuse_scores(" not in source
    assert "score_components(" not in source


def test_outputs_and_raw_data_are_git_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "outputs/" in ignore
    assert "dataset/" in ignore
    assert "*.pt" in ignore


def test_training_code_never_writes_source_csv():
    source = (ROOT / "data" / "swat2026_training.py").read_text(encoding="utf-8")
    assert "to_csv(data_path" not in source
    assert "write_text(data_path" not in source
    assert "shutil" not in source


def test_only_one_seed_is_authorized():
    assert config()["seed"] == 1
    source = inspect.getsource(baseline.run)
    assert 'int(cfg["seed"]) != 1' in source
