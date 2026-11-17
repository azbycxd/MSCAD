from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.mscad import MSCAD


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hai_step3a_train_rec_baseline.py"
CONFIG = ROOT / "configs" / "hai_rec_only_seed1.yaml"
SPEC = importlib.util.spec_from_file_location("hai_rec_baseline", SCRIPT)
baseline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(baseline)


def config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def model():
    return baseline.make_model(config(), 86, torch.device("cpu"))


def test_model_accepts_batch_64_86():
    output = model()(torch.randn(2, 64, 86))
    assert output.shape == (2, 64, 86)


def test_reconstruction_shape_matches_input():
    net = model()
    x = torch.randn(3, 64, 86)
    assert net(x).shape == x.shape


def test_rec_only_epoch_does_not_compute_structure_loss():
    net = model()
    net.struct_loss_fn.forward = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("structure called"))
    loader = DataLoader(TensorDataset(torch.randn(4, 64, 86), torch.zeros(4, dtype=torch.long)), batch_size=2)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss, _ = baseline.train_epoch(net, loader, optimizer, torch.device("cpu"))
    assert np.isfinite(loss)


def test_no_swat_checkpoint_initialization():
    cfg = config()
    assert cfg["training"]["init_checkpoint"] is None
    assert "swat" not in str(cfg["training"]).lower()


def test_channels_are_not_hardcoded_to_51():
    source = (ROOT / "models" / "backbone_ae.py").read_text(encoding="utf-8")
    assert "in_channels" in source
    assert "51" not in source


def test_hai_boundary_safe_loader_is_used():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "build_hai_dataloaders" in source
    assert "bundle.test_sequences" in source
    assert "MultiSequenceWindowDataset([sequence]" in source


def test_best_checkpoint_selected_by_validation_loss():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "improved = val_loss < best_val" in source
    assert '"selection_scope": "normal validation reconstruction loss only"' in source


def test_test_labels_not_used_for_model_selection():
    signature = inspect.signature(baseline.fit_normal_thresholds)
    assert list(signature.parameters) == ["train_scores", "validation_scores", "cfg"]
    source = inspect.getsource(baseline.train_epoch) + inspect.getsource(baseline.validation_loss)
    assert "test" not in source.lower()


def test_rec_score_matches_existing_definition():
    x = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
    x_hat = torch.zeros_like(x)
    expected = (x_hat - x).abs().mean(dim=(1, 2))
    assert torch.equal(baseline.reconstruction_score(x, x_hat), expected)


def test_thresholds_use_only_normal_inputs():
    cfg = config()
    train = np.linspace(0.0, 1.0, 1000)
    val = np.linspace(0.0, 2.0, 500)
    rows = baseline.fit_normal_thresholds(train, val, cfg)
    assert {row["threshold_method"] for row in rows} == {"train_only_pot", "normal_validation_q99"}
    q99 = next(row for row in rows if row["threshold_method"] == "normal_validation_q99")
    assert np.isclose(q99["threshold_value"], np.quantile(val, 0.99))


def test_four_tests_evaluated_separately():
    values = {"score": np.array([0.1, 0.9]), "label": np.array([0, 1])}
    rows = [baseline.metric_row(f"test{i}.csv", "q99", values, 0.5) for i in range(1, 5)]
    assert [row["file_name"] for row in rows] == [f"test{i}.csv" for i in range(1, 5)]


def test_events_never_cross_files():
    events = [
        {"sequence_id": "test1", "file_name": "test1.csv", "event_id_in_file": 1, "global_event_id": 1,
         "start_row": 2, "end_row": 3, "start_timestamp": "t2", "end_timestamp": "t3", "duration_points": 2},
        {"sequence_id": "test2", "file_name": "test2.csv", "event_id_in_file": 1, "global_event_id": 2,
         "start_row": 0, "end_row": 1, "start_timestamp": "u0", "end_timestamp": "u1", "duration_points": 2},
    ]
    values = {
        "test1.csv": {"score": np.array([0.1, 0.8]), "label": np.array([1, 1]), "end_row": np.array([2, 3]), "timestamp": np.array(["t2", "t3"])},
        "test2.csv": {"score": np.array([0.7, 0.1]), "label": np.array([1, 1]), "end_row": np.array([0, 1]), "timestamp": np.array(["u0", "u1"])},
    }
    diagnostics = baseline.event_diagnostics(events, values, [{"status": "ok", "threshold_method": "q99", "threshold_value": 0.5}])
    assert len(diagnostics) == 2
    assert diagnostics.groupby("file_name").size().to_dict() == {"test1.csv": 1, "test2.csv": 1}


def test_parameter_count_is_exact():
    net = model()
    total, trainable = baseline.parameter_counts(net)
    assert total == sum(parameter.numel() for parameter in net.parameters())
    assert trainable == total


def test_benchmark_uses_eval_and_no_grad():
    source = inspect.getsource(baseline.benchmark)
    assert "model.eval()" in source
    assert "torch.no_grad()" in source
    assert "for _ in range(10)" in source and "for _ in range(100)" in source


def test_smoke_and_formal_outputs_are_isolated():
    cfg = config()
    assert cfg["output"]["dir"] != cfg["output"]["smoke_dir"]


def test_output_metrics_remain_finite():
    values = {"score": np.array([0.1, 0.9, 0.2, 0.8]), "label": np.array([0, 1, 0, 1])}
    row = baseline.metric_row("test1.csv", "q99", values, 0.5)
    assert np.isfinite([row[name] for name in ("precision", "recall", "f1", "aupr", "auroc")]).all()
