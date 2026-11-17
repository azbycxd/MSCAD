from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "thesis" / "01_baseline" / "run_lstm_ae_unified.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("baseline_b1_lstm_ae", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_architecture_shape_and_score_semantics():
    runner = load_runner()
    model = runner.LSTMAutoencoder(7)
    x = torch.randn(3, 64, 7)
    y = model(x)
    assert y.shape == x.shape
    expected = ((x[:, -1] - y[:, -1]) ** 2).mean(1)
    assert expected.shape == (3,)
    assert model.encoder.num_layers == model.decoder.num_layers == 1
    assert model.encoder.hidden_size == model.decoder.hidden_size == 64


def test_window_store_respects_common_endpoint_and_segments():
    runner = load_runner()
    a = np.arange(80 * 2, dtype=np.float32).reshape(80, 2)
    b = 10000 + np.arange(70 * 2, dtype=np.float32).reshape(70, 2)
    store = runner.WindowStore.from_segments([a, b], common_endpoint_mask=True)
    assert len(store.starts) == (80 - 64) + (70 - 64)
    windows = store.batch(np.arange(len(store.starts)))
    assert np.array_equal(windows[0], a[1:65])
    assert np.array_equal(windows[80 - 64], b[1:65])
    assert not np.any((windows < 10000).any((1, 2)) & (windows >= 10000).any((1, 2)))


def test_b0_protocol_is_exactly_bound():
    runner = load_runner()
    manifest = json.loads(runner.B0_MANIFEST.read_text(encoding="utf-8"))
    frozen = manifest["baselines"]["LSTM-AE"]
    assert manifest["status"] == "BASELINE_PROTOCOL_FROZEN"
    assert frozen["optimizer"] == "Adam"
    assert frozen["learning_rate"] == 0.001
    assert frozen["weight_decay"] == 0.0
    assert frozen["batch_size"] == 128
    assert frozen["max_epochs"] == 100
    assert manifest["common"]["early_stopping"] == {
        "source": "Normal Validation loss only", "patience": 10, "min_delta": 0.0, "restore_best": True
    }
    assert manifest["common"]["threshold"] == {
        "source": "Normal Validation endpoint scores only", "quantile": 0.99, "comparator": ">"
    }


def test_completed_outputs_if_present():
    runner = load_runner()
    manifest_path = runner.OUT / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    swat = json.loads((runner.OUT / "swat_metrics.json").read_text(encoding="utf-8"))
    hai = json.loads((runner.OUT / "hai_metrics.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "LSTM_AE_BASELINE_COMPLETED"
    assert all(manifest["sanity"].values())
    assert 0.0 <= swat["AUROC"] <= 1.0 and 0.0 <= swat["AUPR"] <= 1.0
    assert 0.0 <= hai["AUROC"] <= 1.0 and 0.0 <= hai["AUPR"] <= 1.0
    assert swat["event_count"] == 36
    assert hai["events"]["ALL"]["event_count"] == 50
    assert hai["events"]["HIGH"]["event_count"] == 19
    assert hai["events"]["DIRECT"]["event_count"] == 31
    assert abs(swat["validation_exceedance_rate"] - 0.01) < 0.001
    assert abs(hai["validation_exceedance_rate"] - 0.01) < 0.001
