from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "thesis" / "01_baseline" / "run_gdn_unified.py"
SPEC = importlib.util.spec_from_file_location("gdn_b3", PATH)
assert SPEC and SPEC.loader
GDN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GDN
SPEC.loader.exec_module(GDN)


def test_forecast_endpoint_alignment_and_boundaries():
    a = np.arange(70 * 2, dtype=np.float32).reshape(70, 2)
    b = 1000 + np.arange(68 * 2, dtype=np.float32).reshape(68, 2)
    store = GDN.ForecastStore.from_segments([a, b])
    assert len(store.starts) == (70 - 64) + (68 - 64)
    x, y = store.batch(np.array([0, 5, 6, 9]))
    assert x.shape == (4, 2, 64) and y.shape == (4, 2)
    assert np.array_equal(x[0].T, a[:64]) and np.array_equal(y[0], a[64])
    assert np.array_equal(x[2].T, b[:64]) and np.array_equal(y[2], b[64])


def test_frozen_gdn_retains_graph_core_and_shape():
    model = GDN.FrozenGDN(7)
    out = model(torch.randn(3, 7, 64))
    assert out.shape == (3, 7)
    assert model.embedding.embedding_dim == 64
    assert model.neighbor_graph().shape == (7, 5)
    assert model.learned_graph is not None
    assert model.output.in_features == 64


def test_official_four_point_trailing_mean_and_top1():
    errors = np.arange(6, dtype=np.float64)[:, None] * np.array([[1.0, 2.0]])
    segment = np.zeros((6 + 64, 2), dtype=np.float32)
    score = GDN.native_scores(errors, np.zeros(2), np.zeros(2), [segment])
    assert np.array_equal(score[:3], np.zeros(3))
    expected = np.mean(np.arange(4) * 2.0) / 0.01
    assert np.isclose(score[3], expected)


def test_protocol_is_frozen_and_no_forbidden_tuning_api():
    protocol = GDN.frozen_protocol()
    assert protocol["top_k_graph_neighbors"] == 5
    assert protocol["history"] == 64 and protocol["dim"] == 64
    assert protocol["batch_size"] == 32 and protocol["max_epochs"] == 30
    assert protocol["patience"] == 10 and protocol["min_delta"] == 0
    source = PATH.read_text(encoding="utf-8").lower()
    assert "test-best" not in source
    assert "point adjustment" not in source
    assert "pot(" not in source
