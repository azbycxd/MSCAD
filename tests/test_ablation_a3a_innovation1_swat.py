from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_ablation" / "innovation1"
SUMMARY = json.loads((OUT / "abl3a_summary.json").read_text(encoding="utf-8"))
VARIANTS = json.loads((OUT / "rec_variant_manifest.json").read_text(encoding="utf-8"))


def _load_runner():
    path = ROOT / "scripts" / "thesis" / "03_ablation" / "run_innovation1_a3a_swat.py"
    spec = importlib.util.spec_from_file_location("abl3a_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _variant_manifest(name: str) -> dict:
    return json.loads((OUT / "swat" / name.lower() / "manifest.json").read_text(encoding="utf-8"))


def test_01_r1_r2_definitions_match_frozen_documents() -> None:
    runner = _load_runner()
    signatures = runner.validate_architecture_signatures(
        runner.BackboneAE(51, 64, 32, 0.1),
        runner.TemporalFreeAE(51, 64, 32, 0.1),
        runner.DenseConvAE(51, 64, 32, 0.1),
    )
    assert signatures["R0 Final"]["parameter_count"] == 11730
    assert signatures["R1"]["parameter_count"] == 11436
    assert signatures["R2"]["parameter_count"] == 25715
    assert "k=3 -> k=1" in VARIANTS["variants"]["R1"]["relative_to_R0"]
    assert "same-width dense k=3" in VARIANTS["variants"]["R2"]["relative_to_R0"]


def test_02_r0_was_not_retrained_or_modified() -> None:
    checkpoint = ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline" / "best.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == VARIANTS["source_hashes"]["r0_checkpoint"]
    assert SUMMARY["r0_checkpoint_unchanged"] is True


def test_03_official_swat_split_is_exact() -> None:
    for name in ("r1", "r2"):
        manifest = _variant_manifest(name)
        assert manifest["splits"]["train"] == [0, 396000]
        assert manifest["splits"]["validation"] == [396000, 495000]
        assert manifest["splits"]["test_segments"] == [[0, 298837], [298837, 449919]]


def test_04_scaler_is_train_only_and_matches_formal_rec_dtype() -> None:
    # E2-R1D distinguishes the float32 S1 Ours-REC scaler from float64 LSTM/GDN baselines.
    for name in ("r1", "r2"):
        manifest = _variant_manifest(name)
        assert manifest["scaler"] == {
            "fit_scope": "Normal Train only",
            "source_dtype": "float32",
            "statistic_dtype": "float32",
            "validation_rows_seen": 0,
            "attack_rows_seen": 0,
        }
        with np.load(OUT / "swat" / name / "scaler.npz", allow_pickle=False) as scaler:
            assert scaler["mean"].dtype == np.float32
            assert scaler["effective_scale"].dtype == np.float32


def test_05_feature_order_matches_frozen_cache() -> None:
    expected = VARIANTS["feature_order"]
    assert len(expected) == 51
    for name in ("r1", "r2"):
        actual = json.loads((OUT / "swat" / name / "feature_order.json").read_text(encoding="utf-8"))
        assert actual["features"] == expected
        assert actual["sha256"] == VARIANTS["feature_order_sha256"]


def test_06_window_and_stride_remain_64_and_1() -> None:
    assert VARIANTS["frozen_data"]["window"] == 64
    assert VARIANTS["frozen_data"]["stride"] == 1
    for name in ("r1", "r2"):
        manifest = _variant_manifest(name)
        assert manifest["window"] == 64 and manifest["stride"] == 1


def test_07_no_window_crosses_the_hard_gap() -> None:
    for name in ("r1", "r2"):
        manifest = _variant_manifest(name)
        assert manifest["hard_gap_crossing"] is False
        with np.load(OUT / "swat" / name / "test_raw_scores.npz", allow_pickle=False) as scores:
            timestamp = scores["endpoint_timestamp"]
            assert len(timestamp) == (298837 - 63) + (151082 - 63)
            # The raw hard gap is 82 seconds. Independent windowing also omits
            # the first 63 endpoints of segment 2, so legal endpoints differ by 145 s.
            differences = np.diff(timestamp)
            assert np.count_nonzero(differences != 1) == 1
            assert np.max(differences) == 145


def test_08_training_config_is_identical_except_architecture() -> None:
    expected = VARIANTS["frozen_training"]
    assert expected["loss"] == "SmoothL1Loss"
    assert expected["optimizer"] == "Adam"
    assert expected["learning_rate"] == 1e-3
    assert expected["weight_decay"] == 1e-4
    assert expected["batch_size"] == 256
    assert expected["max_epochs"] == 15
    assert expected["early_stopping_patience"] == 4
    for name in ("r1", "r2"):
        assert _variant_manifest(name)["training"] == expected


def test_09_checkpoint_selection_uses_only_validation_loss() -> None:
    for name in ("r1", "r2"):
        manifest = _variant_manifest(name)
        assert manifest["checkpoint_selection"] == "minimum Normal Validation reconstruction loss"
        checkpoint = torch.load(OUT / "swat" / name / "best.pt", map_location="cpu", weights_only=False)
        assert checkpoint["selection_scope"] == "Normal v1 validation reconstruction loss only; Attack unopened"


def test_10_q99_threshold_comes_only_from_normal_validation() -> None:
    for name in ("r1", "r2"):
        thresholds = json.loads((OUT / "swat" / name / "thresholds.json").read_text(encoding="utf-8"))
        with np.load(OUT / "swat" / name / "validation_raw_scores.npz", allow_pickle=False) as scores:
            assert thresholds["source"] == "Normal Validation only"
            assert thresholds["quantile"] == 0.99
            assert thresholds["comparator"] == ">"
            assert abs(thresholds["endpoint_mae"] - float(np.quantile(scores["endpoint_mae"], 0.99))) <= 1e-12


def test_11_raw_scores_labels_timeline_and_hashes_are_persisted() -> None:
    for name in ("r1", "r2"):
        manifest = _variant_manifest(name)
        for filename in ("validation_raw_scores.npz", "test_raw_scores.npz"):
            path = OUT / "swat" / name / filename
            assert path.exists()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["artifacts"][filename]["sha256"]
        with np.load(OUT / "swat" / name / "test_raw_scores.npz", allow_pickle=False) as scores:
            assert {"endpoint_timestamp", "label", "endpoint_mae", "full_window_mae"} == set(scores.files)
            assert len(scores["label"]) == 449793


def test_12_no_pa_oracle_test_tuning_or_search() -> None:
    assert SUMMARY["status"] == "ABLATION_A3A_INNOVATION1_SWAT_COMPLETED"
    for name in ("r1", "r2"):
        restrictions = _variant_manifest(name)["restrictions"]
        assert restrictions == {"pa": False, "oracle": False, "test_tuning": False, "hp_search": False, "architecture_search": False}
