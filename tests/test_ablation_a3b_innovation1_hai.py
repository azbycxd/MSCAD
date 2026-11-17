from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_ablation" / "innovation1"
HAI = OUT / "hai"


def load_runner():
    path = ROOT / "scripts" / "thesis" / "03_ablation" / "run_innovation1_a3b_hai.py"
    spec = importlib.util.spec_from_file_location("abl3b_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def summary() -> dict:
    return json.loads((OUT / "abl3b_summary.json").read_text(encoding="utf-8"))


def manifest(name: str) -> dict:
    return json.loads((HAI / name / "manifest.json").read_text(encoding="utf-8"))


def test_01_hai_r1_definition_equals_swat_r1() -> None:
    audit = json.loads((OUT / "hai_variant_parameter_audit.json").read_text(encoding="utf-8"))
    assert audit["variants"]["R1"]["definition_matches_swat_after_C_substitution"] is True
    assert audit["variants"]["R1"]["params"] == 15132


def test_02_hai_r2_definition_equals_swat_r2() -> None:
    audit = json.loads((OUT / "hai_variant_parameter_audit.json").read_text(encoding="utf-8"))
    assert audit["variants"]["R2"]["definition_matches_swat_after_C_substitution"] is True
    assert audit["variants"]["R2"]["params"] == 32911


def test_03_r0_not_retrained() -> None:
    assert summary()["r0_checkpoint_unchanged"] is True
    checkpoint = ROOT / "outputs" / "thesis_hai21_b1_real" / "rec_best.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == "17625f51a6098845956623ced3ddad6fe6fed462544ed91b274430d278fbf30d"


def test_04_formal_hai_ours_rec_scaler_path_preserved() -> None:
    for name in ("r1", "r2"):
        scaler = manifest(name)["scaler"]
        assert scaler["implementation"] == "sklearn.preprocessing.StandardScaler.partial_fit"
        assert scaler["statistic_dtype"] == "float64"
        assert scaler["model_input_dtype"] == "float32"
        with np.load(HAI / name / "scaler.npz", allow_pickle=False) as data:
            assert data["mean"].dtype == np.float64
            assert data["n_samples_seen"][0] == 737280


def test_05_train_validation_split_is_identical() -> None:
    expected = {"train1.csv.gz": 172800, "train2.csv.gz": 181440, "train3.csv.gz": 383040}
    for name in ("r1", "r2"):
        assert manifest(name)["splits"]["train_prefix_boundaries"] == expected


def test_06_no_cross_file_window() -> None:
    for name in ("r1", "r2"):
        assert manifest(name)["cross_file_windows"] is False
        with np.load(HAI / name / "test_raw_scores.npz", allow_pickle=False) as data:
            files = data["source_file"]
            raw = data["raw_index"]
            for file_name in np.unique(files):
                local = raw[files == file_name]
                assert local[0] == 63 and np.all(np.diff(local) == 1)


def test_07_training_config_identical_except_architecture() -> None:
    runner = load_runner()
    for name in ("r1", "r2"):
        assert manifest(name)["training"] == runner.TRAINING


def test_08_checkpoint_by_validation_loss_only() -> None:
    for name in ("r1", "r2"):
        checkpoint = torch.load(HAI / name / "best.pt", map_location="cpu", weights_only=False)
        assert checkpoint["selection_scope"] == "HAI Normal Validation suffix reconstruction loss only; Test unopened"


def test_09_q99_from_normal_validation_only() -> None:
    for name in ("r1", "r2"):
        thresholds = json.loads((HAI / name / "thresholds.json").read_text(encoding="utf-8"))
        with np.load(HAI / name / "validation_raw_scores.npz", allow_pickle=False) as data:
            assert thresholds["source"] == "HAI Normal Validation suffixes only"
            assert thresholds["quantile"] == 0.99
            assert abs(thresholds["endpoint_mae"] - float(np.quantile(data["endpoint_mae"], 0.99))) <= 1e-12


def test_10_raw_score_label_file_and_timeline_persisted() -> None:
    for name in ("r1", "r2"):
        with np.load(HAI / name / "test_raw_scores.npz", allow_pickle=False) as data:
            assert {"endpoint_mae", "label", "source_file", "raw_index", "timestamp"} == set(data.files)
            assert len(data["label"]) == 401690
            assert np.isfinite(data["endpoint_mae"]).all()


def test_11_event_semantics_unchanged() -> None:
    for name in ("r1", "r2"):
        events = json.loads((HAI / name / "event_metrics_50.json").read_text(encoding="utf-8"))
        assert len(events) == 50
        assert sum(row["category"] == "HIGH" for row in events) == 19
        assert sum(row["category"] == "DIRECT" for row in events) == 31


def test_12_no_pa_oracle_or_test_tuning() -> None:
    assert summary()["status"] == "ABLATION_A3_INNOVATION1_COMPLETED"
    for name in ("r1", "r2"):
        restrictions = manifest(name)["restrictions"]
        assert restrictions == {"pa": False, "oracle": False, "test_tuning": False, "hp_search": False, "architecture_search": False, "new_variant": False}
