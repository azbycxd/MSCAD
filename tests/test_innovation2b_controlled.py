"""Artifact-level checks for the completed Innovation-2B controlled validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_innovation2b_controlled"


def manifest() -> dict:
    return json.loads((OUT / "corruption_manifest.json").read_text(encoding="utf-8"))


def metrics() -> pd.DataFrame:
    return pd.read_csv(OUT / "controlled_metrics.csv")


def test_01_attack_not_read() -> None:
    value = manifest()
    assert value["attack_data_read"] is False and value["attack_labels_read"] is False


def test_02_targets_exact() -> None:
    assert manifest()["stable_targets"] == ["AIT202", "FIT201", "DPIT301", "FIT301", "FIT401", "FIT601"]


def test_03_stage_unchanged() -> None:
    assert manifest()["tests"]["stage_definition_unchanged"]


def test_04_self_features_unchanged() -> None:
    assert manifest()["tests"]["self_features_unchanged"]


def test_05_joint_features_unchanged() -> None:
    assert manifest()["tests"]["joint_features_unchanged"]


def test_06_alpha_fixed() -> None:
    assert manifest()["alpha"] == 1.0


def test_07_target_current_unchanged() -> None:
    assert manifest()["tests"]["target_current_unchanged"]


def test_08_target_history_unchanged() -> None:
    assert manifest()["tests"]["target_history_unchanged"]


def test_09_donor_deterministic() -> None:
    assert manifest()["tests"]["donor_deterministic"]


def test_10_donor_separation() -> None:
    assert manifest()["donor_min_separation_seconds"] >= 600


def test_11_donor_from_normal() -> None:
    assert manifest()["tests"]["donor_from_normal"]


def test_12_a0_checkpoint_unchanged() -> None:
    value = manifest()
    assert value["tests"]["a0_checkpoint_unchanged"] and value["a0"]["retrained"] is False


def test_13_threshold_calibration_only() -> None:
    value = manifest()
    assert value["threshold_source"] == "per-target calibration_normal S_collab q99 only"
    assert value["corrupted_labels_used_for_threshold"] is False


def test_14_no_test_tuning() -> None:
    value = manifest()
    assert value["tests"]["no_test_driven_tuning"] and value["threshold_search"] is False


def test_15_no_nan_inf() -> None:
    frame = metrics().select_dtypes(include=["number"])
    semantics = pd.read_csv(OUT / "score_semantics.csv").select_dtypes(include=["number"])
    assert np.isfinite(frame.to_numpy()).all() and np.isfinite(semantics.to_numpy()).all()
