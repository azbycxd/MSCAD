"""Artifact-contract tests for the frozen HAI 21.03 B1 evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_hai21_b1_real"
AUDIT = ROOT / "outputs" / "thesis_hai21_a0_audit"
MANIFEST = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
SUMMARY = json.loads((OUT / "hai21_b1_summary.json").read_text(encoding="utf-8"))
THRESHOLDS = json.loads((OUT / "thresholds.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_01_three_train_five_test() -> None:
    names = list(MANIFEST["source_sha256"])
    assert sum("train" in name and name.endswith(".csv.gz") for name in names) == 3
    assert sum("test" in name and name.endswith(".csv.gz") for name in names) == 5


def test_02_no_window_crosses_file_boundaries() -> None:
    assert SUMMARY["tests"]["no_cross_file_window"] is True
    assert MANIFEST["rec_model"]["window"] == 64
    assert set(MANIFEST["train_prefix_boundaries"]) == {
        "train1.csv.gz", "train2.csv.gz", "train3.csv.gz"
    }


def test_03_scaler_is_train_prefix_only() -> None:
    assert MANIFEST["scaler_fit_rows"] == sum(MANIFEST["train_prefix_boundaries"].values())
    assert SUMMARY["tests"]["scaler_train_only"] is True


def test_04_labels_are_not_features() -> None:
    labels = {"attack", "attack_P1", "attack_P2", "attack_P3"}
    assert labels.isdisjoint(MANIFEST["feature_names"])
    assert len(MANIFEST["feature_names"]) == 79


def test_05_discovery_is_normal_only() -> None:
    assert SUMMARY["tests"]["target_discovery_normal_only"] is True
    assert MANIFEST["test_opened_after_freeze"] is True


def test_06_stable_rules_equal_frozen_2a2_rules() -> None:
    all_targets = pd.read_csv(OUT / "normal_target_discovery_all.csv")
    expected = (
        (all_targets.discovery_gain >= 0.05)
        & (all_targets.validation_gain >= 0.05)
        & (all_targets.validation_first_gain >= 0.02)
        & (all_targets.validation_second_gain >= 0.02)
    )
    assert np.array_equal(expected.to_numpy(), all_targets.stable.to_numpy())
    assert int(expected.sum()) == SUMMARY["frozen_target_count"]


def test_07_target_list_was_frozen_and_hashed() -> None:
    assert digest(OUT / "dual_targets_frozen.csv") == MANIFEST["target_freeze_sha256"]
    assert SUMMARY["tests"]["target_list_written_before_test"] is True


def test_08_high_events_come_from_official_ap_mapping() -> None:
    frozen = pd.read_csv(OUT / "high_collaboration_events_frozen.csv")
    official = pd.read_csv(AUDIT / "attack_events_50.csv")
    merged = frozen.merge(
        official[["file", "event_id", "AP_scenario"]],
        on=["file", "event_id", "AP_scenario"], how="left", indicator=True,
    )
    assert len(frozen) == 19
    assert (merged._merge == "both").all()
    assert digest(OUT / "high_collaboration_events_frozen.csv") == MANIFEST["high_event_freeze_sha256"]


def test_09_self_lags_are_fixed() -> None:
    assert MANIFEST["self_lags"] == [1, 5, 10, 20, 30]


def test_10_cross_lags_are_fixed() -> None:
    assert MANIFEST["cross_lags"] == [0, 1, 5, 10, 20, 30]
    assert MANIFEST["joint_lags"] == [1, 5, 10, 20, 30]


def test_11_target_is_excluded_from_cross_sources() -> None:
    discovery = pd.read_csv(OUT / "normal_target_discovery_all.csv")
    for row in discovery.itertuples(index=False):
        sources = str(row.cross_source_names).split(";")
        assert row.target not in sources


def test_12_ridge_alpha_is_one() -> None:
    assert MANIFEST["alpha"] == 1.0


def test_13_dual_is_predictive_disagreement_only() -> None:
    assert MANIFEST["dual_score"] == "max_j max(0,(abs(pred_self_j-pred_cross_j)-median_j)/scale_j)"
    assert SUMMARY["tests"]["dual_disagreement_only"] is True


def test_14_rec_hyperparameters_are_frozen() -> None:
    assert MANIFEST["rec_model"] == {
        "window": 64, "hidden": 64, "latent": 32, "dropout": 0.1,
        "loss": "SmoothL1", "optimizer": "Adam", "lr": 0.001,
        "weight_decay": 0.0001, "batch": 256, "max_epochs": 15, "patience": 4,
    }


def test_15_thresholds_are_normal_validation_only() -> None:
    assert THRESHOLDS["quantile"] == 0.99
    assert THRESHOLDS["source"] == "Normal Validation suffixes only"
    assert all(np.isfinite(THRESHOLDS[name]) for name in ("rec", "dual", "fusion"))


def test_16_fusion_is_unweighted_max() -> None:
    assert MANIFEST["fusion_score"] == "max(Z_rec,Z_dual)"
    assert SUMMARY["fusion_weight_search"] is False


def test_17_no_point_adjustment() -> None:
    assert SUMMARY["point_adjustment"] is False
    assert SUMMARY["tests"]["no_point_adjustment"] is True


def test_18_test_labels_are_evaluation_only() -> None:
    assert SUMMARY["tests"]["test_labels_evaluation_only"] is True
    assert SUMMARY["no_attack_driven_selection"] is True


def test_19_common_valid_mask_is_used() -> None:
    metrics = pd.read_csv(OUT / "global_metrics.csv")
    totals = metrics[["tn", "fp", "fn", "tp"]].sum(axis=1)
    assert totals.nunique() == 1
    assert SUMMARY["tests"]["common_valid_mask"] is True


def test_20_all_numeric_outputs_are_finite() -> None:
    for name in (
        "global_metrics.csv", "event_metrics_50.csv", "high_event_metrics.csv",
        "point_complementarity.csv", "event_complementarity.csv",
        "target_attribution.csv", "scenario_category_summary.csv",
    ):
        frame = pd.read_csv(OUT / name)
        numeric = frame.select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy()).all(), name
    assert SUMMARY["tests"]["no_nan_inf"] is True
