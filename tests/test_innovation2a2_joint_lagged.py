"""Artifact contract checks for the completed Innovation-2A2 discovery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_innovation2a2_joint_lagged"


def manifest() -> dict:
    return json.loads((OUT / "dependency_manifest.json").read_text(encoding="utf-8"))


def targets() -> pd.DataFrame:
    return pd.read_csv(OUT / "target_metrics.csv")


def test_01_attack_not_read() -> None:
    value = manifest()
    assert value["attack_data_read"] is False and value["attack_labels_read"] is False


def test_02_stage_from_official_schema() -> None:
    value = manifest()
    assert set(value["stage_contract"]) == {f"P{i}" for i in range(1, 7)}
    assert "thesis_t2_swat2015_audit" in value["stage_source"]


def test_03_targets_continuous_nonconstant() -> None:
    assert len(targets()) == 25 and manifest()["tests"]["targets_continuous_nonconstant"]


def test_04_cross_sources_same_stage() -> None:
    assert manifest()["tests"]["cross_sources_same_stage_only"]


def test_05_target_not_cross_source() -> None:
    assert manifest()["tests"]["target_not_cross_source"]


def test_06_constants_excluded() -> None:
    value = manifest()
    assert value["tests"]["constants_excluded"]
    assert all(set(item["excluded_constants"]).isdisjoint(item["nonconstant_channels"]) for item in value["stage_contract"].values())


def test_07_self_lags_fixed() -> None:
    assert manifest()["lags"] == [1, 5, 10, 20, 30]


def test_08_cross_lags_fixed() -> None:
    value = manifest()
    assert value["lags"] == [1, 5, 10, 20, 30] and value["lag_selection"] is False


def test_09_alpha_fixed() -> None:
    assert manifest()["alpha"] == 1.0


def test_10_no_source_selection() -> None:
    value = manifest()
    assert value["source_selection"] is False and value["tests"]["no_source_selection"]


def test_11_discovery_split() -> None:
    value = manifest()
    assert value["discovery_fit_raw_interval"] == [0, 316800]
    assert value["discovery_select_raw_interval"] == [316800, 396000]
    assert value["fit_stride"] == 4


def test_12_validation_not_design() -> None:
    assert manifest()["tests"]["validation_not_used_for_design"]


def test_13_stable_rule_fixed() -> None:
    assert manifest()["stable_rule"] == {
        "discovery_gain_min": 0.05,
        "validation_gain_min": 0.05,
        "first_half_gain_min": 0.02,
        "second_half_gain_min": 0.02,
    }


def test_14_no_anomaly_metrics() -> None:
    value = manifest()
    assert value["anomaly_metrics"] is False and value["threshold"] is None and value["fusion"] is False


def test_15_no_nan_inf() -> None:
    frame = targets()
    columns = ["joint_discovery_gain", "joint_validation_gain", "validation_gain_first", "validation_gain_second"]
    assert np.isfinite(frame[columns].to_numpy(float)).all()
