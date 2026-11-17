"""Artifact-level protocol checks for Innovation-2A."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_innovation2a_lagged_dependency"


def manifest() -> dict:
    return json.loads((OUT / "dependency_manifest.json").read_text(encoding="utf-8"))


def candidates() -> pd.DataFrame:
    return pd.read_csv(OUT / "all_candidate_relations.csv")


def test_01_attack_not_read() -> None:
    value = manifest()
    assert value["attack_data_read"] is False and value["attack_labels_read"] is False


def test_02_stage_from_official_schema() -> None:
    value = manifest()
    assert set(value["stage_contract"]) == {f"P{i}" for i in range(1, 7)}
    assert "thesis_t2_swat2015_audit" in value["stage_source"]


def test_03_targets_continuous_nonconstant() -> None:
    frame = candidates()
    assert set(frame["target_kind"]) == {"continuous_numeric"}
    assert manifest()["tests"]["target_continuous_nonconstant_only"]


def test_04_sources_same_stage_and_not_target() -> None:
    frame = candidates()
    assert (frame["source"] != frame["target"]).all()
    assert manifest()["tests"]["source_same_stage_not_target"]


def test_05_constants_excluded() -> None:
    value = manifest()
    assert value["tests"]["constant_excluded"]
    assert all(set(v["excluded_constants"]).isdisjoint(v["nonconstant_sources"]) for v in value["stage_contract"].values())


def test_06_self_lags_fixed() -> None:
    assert manifest()["self_lags"] == [1, 5, 10, 20, 30]


def test_07_cross_lags_fixed() -> None:
    value = manifest()
    assert value["cross_lags"] == [1, 5, 10, 20, 30] and value["lag_zero_allowed"] is False


def test_08_alpha_fixed() -> None:
    assert manifest()["alpha"] == 1.0


def test_09_lag_selection_train_internal() -> None:
    value = manifest()
    assert value["lag_selection_scope"] == "Discovery Select inside Normal Train only"
    assert value["discovery_select_raw_interval"] == [316800, 396000]


def test_10_validation_not_used_for_lag_selection() -> None:
    assert manifest()["tests"]["validation_not_used_for_lag_selection"]


def test_11_source_selection_no_attack() -> None:
    value = manifest()
    assert value["source_selection_scope"] == "audited same-stage nonconstant channels only"
    assert value["attack_data_read"] is False


def test_12_stable_rule_fixed() -> None:
    assert manifest()["stable_rule"] == {
        "discovery_gain_min": 0.05,
        "validation_gain_min": 0.05,
        "validation_first_gain_min": 0.02,
        "validation_second_gain_min": 0.02,
        "beta_first_times_beta_second_positive": True,
    }


def test_13_top3_fixed() -> None:
    selected = pd.read_csv(OUT / "selected_relations.csv")
    assert manifest()["top_k_per_target"] == 3
    assert selected.groupby("target").size().max() <= 3


def test_14_chronology() -> None:
    value = manifest()
    assert value["discovery_fit_raw_interval"] == [0, 316800]
    assert value["discovery_select_raw_interval"] == [316800, 396000]
    assert value["train_rows"] == 396000 and value["validation_rows"] == 99000
    assert value["tests"]["train_validation_chronology"]


def test_15_no_nan_inf() -> None:
    frame = candidates()
    columns = ["discovery_gain", "validation_gain", "gain_first", "gain_second", "beta_full", "beta_first", "beta_second"]
    assert np.isfinite(frame[columns].to_numpy(float)).all()
