"""Contract tests for the completed SWaT-PCC1 diagnostic artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_swat_pcc1"


def manifest() -> dict:
    return json.loads((OUT / "pcc_reference_manifest.json").read_text(encoding="utf-8"))


def global_metrics() -> dict:
    return json.loads((OUT / "global_metrics.json").read_text(encoding="utf-8"))


def test_01_train_only_scaler() -> None:
    assert manifest()["tests"]["train_only_scaler"]


def test_02_train_only_ridge() -> None:
    frame = pd.read_csv(OUT / "stage_target_contributions.csv")
    assert set(frame["fit_scope"]) == {"Normal Train raw rows [0,396000) only"}


def test_03_stage_schema_source() -> None:
    value = manifest()
    assert value["tests"]["stage_from_t2_schema"] and set(value["stage_contract"]) == {f"P{i}" for i in range(1, 7)}


def test_04_no_attack_feature_selection() -> None:
    assert manifest()["attack_opened_after_fit_and_threshold"]
    assert manifest()["tests"]["no_attack_feature_selection"]


def test_05_target_excluded() -> None:
    frame = pd.read_csv(OUT / "stage_target_contributions.csv")
    assert frame["target_excluded"].all()
    assert all(row.target not in str(row.inputs).split(";") for row in frame.itertuples())


def test_06_continuous_targets_only() -> None:
    assert set(pd.read_csv(OUT / "stage_target_contributions.csv")["target_type"]) == {"continuous"}


def test_07_discrete_predictor_allowed() -> None:
    assert pd.read_csv(OUT / "stage_target_contributions.csv")["discrete_predictors_allowed"].any()


def test_08_constants_excluded() -> None:
    value = manifest()["stage_contract"]
    assert all(set(item["excluded_constants"]).isdisjoint(item["nonconstant_predictors"]) for item in value.values())


def test_09_alpha_fixed() -> None:
    assert set(pd.read_csv(OUT / "stage_target_contributions.csv")["alpha"]) == {1.0}


def test_10_no_lag() -> None:
    value = manifest()
    assert value["lag"] is False and value["pointwise_temporal_context"] == 0


def test_11_q99_validation_only() -> None:
    value = manifest()
    assert value["threshold_quantile"] == 0.99
    assert value["threshold_source"] == "Normal Validation only"
    assert value["attack_opened_after_fit_and_threshold"]


def test_12_no_point_adjustment() -> None:
    assert global_metrics()["pcc"]["point_adjustment"] is False


def test_13_no_fusion() -> None:
    value = json.loads((OUT / "complementarity.json").read_text(encoding="utf-8"))
    assert value["fusion"] is False


def test_14_official_events_unchanged() -> None:
    frame = pd.read_csv(OUT / "event_metrics.csv")
    assert len(frame) == 36 and frame["event_id"].nunique() == 36


def test_15_contextual_ids_exact() -> None:
    frame = pd.read_csv(OUT / "contextual_subset.csv")
    assert sorted(frame["event_id"].tolist()) == [13, 14, 29]


def test_16_no_nan_inf() -> None:
    metrics = global_metrics()["pcc"]
    assert all(np.isfinite(float(metrics[name])) for name in ["auroc", "aupr", "precision", "recall", "f1", "fpr", "threshold"])
    numeric = pd.read_csv(OUT / "stage_target_contributions.csv").select_dtypes(include=["number"])
    assert np.isfinite(numeric.to_numpy()).all()
