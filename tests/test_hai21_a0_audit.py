"""Integrity tests for the HAI 21.03 read-only protocol audit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_hai21_a0_audit"
SUMMARY = json.loads((OUT / "audit_summary.json").read_text(encoding="utf-8"))
MANIFEST = pd.read_csv(OUT / "file_manifest.csv")
SCHEMA = pd.read_csv(OUT / "variable_schema.csv")
EVENTS = pd.read_csv(OUT / "attack_events_50.csv")


def test_version_and_file_counts() -> None:
    assert SUMMARY["version"] == "21.03"
    assert SUMMARY["train_file_count"] == 3
    assert SUMMARY["test_file_count"] == 5


def test_gzip_direct_read_and_timestamp_parsing() -> None:
    assert MANIFEST["gzip_read_direct"].all()
    assert (MANIFEST["nominal_sampling_interval_seconds"] == 1.0).all()
    assert SUMMARY["all_files_internal_timestamp_continuity"]


def test_exact_label_contract_and_feature_exclusion() -> None:
    assert SUMMARY["tests"]["labels_exact"]
    assert SUMMARY["tests"]["labels_excluded_features"]
    assert not {"attack", "attack_P1", "attack_P2", "attack_P3"} & set(SCHEMA["column_name"])


def test_train_normality_audited() -> None:
    train = MANIFEST[MANIFEST["split"] == "train"]
    assert SUMMARY["train_pure_normal"]
    assert int(train[["attack_count", "attack_P1_count", "attack_P2_count", "attack_P3_count"]].sum().sum()) == 0


def test_test_intervals_are_per_file_and_total_50() -> None:
    assert EVENTS["event_id"].str.match(r"T[1-5]-E\d{2}").all()
    assert EVENTS.groupby("file").size().to_dict() == {
        "test1.csv.gz": 5, "test2.csv.gz": 20, "test3.csv.gz": 8,
        "test4.csv.gz": 5, "test5.csv.gz": 12,
    }
    assert len(EVENTS) == 50 and SUMMARY["official_50_match"]


def test_process_grouping_is_prefix_only() -> None:
    assert SUMMARY["tests"]["prefix_grouping_only"]
    assert SUMMARY["process_channel_counts"] == {"P1": 38, "P2": 22, "P3": 7, "P4": 12}
    assert all(name.startswith(prefix + "_") for name, prefix in zip(SCHEMA["column_name"], SCHEMA["prefix"]))


def test_control_loops_and_catalog_are_official_only() -> None:
    assert SUMMARY["tests"]["control_loops_official_docs_only"]
    assert SUMMARY["tests"]["ap_catalog_official_docs_only"]
    assert SUMMARY["hai21_supported_ap_scenario_count"] == 25


def test_interval_to_ap_mapping_is_source_bound() -> None:
    assert SUMMARY["official_interval_to_scenario_mapping"]
    assert EVENTS["AP_scenario"].notna().all()
    assert (EVENTS["official_source"] == "technical-details PDF pp.36-38").all()
    assert SUMMARY["official_local_start_time_conflict_count"] == 1


def test_no_model_or_metric_call() -> None:
    assert SUMMARY["tests"]["no_model_call"]
    assert not SUMMARY["models_loaded"]
    assert not SUMMARY["model_inference"]
    assert not SUMMARY["auroc_aupr_computed"]


def test_no_missing_or_nonfinite_values() -> None:
    assert int(MANIFEST["nan_count"].sum()) == 0
    assert int(MANIFEST["inf_count"].sum()) == 0

