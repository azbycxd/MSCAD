from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "scripts" / "thesis" / "00_audit" / "audit_swat2015_attack_phenomena.py"
    spec = importlib.util.spec_from_file_location("audit_swat2015_attack_phenomena", path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def summary():
    return json.loads((ROOT / "outputs" / "thesis_s6_swat_phenomenon_audit" / "audit_summary.json").read_text(encoding="utf-8"))


def test_01_official_event_count_is_36(): assert summary()["official_physical_event_count"] == 36
def test_02_excluded_ids_are_exact(): assert set(summary()["excluded_no_physical_impact_ids"]) == {5, 9, 12, 15, 18}


def test_03_ids_37_to_41_use_2016():
    rows = list(csv.DictReader((ROOT / "outputs" / "thesis_s6_swat_phenomenon_audit" / "event_audit.csv").open(encoding="utf-8")))
    assert all(row["start"].startswith("2016-") for row in rows if 37 <= int(row["event_id"]) <= 41)


def test_04_attack_mapping_is_exact_and_errors_are_recorded():
    m = module(); assert m.canonical_attack_point("Mv-303") == "MV303"
    rows = list(csv.DictReader((ROOT / "outputs" / "thesis_s6_swat_phenomenon_audit" / "attacked_variable_audit.csv").open(encoding="utf-8")))
    assert all(row["mapping_status"] in {"MAPPED", "MAPPING_ERROR"} for row in rows)


def test_05_normal_statistics_are_train_only():
    with pytest.raises(Exception): module().robust_reference(np.arange(3), source_split="Normal Validation")


def test_06_q995_is_validation_only():
    with pytest.raises(Exception): module().calibration_q995(np.arange(3), source_split="Normal Train")


def test_07_attack_labels_do_not_estimate_threshold(): assert summary()["attack_labels_used_for_threshold"] is False


def test_08_a0_checkpoint_unchanged():
    path = ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline" / "best.pt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == summary()["a0_checkpoint_sha256"]


def test_09_direct_rules_are_fixed():
    m = module(); assert m.direct_class_continuous(.20, 0, 0) == "STRONG_DIRECT" and m.direct_class_continuous(.05, 0, 0) == "MEDIUM_DIRECT" and m.direct_class_continuous(.049, 0, 0) == "WEAK_DIRECT"


def test_10_contextual_rules_are_fixed():
    m = module(); assert m.contextual_class("WEAK_DIRECT", "SENSOR_SPOOFING", False) == "HIGH_CONTEXTUAL_CANDIDATE" and m.contextual_class("STRONG_DIRECT", "SENSOR_SPOOFING", False) == "DIRECT_DOMINATED"


def test_11_no_point_adjustment(): assert summary()["point_adjustment"] is False
def test_12_no_model_or_fusion_tuning(): assert not summary()["new_detector"] and not summary()["parameter_search"] and not summary()["fusion"]


def test_13_unmapped_interval_is_separate():
    item = summary()["unmapped_physical_label_interval"]
    assert item["official_event_id"] is None and item["start"] == "2015-12-29 06:30:00"


def test_14_no_nan_or_inf():
    rows = list(csv.DictReader((ROOT / "outputs" / "thesis_s6_swat_phenomenon_audit" / "event_audit.csv").open(encoding="utf-8")))
    keys = ("event_score_median", "event_score_max", "median_percentile", "max_percentile", "formal_q99_hit_ratio")
    assert np.isfinite([float(row[key]) for row in rows for key in keys]).all()
