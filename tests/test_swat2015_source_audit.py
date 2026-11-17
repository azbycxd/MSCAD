from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import datetime, time
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "thesis" / "00_audit" / "audit_swat2015_source.py"


@pytest.fixture(scope="module")
def audit_module():
    spec = importlib.util.spec_from_file_location("swat2015_source_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _identity_root(tmp_path: Path) -> Path:
    root = tmp_path / "Swat_2015"
    physical = root / "Physical"
    physical.mkdir(parents=True)
    (root / "List_of_attacks_Final.xlsx").write_bytes(b"xlsx-placeholder")
    (root / "List_of_attacks_Final.pdf").write_bytes(b"pdf-placeholder")
    (root / "A Dataset to Support Research in the Design of Secure Water Treatment Systems.pdf").write_bytes(b"pdf-placeholder")
    return root


def _physical_workbook(path: Path, rows: list[list[object]]) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Combined Data"
    sheet.append([" Timestamp", "FIT101", "MV101", "Normal/Attack"])
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    workbook.close()


def _attack_workbook(path: Path, rows: list[list[object]]) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append([
        "Attack #", "Start Time", "End Time", "Attack Point", "Start State",
        "Attack", "Actual Change", "Expected Impact or attacker intent", "Unexpected Outcome",
    ])
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    workbook.close()


def test_official_root_must_be_explicit(audit_module):
    with pytest.raises(audit_module.AuditError, match="explicitly"):
        audit_module.resolve_official_root(None)


def test_legacy_swat_directory_is_rejected(tmp_path, audit_module):
    legacy = tmp_path / "Swat"
    legacy.mkdir()
    with pytest.raises(audit_module.AuditError, match="Legacy"):
        audit_module.resolve_official_root(legacy)


def test_no_fallback_when_identity_files_are_missing(tmp_path, audit_module):
    root = tmp_path / "Swat_2015"
    root.mkdir()
    with pytest.raises(audit_module.AuditError, match="identity files missing"):
        audit_module.resolve_official_root(root)


def test_legacy_csv_inside_official_root_is_rejected(tmp_path, audit_module):
    root = _identity_root(tmp_path)
    (root / "merged.csv").write_text("legacy", encoding="utf-8")
    with pytest.raises(audit_module.AuditError, match="Legacy normal.csv/merged.csv"):
        audit_module.resolve_official_root(root)


def test_valid_identity_root_resolves_exactly(tmp_path, audit_module):
    root = _identity_root(tmp_path)
    assert audit_module.resolve_official_root(root) == root.resolve()


def test_timestamp_parser_accepts_official_format(audit_module):
    assert audit_module.parse_timestamp(" 28/12/2015 10:00:00 AM") == datetime(2015, 12, 28, 10, 0, 0)


def test_timestamp_parser_rejects_malformed_value(audit_module):
    with pytest.raises(ValueError, match="unparseable"):
        audit_module.parse_timestamp("2015/99/99")


def test_schema_identifies_timestamp_label_and_variables(audit_module):
    profile = audit_module.identify_schema([" Timestamp", "FIT101", " MV101", "Normal/Attack"])
    assert profile["timestamp_index"] == 0
    assert profile["label_index"] == 3
    assert profile["variable_names"] == ["FIT101", "MV101"]
    assert profile["paper_literal_sensor_count"] == 1
    assert profile["paper_literal_actuator_count"] == 1


def test_duplicate_timestamp_detection(tmp_path, audit_module):
    path = tmp_path / "duplicate.xlsx"
    _physical_workbook(path, [
        ["22/12/2015 04:00:00 PM", 1.0, 1, "Normal"],
        ["22/12/2015 04:00:00 PM", 1.1, 1, "Normal"],
    ])
    assert audit_module.audit_physical_workbook(path)["duplicate_timestamp_count"] == 1


def test_gap_detection(tmp_path, audit_module):
    path = tmp_path / "gap.xlsx"
    _physical_workbook(path, [
        ["22/12/2015 04:00:00 PM", 1.0, 1, "Normal"],
        ["22/12/2015 04:00:04 PM", 1.1, 1, "Normal"],
    ])
    result = audit_module.audit_physical_workbook(path)
    assert result["large_gap_count"] == 1
    assert result["large_gap_examples"][0]["delta_seconds"] == 4


def test_invalid_value_is_recorded_not_zero_filled(tmp_path, audit_module):
    path = tmp_path / "invalid.xlsx"
    _physical_workbook(path, [
        ["22/12/2015 04:00:00 PM", "bad-token", 1, "Normal"],
        ["22/12/2015 04:00:01 PM", None, 1, "Normal"],
    ])
    result = audit_module.audit_physical_workbook(path)
    fit = next(item for item in result["variable_profiles"] if item["name"] == "FIT101")
    assert fit["malformed_count"] == 1
    assert fit["missing_count"] == 1
    assert fit["numeric_count"] == 0
    assert {item["raw"] for item in result["invalid_examples"]} == {"'bad-token'", "None"}


def test_label_unique_spellings_remain_distinct(tmp_path, audit_module):
    path = tmp_path / "labels.xlsx"
    _physical_workbook(path, [
        ["22/12/2015 04:00:00 PM", 1.0, 1, "Normal"],
        ["22/12/2015 04:00:01 PM", 1.0, 1, "Attack"],
        ["22/12/2015 04:00:02 PM", 1.0, 1, "A ttack"],
    ])
    counts = audit_module.audit_physical_workbook(path)["raw_label_counts"]
    assert counts == {"Normal": 1, "Attack": 1, "A ttack": 1}


def test_attack_list_reading_preserves_official_fields(tmp_path, audit_module):
    path = tmp_path / "attacks.xlsx"
    _attack_workbook(path, [[
        1, datetime(2015, 12, 28, 10, 29, 14), time(10, 44, 53), "MV-101",
        "closed", "Open MV-101", "Yes", "Tank overflow", None,
    ]])
    result = audit_module.read_attack_list(path)
    assert result["listed_item_count"] == 1
    assert result["physical_interval_count"] == 1
    assert result["attacks"][0]["attack_point"] == "MV-101"


def _alignment_inputs(start: str, end: str, label: str = "Attack"):
    attack_list = {"attacks": [{
        "attack_id": 1, "attack_point": "MV-101", "start": start, "end": end,
        "no_physical_impact_marker": False,
    }]}
    physical = {
        "first_timestamp": "2015-12-28 10:00:00",
        "last_timestamp": "2016-01-02 14:59:59",
        "label_segments": [{"label": label, "start": start, "end_inclusive": end}],
    }
    return attack_list, physical


def test_attack_time_alignment_exact(audit_module):
    attack_list, physical = _alignment_inputs("2015-12-28 10:29:14", "2015-12-28 10:44:53")
    assert audit_module.align_attack_intervals(attack_list, physical)[0]["classification"] == "exact"


def test_label_typo_is_not_silently_normalized(audit_module):
    attack_list, physical = _alignment_inputs("2015-12-28 10:29:14", "2015-12-28 10:44:53", "A ttack")
    row = audit_module.align_attack_intervals(attack_list, physical)[0]
    assert row["classification"] == "substantive_label_mismatch"
    assert row["label_at_official_start"] == "A ttack"


def test_calendar_year_conflict_is_explicit(audit_module):
    attack_list, physical = _alignment_inputs("2015-01-02 11:17:02", "2015-01-02 11:24:50")
    physical["label_segments"] = [{
        "label": "Attack", "start": "2016-01-02 11:17:02", "end_inclusive": "2016-01-02 11:24:50",
    }]
    row = audit_module.align_attack_intervals(attack_list, physical)[0]
    assert row["classification"] == "official_metadata_calendar_year_conflict"
    assert row["alignment_after_explicit_year_diagnostic"] == "exact"
    assert row["official_start_raw"].startswith("2015-")


def test_source_workbook_fingerprint_is_unchanged(tmp_path, audit_module):
    path = tmp_path / "readonly.xlsx"
    _physical_workbook(path, [["22/12/2015 04:00:00 PM", 1.0, 1, "Normal"]])
    before = audit_module.fingerprint(path)
    result = audit_module.audit_physical_workbook(path)
    after = audit_module.fingerprint(path)
    assert before == after
    assert result["read_only_integrity"].startswith("size, mtime_ns")


def test_network_pcap_is_metadata_only(tmp_path, monkeypatch, audit_module):
    root = _identity_root(tmp_path)
    network = root / "Network"
    network.mkdir()
    pcap = network / "capture.pcap"
    pcap.write_bytes(b"must-not-be-read")
    original = audit_module.sha256_file

    def guarded_hash(path):
        if path.suffix.casefold() == ".pcap":
            raise AssertionError("PCAP content was read")
        return original(path)

    monkeypatch.setattr(audit_module, "sha256_file", guarded_hash)
    row = next(item for item in audit_module.inventory_source(root) if item["relative_path"].endswith("capture.pcap"))
    assert row["sha256"] is None
    assert row["inspection"] == "metadata_only_not_parsed"


def test_module_has_no_training_window_builder(audit_module):
    assert not hasattr(audit_module, "build_windows")
    assert not hasattr(audit_module, "train")


def test_outputs_are_git_ignored():
    repo = SCRIPT.parents[3]
    completed = subprocess.run(
        ["git", "check-ignore", "outputs/thesis_t2_swat2015_audit/schema_profile.json"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip().startswith("outputs/")
