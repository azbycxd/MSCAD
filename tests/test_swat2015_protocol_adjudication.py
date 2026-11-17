from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "thesis" / "00_audit" / "adjudicate_swat2015_protocol.py"


@pytest.fixture(scope="module")
def protocol_module():
    spec = importlib.util.spec_from_file_location("swat2015_protocol_adjudication", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _raw_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    no_impact = {5, 9, 12, 15, 18}
    for attack_id in range(1, 42):
        if attack_id == 36:
            start = datetime(2016, 1, 1, 22, 16, 1)
            end = datetime(2016, 1, 1, 22, 25, 0)
        elif 37 <= attack_id <= 40:
            start = datetime(2015, 1, 2, 11, attack_id - 30, 0)
            end = start + timedelta(minutes=4)
        elif attack_id == 41:
            start = datetime(2015, 1, 2, 13, 13, 2)
            end = datetime(2015, 1, 2, 13, 40, 56)
        else:
            start = datetime(2015, 12, 28, 10, 0, 0) + timedelta(minutes=attack_id * 10)
            end = start + timedelta(minutes=5)
        marker = attack_id in no_impact
        records.append({
            "attack_id": attack_id,
            "start": start.isoformat(sep=" "),
            "end": None if marker else end.isoformat(sep=" "),
            "attack_point": "No Physical Impact Attack" if marker else f"P-{attack_id}",
            "start_state": None,
            "description": None,
            "actual_change": None,
            "expected_impact_or_intent": None,
            "unexpected_outcome": None,
            "no_physical_impact_marker": marker,
        })
    return records


def _normal_profile() -> dict[str, object]:
    return {
        "file_name": "SWaT_Dataset_Normal_v1.xlsx",
        "row_count": 495000,
        "first_timestamp": "2015-12-22 16:30:00",
        "last_timestamp": "2015-12-28 09:59:59",
        "timestamp_parse_errors": 0,
    }


def _attack_profile() -> dict[str, object]:
    return {
        "file_name": "SWaT_Dataset_Attack_v0.xlsx",
        "row_count": 449919,
        "first_timestamp": "2015-12-28 10:00:00",
        "last_timestamp": "2016-01-02 14:59:59",
        "raw_label_counts": {"Normal": 395298, "Attack": 54584, "A ttack": 37},
        "large_gap_examples": [{
            "previous": "2015-12-31 21:00:36",
            "current": "2015-12-31 21:01:58",
            "delta_seconds": 82,
        }],
    }


def test_normal_exact_mapping(protocol_module):
    assert protocol_module.canonicalize_point_label("Normal") == 0


def test_attack_exact_mapping(protocol_module):
    assert protocol_module.canonicalize_point_label("Attack") == 1


def test_a_space_ttack_exact_mapping(protocol_module):
    assert protocol_module.canonicalize_point_label("A ttack") == 1


@pytest.mark.parametrize("unknown", [" attack", "attack", "Attack ", "A  ttack", "", None, 1])
def test_unknown_label_raises_without_strip_or_guess(protocol_module, unknown):
    with pytest.raises(protocol_module.ProtocolAdjudicationError, match="unknown exact"):
        protocol_module.canonicalize_point_label(unknown)


def test_ids_37_to_41_keep_raw_year_and_add_canonical_2016(protocol_module):
    canonical = protocol_module.canonicalize_attack_records(_raw_records())
    by_id = {row["attack_id"]: row for row in canonical}
    for attack_id in range(37, 42):
        assert by_id[attack_id]["raw_start"].startswith("2015-01-02")
        assert by_id[attack_id]["canonical_start"].startswith("2016-01-02")
        assert by_id[attack_id]["chronology_correction"] is True


def test_raw_attack_records_are_not_mutated(protocol_module):
    raw = _raw_records()
    before = copy.deepcopy(raw)
    protocol_module.canonicalize_attack_records(raw)
    assert raw == before


def test_five_no_impact_markers_have_no_fabricated_end(protocol_module):
    canonical = protocol_module.canonicalize_attack_records(_raw_records())
    markers = [row for row in canonical if row["no_physical_impact_marker"]]
    assert {row["attack_id"] for row in markers} == {5, 9, 12, 15, 18}
    assert all(row["event_id"] is None and row["canonical_end"] is None for row in markers)


def test_exactly_36_evaluable_event_intervals(protocol_module):
    canonical = protocol_module.canonicalize_attack_records(_raw_records())
    assert sum(row["evaluable_physical_attack_interval"] for row in canonical) == 36


def test_ids_10_and_11_remain_independent_events(protocol_module):
    by_id = {row["attack_id"]: row for row in protocol_module.canonicalize_attack_records(_raw_records())}
    assert by_id[10]["event_id"] == 10
    assert by_id[11]["event_id"] == 11


def test_ids_27_and_28_remain_independent_events(protocol_module):
    by_id = {row["attack_id"]: row for row in protocol_module.canonicalize_attack_records(_raw_records())}
    assert by_id[27]["event_id"] == 27
    assert by_id[28]["event_id"] == 28


def test_physical_only_interval_does_not_create_event(protocol_module):
    policy = protocol_module.build_point_label_policy(_attack_profile()["raw_label_counts"])
    interval = policy["physical_only_intervals"][0]
    assert interval["physical_only_label_interval"] is True
    assert interval["event_id"] is None
    assert interval["start"] == "2015-12-29 06:30:00"


def test_id41_event_end_comes_from_canonical_attack_list(protocol_module):
    by_id = {row["attack_id"]: row for row in protocol_module.canonicalize_attack_records(_raw_records())}
    assert by_id[41]["canonical_start"] == "2016-01-02 13:13:02"
    assert by_id[41]["canonical_end"] == "2016-01-02 13:40:56"
    policy = protocol_module.build_point_label_policy(_attack_profile()["raw_label_counts"])
    assert policy["id41_pointwise_rule"]["extra_pointwise_seconds"] == 15


def test_82_second_gap_creates_two_hard_segments(protocol_module):
    manifest = protocol_module.build_segment_manifest(_normal_profile(), _attack_profile())
    assert manifest["gap_policy"] == "hard_session_boundary"
    assert len(manifest["test_segments"]) == 2
    assert manifest["test_segments"][0]["end_inclusive"] == "2015-12-31 21:00:36"
    assert manifest["test_segments"][1]["start"] == "2015-12-31 21:01:58"


def test_window_cannot_cross_gap(protocol_module):
    timestamps = [datetime(2015, 12, 31, 21, 0, 36), datetime(2015, 12, 31, 21, 1, 58)]
    with pytest.raises(protocol_module.ProtocolAdjudicationError, match="hard boundary"):
        protocol_module.assert_contiguous_window_timestamps(timestamps)


def test_contiguous_candidate_window_is_only_validated_not_generated(protocol_module):
    timestamps = [datetime(2015, 12, 31, 20, 0, 0) + timedelta(seconds=i) for i in range(3)]
    assert protocol_module.assert_contiguous_window_timestamps(timestamps) is None
    assert not hasattr(protocol_module, "build_windows")


def test_normal_v1_keeps_all_rows_without_extra_startup_trim(protocol_module):
    manifest = protocol_module.build_segment_manifest(_normal_profile(), _attack_profile())
    assert manifest["normal_source"]["row_count"] == 495000
    assert manifest["normal_source"]["startup_rows_removed_after_official_v1"] == 0


def test_protocol_does_not_choose_window_threshold_or_training(protocol_module):
    decision = protocol_module.build_protocol_decision()
    assert decision["not_decided_here"] == ["window_length", "zero_variance_training_policy", "threshold"]
    assert decision["formal_benchmark_run"] is False
    assert decision["prohibitions_observed"]["training"] is False
    assert not hasattr(protocol_module, "train")


def test_event_hit_and_delay_use_first_alarm_inside_interval(protocol_module):
    start = datetime(2016, 1, 2, 13, 13, 2)
    end = datetime(2016, 1, 2, 13, 40, 56)
    result = protocol_module.evaluate_event_hit(
        [start - timedelta(seconds=1), start + timedelta(seconds=12), start + timedelta(seconds=3), end + timedelta(seconds=1)],
        start,
        end,
    )
    assert result["event_hit"] is True
    assert result["detection_delay_seconds"] == 3
    assert result["point_predictions_modified"] is False


def test_event_without_alarm_inside_interval_is_not_hit(protocol_module):
    start = datetime(2016, 1, 2, 13, 13, 2)
    end = datetime(2016, 1, 2, 13, 40, 56)
    result = protocol_module.evaluate_event_hit([start - timedelta(seconds=1), end + timedelta(seconds=1)], start, end)
    assert result["event_hit"] is False
    assert result["detection_delay_seconds"] is None


def test_run_adjudication_keeps_raw_attack_file_unchanged(tmp_path, protocol_module):
    root = tmp_path / "Swat_2015"
    root.mkdir()
    attack_list = root / "List_of_attacks_Final.xlsx"
    attack_list.write_bytes(b"immutable-official-attack-list")
    before = protocol_module.sha256_file(attack_list)
    audit = {
        "official_root": str(root.resolve()),
        "physical_files": [_normal_profile(), _attack_profile()],
        "attack_list": {"sha256": before, "attacks": _raw_records()},
    }
    audit_json = tmp_path / "schema_profile.json"
    audit_json.write_text(json.dumps(audit), encoding="utf-8")
    output = tmp_path / "outputs"
    summary = protocol_module.run_adjudication(root, audit_json, output)
    assert protocol_module.sha256_file(attack_list) == before
    assert summary["raw_attack_list_unchanged"] is True
    assert summary["raw_record_count"] == 41
    assert summary["evaluable_event_count"] == 36
    assert {path.name for path in output.iterdir()} == {
        "canonical_attack_events.csv", "point_label_policy.json",
        "segment_manifest.json", "protocol_decision.json",
    }


def test_protocol_outputs_are_git_ignored():
    repo = SCRIPT.parents[3]
    completed = subprocess.run(
        ["git", "check-ignore", "outputs/thesis_t25_swat2015_protocol/protocol_decision.json"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip().startswith("outputs/")
