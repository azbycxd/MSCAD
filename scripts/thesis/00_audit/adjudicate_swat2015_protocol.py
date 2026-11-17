"""Freeze SWaT 2015 label semantics as metadata without changing raw data.

The adjudication consumes the machine-readable T2 audit.  It does not load
model outputs, fit a scaler, choose a window length, generate windows, or train
anything.  Its outputs are protocol metadata under a Git-ignored directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence


EXACT_POINT_LABEL_MAP = {"Normal": 0, "Attack": 1, "A ttack": 1}
NO_PHYSICAL_IMPACT_IDS = {5, 9, 12, 15, 18}
YEAR_CORRECTION_IDS = {37, 38, 39, 40, 41}
PHYSICAL_ONLY_INTERVAL = {
    "start": "2015-12-29 06:30:00",
    "end_inclusive": "2015-12-29 06:42:00",
    "physical_only_label_interval": True,
    "event_id": None,
}
ID41_CANONICAL_START = "2016-01-02 13:13:02"
ID41_CANONICAL_END = "2016-01-02 13:40:56"
EXPECTED_NORMAL_V1_ROWS = 495_000


class ProtocolAdjudicationError(RuntimeError):
    """Raised when adjudication would require an unapproved inference."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_point_label(raw_label: Any) -> int:
    """Apply only the three frozen exact mappings; never strip or guess."""
    if not isinstance(raw_label, str) or raw_label not in EXACT_POINT_LABEL_MAP:
        raise ProtocolAdjudicationError(f"unknown exact Physical label: {raw_label!r}")
    return EXACT_POINT_LABEL_MAP[raw_label]


def build_point_label_policy(observed_counts: dict[str, int]) -> dict[str, Any]:
    unknown = sorted(label for label in observed_counts if label not in EXACT_POINT_LABEL_MAP)
    if unknown:
        raise ProtocolAdjudicationError(f"unadjudicated Physical labels present: {unknown!r}")
    mappings = [
        {
            "raw_label": raw,
            "canonical_label": "Normal" if canonical == 0 else "Attack",
            "canonical_value": canonical,
            "exact_match_required": True,
            "observed_row_count": int(observed_counts.get(raw, 0)),
        }
        for raw, canonical in EXACT_POINT_LABEL_MAP.items()
    ]
    return {
        "primary_point_ground_truth": "SWaT_Dataset_Attack_v0.xlsx::Normal/Attack",
        "mapping_mode": "exact_only_no_strip_no_casefold_no_fallback",
        "mappings": mappings,
        "raw_label_preserved": True,
        "canonical_label_stored_separately": True,
        "physical_only_intervals": [dict(PHYSICAL_ONLY_INTERVAL)],
        "id41_pointwise_rule": {
            "policy": "retain Physical raw point labels",
            "physical_label_end_inclusive": "2016-01-02 13:41:11",
            "event_metadata_end_inclusive": ID41_CANONICAL_END,
            "extra_pointwise_seconds": 15,
        },
        "point_adjustment": False,
    }


def _parse_iso(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def canonicalize_attack_records(raw_records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return new canonical metadata while retaining every raw record verbatim."""
    source_snapshot = copy.deepcopy(list(raw_records))
    ids = [int(record["attack_id"]) for record in source_snapshot]
    if ids != list(range(1, 42)):
        raise ProtocolAdjudicationError(f"expected ordered attack IDs 1..41, got {ids!r}")

    canonical_records: list[dict[str, Any]] = []
    for raw_record in source_snapshot:
        attack_id = int(raw_record["attack_id"])
        raw_start = raw_record.get("start")
        raw_end = raw_record.get("end")
        is_marker = attack_id in NO_PHYSICAL_IMPACT_IDS
        recorded_marker = bool(raw_record.get("no_physical_impact_marker"))
        if is_marker != recorded_marker:
            raise ProtocolAdjudicationError(f"no-impact marker mismatch for ID {attack_id}")

        chronology_correction = attack_id in YEAR_CORRECTION_IDS
        canonical_start = raw_start
        canonical_end = raw_end
        if is_marker:
            if raw_end is not None:
                raise ProtocolAdjudicationError(f"no-impact marker ID {attack_id} unexpectedly has an end time")
            canonical_end = None
        else:
            if raw_start is None or raw_end is None:
                raise ProtocolAdjudicationError(f"physical attack ID {attack_id} lacks a complete interval")
            start_dt = _parse_iso(str(raw_start))
            end_dt = _parse_iso(str(raw_end))
            assert start_dt and end_dt
            if chronology_correction:
                if start_dt.year != 2015 or end_dt.year != 2015 or start_dt.month != 1 or start_dt.day != 2:
                    raise ProtocolAdjudicationError(f"ID {attack_id} does not match the adjudicated raw-year pattern")
                canonical_start = start_dt.replace(year=2016).isoformat(sep=" ")
                canonical_end = end_dt.replace(year=2016).isoformat(sep=" ")

        canonical_records.append({
            "raw_record_index": len(canonical_records) + 1,
            "attack_id": attack_id,
            "event_id": None if is_marker else attack_id,
            "no_physical_impact_marker": is_marker,
            "evaluable_physical_attack_interval": not is_marker,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "canonical_start": canonical_start,
            "canonical_end": canonical_end,
            "chronology_correction": chronology_correction,
            "attack_point": raw_record.get("attack_point"),
            "start_state": raw_record.get("start_state"),
            "description": raw_record.get("description"),
            "actual_change": raw_record.get("actual_change"),
            "expected_impact_or_intent": raw_record.get("expected_impact_or_intent"),
            "unexpected_outcome": raw_record.get("unexpected_outcome"),
            "raw_record_json": json.dumps(raw_record, ensure_ascii=False, sort_keys=True),
        })

    if list(raw_records) != source_snapshot:
        raise ProtocolAdjudicationError("raw attack records were mutated")
    validate_canonical_events(canonical_records)
    return canonical_records


def validate_canonical_events(records: Sequence[dict[str, Any]]) -> None:
    if len(records) != 41:
        raise ProtocolAdjudicationError(f"expected 41 retained raw records, got {len(records)}")
    markers = [record for record in records if record["no_physical_impact_marker"]]
    events = [record for record in records if record["evaluable_physical_attack_interval"]]
    if {record["attack_id"] for record in markers} != NO_PHYSICAL_IMPACT_IDS:
        raise ProtocolAdjudicationError("the five no-impact marker IDs were not preserved")
    if any(record["canonical_end"] is not None for record in markers):
        raise ProtocolAdjudicationError("a no-impact marker received a fabricated end time")
    if len(events) != 36:
        raise ProtocolAdjudicationError(f"expected 36 evaluable physical attack intervals, got {len(events)}")
    if len({record["event_id"] for record in events}) != 36:
        raise ProtocolAdjudicationError("event IDs are not independent")
    by_id = {record["attack_id"]: record for record in records}
    if by_id[10]["event_id"] == by_id[11]["event_id"]:
        raise ProtocolAdjudicationError("IDs 10 and 11 were merged")
    if by_id[27]["event_id"] == by_id[28]["event_id"]:
        raise ProtocolAdjudicationError("IDs 27 and 28 were merged")
    for attack_id in YEAR_CORRECTION_IDS:
        if not str(by_id[attack_id]["raw_start"]).startswith("2015-01-02"):
            raise ProtocolAdjudicationError(f"ID {attack_id} raw year was not retained")
        if not str(by_id[attack_id]["canonical_start"]).startswith("2016-01-02"):
            raise ProtocolAdjudicationError(f"ID {attack_id} canonical year was not corrected")
        if not by_id[attack_id]["chronology_correction"]:
            raise ProtocolAdjudicationError(f"ID {attack_id} lacks correction provenance")
    if by_id[41]["canonical_start"] != ID41_CANONICAL_START or by_id[41]["canonical_end"] != ID41_CANONICAL_END:
        raise ProtocolAdjudicationError("ID 41 canonical event boundary differs from the attack list adjudication")


def build_segment_manifest(normal_profile: dict[str, Any], attack_profile: dict[str, Any]) -> dict[str, Any]:
    if normal_profile["file_name"] != "SWaT_Dataset_Normal_v1.xlsx":
        raise ProtocolAdjudicationError("primary normal source must be Normal v1")
    if int(normal_profile["row_count"]) != EXPECTED_NORMAL_V1_ROWS:
        raise ProtocolAdjudicationError("Normal v1 must retain all 495000 rows")
    if int(normal_profile.get("timestamp_parse_errors", 0)) != 0:
        raise ProtocolAdjudicationError("Normal v1 contains unadjudicated timestamps")

    segment_start = _parse_iso(attack_profile["first_timestamp"])
    final_end = _parse_iso(attack_profile["last_timestamp"])
    assert segment_start and final_end
    test_segments: list[dict[str, Any]] = []
    gaps = attack_profile.get("large_gap_examples", [])
    for gap_index, gap in enumerate(gaps, start=1):
        previous = _parse_iso(gap["previous"])
        current = _parse_iso(gap["current"])
        assert previous and current
        delta_seconds = int((current - previous).total_seconds())
        if delta_seconds <= 1:
            raise ProtocolAdjudicationError("declared gap is not larger than one second")
        test_segments.append({
            "segment_id": f"attack_segment_{gap_index:03d}",
            "source_file": attack_profile["file_name"],
            "start": segment_start.isoformat(sep=" "),
            "end_inclusive": previous.isoformat(sep=" "),
            "row_count": int((previous - segment_start).total_seconds()) + 1,
            "ends_at_hard_session_boundary": True,
        })
        segment_start = current
    test_segments.append({
        "segment_id": f"attack_segment_{len(gaps) + 1:03d}",
        "source_file": attack_profile["file_name"],
        "start": segment_start.isoformat(sep=" "),
        "end_inclusive": final_end.isoformat(sep=" "),
        "row_count": int((final_end - segment_start).total_seconds()) + 1,
        "ends_at_hard_session_boundary": False,
    })
    if sum(segment["row_count"] for segment in test_segments) != int(attack_profile["row_count"]):
        raise ProtocolAdjudicationError("segment row counts do not reproduce the Attack row count")
    return {
        "normal_source": {
            "source_file": normal_profile["file_name"],
            "row_count": int(normal_profile["row_count"]),
            "first_timestamp": normal_profile["first_timestamp"],
            "last_timestamp": normal_profile["last_timestamp"],
            "startup_rows_removed_after_official_v1": 0,
            "policy": "retain complete official Normal v1; startup trim only as future sensitivity",
        },
        "test_source": attack_profile["file_name"],
        "gap_policy": "hard_session_boundary",
        "forbidden_gap_actions": ["interpolate", "forward_fill", "zero_fill", "cross_gap_window"],
        "test_segments": test_segments,
        "window_policy": "a future window must be wholly contained in exactly one segment",
        "window_length": None,
    }


def assert_contiguous_window_timestamps(timestamps: Sequence[datetime]) -> None:
    """Validate a proposed window without selecting or constructing windows."""
    for previous, current in zip(timestamps, timestamps[1:]):
        if current - previous != timedelta(seconds=1):
            raise ProtocolAdjudicationError(
                f"window crosses a hard boundary or non-1s interval: {previous!s} -> {current!s}"
            )


def evaluate_event_hit(
    legal_alarm_timestamps: Sequence[datetime], event_start: datetime, event_end: datetime
) -> dict[str, Any]:
    """Apply the frozen event hit/delay definition without point adjustment."""
    alarms_inside = sorted(
        timestamp for timestamp in legal_alarm_timestamps if event_start <= timestamp <= event_end
    )
    first_alarm = alarms_inside[0] if alarms_inside else None
    return {
        "event_hit": first_alarm is not None,
        "first_alarm_timestamp": first_alarm.isoformat(sep=" ") if first_alarm else None,
        "detection_delay_seconds": int((first_alarm - event_start).total_seconds()) if first_alarm else None,
        "point_predictions_modified": False,
    }


def build_protocol_decision() -> dict[str, Any]:
    return {
        "phase": "T2.5 SWaT 2015 label and benchmark protocol adjudication",
        "scientific_status": "VALID_DIAGNOSTIC",
        "historical_t2_readiness_preserved": {
            "source_readiness": "E. label_alignment_conflict",
            "protocol_readiness": "D. needs_official_semantics",
        },
        "pointwise_ground_truth": "Physical Normal/Attack with exact three-value mapping",
        "event_ground_truth": "36 canonical attack-list intervals; 41 raw records retained",
        "event_identity": "attack-list ID; IDs 10/11 and 27/28 remain distinct",
        "event_hit": "at least one legal alarm timestamp inside the canonical event interval",
        "detection_delay": "first legal alarm timestamp minus canonical event start",
        "point_adjustment": False,
        "gap_policy": "82-second gap is a hard_session_boundary",
        "primary_normal_source": "complete SWaT_Dataset_Normal_v1.xlsx (495000 rows)",
        "primary_input_dimension": 51,
        "test_driven_channel_removal": False,
        "formal_benchmark_run": False,
        "next_gate": "T3 protocol freeze",
        "not_decided_here": ["window_length", "zero_variance_training_policy", "threshold"],
        "prohibitions_observed": {
            "training": False,
            "scaler_fit": False,
            "raw_data_modified": False,
            "model_results_read": False,
            "f1_optimized": False,
            "v3_implemented": False,
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ProtocolAdjudicationError("refusing to write an empty canonical event file")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_adjudication(official_root: Path, audit_json: Path, output_dir: Path) -> dict[str, Any]:
    official_root = official_root.resolve()
    audit_json = audit_json.resolve()
    output_dir = output_dir.resolve()
    if not official_root.is_dir() or not audit_json.is_file():
        raise ProtocolAdjudicationError("official root and T2 audit JSON must exist explicitly")
    if official_root == output_dir or official_root in output_dir.parents:
        raise ProtocolAdjudicationError("canonical metadata output must not be inside the raw source root")

    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    if Path(audit["official_root"]).resolve() != official_root:
        raise ProtocolAdjudicationError("T2 audit root does not match the explicit official root")
    physical_profiles = audit["physical_files"]
    attack_profile = next(item for item in physical_profiles if item["file_name"] == "SWaT_Dataset_Attack_v0.xlsx")
    normal_profile = next(item for item in physical_profiles if item["file_name"] == "SWaT_Dataset_Normal_v1.xlsx")
    raw_attack_path = official_root / "List_of_attacks_Final.xlsx"
    before_hash = sha256_file(raw_attack_path)
    if before_hash != audit["attack_list"]["sha256"]:
        raise ProtocolAdjudicationError("raw attack list hash differs from the T2 audit")

    point_policy = build_point_label_policy(attack_profile["raw_label_counts"])
    canonical_records = canonicalize_attack_records(audit["attack_list"]["attacks"])
    segment_manifest = build_segment_manifest(normal_profile, attack_profile)
    protocol_decision = build_protocol_decision()
    protocol_decision["source_integrity"] = {
        "official_root": str(official_root),
        "t2_audit_json": str(audit_json),
        "raw_attack_list_sha256": before_hash,
    }

    write_csv(output_dir / "canonical_attack_events.csv", canonical_records)
    write_json(output_dir / "point_label_policy.json", point_policy)
    write_json(output_dir / "segment_manifest.json", segment_manifest)
    write_json(output_dir / "protocol_decision.json", protocol_decision)
    after_hash = sha256_file(raw_attack_path)
    if after_hash != before_hash:
        raise ProtocolAdjudicationError("raw attack list changed during adjudication")
    return {
        "raw_record_count": len(canonical_records),
        "evaluable_event_count": sum(record["evaluable_physical_attack_interval"] for record in canonical_records),
        "test_segment_count": len(segment_manifest["test_segments"]),
        "output_dir": str(output_dir),
        "raw_attack_list_unchanged": True,
        "training": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_adjudication(Path(args.official_root), Path(args.audit_json), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
