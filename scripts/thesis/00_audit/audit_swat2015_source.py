"""Read-only source audit for the official SWaT 2015 Physical release.

This module deliberately does not expose a training loader or window builder.  It
requires an explicit source root and writes only diagnostic artifacts beneath an
explicit output directory.  Raw workbooks are opened with openpyxl's read-only
mode and are never saved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


EXPECTED_ATTACK_XLSX = "List_of_attacks_Final.xlsx"
EXPECTED_ATTACK_PDF = "List_of_attacks_Final.pdf"
EXPECTED_PAPER_PDF = "A Dataset to Support Research in the Design of Secure Water Treatment Systems.pdf"
EXPECTED_PHYSICAL = (
    "SWaT_Dataset_Attack_v0.xlsx",
    "SWaT_Dataset_Normal_v0.xlsx",
    "SWaT_Dataset_Normal_v1.xlsx",
)
LEGACY_BASENAMES = {"normal.csv", "merged.csv"}

# This is a transcription of Table 2's Type column in the dataset paper.  The
# paper itself calls LIT401 an actuator although its name and description say
# "Level Transmitter".  Both the literal-paper and semantic counts are emitted;
# the audit never silently repairs that official metadata inconsistency.
PAPER_SENSOR_NAMES = {
    "FIT101", "LIT101", "AIT201", "AIT202", "AIT203", "FIT201",
    "DPIT301", "FIT301", "LIT301", "AIT401", "AIT402", "FIT401",
    "AIT501", "AIT502", "AIT503", "AIT504", "FIT501", "FIT502",
    "FIT503", "FIT504", "PIT501", "PIT502", "PIT503", "FIT601",
}
PAPER_ACTUATOR_NAMES = {
    "MV101", "P101", "P102", "MV201", "P201", "P202", "P203",
    "P204", "P205", "P206", "MV301", "MV302", "MV303", "MV304",
    "P301", "P302", "LIT401", "P401", "P402", "P403", "P404",
    "UV401", "P501", "P502", "P601", "P602", "P603",
}
SEMANTIC_SENSOR_NAMES = PAPER_SENSOR_NAMES | {"LIT401"}
SEMANTIC_ACTUATOR_NAMES = PAPER_ACTUATOR_NAMES - {"LIT401"}

TIMESTAMP_FORMATS = (
    "%d/%m/%Y %I:%M:%S %p",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)


class AuditError(RuntimeError):
    """Raised when the source cannot be audited without guessing."""


@dataclass(frozen=True)
class SourceFingerprint:
    size: int
    mtime_ns: int
    sha256: str


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> SourceFingerprint:
    stat = path.stat()
    return SourceFingerprint(stat.st_size, stat.st_mtime_ns, sha256_file(path))


def resolve_official_root(value: str | os.PathLike[str] | None) -> Path:
    """Resolve an explicit official root; never search or fall back to Legacy."""
    if value is None or not str(value).strip():
        raise AuditError("SWaT 2015 official root must be provided explicitly")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise AuditError(f"official root is not a directory: {root}")
    if root.name.casefold() == "swat":
        raise AuditError("refusing ambiguous Legacy directory named 'Swat'")
    legacy_hits = [p for p in root.rglob("*") if p.is_file() and p.name.casefold() in LEGACY_BASENAMES]
    if legacy_hits:
        joined = ", ".join(str(p) for p in legacy_hits[:5])
        raise AuditError(f"Legacy normal.csv/merged.csv found inside official root: {joined}")
    required = [root / "Physical", root / EXPECTED_ATTACK_XLSX, root / EXPECTED_ATTACK_PDF, root / EXPECTED_PAPER_PDF]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise AuditError("official package identity files missing: " + ", ".join(missing))
    return root


def normalize_name(value: Any) -> str:
    return "" if value is None else str(value).strip().replace("-", "").replace(" ", "")


def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, time())
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"blank or non-text timestamp: {value!r}")
    raw = " ".join(value.strip().split())
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    raise ValueError(f"unparseable timestamp: {value!r}")


def identify_schema(headers: Sequence[Any]) -> dict[str, Any]:
    stripped = ["" if h is None else str(h).strip() for h in headers]
    normalized = [normalize_name(h).casefold() for h in headers]
    timestamp_candidates = [i for i, name in enumerate(normalized) if name == "timestamp"]
    label_candidates = [i for i, name in enumerate(normalized) if name in {"normal/attack", "label", "attack"}]
    if len(timestamp_candidates) != 1:
        raise AuditError(f"expected one timestamp column, got {timestamp_candidates}")
    if len(label_candidates) != 1:
        raise AuditError(f"expected one label column, got {label_candidates}")
    timestamp_index = timestamp_candidates[0]
    label_index = label_candidates[0]
    variable_indices = [i for i in range(len(headers)) if i not in {timestamp_index, label_index}]
    variable_names = [normalize_name(headers[i]) for i in variable_indices]
    paper_sensors = [name for name in variable_names if name in PAPER_SENSOR_NAMES]
    paper_actuators = [name for name in variable_names if name in PAPER_ACTUATOR_NAMES]
    unclassified = [name for name in variable_names if name not in PAPER_SENSOR_NAMES | PAPER_ACTUATOR_NAMES]
    return {
        "column_count": len(headers),
        "column_names_raw": stripped,
        "column_names_normalized": [normalize_name(h) for h in headers],
        "timestamp_column": stripped[timestamp_index],
        "timestamp_index": timestamp_index,
        "label_column": stripped[label_index],
        "label_index": label_index,
        "variable_indices": variable_indices,
        "variable_names": variable_names,
        "physical_variable_count": len(variable_names),
        "paper_literal_sensor_names": paper_sensors,
        "paper_literal_actuator_names": paper_actuators,
        "paper_literal_sensor_count": len(paper_sensors),
        "paper_literal_actuator_count": len(paper_actuators),
        "semantic_sensor_names": [name for name in variable_names if name in SEMANTIC_SENSOR_NAMES],
        "semantic_actuator_names": [name for name in variable_names if name in SEMANTIC_ACTUATOR_NAMES],
        "semantic_sensor_count": sum(name in SEMANTIC_SENSOR_NAMES for name in variable_names),
        "semantic_actuator_count": sum(name in SEMANTIC_ACTUATOR_NAMES for name in variable_names),
        "unclassified_names": unclassified,
        "official_type_conflicts": [
            {
                "variable": "LIT401",
                "paper_type": "Actuator",
                "paper_description": "Level Transmitter; RO feed water tank level",
                "audit_resolution": "not silently resolved",
            }
        ] if "LIT401" in variable_names else [],
    }


def _openpyxl() -> Any:
    try:
        import openpyxl  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific message
        raise AuditError("read-only XLSX audit requires openpyxl") from exc
    return openpyxl


def iter_workbook_rows(path: Path) -> tuple[list[Any], Iterator[tuple[Any, ...]], Any, str]:
    """Return headers and a streaming iterator; caller must close workbook."""
    openpyxl = _openpyxl()
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if len(workbook.worksheets) != 1:
        workbook.close()
        raise AuditError(f"expected exactly one worksheet in {path.name}")
    sheet = workbook.active
    row_iter = sheet.iter_rows(values_only=True)
    header = None
    for row_number, row in enumerate(row_iter, start=1):
        normalized = {normalize_name(v).casefold() for v in row if v is not None}
        if "timestamp" in normalized and "normal/attack" in normalized:
            header = list(row)
            break
        if row_number >= 10:
            break
    if header is None:
        workbook.close()
        raise AuditError(f"header row not found in first 10 rows: {path.name}")
    return header, row_iter, workbook, sheet.title


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _jsonable_time(value: datetime | None) -> str | None:
    return value.isoformat(sep=" ") if value else None


def audit_physical_workbook(path: Path) -> dict[str, Any]:
    before = fingerprint(path)
    headers, rows, workbook, worksheet_title = iter_workbook_rows(path)
    schema = identify_schema(headers)
    ts_idx = schema["timestamp_index"]
    label_idx = schema["label_index"]
    variable_indices: list[int] = schema["variable_indices"]
    variable_names: list[str] = schema["variable_names"]

    numeric_count = [0] * len(variable_indices)
    missing_count = [0] * len(variable_indices)
    inf_count = [0] * len(variable_indices)
    malformed_count = [0] * len(variable_indices)
    means = [0.0] * len(variable_indices)
    m2s = [0.0] * len(variable_indices)
    minima = [math.inf] * len(variable_indices)
    maxima = [-math.inf] * len(variable_indices)
    changes = [0] * len(variable_indices)
    previous_values: list[float | None] = [None] * len(variable_indices)
    small_unique_values: list[set[float] | None] = [set() for _ in variable_indices]
    invalid_examples: list[dict[str, Any]] = []

    raw_label_counts: Counter[str] = Counter()
    normalized_label_counts: Counter[str] = Counter()
    transitions: list[dict[str, Any]] = []
    label_intervals: list[dict[str, Any]] = []
    active_attack_start: datetime | None = None
    last_attack_timestamp: datetime | None = None
    previous_label: str | None = None
    label_segments: list[dict[str, Any]] = []
    active_segment_start: datetime | None = None

    row_count = 0
    first_data_row: list[str | None] | None = None
    last_data_row: list[str | None] | None = None
    timestamp_parse_errors = 0
    timestamp_error_examples: list[dict[str, Any]] = []
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    previous_timestamp: datetime | None = None
    delta_counts: Counter[int] = Counter()
    duplicate_count = 0
    nonmonotonic_count = 0
    gaps: list[dict[str, Any]] = []

    try:
        for excel_row, row in enumerate(rows, start=3):
            if all(_blank(v) for v in row):
                continue
            row_count += 1
            rendered_row = [None if value is None else str(value).strip() for value in row]
            if first_data_row is None:
                first_data_row = rendered_row
            last_data_row = rendered_row
            prior_timestamp = previous_timestamp
            try:
                current_timestamp = parse_timestamp(row[ts_idx] if ts_idx < len(row) else None)
            except ValueError as exc:
                timestamp_parse_errors += 1
                if len(timestamp_error_examples) < 20:
                    timestamp_error_examples.append({"row": excel_row, "raw": repr(row[ts_idx]), "error": str(exc)})
                current_timestamp = None

            if current_timestamp is not None:
                if first_timestamp is None:
                    first_timestamp = current_timestamp
                last_timestamp = current_timestamp
                if previous_timestamp is not None:
                    delta = int((current_timestamp - previous_timestamp).total_seconds())
                    delta_counts[delta] += 1
                    if delta == 0:
                        duplicate_count += 1
                    elif delta < 0:
                        nonmonotonic_count += 1
                    elif delta > 1 and len(gaps) < 100:
                        gaps.append({
                            "previous": _jsonable_time(previous_timestamp),
                            "current": _jsonable_time(current_timestamp),
                            "delta_seconds": delta,
                        })
                previous_timestamp = current_timestamp

            raw_label = row[label_idx] if label_idx < len(row) else None
            raw_label_text = "<MISSING>" if raw_label is None else str(raw_label)
            normalized_label = raw_label_text.strip()
            raw_label_counts[raw_label_text] += 1
            normalized_label_counts[normalized_label] += 1
            if previous_label is not None and normalized_label != previous_label:
                label_segments.append({
                    "label": previous_label,
                    "start": _jsonable_time(active_segment_start),
                    "end_inclusive": _jsonable_time(prior_timestamp),
                })
                transitions.append({
                    "timestamp": _jsonable_time(current_timestamp),
                    "from": previous_label,
                    "to": normalized_label,
                    "row": excel_row,
                })
                active_segment_start = current_timestamp
            elif previous_label is None:
                active_segment_start = current_timestamp
            if normalized_label.casefold() == "attack":
                if previous_label is None or previous_label.casefold() != "attack":
                    active_attack_start = current_timestamp
                last_attack_timestamp = current_timestamp
            elif previous_label is not None and previous_label.casefold() == "attack" and active_attack_start is not None:
                label_intervals.append({
                    "start": _jsonable_time(active_attack_start),
                    "end_inclusive": _jsonable_time(last_attack_timestamp),
                    "next_normal": _jsonable_time(current_timestamp),
                })
                active_attack_start = None
                last_attack_timestamp = None
            previous_label = normalized_label

            for pos, col_idx in enumerate(variable_indices):
                value = row[col_idx] if col_idx < len(row) else None
                if _blank(value):
                    missing_count[pos] += 1
                    if len(invalid_examples) < 50:
                        invalid_examples.append({"row": excel_row, "column": variable_names[pos], "raw": repr(value), "kind": "missing"})
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    malformed_count[pos] += 1
                    if len(invalid_examples) < 50:
                        invalid_examples.append({"row": excel_row, "column": variable_names[pos], "raw": repr(value), "kind": "nonnumeric"})
                    continue
                numeric = float(value)
                if not math.isfinite(numeric):
                    inf_count[pos] += 1
                    if len(invalid_examples) < 50:
                        invalid_examples.append({"row": excel_row, "column": variable_names[pos], "raw": repr(value), "kind": "nonfinite"})
                    continue
                numeric_count[pos] += 1
                unique_values = small_unique_values[pos]
                if unique_values is not None:
                    unique_values.add(numeric)
                    if len(unique_values) > 20:
                        small_unique_values[pos] = None
                delta_from_mean = numeric - means[pos]
                means[pos] += delta_from_mean / numeric_count[pos]
                m2s[pos] += delta_from_mean * (numeric - means[pos])
                minima[pos] = min(minima[pos], numeric)
                maxima[pos] = max(maxima[pos], numeric)
                if previous_values[pos] is not None and numeric != previous_values[pos]:
                    changes[pos] += 1
                previous_values[pos] = numeric
    finally:
        workbook.close()

    if active_attack_start is not None:
        label_intervals.append({
            "start": _jsonable_time(active_attack_start),
            "end_inclusive": _jsonable_time(last_attack_timestamp),
            "next_normal": None,
        })
    if previous_label is not None:
        label_segments.append({
            "label": previous_label,
            "start": _jsonable_time(active_segment_start),
            "end_inclusive": _jsonable_time(last_timestamp),
        })

    after = fingerprint(path)
    if before != after:
        raise AuditError(f"source changed while being audited: {path}")

    profiles = []
    for pos, name in enumerate(variable_names):
        count = numeric_count[pos]
        variance = m2s[pos] / (count - 1) if count > 1 else 0.0
        comparisons = max(0, count - 1)
        change_rate = changes[pos] / comparisons if comparisons else 0.0
        is_constant = count > 0 and minima[pos] == maxima[pos]
        is_near_constant = count > 1 and not is_constant and change_rate < 0.001
        unique_values = small_unique_values[pos]
        if is_constant:
            observed_kind = "constant_numeric"
        elif unique_values is not None and all(value.is_integer() for value in unique_values):
            observed_kind = "discrete_numeric"
        else:
            observed_kind = "continuous_numeric"
        profiles.append({
            "name": name,
            "numeric_count": count,
            "missing_count": missing_count[pos],
            "nonfinite_count": inf_count[pos],
            "malformed_count": malformed_count[pos],
            "min": None if count == 0 else minima[pos],
            "max": None if count == 0 else maxima[pos],
            "mean": None if count == 0 else means[pos],
            "std": math.sqrt(max(0.0, variance)) if count else None,
            "change_count": changes[pos],
            "change_rate": change_rate,
            "constant": is_constant,
            "near_constant_definition": "nonconstant and exact-value change_rate < 0.001",
            "near_constant": is_near_constant,
            "observed_value_kind": observed_kind,
            "unique_values": sorted(unique_values) if unique_values is not None else None,
            "unique_count": len(unique_values) if unique_values is not None else ">20",
            "paper_literal_type": "sensor" if name in PAPER_SENSOR_NAMES else "actuator" if name in PAPER_ACTUATOR_NAMES else "unclassified",
            "semantic_type": "sensor" if name in SEMANTIC_SENSOR_NAMES else "actuator" if name in SEMANTIC_ACTUATOR_NAMES else "unclassified",
        })

    return {
        "path": str(path),
        "file_name": path.name,
        "size_bytes": before.size,
        "sha256": before.sha256,
        "worksheet": worksheet_title,
        "row_count": row_count,
        "first_data_row": first_data_row,
        "last_data_row": last_data_row,
        "schema": schema,
        "first_timestamp": _jsonable_time(first_timestamp),
        "last_timestamp": _jsonable_time(last_timestamp),
        "timestamp_parse_errors": timestamp_parse_errors,
        "timestamp_error_examples": timestamp_error_examples,
        "monotonic_increasing": nonmonotonic_count == 0 and duplicate_count == 0,
        "duplicate_timestamp_count": duplicate_count,
        "nonmonotonic_timestamp_count": nonmonotonic_count,
        "sampling_interval_counts_seconds": {str(k): v for k, v in sorted(delta_counts.items())},
        "large_gap_count": sum(count for seconds, count in delta_counts.items() if seconds > 1),
        "large_gap_examples": gaps,
        "timezone_information": "not present in timestamp cells; local-clock timezone unspecified",
        "raw_label_counts": dict(raw_label_counts),
        "normalized_label_counts": dict(normalized_label_counts),
        "label_transitions": transitions,
        "label_segments": label_segments,
        "label_intervals": label_intervals,
        "variable_profiles": profiles,
        "invalid_examples": invalid_examples,
        "read_only_integrity": "size, mtime_ns and SHA-256 unchanged across audit",
    }


def _excel_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, time):
        return value.isoformat()
    return str(value)


def _combine_end(start: datetime, end_value: Any) -> datetime | None:
    if end_value is None:
        return None
    if isinstance(end_value, datetime):
        return end_value.replace(tzinfo=None)
    if isinstance(end_value, time):
        combined = datetime.combine(start.date(), end_value)
        if combined < start:
            combined += timedelta(days=1)
        return combined
    if isinstance(end_value, str):
        raw = " ".join(end_value.strip().split())
        for fmt in ("%I:%M:%S %p", "%H:%M:%S", "%I:%M %p", "%H:%M"):
            try:
                parsed = datetime.strptime(raw, fmt).time()
                return _combine_end(start, parsed)
            except ValueError:
                pass
    raise AuditError(f"cannot parse attack end time: {end_value!r}")


def read_attack_list(path: Path) -> dict[str, Any]:
    before = fingerprint(path)
    openpyxl = _openpyxl()
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    attacks: list[dict[str, Any]] = []
    try:
        rows = sheet.iter_rows(values_only=True)
        headers = [str(v).strip() if v is not None else "" for v in next(rows)]
        for excel_row, row in enumerate(rows, start=2):
            attack_id = row[0] if row else None
            if not isinstance(attack_id, (int, float)):
                continue
            start_value = row[1]
            if not isinstance(start_value, datetime):
                raise AuditError(f"attack start is not a datetime at row {excel_row}: {start_value!r}")
            start = start_value.replace(tzinfo=None)
            end = _combine_end(start, row[2])
            attack_point = _excel_text(row[3])
            no_physical = end is None or (attack_point is not None and "no physical impact" in attack_point.casefold())
            attacks.append({
                "attack_id": int(attack_id),
                "source_row": excel_row,
                "start_raw": _excel_text(row[1]),
                "end_raw": _excel_text(row[2]),
                "start": _jsonable_time(start),
                "end": _jsonable_time(end),
                "attack_point": attack_point,
                "start_state": _excel_text(row[4]),
                "description": _excel_text(row[5]),
                "actual_change": _excel_text(row[6]),
                "expected_impact_or_intent": _excel_text(row[7]),
                "unexpected_outcome": _excel_text(row[8]),
                "no_physical_impact_marker": no_physical,
            })
    finally:
        workbook.close()
    after = fingerprint(path)
    if before != after:
        raise AuditError(f"attack list changed while being audited: {path}")

    intervals = [item for item in attacks if not item["no_physical_impact_marker"]]
    overlaps = []
    boundary_touches = []
    chronology_reversals = []
    previous: dict[str, Any] | None = None
    for item in intervals:
        if previous is not None:
            previous_start = datetime.fromisoformat(previous["start"])
            previous_end = datetime.fromisoformat(previous["end"])
            current_start = datetime.fromisoformat(item["start"])
            if current_start < previous_start:
                chronology_reversals.append({"previous_id": previous["attack_id"], "current_id": item["attack_id"]})
            elif current_start < previous_end:
                overlaps.append({"previous_id": previous["attack_id"], "current_id": item["attack_id"]})
            elif current_start == previous_end:
                boundary_touches.append({"previous_id": previous["attack_id"], "current_id": item["attack_id"]})
        previous = item
    return {
        "path": str(path),
        "size_bytes": before.size,
        "sha256": before.sha256,
        "headers": headers,
        "listed_item_count": len(attacks),
        "physical_interval_count": len(intervals),
        "no_physical_impact_marker_count": sum(item["no_physical_impact_marker"] for item in attacks),
        "no_physical_impact_ids": [item["attack_id"] for item in attacks if item["no_physical_impact_marker"]],
        "raw_interval_overlap_count_in_sheet_order": len(overlaps),
        "raw_interval_overlaps_in_sheet_order": overlaps,
        "raw_sheet_order_chronology_reversal_count": len(chronology_reversals),
        "raw_sheet_order_chronology_reversals": chronology_reversals,
        "raw_boundary_touch_count_in_sheet_order": len(boundary_touches),
        "raw_boundary_touches_in_sheet_order": boundary_touches,
        "attacks": attacks,
        "read_only_integrity": "size, mtime_ns and SHA-256 unchanged across audit",
    }


def _parse_iso(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def align_attack_intervals(attack_list: dict[str, Any], physical: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare official intervals to exact, unmodified Physical label segments.

    Matching is time-based rather than sequence-based because two pairs of
    back-to-back official attacks form one continuous ``Attack`` segment in the
    workbook.  The 2015->2016 year substitution for IDs 37-41 is emitted only as
    a diagnostic comparison and never changes the official metadata.
    """
    official = [item for item in attack_list["attacks"] if not item["no_physical_impact_marker"]]
    segments = [
        {**item, "start_dt": _parse_iso(item["start"]), "end_dt": _parse_iso(item["end_inclusive"])}
        for item in physical["label_segments"]
    ]
    attack_segments = [item for item in segments if item["label"].casefold() == "attack"]
    matched_segment_indices: set[int] = set()
    rows: list[dict[str, Any]] = []
    physical_first = _parse_iso(physical["first_timestamp"])
    physical_last = _parse_iso(physical["last_timestamp"])
    assert physical_first and physical_last

    effective_intervals: list[tuple[datetime, datetime, bool]] = []
    for item in official:
        raw_start = _parse_iso(item["start"])
        raw_end = _parse_iso(item["end"])
        assert raw_start and raw_end
        year_conflict = raw_end < physical_first or raw_start > physical_last
        if year_conflict:
            effective_intervals.append((raw_start.replace(year=physical_last.year), raw_end.replace(year=physical_last.year), True))
        else:
            effective_intervals.append((raw_start, raw_end, False))

    def segment_at(moment: datetime) -> dict[str, Any] | None:
        return next((item for item in segments if item["start_dt"] <= moment <= item["end_dt"]), None)

    for index, source in enumerate(official, start=1):
        raw_start = _parse_iso(source["start"])
        raw_end = _parse_iso(source["end"])
        assert raw_start and raw_end
        effective_start, effective_end, calendar_year_conflict = effective_intervals[index - 1]

        overlapping: list[tuple[int, dict[str, Any]]] = []
        for seg_index, segment in enumerate(attack_segments):
            if segment["start_dt"] <= effective_end and segment["end_dt"] >= effective_start:
                overlapping.append((seg_index, segment))
                matched_segment_indices.add(seg_index)
        start_segment = segment_at(effective_start)
        end_segment = segment_at(effective_end)
        containing = next(
            (segment for _, segment in overlapping if segment["start_dt"] <= effective_start and segment["end_dt"] >= effective_end),
            None,
        )
        start_label = start_segment["label"] if start_segment else None
        end_label = end_segment["label"] if end_segment else None
        start_delta = int((containing["start_dt"] - effective_start).total_seconds()) if containing else None
        end_delta = int((containing["end_dt"] - effective_end).total_seconds()) if containing else None

        merges_previous = False
        merges_next = False
        if containing is not None and (start_delta != 0 or end_delta != 0):
            previous_end = effective_intervals[index - 2][1] if index > 1 else None
            next_start = effective_intervals[index][0] if index < len(effective_intervals) else None
            merges_previous = start_delta is not None and start_delta < 0 and previous_end is not None and previous_end >= effective_start - timedelta(seconds=1)
            merges_next = end_delta is not None and end_delta > 0 and next_start is not None and next_start <= effective_end + timedelta(seconds=1)
        if start_label != "Attack" or end_label != "Attack" or containing is None:
            alignment = "substantive_label_mismatch"
        elif start_delta == 0 and end_delta == 0:
            alignment = "exact"
        elif merges_previous or merges_next:
            alignment = "continuous_attack_label_merges_adjacent_list_intervals"
        elif abs(start_delta or 0) <= 1 and abs(end_delta or 0) <= 1:
            alignment = "boundary_only"
        else:
            alignment = "substantive_boundary_mismatch"
        classification = "official_metadata_calendar_year_conflict" if calendar_year_conflict else alignment
        rows.append({
            "sequence": index,
            "attack_id": source["attack_id"],
            "attack_point": source["attack_point"],
            "official_start_raw": source["start"],
            "official_end_raw": source["end"],
            "diagnostic_effective_start": _jsonable_time(effective_start),
            "diagnostic_effective_end": _jsonable_time(effective_end),
            "calendar_year_conflict": calendar_year_conflict,
            "label_at_official_start": start_label,
            "label_at_official_end": end_label,
            "observed_attack_segment_start": containing["start"] if containing else None,
            "observed_attack_segment_end_inclusive": containing["end_inclusive"] if containing else None,
            "observed_segment_start_delta_seconds": start_delta,
            "observed_segment_end_delta_seconds": end_delta,
            "alignment_after_explicit_year_diagnostic": alignment,
            "classification": classification,
            "boundary_convention": "not_frozen",
        })

    for seg_index, segment in enumerate(attack_segments):
        if seg_index not in matched_segment_indices:
            rows.append({
                "sequence": None,
                "attack_id": None,
                "attack_point": None,
                "official_start_raw": None,
                "official_end_raw": None,
                "diagnostic_effective_start": None,
                "diagnostic_effective_end": None,
                "calendar_year_conflict": False,
                "label_at_official_start": None,
                "label_at_official_end": None,
                "observed_attack_segment_start": segment["start"],
                "observed_attack_segment_end_inclusive": segment["end_inclusive"],
                "observed_segment_start_delta_seconds": None,
                "observed_segment_end_delta_seconds": None,
                "alignment_after_explicit_year_diagnostic": "unmatched_physical_label_interval",
                "classification": "unmatched_physical_label_interval",
                "boundary_convention": "not_frozen",
            })
    return rows


def inventory_source(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: str(p.relative_to(root)).casefold()):
        relative = path.relative_to(root)
        in_network = bool(relative.parts and relative.parts[0].casefold() == "network")
        stat = path.stat()
        digest = None if in_network else sha256_file(path)
        rows.append({
            "relative_path": str(relative),
            "size_bytes": stat.st_size,
            "suffix": path.suffix.casefold(),
            "sha256": digest,
            "inspection": "metadata_only_not_parsed" if in_network else "identity_and_format_audit",
        })
    return rows


def compare_legacy_candidates(official_profiles: Sequence[dict[str, Any]], candidates: Sequence[Path]) -> list[dict[str, Any]]:
    official_by_hash = {item["sha256"]: item["file_name"] for item in official_profiles}
    rows = []
    for path in candidates:
        if not path.exists() or not path.is_file():
            rows.append({"path": str(path), "exists": False, "same_hash_as_official": False})
            continue
        digest = sha256_file(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            legacy_header = next(reader, [])
            legacy_first_row = next(reader, [])
        normalized_header = [normalize_name(value) for value in legacy_header]
        header_matches = [
            item for item in official_profiles
            if item["schema"]["column_names_normalized"] == normalized_header
        ]
        first_row_match_ratio = None
        matching_first_row_file = None
        for item in header_matches:
            official_first = item.get("first_data_row") or []
            if len(official_first) != len(legacy_first_row):
                continue
            matches = 0
            for index, (left, right) in enumerate(zip(official_first, legacy_first_row)):
                if index == item["schema"]["timestamp_index"]:
                    try:
                        equal = parse_timestamp(left) == parse_timestamp(right)
                    except ValueError:
                        equal = str(left).strip() == str(right).strip()
                elif index == item["schema"]["label_index"]:
                    equal = str(left).strip() == str(right).strip()
                else:
                    try:
                        equal = float(left) == float(right)
                    except (TypeError, ValueError):
                        equal = str(left).strip() == str(right).strip()
                matches += int(equal)
            ratio = matches / len(official_first) if official_first else 0.0
            if first_row_match_ratio is None or ratio > first_row_match_ratio:
                first_row_match_ratio = ratio
                matching_first_row_file = item["file_name"]
        rows.append({
            "path": str(path),
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256": digest,
            "same_hash_as_official": digest in official_by_hash,
            "matching_official_file": official_by_hash.get(digest),
            "normalized_header_matches_official": bool(header_matches),
            "first_row_value_match_ratio": first_row_match_ratio,
            "matching_official_first_row_file": matching_first_row_file,
            "comparison_scope": "hash/size plus header and first-row probe; no Legacy data loaded into protocol",
        })
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def timeline_row(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "file_name": profile["file_name"],
        "row_count": profile["row_count"],
        "first_timestamp": profile["first_timestamp"],
        "last_timestamp": profile["last_timestamp"],
        "timestamp_parse_errors": profile["timestamp_parse_errors"],
        "monotonic_increasing": profile["monotonic_increasing"],
        "duplicate_timestamp_count": profile["duplicate_timestamp_count"],
        "nonmonotonic_timestamp_count": profile["nonmonotonic_timestamp_count"],
        "sampling_interval_counts_seconds": json.dumps(profile["sampling_interval_counts_seconds"], sort_keys=True),
        "large_gap_count": profile["large_gap_count"],
        "large_gap_examples": json.dumps(profile["large_gap_examples"], ensure_ascii=False),
        "timezone_information": profile["timezone_information"],
    }


def run_audit(root: Path, output_dir: Path, legacy_candidates: Sequence[Path]) -> dict[str, Any]:
    physical_dir = root / "Physical"
    physical_files = sorted(physical_dir.glob("*.xlsx"), key=lambda p: p.name)
    missing_expected = [name for name in EXPECTED_PHYSICAL if not (physical_dir / name).is_file()]
    if missing_expected:
        raise AuditError("expected Physical workbooks missing: " + ", ".join(missing_expected))
    profiles = [audit_physical_workbook(path) for path in physical_files]
    attack_profile = next(item for item in profiles if "Attack" in item["file_name"])
    attack_list = read_attack_list(root / EXPECTED_ATTACK_XLSX)
    alignment = align_attack_intervals(attack_list, attack_profile)
    inventory = inventory_source(root)
    legacy_comparison = compare_legacy_candidates(profiles, legacy_candidates)

    payload = {
        "audit_scope": "read-only source/schema/timeline/label audit; no training, windows, scaling, or label rewriting",
        "official_root": str(root),
        "official_package_identity": {
            "physical_directory_present": physical_dir.is_dir(),
            "network_directory_present": (root / "Network").is_dir(),
            "expected_attack_xlsx_present": (root / EXPECTED_ATTACK_XLSX).is_file(),
            "expected_attack_pdf_present": (root / EXPECTED_ATTACK_PDF).is_file(),
            "expected_paper_pdf_present": (root / EXPECTED_PAPER_PDF).is_file(),
            "legacy_basename_hits": [],
        },
        "inventory": inventory,
        "physical_files": profiles,
        "attack_list": attack_list,
        "label_alignment_summary": dict(Counter(row["classification"] for row in alignment)),
        "legacy_comparison": legacy_comparison,
        "prohibitions_observed": {
            "training_run": False,
            "windows_created": False,
            "scaler_fit": False,
            "labels_rewritten": False,
            "network_pcap_parsed": False,
        },
    }
    write_csv(output_dir / "timeline_audit.csv", [timeline_row(item) for item in profiles])
    write_json(output_dir / "schema_profile.json", payload)
    write_csv(output_dir / "label_alignment_report.csv", alignment)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", required=True, help="Explicit SWaT 2015 official package root")
    parser.add_argument("--output-dir", required=True, help="Ignored diagnostic output directory")
    parser.add_argument("--legacy-candidate", action="append", default=[], help="Explicit Legacy file to compare by hash only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = resolve_official_root(args.official_root)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if root == output_dir or root in output_dir.parents:
        raise AuditError("output directory must not be inside the immutable official source root")
    payload = run_audit(root, output_dir, [Path(value).expanduser().resolve() for value in args.legacy_candidate])
    print(json.dumps({
        "official_root": payload["official_root"],
        "physical_file_count": len(payload["physical_files"]),
        "label_alignment_summary": payload["label_alignment_summary"],
        "outputs": str(output_dir),
        "training_run": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
