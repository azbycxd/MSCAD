from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import re
import struct
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from zipfile import ZipFile
import xml.etree.ElementTree as ET


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"w": W, "a": A, "r": R, "wp": WP, "rel": REL}
KNOWN_SCHEDULE_IMAGE_SHA256 = "faafa2b8e702fd7a312a409bc43f9785a41de7ce93caa75ac6de685c8230af31"
DOC_DATE = "2026-03-11"
PHASE_START = datetime.fromisoformat("2026-03-11T13:00:00")
PHASE_END = datetime.fromisoformat("2026-03-11T17:00:00")
SESSION_START = datetime.fromisoformat("2026-03-11T09:00:00")
SESSION_END = datetime.fromisoformat("2026-03-11T17:00:59")


TRANSCRIPTION = [
    ("1", "Stage 5 Valve Manipulation — MV503/504 Open, MV501/502 Close", "13:00:00", "13:05:00", "MV501–504"),
    ("2", "Stage 1 Flow Disruption — MV101 Open, P101 & P102 Stop", "13:40:00", "13:45:00", "MV101, P101, P102"),
    ("3", "Florida Water Plant Scenario — Dosing Pump Activation & Sensor Spoofing", "14:20:00", "14:25:00", "P201–P206, MV201"),
    ("4", "Tank Overflow via LIT101 Spoofing — Schneider Demo Attack 01", "14:30:00", "14:35:00", "LIT101"),
    ("5", "Stage 5 Valve Manipulation — MV503/504 Open, MV501/502 Close (Repeat)", "14:40:00", "14:45:00", "MV501–504"),
    ("6", "Tank Overflow via LIT101 Spoofing — Schneider Demo Attack 01", "15:00:00", "15:02:00", "LIT101"),
    ("7", "Stage 2 Parallel Pump Override — MV201 Open, P101 & P102 Run", "15:02:00", "15:07:00", "MV201, P101, P102"),
    ("8", "Reverse Osmosis Backwash Diversion — MV302 Close, MV303 Open", "15:20:00", "15:25:00", "MV302, MV303"),
    ("9", "Forced Backwash Trigger via DPIT301 Spoofing — Ensign Pre-UAT Attack 1", "15:45:00", "15:50:00", "DPIT301"),
    ("10", "Multi-Valve Level Oscillation — LIT601 Spoofing Sequence — Ensign Pre-UAT Attack 3", "16:10:00", "16:15:00", "LIT601"),
    ("11", "AIT402 High-Value Spoof Start – Using script", "16:35:00", "16:40:00", "AIT402"),
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def image_dimensions(data: bytes, suffix: str) -> tuple[int | None, int | None]:
    if suffix.lower() == ".png" and data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if suffix.lower() in {".jpg", ".jpeg"} and data.startswith(b"\xff\xd8"):
        offset = 2
        while offset + 9 < len(data):
            if data[offset] != 0xFF:
                offset += 1
                continue
            marker = data[offset + 1]
            length = int.from_bytes(data[offset + 2:offset + 4], "big")
            if marker in range(0xC0, 0xC4):
                return int.from_bytes(data[offset + 7:offset + 9], "big"), int.from_bytes(data[offset + 5:offset + 7], "big")
            offset += 2 + max(length, 2)
    return None, None


def relationship_map(archive: ZipFile, part: str) -> dict[str, dict[str, str]]:
    pure = PurePosixPath(part)
    rels_name = str(pure.parent / "_rels" / f"{pure.name}.rels")
    if rels_name not in archive.namelist():
        return {}
    root = ET.fromstring(archive.read(rels_name))
    result = {}
    for node in root.findall("rel:Relationship", NS):
        target = node.attrib.get("Target", "")
        if not target.startswith("/"):
            target = str(PurePosixPath(pure.parent, target))
        result[node.attrib["Id"]] = {"target": target.lstrip("/"), "type": node.attrib.get("Type", "")}
    return result


def node_text(node: ET.Element) -> str:
    return "".join(item.text or "" for item in node.findall(".//w:t", NS)).strip()


def parse_word_part(archive: ZipFile, part: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root = ET.fromstring(archive.read(part))
    rels = relationship_map(archive, part)
    paragraphs, tables, references = [], [], []
    for paragraph_index, paragraph in enumerate(root.findall(".//w:p", NS)):
        ids = [node.attrib.get(f"{{{R}}}embed", "") for node in paragraph.findall(".//a:blip", NS)]
        ids = [value for value in ids if value]
        names = [node.attrib.get("name", "") for node in paragraph.findall(".//wp:docPr", NS)]
        paragraphs.append({
            "source_part": part, "paragraph_index": paragraph_index, "text": node_text(paragraph),
            "relationship_ids": ids, "drawing_names": [name for name in names if name],
        })
        for rel_id in ids:
            references.append({
                "source_part": part, "paragraph_index": paragraph_index, "relationship_id": rel_id,
                "target": rels.get(rel_id, {}).get("target", ""), "drawing_names": [name for name in names if name],
            })
    for table_index, table in enumerate(root.findall(".//w:tbl", NS)):
        for row_index, row in enumerate(table.findall("./w:tr", NS)):
            for column_index, cell in enumerate(row.findall("./w:tc", NS)):
                tables.append({"source_part": part, "table_index": table_index, "row_index": row_index,
                               "column_index": column_index, "content": node_text(cell)})
    return paragraphs, tables, references


def extract_docx(docx: Path, output_dir: Path | None = None) -> dict[str, Any]:
    before_hash = sha256_file(docx)
    with ZipFile(docx) as archive:
        names = set(archive.namelist())
        parts = ["word/document.xml"] + sorted(name for name in names if re.fullmatch(r"word/(?:header|footer)\d+\.xml", name))
        paragraphs: list[dict[str, Any]] = []
        tables: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []
        for part in parts:
            part_paragraphs, part_tables, part_references = parse_word_part(archive, part)
            paragraphs.extend(part_paragraphs)
            tables.extend(part_tables)
            references.extend(part_references)
        ref_by_target: dict[str, list[dict[str, Any]]] = {}
        for reference in references:
            ref_by_target.setdefault(reference["target"], []).append(reference)
        media = []
        for name in sorted(item for item in names if item.startswith("word/media/") and not item.endswith("/")):
            data = archive.read(name)
            suffix = Path(name).suffix.lower()
            width, height = image_dimensions(data, suffix)
            item_refs = ref_by_target.get(name, [])
            extracted_path = ""
            if output_dir is not None:
                media_dir = output_dir / "media"
                media_dir.mkdir(parents=True, exist_ok=True)
                destination = media_dir / Path(name).name
                destination.write_bytes(data)
                extracted_path = str(destination)
            media.append({
                "original_filename": Path(name).name, "archive_path": name,
                "media_type": mimetypes.guess_type(name)[0] or "application/octet-stream",
                "size_bytes": len(data), "width": width, "height": height, "sha256": sha256_bytes(data),
                "relationship_ids": ";".join(sorted({ref["relationship_id"] for ref in item_refs})),
                "source_parts": ";".join(sorted({ref["source_part"] for ref in item_refs})),
                "referenced_paragraph_indices": ";".join(str(ref["paragraph_index"]) for ref in item_refs),
                "drawing_names": ";".join(name for ref in item_refs for name in ref["drawing_names"]),
                "extracted_path": extracted_path, "readable": width is not None and height is not None,
            })
    after_hash = sha256_file(docx)
    if before_hash != after_hash:
        raise RuntimeError("Source DOCX hash changed during read-only extraction")
    return {"paragraphs": paragraphs, "tables": tables, "media": media, "source_sha256": before_hash,
            "source_hash_unchanged": True}


def make_schedule(image: dict[str, Any] | None, docx: Path) -> list[dict[str, Any]]:
    known = bool(image and image["sha256"] == KNOWN_SCHEDULE_IMAGE_SHA256)
    source_image = image.get("extracted_path", "") if image else ""
    source_sha = image.get("sha256", "") if image else ""
    rows = []
    values = TRANSCRIPTION if known else [(str(index), "", "", "", "") for index in range(1, 12)]
    for row_number, (attack_id, name, start, end, target) in enumerate(values, 1):
        note = ""
        if known:
            note = ("Date is supplied from the A.12 document context because the image has no date column; "
                    "the image has one Attack Name field, copied verbatim into attack_name and description.")
        else:
            note = "Schedule image is not the visually reviewed known image; fields intentionally left blank."
        rows.append({
            "row_number": row_number, "attack_id": attack_id, "attack_name": name,
            "date": DOC_DATE if known else "", "start_time": start, "end_time": end, "target": target,
            "description": name, "source_document": str(docx), "source_image": source_image,
            "source_image_sha256": source_sha,
            "transcription_method": "manual_visual_transcription_by_codex" if known else "blank_human_review_template",
            "field_confidence": "high; date=contextual" if known else "unreadable_or_unreviewed",
            "ambiguity_note": note, "human_review_status": "needs_human_confirmation",
        })
    return rows


def parse_candidate_time(row: dict[str, Any], field: str) -> datetime | None:
    try:
        return datetime.fromisoformat(f"{row['date']}T{row[field]}")
    except (KeyError, TypeError, ValueError):
        return None


def validate_schedule(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    id_counts = Counter(str(row.get("attack_id", "")) for row in rows if row.get("attack_id"))
    parsed = [(parse_candidate_time(row, "start_time"), parse_candidate_time(row, "end_time")) for row in rows]
    validation = []
    closed_overlaps, strict_overlaps, touches = [], [], []
    for index, row in enumerate(rows):
        start, end = parsed[index]
        overlap_ids, touch_ids = [], []
        if start and end:
            for other_index, other in enumerate(rows):
                if index == other_index:
                    continue
                other_start, other_end = parsed[other_index]
                if not other_start or not other_end:
                    continue
                left, right = max(start, other_start), min(end, other_end)
                if left < right:
                    overlap_ids.append(str(other.get("attack_id", "")))
                elif left == right:
                    touch_ids.append(str(other.get("attack_id", "")))
        required = ["attack_id", "attack_name", "date", "start_time", "end_time", "target", "description"]
        missing = [field for field in required if not str(row.get(field, "")).strip()]
        issues = []
        if id_counts.get(str(row.get("attack_id", "")), 0) > 1:
            issues.append("duplicate_attack_id")
        if start is None:
            issues.append("invalid_start_time")
        if end is None:
            issues.append("invalid_end_time")
        if start and end and start > end:
            issues.append("reversed_interval")
        if start and start.date().isoformat() != DOC_DATE or end and end.date().isoformat() != DOC_DATE:
            issues.append("date_outside_2026_03_11")
        within_phase = bool(start and end and PHASE_START <= start <= end <= PHASE_END)
        within_session = bool(start and end and SESSION_START <= start <= end <= SESSION_END)
        if start and end and not within_phase:
            issues.append("outside_attack_experiment_phase")
        if start and end and not within_session:
            issues.append("outside_csv_session")
        if overlap_ids:
            issues.append("positive_duration_overlap_needs_review")
        if touch_ids:
            issues.append("shared_boundary_needs_policy_review")
        issues.extend(f"missing_{field}" for field in missing)
        validation.append({
            "row_number": row.get("row_number", ""), "attack_id": row.get("attack_id", ""),
            "attack_id_unique": id_counts.get(str(row.get("attack_id", "")), 0) == 1,
            "start_time_valid": start is not None, "end_time_valid": end is not None,
            "start_before_or_equal_end": bool(start and end and start <= end),
            "date_is_2026_03_11": bool(start and end and start.date().isoformat() == DOC_DATE and end.date().isoformat() == DOC_DATE),
            "within_attack_phase": within_phase, "within_csv_session": within_session,
            "positive_overlap_with": ";".join(sorted(set(overlap_ids))),
            "shared_boundary_with": ";".join(sorted(set(touch_ids))),
            "missing_field_count": len(missing), "issues": ";".join(issues),
            "candidate_valid_pending_human_review": not any(issue for issue in issues if issue not in {"shared_boundary_needs_policy_review"}),
        })
    for left_index in range(len(rows)):
        for right_index in range(left_index + 1, len(rows)):
            left_start, left_end = parsed[left_index]
            right_start, right_end = parsed[right_index]
            if not all((left_start, left_end, right_start, right_end)):
                continue
            pair = f"{rows[left_index].get('attack_id', '')}-{rows[right_index].get('attack_id', '')}"
            if max(left_start, right_start) < min(left_end, right_end):
                strict_overlaps.append(pair)
                closed_overlaps.append(pair)
            elif max(left_start, right_start) == min(left_end, right_end):
                touches.append(pair)
                closed_overlaps.append(pair)
    valid_intervals = sorted((start, end) for start, end in parsed if start and end and start <= end)
    gaps = [(valid_intervals[i][1], valid_intervals[i + 1][0]) for i in range(len(valid_intervals) - 1)
            if valid_intervals[i + 1][0] > valid_intervals[i][1]]
    total_seconds = sum((end - start).total_seconds() for start, end in valid_intervals)
    summary = {
        "candidate_row_count": len(rows), "duplicate_attack_ids": sorted(key for key, count in id_counts.items() if count > 1),
        "invalid_start_count": sum(start is None for start, _ in parsed),
        "invalid_end_count": sum(end is None for _, end in parsed),
        "reversed_interval_count": sum(bool(start and end and start > end) for start, end in parsed),
        "outside_attack_phase_count": sum(not row["within_attack_phase"] for row in validation),
        "outside_csv_session_count": sum(not row["within_csv_session"] for row in validation),
        "positive_duration_overlap_pairs": strict_overlaps, "closed_interval_overlap_pairs": closed_overlaps,
        "shared_boundary_pairs": touches, "normal_gap_count_between_attacks": len(gaps),
        "normal_gaps_between_attacks": [{"start": start.isoformat(), "end": end.isoformat(),
                                          "duration_seconds": (end - start).total_seconds()} for start, end in gaps],
        "phase_edge_normal_gap_count": int(bool(valid_intervals and valid_intervals[-1][1] < PHASE_END)),
        "attack_total_duration_seconds_interval_sum": total_seconds,
        "missing_field_count": sum(row["missing_field_count"] for row in validation),
        "human_approval_required": True,
    }
    return validation, summary


def phase_rows(docx: Path) -> list[dict[str, Any]]:
    return [
        {"session_id": "11-Mar-2026", "phase_type": "normal_phase", "start_time": "2026-03-11T09:00:00",
         "end_time": "2026-03-11T13:00:00", "official_evidence": str(docx),
         "interpretation": "Official normal operating phase; phase metadata is not a point label.",
         "human_review_status": "needs_human_confirmation"},
        {"session_id": "11-Mar-2026", "phase_type": "attack_experiment_phase", "start_time": "2026-03-11T13:00:00",
         "end_time": "2026-03-11T17:00:00", "official_evidence": str(docx),
         "interpretation": "Attacks are intermittent; do not label the complete phase as anomalous.",
         "human_review_status": "needs_human_confirmation"},
        {"session_id": "19-Feb-2026", "phase_type": "unresolved", "start_time": "2026-02-19T09:30:00",
         "end_time": "2026-02-19T17:35:59", "official_evidence": "",
         "interpretation": "A.12 Info Sheet does not assign semantics to this A.10 session.",
         "human_review_status": "needs_human_confirmation"},
        {"session_id": "20-Feb-2026", "phase_type": "unresolved", "start_time": "2026-02-20T09:05:00",
         "end_time": "2026-02-20T17:10:59", "official_evidence": "",
         "interpretation": "A.12 Info Sheet does not assign semantics to this A.10 session.",
         "human_review_status": "needs_human_confirmation"},
    ]


def iter_seconds(start: datetime, end: datetime, closed: bool) -> Iterable[datetime]:
    current = start
    while current < end or closed and current == end:
        yield current
        current += timedelta(seconds=1)


def label_policy_candidates(rows: list[dict[str, Any]], timestamps: set[datetime] | None = None) -> dict[str, Any]:
    intervals = [(parse_candidate_time(row, "start_time"), parse_candidate_time(row, "end_time")) for row in rows]
    intervals = [(start, end) for start, end in intervals if start and end and start <= end]
    closed_points = [point for start, end in intervals for point in iter_seconds(start, end, True)]
    left_open_points = [point for start, end in intervals for point in iter_seconds(start, end, False)]
    closed_union, left_union = set(closed_points), set(left_open_points)
    if timestamps is not None:
        closed_union &= timestamps
        left_union &= timestamps
    return {
        "timezone_policy": "local_clock_unspecified",
        "timezone_interpretation": "CSV and schedule use the same literal local clock; no UTC conversion is applied.",
        "boundary_policy": "pending_human_review",
        "closed_interval": {"rule": "start <= timestamp <= end", "interval_sum_point_count": len(closed_points),
                            "unique_session_point_count": len(closed_union)},
        "left_closed_right_open": {"rule": "start <= timestamp < end", "interval_sum_point_count": len(left_open_points),
                                   "unique_session_point_count": len(left_union)},
        "unique_point_count_difference": len(closed_union) - len(left_union),
        "formal_labels_generated": False,
    }


def load_timestamps(path: Path) -> set[datetime]:
    result: set[datetime] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        timestamp_column = next((name for name in reader.fieldnames or [] if name.strip().lower() in {"t_stamp", "timestamp"}), None)
        if not timestamp_column:
            return result
        for row in reader:
            try:
                result.add(datetime.strptime(row[timestamp_column].strip(), "%d-%b-%Y %H:%M:%S"))
            except (KeyError, ValueError):
                continue
    return result


def schedule_membership(timestamp: datetime, intervals: list[tuple[datetime, datetime]]) -> tuple[bool, bool]:
    closed = any(start <= timestamp <= end for start, end in intervals)
    left_open = any(start <= timestamp < end for start, end in intervals)
    return closed, left_open


def scan_bad_inputs(data_dir: Path, schedule: list[dict[str, Any]], destination: Path | None) -> dict[str, Any]:
    intervals = [(parse_candidate_time(row, "start_time"), parse_candidate_time(row, "end_time")) for row in schedule]
    intervals = [(start, end) for start, end in intervals if start and end and start <= end]
    files = [
        ("11-Mar-2026", data_dir / "11-Mar-2026_0900_1700.csv"),
        ("19-Feb-2026", data_dir / "SWaT.A10_OTDataset_19-Feb-2026_0930_1735.csv"),
        ("20-Feb-2026", data_dir / "SWaT.A10_OTDataset_20-Feb-2026_0905_1710.csv"),
    ]
    fields = ["session_id", "timestamp", "column", "raw_value", "in_attack_closed", "in_attack_left_open", "location_category"]
    handle = destination.open("w", newline="", encoding="utf-8-sig") if destination else None
    writer = csv.DictWriter(handle, fieldnames=fields) if handle else None
    if writer:
        writer.writeheader()
    counts: Counter[str] = Counter()
    source_hashes = {}
    try:
        for session_id, path in files:
            if not path.exists():
                continue
            before = sha256_file(path)
            with path.open(newline="", encoding="utf-8-sig") as source:
                reader = csv.DictReader(source)
                timestamp_column = next((name for name in reader.fieldnames or [] if name.strip().lower() in {"t_stamp", "timestamp"}), None)
                for row in reader:
                    raw_time = row.get(timestamp_column or "", "")
                    try:
                        timestamp = datetime.strptime(raw_time.strip(), "%d-%b-%Y %H:%M:%S")
                    except ValueError:
                        continue
                    closed, left_open = schedule_membership(timestamp, intervals) if session_id == "11-Mar-2026" else (False, False)
                    if session_id != "11-Mar-2026":
                        category = "unresolved_session"
                    elif timestamp < PHASE_START:
                        category = "normal_phase"
                    elif timestamp > PHASE_END:
                        category = "outside_declared_phase"
                    elif left_open:
                        category = "attack_interval_both_policies"
                    elif closed:
                        category = "attack_end_boundary_closed_only"
                    else:
                        category = "attack_experiment_phase_between_attacks"
                    for column, value in row.items():
                        if isinstance(value, str) and value.strip().lower() == "bad input":
                            result = {"session_id": session_id, "timestamp": timestamp.isoformat(), "column": column,
                                      "raw_value": value, "in_attack_closed": closed, "in_attack_left_open": left_open,
                                      "location_category": category}
                            if writer:
                                writer.writerow(result)
                            counts[category] += 1
                            counts["total"] += 1
            after = sha256_file(path)
            if before != after:
                raise RuntimeError(f"Source CSV changed during read-only scan: {path}")
            source_hashes[str(path)] = before
    finally:
        if handle:
            handle.close()
    return {"occurrence_count": counts["total"], "category_counts": dict(counts), "source_hashes_unchanged": True,
            "source_sha256": source_hashes, "values_modified": False,
            "n2_recommendation": "Prefer an invalid_mask that preserves the timeline; finalize only after schedule review."}


def checklist_text() -> str:
    items = [
        "The image contains exactly 11 attacks.", "Every attack_id is correct.", "Every start time is correct.",
        "Every end time is correct.", "Every primary target is correct.", "Every description preserves the image wording.",
        "The shared 15:02 boundary between attacks 6 and 7 is understood.", "Every date is 11-Mar-2026.",
        "No timezone is specified elsewhere in the official material.", "Confirm whether end timestamps are inclusive.",
        "Confirm whether a separate official Bad Input policy exists.",
    ]
    return "# SWaT A.12 human review checklist\n\n" + "\n".join(f"- [ ] {item}" for item in items) + "\n"


def readiness_text() -> str:
    return """# SWaT-N1R readiness reassessment

The official files are not mutually conflicting. The earlier F result arose because the attack table is embedded as an image and was not structurally parsed or human-confirmed.

- 11-Mar has an official normal phase (09:00–13:00) and attack experiment phase (13:00–17:00).
- Attacks are intermittent; the whole attack experiment phase must not be labeled anomalous.
- 19-Feb and 20-Feb remain unresolved and can be excluded from the initial split design.
- After all 11 candidate rows and the boundary policy are explicitly approved by a human, the evidence can advance to **H. ready_for_label_generation_review**.
- This supports only an initial official 11-Mar experiment. Stronger normal-operation coverage still requires semantics for 19/20 or additional official normal sessions.

Current state: **candidate transcribed, pending row-by-row human review; not ready for N2**.
"""


def run(docx: Path, data_dir: Path, output_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    extracted = extract_docx(docx, None if dry_run else output_dir)
    media = extracted["media"]
    selected = media[0] if len(media) == 1 else None
    if selected and not dry_run:
        selected["extracted_path"] = str(output_dir / "media" / selected["original_filename"])
    schedule = make_schedule(selected, docx)
    validation, validation_summary = validate_schedule(schedule)
    image_known = bool(selected and selected["sha256"] == KNOWN_SCHEDULE_IMAGE_SHA256)
    human_state = "B. candidate_transcribed_pending_review" if image_known and validation_summary["missing_field_count"] == 0 else (
        "C. candidate_incomplete_pending_review" if selected else "A. schedule_not_readable")
    eleven_mar = data_dir / "11-Mar-2026_0900_1700.csv"
    timestamps = load_timestamps(eleven_mar) if eleven_mar.exists() else None
    policy = label_policy_candidates(schedule, timestamps)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        text_lines = ["# DOCX text extraction", "", f"Source: `{docx}`", f"SHA-256: `{extracted['source_sha256']}`", ""]
        for item in extracted["paragraphs"]:
            if item["text"] or item["relationship_ids"]:
                refs = f" [media refs: {', '.join(item['relationship_ids'])}]" if item["relationship_ids"] else ""
                text_lines.append(f"- `{item['source_part']}#{item['paragraph_index']}`: {item['text']}{refs}")
        (output_dir / "docx_text_extraction.md").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
        write_csv(output_dir / "docx_table_extraction.csv", extracted["tables"],
                  ["source_part", "table_index", "row_index", "column_index", "content"])
        media_fields = ["original_filename", "archive_path", "media_type", "size_bytes", "width", "height", "sha256",
                        "relationship_ids", "source_parts", "referenced_paragraph_indices", "drawing_names", "extracted_path", "readable"]
        write_csv(output_dir / "docx_media_inventory.csv", media, media_fields)
        manifest = {
            "source_document": str(docx), "source_document_sha256": extracted["source_sha256"],
            "media_count": len(media), "selection_status": "unique_attack_schedule_image" if selected else "multiple_or_missing_candidates",
            "selected_image": selected, "visually_reviewed_sha256": KNOWN_SCHEDULE_IMAGE_SHA256,
            "is_clearly_readable": image_known, "candidate_images": media if not selected else [],
            "source_hash_unchanged": extracted["source_hash_unchanged"],
        }
        write_json(output_dir / "attack_schedule_image_manifest.json", manifest)
        schedule_fields = ["row_number", "attack_id", "attack_name", "date", "start_time", "end_time", "target", "description",
                           "source_document", "source_image", "source_image_sha256", "transcription_method", "field_confidence",
                           "ambiguity_note", "human_review_status"]
        write_csv(output_dir / "official_attack_schedule_candidate.csv", schedule, schedule_fields)
        write_csv(output_dir / "official_session_phase_candidate.csv", phase_rows(docx),
                  ["session_id", "phase_type", "start_time", "end_time", "official_evidence", "interpretation", "human_review_status"])
        validation_fields = ["row_number", "attack_id", "attack_id_unique", "start_time_valid", "end_time_valid",
                             "start_before_or_equal_end", "date_is_2026_03_11", "within_attack_phase", "within_csv_session",
                             "positive_overlap_with", "shared_boundary_with", "missing_field_count", "issues",
                             "candidate_valid_pending_human_review"]
        write_csv(output_dir / "attack_schedule_validation.csv", validation, validation_fields)
        write_json(output_dir / "attack_schedule_validation_summary.json", validation_summary)
        write_json(output_dir / "label_policy_candidates.json", policy)
        bad_input = scan_bad_inputs(data_dir, schedule, output_dir / "bad_input_schedule_overlap.csv")
        write_json(output_dir / "bad_input_schedule_overlap_summary.json", bad_input)
        (output_dir / "human_review_checklist.md").write_text(checklist_text(), encoding="utf-8")
        write_json(output_dir / "human_review_status.json", {
            "status": human_state, "human_approved": False, "ready_for_n2": False,
            "reason": "All candidate rows and the boundary policy require explicit human confirmation.",
        })
        (output_dir / "readiness_reassessment.md").write_text(readiness_text(), encoding="utf-8")
    else:
        bad_input = scan_bad_inputs(data_dir, schedule, None)
    result = {
        "dry_run": dry_run, "source_hash_unchanged": extracted["source_hash_unchanged"], "paragraph_count": len(extracted["paragraphs"]),
        "table_cell_count": len(extracted["tables"]), "media_count": len(media), "selected_image_sha256": selected["sha256"] if selected else None,
        "candidate_row_count": len(schedule), "missing_field_count": validation_summary["missing_field_count"],
        "human_review_status": human_state, "normal_gap_count_between_attacks": validation_summary["normal_gap_count_between_attacks"],
        "unique_boundary_point_difference": policy["unique_point_count_difference"],
        "bad_input_occurrence_count": bad_input["occurrence_count"], "ready_for_n2": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract and review SWaT A.12 Info Sheet without python-docx")
    parser.add_argument("--docx", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(Path(args.docx), Path(args.data_dir), Path(args.output_dir), args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
