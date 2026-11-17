from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.project_paths import get_repo_root  # noqa: E402


SUPPORTED = {".csv", ".xlsx", ".xls", ".docx", ".pdf", ".txt", ".json", ".yaml", ".yml", ".zip", ".7z", ".rar", ".pcap", ".pcapng"}
DOWNLOAD_KEYWORDS = re.compile(r"swat|a12|dataset.?info|mar[_ -]?26|mar 2026", re.I)
TIMESTAMP_NAMES = ("t_stamp", "timestamp", "datetime", "time", "date")
LABEL_NAMES = ("label", "attack", "attack_label", "anomaly", "normal/attack")
MAIN_ROOT = get_repo_root().parent / "mscad"


@dataclass
class SessionProfile:
    row: dict[str, Any]
    columns: list[str]
    timestamps: pd.DatetimeIndex


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def empty_csv(path: Path, columns: list[str]) -> None:
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def resolve_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    return next((lookup[name] for name in candidates if name in lookup), None)


def classify_file(path: Path, data_dir: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in {".pcap", ".pcapng"}:
        return "network_pcap"
    if suffix in {".zip", ".7z", ".rar"}:
        return "archive"
    if "label" in name:
        return "label_file"
    if any(term in name for term in ("attack schedule", "attack_schedule", "attack list", "attack_list", "experiment schedule")):
        return "attack_schedule"
    if any(term in name for term in ("dataset info", "dataset_info", "info sheet", "infosheet")) or "a12" in name and suffix != ".csv":
        return "dataset_info_sheet"
    if suffix == ".csv" and (path.parent.resolve() == data_dir.resolve() or "swat" in name):
        return "ot_timeseries"
    if suffix in {".docx", ".pdf", ".xlsx", ".xls", ".txt", ".json", ".yaml", ".yml"}:
        return "unknown_document"
    return "unrelated"


def discover_files(data_dir: Path, dataset_root: Path, downloads: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    locations: list[dict[str, Any]] = []
    found: dict[str, Path] = {}

    def scan(root: Path, mode: str, predicate) -> None:
        if not root.exists():
            locations.append({"location": str(root), "search_mode": mode, "exists": False, "candidate_count": 0, "error": "missing"})
            return
        try:
            candidates = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED and predicate(path)]
            for path in candidates:
                found[str(path.resolve()).lower()] = path.resolve()
            locations.append({"location": str(root), "search_mode": mode, "exists": True, "candidate_count": len(candidates), "error": ""})
        except Exception as exc:
            locations.append({"location": str(root), "search_mode": mode, "exists": True, "candidate_count": 0, "error": str(exc)})

    scan(data_dir, "official_data_directory_all_supported", lambda path: True)
    scan(dataset_root, "dataset_tree_swat_named_only", lambda path: data_dir.resolve() not in path.resolve().parents and "swat" in str(path).lower())
    scan(downloads, "downloads_keyword_filtered", lambda path: bool(DOWNLOAD_KEYWORDS.search(path.name)))
    return sorted(found.values(), key=lambda path: str(path).lower()), locations


def git_state(path: Path) -> tuple[bool, bool]:
    try:
        rel = path.resolve().relative_to(MAIN_ROOT.resolve()).as_posix()
    except ValueError:
        return False, False
    ignored = subprocess.run(["git", "-C", str(MAIN_ROOT), "check-ignore", "-q", "--", rel], check=False).returncode == 0
    tracked = subprocess.run(["git", "-C", str(MAIN_ROOT), "ls-files", "--error-unmatch", "--", rel], capture_output=True, check=False).returncode == 0
    return ignored, tracked


def inventory_files(paths: list[Path], data_dir: Path) -> pd.DataFrame:
    rows = []
    for index, path in enumerate(paths, 1):
        ignored, tracked = git_state(path)
        readable, error, digest = True, "", ""
        try:
            digest = sha256(path)
        except Exception as exc:
            readable, error = False, str(exc)
        rows.append({
            "file_id": f"F{index:04d}", "absolute_path": str(path), "filename": path.name,
            "extension": path.suffix.lower(), "size_bytes": path.stat().st_size,
            "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(), "sha256": digest,
            "possible_role": classify_file(path, data_dir), "readable": readable, "read_error": error,
            "git_ignored": ignored, "git_tracked": tracked,
        })
    return pd.DataFrame(rows)


def profile_ot_csv(path: Path, session_id: str, n0_columns: list[str] | None = None) -> SessionProfile:
    try:
        header = pd.read_csv(path, nrows=0).columns.astype(str).str.strip().tolist()
        timestamp_column = resolve_column(header, TIMESTAMP_NAMES)
        label_column = resolve_column(header, LABEL_NAMES)
        if timestamp_column is None:
            raise ValueError("timestamp column not found")
        row_count = 0
        first_time = last_time = previous_time = None
        monotonic = True
        duplicate_timestamp_count = 0
        one_second = interval_count = gap_count = 0
        maximum_gap = None
        seen_timestamps: set[int] = set()
        seen_rows: set[int] = set()
        duplicate_rows = 0
        bad_rows = 0
        bad_columns: Counter[str] = Counter()
        label_values: set[str] = set()
        timestamp_values: list[np.datetime64] = []
        for chunk in pd.read_csv(path, chunksize=25_000, low_memory=False):
            chunk.columns = [str(column).strip() for column in chunk.columns]
            parsed = pd.to_datetime(chunk[timestamp_column], format="%d-%b-%Y %H:%M:%S", errors="coerce")
            if parsed.isna().any():
                parsed = pd.to_datetime(chunk[timestamp_column], errors="coerce")
            if parsed.isna().any():
                raise ValueError(f"{int(parsed.isna().sum())} unparseable timestamps")
            for current in parsed:
                ns = int(current.value)
                if ns in seen_timestamps:
                    duplicate_timestamp_count += 1
                seen_timestamps.add(ns)
                timestamp_values.append(np.datetime64(current.to_datetime64()))
                if previous_time is not None:
                    delta = float((current - previous_time).total_seconds())
                    interval_count += 1
                    one_second += int(delta == 1.0)
                    gap_count += int(delta != 1.0)
                    maximum_gap = delta if maximum_gap is None else max(maximum_gap, delta)
                    monotonic &= delta >= 0
                previous_time = current
            first_time = parsed.iloc[0] if first_time is None else first_time
            last_time = parsed.iloc[-1]
            text = chunk.astype("string").apply(lambda column: column.str.strip().str.lower())
            bad_mask = text.eq("bad input")
            bad_rows += int(bad_mask.any(axis=1).sum())
            for column in chunk.columns:
                count = int(bad_mask[column].sum())
                if count:
                    bad_columns[column] += count
            if label_column is not None:
                label_values.update(chunk[label_column].dropna().astype(str).str.strip().tolist())
            hashes = pd.util.hash_pandas_object(chunk, index=False).astype("uint64")
            for value in hashes:
                integer = int(value)
                duplicate_rows += int(integer in seen_rows)
                seen_rows.add(integer)
            row_count += len(chunk)
        schema_hash = hashlib.sha256(json.dumps(header, ensure_ascii=False).encode("utf-8")).hexdigest()
        row = {
            "session_id": session_id, "file_path": str(path), "row_count": row_count, "column_count": len(header),
            "timestamp_column": timestamp_column, "start_time": first_time.isoformat(), "end_time": last_time.isoformat(),
            "duration_seconds": float((last_time - first_time).total_seconds()), "monotonic": monotonic,
            "duplicate_timestamp_count": duplicate_timestamp_count,
            "one_second_interval_ratio": float(one_second / interval_count) if interval_count else None,
            "gap_count": gap_count, "maximum_gap_seconds": maximum_gap, "schema_hash": schema_hash,
            "schema_matches_n0": header == n0_columns if n0_columns is not None else True,
            "pv_count": sum(column.lower().endswith(".pv") for column in header),
            "stat_count": sum(column.lower().endswith((".stat", ".status")) for column in header),
            "alarm_count": sum(column.lower().endswith((".al", ".alarm")) for column in header),
            "process_state_count": sum(column.lower().endswith("_state") for column in header),
            "label_column": label_column or "", "label_unique_values": json.dumps(sorted(label_values), ensure_ascii=False),
            "bad_input_row_count": bad_rows, "bad_input_columns": json.dumps(sorted(bad_columns), ensure_ascii=False),
            "duplicate_row_count": duplicate_rows, "readable": True, "error": "",
        }
        row["bad_input_counts"] = dict(bad_columns)
        return SessionProfile(row, header, pd.DatetimeIndex(timestamp_values))
    except Exception as exc:
        row = {"session_id": session_id, "file_path": str(path), "readable": False, "error": str(exc)}
        return SessionProfile(row, [], pd.DatetimeIndex([]))


def build_session_timeline(profiles: list[SessionProfile]) -> pd.DataFrame:
    valid = sorted((profile for profile in profiles if profile.row.get("readable")), key=lambda item: item.row["start_time"])
    rows = []
    for index, profile in enumerate(valid):
        row = profile.row
        following = valid[index + 1] if index + 1 < len(valid) else None
        next_start = pd.Timestamp(following.row["start_time"]) if following else None
        end = pd.Timestamp(row["end_time"])
        overlap_count = int(profile.timestamps.intersection(following.timestamps).size) if following else 0
        rows.append({
            "session_id": row["session_id"], "start_time": row["start_time"], "end_time": row["end_time"],
            "duration_seconds": row["duration_seconds"], "next_session_id": following.row["session_id"] if following else "",
            "gap_to_next_seconds": float((next_start - end).total_seconds()) if following else None,
            "overlaps_next": bool(next_start <= end) if following else False,
            "duplicate_timestamp_count_between_files": overlap_count,
            "same_day": bool(next_start.date() == end.date()) if following else False,
            "strictly_continuous_next": bool((next_start - end).total_seconds() == 1.0) if following else False,
            "schema_compatible": bool(profile.columns == following.columns) if following else None,
            "cross_file_window_allowed": False,
            "independent_session_default": True,
        })
    return pd.DataFrame(rows)


def align_attack_to_session(attack: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
    start = pd.Timestamp(attack["attack_start"])
    end = pd.Timestamp(attack["attack_end"])
    session_start, session_end = pd.Timestamp(session["start_time"]), pd.Timestamp(session["end_time"])
    overlap_start, overlap_end = max(start, session_start), min(end, session_end)
    overlap = max(0.0, float((overlap_end - overlap_start).total_seconds())) if overlap_end >= overlap_start else 0.0
    fully = start >= session_start and end <= session_end
    partial = overlap > 0 and not fully
    return {
        "attack_id": attack.get("attack_id", ""), "session_id": session["session_id"],
        "attack_start": start.isoformat(), "attack_end": end.isoformat(),
        "session_start": session_start.isoformat(), "session_end": session_end.isoformat(),
        "overlap_seconds": overlap, "fully_contained": fully, "partially_contained": partial,
        "outside_session": overlap == 0.0, "boundary_ambiguous": attack.get("boundary_ambiguous", False),
        "candidate_for_labeling": bool(fully and not attack.get("boundary_ambiguous", False)),
    }


def classify_session(session_id: str, official_normal: bool = False, official_attack: bool = False, evidence: str = "") -> dict[str, Any]:
    return {
        "session_id": session_id, "confirmed_normal_session": bool(official_normal and not official_attack),
        "confirmed_attack_session": bool(official_attack), "mixed_session": bool(official_normal and official_attack),
        "evidence_source": evidence, "evidence_summary": evidence,
        "confidence": "official" if evidence else "unresolved",
        "unresolved_reason": "No official session semantics, label file, or attack schedule found" if not evidence else "",
    }


def determine_readiness(session_count: int, semantics: list[dict[str, Any]], schedule_count: int, info_sheet_found: bool) -> str:
    normal = any(row["confirmed_normal_session"] for row in semantics)
    attack = any(row["confirmed_attack_session"] for row in semantics) and schedule_count > 0
    if normal and attack:
        return "H. ready_for_label_generation_review"
    if normal:
        return "B. normal_source_ready_attack_schedule_missing"
    if attack:
        return "C. attack_source_ready_normal_source_missing"
    if session_count == 1 and not info_sheet_found:
        return "E. only_single_unlabelled_session"
    if session_count > 1 and not info_sheet_found:
        return "D. multiple_sessions_found_info_sheet_missing"
    return "F. official_files_conflict" if info_sheet_found else "G. source_files_unreadable"


def extract_info_documents(files: pd.DataFrame, output_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    candidates = files[files.possible_role.isin(["dataset_info_sheet", "attack_schedule", "label_file"])]
    if candidates.empty:
        structured = {"status": "missing", "source_files": [], "dataset_version": None, "sampling_frequency": None,
                      "normal_dates": [], "attack_dates": [], "timezone": "unknown", "bad_input_explanation": None,
                      "ot_pcap_relationship": None, "manual_review_required": False}
        (output_dir / "info_sheet_extraction.md").write_text("# Dataset Info Sheet extraction\n\nStatus: **missing**. No official information sheet, attack schedule, or label document was found in the allowed search locations.\n", encoding="utf-8")
        return structured, pd.DataFrame(columns=["source_file", "sheet_or_table", "row_index", "content"]), []
    # Conservative extraction: preserve text/tables for review; do not infer attack semantics.
    table_rows: list[dict[str, Any]] = []
    extracted: list[str] = []
    for _, item in candidates.iterrows():
        path = Path(item.absolute_path)
        try:
            if path.suffix.lower() == ".docx":
                from docx import Document
                document = Document(path)
                extracted.extend(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
                for table_index, table in enumerate(document.tables):
                    for row_index, row in enumerate(table.rows):
                        content = " | ".join(cell.text.strip() for cell in row.cells)
                        table_rows.append({"source_file": str(path), "sheet_or_table": f"table_{table_index}", "row_index": row_index, "content": content})
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                book = pd.ExcelFile(path)
                for sheet in book.sheet_names:
                    frame = pd.read_excel(path, sheet_name=sheet, header=None)
                    for row_index, values in frame.iterrows():
                        content = " | ".join(str(value) for value in values.dropna().tolist())
                        if content:
                            table_rows.append({"source_file": str(path), "sheet_or_table": sheet, "row_index": row_index, "content": content})
                            extracted.append(content)
            elif path.suffix.lower() == ".pdf":
                from pypdf import PdfReader
                extracted.extend((page.extract_text() or "") for page in PdfReader(path).pages)
            elif path.suffix.lower() in {".txt", ".json", ".yaml", ".yml", ".csv"}:
                extracted.append(path.read_text(encoding="utf-8-sig", errors="replace"))
        except Exception as exc:
            table_rows.append({"source_file": str(path), "sheet_or_table": "parse_error", "row_index": -1, "content": str(exc)})
    structured = {"status": "found_requires_manual_semantic_review", "source_files": candidates.absolute_path.tolist(),
                  "dataset_version": None, "sampling_frequency": None, "normal_dates": [], "attack_dates": [],
                  "timezone": "unknown", "bad_input_explanation": None, "ot_pcap_relationship": None,
                  "manual_review_required": True}
    (output_dir / "info_sheet_extraction.md").write_text("# Dataset Info Sheet extraction\n\n" + "\n\n".join(extracted), encoding="utf-8")
    return structured, pd.DataFrame(table_rows), []


def archive_inventory(files: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, item in files[files.possible_role.eq("archive")].iterrows():
        path = Path(item.absolute_path)
        if path.suffix.lower() == ".zip":
            try:
                with ZipFile(path) as archive:
                    for member in archive.infolist():
                        rows.append({"archive_path": str(path), "member_name": member.filename, "member_size": member.file_size, "readable": True, "error": ""})
            except Exception as exc:
                rows.append({"archive_path": str(path), "member_name": "", "member_size": None, "readable": False, "error": str(exc)})
        else:
            rows.append({"archive_path": str(path), "member_name": "", "member_size": None, "readable": False, "error": "Archive content not parsed at N1"})
    return pd.DataFrame(rows, columns=["archive_path", "member_name", "member_size", "readable", "error"])


def run(data_dir: Path, output_dir: Path, dry_run: bool = False, downloads: Path | None = None) -> dict[str, Any]:
    dataset_root = data_dir.parent
    downloads = downloads or Path.home() / "Downloads"
    paths, locations = discover_files(data_dir, dataset_root, downloads)
    inventory = inventory_files(paths, data_dir)
    unique_ot = inventory[inventory.possible_role.eq("ot_timeseries")].copy()
    # Prefer the official data directory and de-duplicate byte-identical Downloads copies.
    unique_ot["in_official_dir"] = unique_ot.absolute_path.map(lambda value: data_dir.resolve() in Path(value).resolve().parents)
    unique_ot = unique_ot.sort_values("in_official_dir", ascending=False).drop_duplicates("sha256")
    n0_match = unique_ot.filename.eq("11-Mar-2026_0900_1700.csv")
    if not n0_match.any() and len(unique_ot):
        n0_index = unique_ot.index[0]
    else:
        n0_index = unique_ot[n0_match].index[0] if n0_match.any() else None
    n0_columns = pd.read_csv(unique_ot.loc[n0_index, "absolute_path"], nrows=0).columns.astype(str).str.strip().tolist() if n0_index is not None else None
    profiles: list[SessionProfile] = []
    for number, (_, item) in enumerate(unique_ot.iterrows(), 1):
        profiles.append(profile_ot_csv(Path(item.absolute_path), f"S{number:03d}", n0_columns))
    summary = {"dry_run": dry_run, "candidate_file_count": len(inventory), "unique_ot_csv_count": len(profiles),
               "info_sheet_found": bool(inventory.possible_role.eq("dataset_info_sheet").any()),
               "attack_schedule_found": bool(inventory.possible_role.eq("attack_schedule").any())}
    if dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.drop(columns=[], errors="ignore").to_csv(output_dir / "source_file_inventory.csv", index=False)
    pd.DataFrame(locations).to_csv(output_dir / "search_locations.csv", index=False)
    archive_inventory(inventory).to_csv(output_dir / "archive_inventory.csv", index=False)
    profile_rows = []
    bad_rows = []
    schema_rows = []
    for profile in profiles:
        row = dict(profile.row)
        counts = row.pop("bad_input_counts", {})
        profile_rows.append(row)
        for column, count in counts.items():
            bad_rows.append({"session_id": row["session_id"], "file_path": row["file_path"], "column_name": column, "bad_input_count": count, "official_explanation_found": False})
        schema_rows.append({"session_id": row["session_id"], "file_path": row["file_path"], "schema_hash": row.get("schema_hash"),
                            "schema_matches_n0": row.get("schema_matches_n0"), "missing_from_n0": json.dumps(sorted(set(n0_columns or []) - set(profile.columns))),
                            "extra_vs_n0": json.dumps(sorted(set(profile.columns) - set(n0_columns or [])))})
    pd.DataFrame(profile_rows).to_csv(output_dir / "ot_session_profiles.csv", index=False)
    pd.DataFrame(schema_rows).to_csv(output_dir / "ot_schema_comparison.csv", index=False)
    pd.DataFrame(bad_rows, columns=["session_id", "file_path", "column_name", "bad_input_count", "official_explanation_found"]).to_csv(output_dir / "bad_input_inventory.csv", index=False)
    timeline = build_session_timeline(profiles)
    timeline.to_csv(output_dir / "session_timeline.csv", index=False)
    gap_summary = {"session_count": len(profiles), "all_files_independent_by_default": True, "cross_file_windows_allowed": False,
                   "overlap_pair_count": int(timeline.overlaps_next.sum()) if len(timeline) else 0,
                   "strictly_continuous_pair_count": int(timeline.strictly_continuous_next.sum()) if len(timeline) else 0,
                   "schema_compatible_pair_count": int(timeline["schema_compatible"].eq(True).sum()) if len(timeline) else 0,
                   "direct_concatenation_approved": False, "reason": "No official document confirms that separate CSV files form one continuous session."}
    write_json(output_dir / "session_gap_summary.json", gap_summary)
    info, info_tables, schedules = extract_info_documents(inventory, output_dir)
    write_json(output_dir / "info_sheet_structured.json", info)
    info_tables.to_csv(output_dir / "info_sheet_tables.csv", index=False)
    schedule_columns = ["attack_id", "attack_name", "date", "start_time", "end_time", "target", "description", "stated_label", "timezone", "source_file", "source_location", "parsing_confidence", "ambiguity_note"]
    pd.DataFrame(schedules, columns=schedule_columns).to_csv(output_dir / "attack_schedule_candidates.csv", index=False)
    alignment_columns = ["attack_id", "session_id", "attack_start", "attack_end", "session_start", "session_end", "overlap_seconds", "fully_contained", "partially_contained", "outside_session", "boundary_ambiguous", "candidate_for_labeling"]
    alignment_rows = [align_attack_to_session(attack, profile.row) for attack in schedules for profile in profiles if profile.row.get("readable")]
    pd.DataFrame(alignment_rows, columns=alignment_columns).to_csv(output_dir / "attack_session_alignment.csv", index=False)
    write_json(output_dir / "attack_coverage_summary.json", {"schedule_count": len(schedules), "aligned_row_count": len(alignment_rows), "fully_covered_attack_count": 0, "partially_covered_attack_count": 0, "label_generation_approved": False})
    semantics = [classify_session(profile.row["session_id"]) for profile in profiles]
    pd.DataFrame(semantics).to_csv(output_dir / "session_semantic_classification.csv", index=False)
    split_rows = [{"split_role": "unresolved", "session_id": profile.row["session_id"], "file_path": profile.row["file_path"],
                   "start_time": profile.row.get("start_time", ""), "end_time": profile.row.get("end_time", ""), "label_source": "",
                   "normal_only_required": True, "cross_file_window_allowed": False, "status": "unresolved",
                   "exclusion_reason": "Official normal/attack semantics and attack schedule are missing"} for profile in profiles]
    pd.DataFrame(split_rows).to_csv(output_dir / "split_plan_draft.csv", index=False)
    readiness = determine_readiness(len(profiles), semantics, len(schedules), info["status"] != "missing")
    readiness_payload = {"classification": readiness, "official_directory": str(data_dir), "unique_ot_session_count": len(profiles),
                         "info_sheet_status": info["status"], "attack_schedule_count": len(schedules),
                         "confirmed_normal_session_count": 0, "confirmed_attack_session_count": 0,
                         "can_generate_point_labels": False, "ready_for_split_design": False,
                         "available_materials": ["Three unique OT CSV sessions"] if len(profiles) == 3 else [f"{len(profiles)} unique OT CSV sessions"],
                         "missing_exact_materials": ["Official Dataset Info Sheet A.12", "Official attack schedule with date/start/end/target/timezone", "Official normal-session designation", "Official explanation of Bad Input or data-quality codes"],
                         "next_required_condition": "Obtain official session semantics and an attack schedule covering at least one continuous OT session, plus one officially confirmed normal session."}
    write_json(output_dir / "swat2026_source_readiness.json", readiness_payload)
    report_lines = ["# SWaT-N1 source alignment", "", f"Readiness: **{readiness}**", "", f"Unique OT sessions: {len(profiles)}.",
                    f"Dataset Info Sheet: {info['status']}.", f"Official attack schedule rows: {len(schedules)}.",
                    "No session is classified as normal or attack without official evidence.",
                    "The 11-Mar session remains semantically unresolved.",
                    "Bad Input is observed in OT data, but no official explanation was found.",
                    "No point labels, splits, concatenated data, scaler, model, or checkpoint were created."]
    (output_dir / "final_source_alignment_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "readiness": readiness, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return {**summary, "readiness": readiness}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory and align official SWaT 2026 source files without labeling or splitting")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--downloads-dir", default=str(Path.home() / "Downloads"))
    args = parser.parse_args()
    run(Path(args.data_dir), Path(args.output_dir), args.dry_run, Path(args.downloads_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
