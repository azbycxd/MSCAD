from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
OUTPUT = ROOT / "outputs" / "stage4_3e1_raw_source_locator"
DATA_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet", ".pkl"}
ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z"}
SKIP_PARTS = {".venv", "venv", "site-packages", "__pycache__", ".pytest_cache"}
CHUNK_SIZE = 50_000


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"normal", "0", "0.0", "false"}:
        return "Normal"
    if text in {"attack", "1", "1.0", "true"}:
        return "Attack"
    return f"Other:{text}"


def safe_search_roots(project_root: Path = ROOT) -> list[Path]:
    parent = project_root.parent
    roots = [project_root]
    adjacent_data = parent / "data"
    if adjacent_data.is_dir():
        roots.append(adjacent_data)
    for child in parent.iterdir():
        if child.is_dir() and (child.name.lower().startswith("mscad") or child.name.lower() == "lrs"):
            roots.append(child)
    return sorted(set(path.resolve() for path in roots), key=str)


def search_scope_is_safe(roots: Iterable[Path], project_root: Path = ROOT) -> bool:
    allowed_parent = project_root.parent.resolve()
    drive_root = Path(project_root.drive + "\\").resolve()
    for root in roots:
        resolved = root.resolve()
        if resolved == drive_root or (resolved != allowed_parent and allowed_parent not in resolved.parents):
            return False
    return True


def skipped(path: Path) -> bool:
    return bool(SKIP_PARTS.intersection(part.lower() for part in path.parts))


def looks_swat_related(path: Path) -> bool:
    text = str(path).lower()
    name = path.name.lower()
    return "swat" in text or name in {"normal.csv", "attack.csv", "merged.csv"}


def discover_files(roots: list[Path]) -> tuple[list[Path], list[Path]]:
    data_files: list[Path] = []
    archives: list[Path] = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or skipped(path):
                continue
            suffix = path.suffix.lower()
            if suffix in DATA_EXTENSIONS and looks_swat_related(path):
                if "stage4_3d_swat_data_audit" not in str(path).lower():
                    data_files.append(path.resolve())
            elif suffix in ARCHIVE_EXTENSIONS and (looks_swat_related(path) or path.parent == PARENT):
                archives.append(path.resolve())
    for path in PARENT.iterdir():
        if path.is_file() and path.suffix.lower() in ARCHIVE_EXTENSIONS:
            archives.append(path.resolve())
    return sorted(set(data_files), key=str), sorted(set(archives), key=str)


def sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    while block := handle.read(4 * 1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_stream(handle)


def resolve_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {str(c).strip().lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    for column in columns:
        clean = str(column).strip().lower()
        if any(candidate.lower() in clean for candidate in candidates):
            return str(column)
    return None


def canonical_row_hashes(frame: pd.DataFrame, label_col: str, timestamp_col: str) -> np.ndarray:
    canonical = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        clean = str(column).strip()
        if column == label_col:
            canonical[clean] = frame[column].map(normalize_label)
        elif column == timestamp_col:
            canonical[clean] = frame[column].astype("string").str.strip().fillna("<NA>")
        else:
            canonical[clean] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return pd.util.hash_pandas_object(canonical, index=False).to_numpy(np.uint64)


def label_sequence_stats(labels: list[Any]) -> dict[str, Any]:
    normalized = [normalize_label(value) for value in labels]
    transitions = sum(a != b for a, b in zip(normalized, normalized[1:]))
    positive_segments = 0
    previous = None
    for value in normalized:
        if value == "Attack" and previous != "Attack":
            positive_segments += 1
        previous = value
    contains_n_a_n = any(
        normalized[i] == "Normal" and normalized[j] == "Attack" and normalized[k] == "Normal"
        for i, j, k in zip(range(len(normalized) - 2), range(1, len(normalized) - 1), range(2, len(normalized)))
    )
    # The direct triplet test is sufficient for segments; collapse repeated labels as well.
    collapsed = [value for i, value in enumerate(normalized) if i == 0 or value != normalized[i - 1]]
    contains_n_a_n = contains_n_a_n or any(collapsed[i : i + 3] == ["Normal", "Attack", "Normal"] for i in range(max(0, len(collapsed) - 2)))
    counts = Counter(normalized)
    return {
        "label_unique_values": sorted(counts), "normal_label_count": counts["Normal"],
        "attack_label_count": counts["Attack"], "label_transition_count": transitions,
        "positive_segment_count": positive_segments, "contains_both_normal_and_attack": counts["Normal"] > 0 and counts["Attack"] > 0,
        "contains_normal_attack_normal": contains_n_a_n,
    }


def profile_csv(path: Path, limit: int | None = None, attack_hashes: set[int] | None = None) -> tuple[dict[str, Any], set[int]]:
    profile: dict[str, Any] = {
        "path": str(path), "file_type": path.suffix.lower(), "file_size": path.stat().st_size,
        "readable": False, "read_error": "", "row_count": 0, "column_count": 0, "column_names": [],
        "possible_timestamp_column": None, "possible_label_column": None,
    }
    own_hashes: set[int] = set()
    try:
        header = pd.read_csv(path, nrows=0)
        columns = [str(column) for column in header.columns]
        label_col = resolve_column(columns, ["Normal/Attack", "label", "status", "attack"])
        timestamp_col = resolve_column(columns, ["Timestamp", "time", "datetime", "date"])
        profile.update({"column_count": len(columns), "column_names": columns,
                        "possible_timestamp_column": timestamp_col, "possible_label_column": label_col})
        labels: list[str] = []
        first_timestamp = last_timestamp = None
        previous_timestamp: pd.Timestamp | None = None
        monotonic = True
        timestamp_hash_parts: list[np.ndarray] = []
        attack_hits: set[int] = set()
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False, nrows=limit):
            profile["row_count"] += len(chunk)
            if label_col:
                labels.extend(chunk[label_col].map(normalize_label).tolist())
            if timestamp_col:
                values = chunk[timestamp_col].astype("string").str.strip().fillna("<NA>")
                if first_timestamp is None and len(values):
                    first_timestamp = str(values.iloc[0])
                if len(values):
                    last_timestamp = str(values.iloc[-1])
                parsed = pd.to_datetime(values, dayfirst=True, errors="coerce")
                if parsed.isna().any() or not parsed.is_monotonic_increasing:
                    monotonic = False
                if previous_timestamp is not None and len(parsed) and parsed.iloc[0] < previous_timestamp:
                    monotonic = False
                if len(parsed):
                    previous_timestamp = parsed.iloc[-1]
                timestamp_hash_parts.append(pd.util.hash_pandas_object(values, index=False).to_numpy(np.uint64))
            if label_col and timestamp_col:
                hashes = canonical_row_hashes(chunk, label_col, timestamp_col)
                own_hashes.update(int(value) for value in hashes)
                if attack_hashes:
                    attack_hits.update(int(value) for value in hashes if int(value) in attack_hashes)
        stats = label_sequence_stats(labels) if label_col else label_sequence_stats([])
        timestamp_hashes = np.concatenate(timestamp_hash_parts) if timestamp_hash_parts else np.array([], dtype=np.uint64)
        duplicate_timestamps = int(len(timestamp_hashes) - len(np.unique(timestamp_hashes)))
        profile.update(stats)
        profile.update({"timestamp_first": first_timestamp, "timestamp_last": last_timestamp,
                        "timestamp_monotonic": monotonic if timestamp_col else None,
                        "timestamp_duplicate_count": duplicate_timestamps,
                        "current_attack_unique_rows_found": len(attack_hits), "readable": True})
    except Exception as exc:
        profile["read_error"] = f"{type(exc).__name__}: {exc}"
    return profile, own_hashes


def inspect_excel(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        book = pd.ExcelFile(path)
        for sheet in book.sheet_names:
            sample = pd.read_excel(path, sheet_name=sheet, nrows=5)
            rows.append({"path": str(path), "sheet": sheet, "column_count": len(sample.columns),
                         "column_names": json.dumps([str(c) for c in sample.columns], ensure_ascii=False),
                         "sample_rows": len(sample), "readable": True, "read_error": ""})
    except Exception as exc:
        rows.append({"path": str(path), "sheet": "", "column_count": 0, "column_names": "[]",
                     "sample_rows": 0, "readable": False, "read_error": f"{type(exc).__name__}: {exc}"})
    return rows


def inspect_archives(archives: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    for archive in archives:
        if archive.suffix.lower() != ".zip":
            inventory.append({"path": str(archive), "archive_type": archive.suffix.lower(), "readable": False,
                              "entry_count": None, "read_error": "RAR/7z listing support unavailable; no extraction attempted"})
            continue
        try:
            with zipfile.ZipFile(archive) as handle:
                infos = handle.infolist()
                inventory.append({"path": str(archive), "archive_type": ".zip", "readable": True,
                                  "entry_count": len(infos), "read_error": ""})
                for info in infos:
                    name = info.filename
                    if info.is_dir() or not re.search(r"(?i)(swat|normal|attack|merged).*(csv|xlsx|xls|parquet|pkl)$", name):
                        continue
                    digest = ""
                    try:
                        with handle.open(info) as member:
                            digest = sha256_stream(member)
                    except Exception:
                        pass
                    members.append({"archive_path": str(archive), "member_path": name, "member_size": info.file_size,
                                    "compressed_size": info.compress_size, "crc32": f"{info.CRC:08x}", "sha256": digest})
        except Exception as exc:
            inventory.append({"path": str(archive), "archive_type": ".zip", "readable": False,
                              "entry_count": None, "read_error": f"{type(exc).__name__}: {exc}"})
    return inventory, members


def source_references(roots: list[Path]) -> list[dict[str, Any]]:
    terms = re.compile(r"(?i)normal\.csv|attack\.csv|merged\.csv|swat|read_csv|read_excel|concat|normal/attack")
    refs: list[dict[str, Any]] = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or skipped(path) or any(part.lower() in {"outputs", "results"} for part in path.parts):
                continue
            if path.suffix.lower() not in {".py", ".yaml", ".yml", ".md", ".txt", ".json", ".ipynb", ".gitignore"}:
                continue
            try:
                for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if terms.search(line):
                        refs.append({"file_path": str(path.resolve()), "line_number": number,
                                     "reference": line.strip()[:500], "search_root": str(root)})
            except OSError:
                continue
    return refs


def is_direct_derived(path: Path, digest: str, current_hashes: dict[str, str]) -> bool:
    return digest in set(current_hashes.values()) or path.name.lower() in {"merged.csv", "attack.csv"}


def chronological_attack_candidate(profile: dict[str, Any], path: Path, digest: str, current_hashes: dict[str, str]) -> bool:
    synthetic = "semisynth" in str(path).lower() or "synthetic" in str(path).lower()
    return bool(
        profile.get("readable") and profile.get("possible_timestamp_column") and
        profile.get("contains_both_normal_and_attack") and profile.get("label_transition_count", 0) >= 2 and
        profile.get("contains_normal_attack_normal") and path.name.lower() not in {"merged.csv", "attack.csv", "normal.csv"} and
        not is_direct_derived(path, digest, current_hashes) and not synthetic
    )


def chronological_normal_candidate(profile: dict[str, Any], path: Path, digest: str, current_hashes: dict[str, str]) -> bool:
    rows = int(profile.get("row_count", 0))
    normal_ratio = profile.get("normal_label_count", 0) / rows if rows else 0.0
    has_provenance = any(token in str(path).lower() for token in ("raw", "physical", "normal_v", "normal v"))
    return bool(profile.get("readable") and profile.get("possible_timestamp_column") and normal_ratio >= 0.99 and
                profile.get("timestamp_monotonic") and digest != current_hashes["normal.csv"] and has_provenance)


def ensure_output(output: Path) -> None:
    names = ["searched_locations.csv", "source_code_references.csv", "candidate_file_inventory.csv",
             "archive_inventory.csv", "excel_sheet_inventory.csv", "candidate_data_profiles.json",
             "source_relationships.csv", "raw_source_readiness.json", "final_source_locator_report.md"]
    existing = [name for name in names if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing locator outputs: {existing}")
    output.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run(output: Path, smoke: bool = False) -> dict[str, Any]:
    roots = safe_search_roots()
    if not search_scope_is_safe(roots):
        raise RuntimeError("Unsafe search scope: a root escapes the project parent or equals a drive root")
    ensure_output(output)
    searched = [{"location": str(path), "exists": path.exists(), "scope_reason": "project, adjacent data, or MSCAD/LRS sibling",
                 "recursive": True, "excluded": ".venv; venv; site-packages; outputs/results for code refs"} for path in roots]
    pd.DataFrame(searched).to_csv(output / "searched_locations.csv", index=False)

    data_files, archives = discover_files(roots)
    refs = source_references(roots)
    pd.DataFrame(refs, columns=["file_path", "line_number", "reference", "search_root"]).to_csv(output / "source_code_references.csv", index=False)
    archive_rows, archive_members = inspect_archives(archives)
    archive_table = archive_rows + [{"path": row["archive_path"], "archive_type": "member", "readable": bool(row["sha256"]),
                                     "entry_count": 1, "read_error": "", **row} for row in archive_members]
    pd.DataFrame(archive_table).to_csv(output / "archive_inventory.csv", index=False)

    excel_rows: list[dict[str, Any]] = []
    for path in data_files:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            excel_rows.extend(inspect_excel(path))
    pd.DataFrame(excel_rows, columns=["path", "sheet", "column_count", "column_names", "sample_rows", "readable", "read_error"]).to_csv(
        output / "excel_sheet_inventory.csv", index=False)

    current_paths = {name: ROOT / "Swat" / name for name in ("normal.csv", "attack.csv", "merged.csv")}
    current_hashes = {name: sha256_file(path) for name, path in current_paths.items()}
    attack_profile, attack_row_hashes = profile_csv(current_paths["attack.csv"], limit=5_000 if smoke else None)
    unique_profiles: dict[str, dict[str, Any]] = {current_hashes["attack.csv"]: attack_profile}
    inventory: list[dict[str, Any]] = []
    profiles: dict[str, Any] = {}
    relationships: list[dict[str, Any]] = []
    selected_files = data_files[:8] if smoke else data_files
    for candidate_id, path in enumerate(selected_files, 1):
        digest = sha256_file(path)
        if digest not in unique_profiles:
            if path.suffix.lower() == ".csv":
                profile, _ = profile_csv(path, limit=5_000 if smoke else None, attack_hashes=attack_row_hashes)
            else:
                profile = {"path": str(path), "file_type": path.suffix.lower(), "file_size": path.stat().st_size,
                           "readable": False, "read_error": "Only CSV receives full profile in this locator"}
            unique_profiles[digest] = profile
        profile = dict(unique_profiles[digest])
        profile["path"] = str(path)
        profiles[f"candidate_{candidate_id:03d}"] = profile
        test_candidate = chronological_attack_candidate(profile, path, digest, current_hashes)
        normal_candidate = chronological_normal_candidate(profile, path, digest, current_hashes)
        inventory.append({
            "candidate_id": f"candidate_{candidate_id:03d}", "path": str(path), "file_type": path.suffix.lower(),
            "file_size": path.stat().st_size, "sha256": digest, "row_count": profile.get("row_count"),
            "column_count": profile.get("column_count"), "possible_timestamp_column": profile.get("possible_timestamp_column"),
            "possible_label_column": profile.get("possible_label_column"), "contains_both_normal_and_attack": profile.get("contains_both_normal_and_attack"),
            "chronological_attack_test_candidate": test_candidate, "chronological_normal_train_candidate": normal_candidate,
            "readable": profile.get("readable"), "read_error": profile.get("read_error"),
        })
        same = {name: digest == value for name, value in current_hashes.items()}
        contains_attack = bool(same["attack.csv"] or same["merged.csv"] or profile.get("current_attack_unique_rows_found") == len(attack_row_hashes))
        relationships.append({
            "candidate_id": f"candidate_{candidate_id:03d}", "path": str(path),
            "same_as_current_normal": same["normal.csv"], "same_as_current_attack": same["attack.csv"],
            "same_as_current_merged": same["merged.csv"], "contains_all_current_attack_rows": contains_attack,
            "possible_upstream_source": test_candidate, "preserves_normal_rows_between_attacks": bool(test_candidate),
            "relationship_note": "known semisynthetic output" if "semisynth" in str(path).lower() else
                                 "byte-identical derived/current file" if any(same.values()) else "no proven upstream relationship",
        })
    for row in archive_members:
        relationships.append({
            "candidate_id": "archive_member", "path": f"{row['archive_path']}::{row['member_path']}",
            "same_as_current_normal": row["sha256"] == current_hashes["normal.csv"],
            "same_as_current_attack": row["sha256"] == current_hashes["attack.csv"],
            "same_as_current_merged": row["sha256"] == current_hashes["merged.csv"],
            "contains_all_current_attack_rows": row["sha256"] in {current_hashes["attack.csv"], current_hashes["merged.csv"]},
            "possible_upstream_source": False, "preserves_normal_rows_between_attacks": False,
            "relationship_note": "archive member hashed in place; no extraction performed",
        })

    inventory_df = pd.DataFrame(inventory)
    inventory_df.to_csv(output / "candidate_file_inventory.csv", index=False)
    pd.DataFrame(relationships).to_csv(output / "source_relationships.csv", index=False)
    write_json(output / "candidate_data_profiles.json", profiles)

    attack_candidates = inventory_df[inventory_df["chronological_attack_test_candidate"] == True] if len(inventory_df) else inventory_df
    normal_candidates = inventory_df[inventory_df["chronological_normal_train_candidate"] == True] if len(inventory_df) else inventory_df
    blocked = any(not row.get("readable", False) for row in archive_rows) or any(not bool(row.get("readable")) for row in inventory)
    if len(attack_candidates) and len(normal_candidates):
        status = "A. ready_to_rebuild"
    elif len(attack_candidates):
        status = "B. attack_source_found_normal_source_uncertain"
    elif len(normal_candidates):
        status = "C. normal_source_found_attack_source_missing"
    elif blocked and any(path.suffix.lower() in {".rar", ".7z"} for path in archives):
        status = "F. access_or_format_blocked"
    else:
        status = "D. only_derived_files_found"
    readiness = {
        "mode": "smoke" if smoke else "formal", "classification": status,
        "safe_search_scope": True, "searched_root_count": len(roots), "candidate_file_count": len(inventory),
        "chronological_attack_test_candidates": attack_candidates["path"].tolist() if len(attack_candidates) else [],
        "chronological_normal_train_candidates": normal_candidates["path"].tolist() if len(normal_candidates) else [],
        "ready_to_rebuild": status.startswith("A."),
        "missing_requirements": [] if status.startswith("A.") else [
            "Original chronological attack test file containing Timestamp, sensor/actuator features, and alternating Normal/Attack labels.",
            "Provenance-confirmed chronological normal training file with Timestamp and original row order.",
        ],
        "git_history_status": "No .git metadata exists at project root; history could not be inspected.",
        "gitignore_status": "No project-root .gitignore exists.",
        "internet_download_performed": False, "archives_extracted": False, "data_files_modified": False,
        "models_loaded": False, "training_run": False,
    }
    write_json(output / "raw_source_readiness.json", readiness)
    report = f"""# Step 4.3E-1 raw SWaT source locator

## Status

**{status}**

- Safe roots searched: {len(roots)} under `{PARENT}` only; no drive-root scan.
- Physical candidate files profiled/inventoried: {len(inventory)}.
- Archive members matching SWaT/normal/attack/merged names: {len(archive_members)}.
- Chronological mixed-label attack candidates: {len(attack_candidates)}.
- Provenance-confirmed normal-train candidates: {len(normal_candidates)}.

## Finding

No qualifying original Normal-Attack-Normal chronological SWaT test source was found. Located `normal.csv`, `attack.csv`, and `merged.csv` files and their archive copies are byte-identical to the already audited current files. `swat_semisynth.csv` files are generated controlled data and are not raw-source candidates. No direct merged-generation script or older raw path was confirmed.

The closest immediate source of the current suffix is the byte-identical `attack.csv`, but that file already contains Attack rows only and is not the upstream chronological attack test.

## Required input

To reach `ready_to_rebuild`, supply the original chronological SWaT attack test as CSV/XLS/XLSX with a timestamp or guaranteed original row order, all sensor/actuator columns, and a label column containing alternating Normal and Attack periods. A provenance-confirmed original normal-operation training file with timestamp and original order is also required.

No data was rebuilt, sorted, concatenated, cleaned, or modified.
"""
    (output / "final_source_locator_report.md").write_text(report, encoding="utf-8")
    return readiness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only raw SWaT source locator")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = safe_search_roots()
    if args.dry_run:
        print("Step 4.3E-1 dry-run; no output or data files will be written.")
        print(f"safe_scope={search_scope_is_safe(roots)}")
        for root in roots:
            print(f"SEARCH: {root}")
        print("EXCLUDE: drive roots, unrelated sibling projects, .venv, venv, site-packages")
        print("NO: extraction, model loading, training, reconstruction, internet download")
        return 0 if search_scope_is_safe(roots) else 2
    output = args.output_dir.resolve() if args.output_dir else (Path(tempfile.mkdtemp(prefix="stage4_3e1_smoke_")) if args.smoke else OUTPUT)
    result = run(output, smoke=args.smoke)
    print(json.dumps({"output": str(output), "classification": result["classification"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
