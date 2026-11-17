"""One-time deterministic XLSX-to-NumPy cache conversion for Sprint S1."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat2015_official import (  # noqa: E402
    CACHE_FORMAT_VERSION,
    EXPECTED_ATTACK_ROWS,
    EXPECTED_FEATURES,
    EXPECTED_NORMAL_ROWS,
    OfficialSWaTError,
    canonicalize_point_label,
    load_cache,
    protocol_fingerprint,
    reject_legacy_path,
    resolve_repo_path,
    validate_config_contract,
)
from utils.config_io import load_config  # noqa: E402


SCRIPT_VERSION = "s1-cache-v1"
CACHE_FILES = (
    "normal_v1_features.npy",
    "attack_v0_features.npy",
    "attack_v0_labels.npy",
    "attack_v0_timestamps.npy",
    "cache_manifest.json",
)


def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if not isinstance(value, str):
        raise OfficialSWaTError(f"timestamp is not text/datetime: {value!r}")
    raw = " ".join(value.strip().split())
    for fmt in ("%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    raise OfficialSWaTError(f"unparseable timestamp: {value!r}")


def timestamp_int64(value: Any) -> np.int64:
    parsed = parse_timestamp(value)
    return np.int64(np.datetime64(parsed, "s").astype(np.int64))


def _find_header(sheet: Any) -> tuple[list[Any], Any]:
    rows = sheet.iter_rows(values_only=True)
    for row_number, row in enumerate(rows, start=1):
        names = {str(value).strip() for value in row if value is not None}
        if "Timestamp" in names and "Normal/Attack" in names:
            return list(row), rows
        if row_number >= 10:
            break
    raise OfficialSWaTError("Physical workbook header not found in first 10 rows")


def convert_workbook(
    source: Path,
    feature_output: Path,
    expected_rows: int,
    *,
    label_output: Path | None = None,
    timestamp_output: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    try:
        import openpyxl  # type: ignore
    except ImportError as exc:
        raise OfficialSWaTError("cache conversion requires the approved read-only openpyxl runtime") from exc

    reject_legacy_path(source)
    before = source.stat()
    workbook = openpyxl.load_workbook(source, read_only=True, data_only=True)
    if len(workbook.worksheets) != 1:
        workbook.close()
        raise OfficialSWaTError(f"expected one worksheet: {source}")
    sheet = workbook.active
    headers, rows = _find_header(sheet)
    stripped = ["" if value is None else str(value).strip() for value in headers]
    try:
        timestamp_index = stripped.index("Timestamp")
        label_index = stripped.index("Normal/Attack")
    except ValueError as exc:
        workbook.close()
        raise OfficialSWaTError("required Physical columns are missing") from exc
    feature_indices = [i for i in range(len(headers)) if i not in {timestamp_index, label_index}]
    feature_names = [stripped[i].replace(" ", "") for i in feature_indices]
    if len(headers) != 53 or len(feature_names) != EXPECTED_FEATURES or len(set(feature_names)) != EXPECTED_FEATURES:
        workbook.close()
        raise OfficialSWaTError("Physical workbook does not have 51 unique input variables")

    feature_array = np.lib.format.open_memmap(
        feature_output, mode="w+", dtype=np.float32, shape=(expected_rows, EXPECTED_FEATURES)
    )
    label_array = (
        np.lib.format.open_memmap(label_output, mode="w+", dtype=np.int8, shape=(expected_rows,))
        if label_output is not None else None
    )
    timestamp_array = (
        np.lib.format.open_memmap(timestamp_output, mode="w+", dtype=np.int64, shape=(expected_rows,))
        if timestamp_output is not None else None
    )
    raw_label_counts: dict[str, int] = {}
    row_count = 0
    try:
        for row in rows:
            if all(value is None or (isinstance(value, str) and not value.strip()) for value in row):
                continue
            if row_count >= expected_rows:
                raise OfficialSWaTError(f"workbook contains more than {expected_rows} data rows: {source.name}")
            values = []
            for col_index in feature_indices:
                value = row[col_index] if col_index < len(row) else None
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise OfficialSWaTError(
                        f"nonnumeric Physical value at data row {row_count}, column {feature_names[len(values)]}: {value!r}"
                    )
                numeric = float(value)
                if not np.isfinite(numeric):
                    raise OfficialSWaTError(
                        f"non-finite Physical value at data row {row_count}, column {feature_names[len(values)]}"
                    )
                values.append(numeric)
            feature_array[row_count] = values
            raw_label = row[label_index]
            canonical = canonicalize_point_label(raw_label)
            raw_label_counts[str(raw_label)] = raw_label_counts.get(str(raw_label), 0) + 1
            if label_array is None:
                if canonical != 0:
                    raise OfficialSWaTError("Normal v1 contains a non-Normal label")
            else:
                label_array[row_count] = canonical
            if timestamp_array is not None:
                timestamp_array[row_count] = timestamp_int64(row[timestamp_index])
            row_count += 1
    finally:
        workbook.close()
        feature_array.flush()
        if label_array is not None:
            label_array.flush()
        if timestamp_array is not None:
            timestamp_array.flush()
    after = source.stat()
    if row_count != expected_rows:
        raise OfficialSWaTError(f"row count mismatch for {source.name}: {row_count} != {expected_rows}")
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OfficialSWaTError(f"raw workbook changed during read-only conversion: {source}")
    return feature_names, {
        "source_path": str(source),
        "source_size_bytes": before.st_size,
        "source_mtime_ns_before": before.st_mtime_ns,
        "source_mtime_ns_after": after.st_mtime_ns,
        "row_count": row_count,
        "column_count": EXPECTED_FEATURES,
        "feature_dtype": "float32",
        "raw_label_counts": raw_label_counts,
        "source_unchanged_by_size_and_mtime": True,
    }


def expected_source_profiles(t2_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {profile["file_name"]: profile for profile in t2_audit["physical_files"]}


def reuse_if_valid(cache_dir: Path, config: dict[str, Any], t2_audit: dict[str, Any]) -> bool:
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profiles = expected_source_profiles(t2_audit)
    expected_hashes = {
        name: profiles[name]["sha256"]
        for name in ("SWaT_Dataset_Normal_v1.xlsx", "SWaT_Dataset_Attack_v0.xlsx")
    }
    if manifest.get("known_t2_source_sha256") != expected_hashes:
        return False
    load_cache(cache_dir, config)
    return True


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(str(config_path))
    validate_config_contract(config)
    official_root = resolve_repo_path(ROOT, config["paths"]["official_root"])
    reject_legacy_path(official_root)
    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    t2_path = resolve_repo_path(ROOT, config["paths"]["t2_audit_json"])
    point_policy_path = resolve_repo_path(ROOT, config["paths"]["t25_point_policy"])
    segment_path = resolve_repo_path(ROOT, config["paths"]["t25_segment_manifest"])
    t2_audit = json.loads(t2_path.read_text(encoding="utf-8"))
    point_policy = json.loads(point_policy_path.read_text(encoding="utf-8"))
    segment_manifest = json.loads(segment_path.read_text(encoding="utf-8"))
    if point_policy["mapping_mode"] != "exact_only_no_strip_no_casefold_no_fallback":
        raise OfficialSWaTError("T2.5 exact label policy is missing")
    if reuse_if_valid(cache_dir, config, t2_audit):
        result = {"status": "reused", "cache_dir": str(cache_dir), "training": False}
        print(json.dumps(result, indent=2))
        return result

    existing = [name for name in CACHE_FILES if (cache_dir / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite incomplete or mismatched cache: {existing}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    profiles = expected_source_profiles(t2_audit)
    normal_source = official_root / config["data"]["normal_file"]
    attack_source = official_root / config["data"]["attack_file"]
    normal_profile = profiles[normal_source.name]
    attack_profile = profiles[attack_source.name]
    if normal_source.stat().st_size != normal_profile["size_bytes"] or attack_source.stat().st_size != attack_profile["size_bytes"]:
        raise OfficialSWaTError("raw source size differs from T2; full SHA is intentionally not recomputed")

    normal_columns, normal_meta = convert_workbook(
        normal_source, cache_dir / "normal_v1_features.npy", EXPECTED_NORMAL_ROWS
    )
    attack_columns, attack_meta = convert_workbook(
        attack_source,
        cache_dir / "attack_v0_features.npy",
        EXPECTED_ATTACK_ROWS,
        label_output=cache_dir / "attack_v0_labels.npy",
        timestamp_output=cache_dir / "attack_v0_timestamps.npy",
    )
    if normal_columns != attack_columns:
        raise OfficialSWaTError("Normal and Attack column order differs")
    if attack_meta["raw_label_counts"] != attack_profile["raw_label_counts"]:
        raise OfficialSWaTError("cache label counts differ from T2")
    manifest = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "creation_script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_fingerprint": protocol_fingerprint(config),
        "column_order": normal_columns,
        "label_mapping": config["data"]["point_label_mapping"],
        "unknown_label_policy": "hard_error",
        "timestamp_representation": "int64 seconds from naive np.datetime64; source timezone remains unspecified",
        "known_t2_source_sha256": {
            normal_source.name: normal_profile["sha256"],
            attack_source.name: attack_profile["sha256"],
        },
        "raw_sources": {"normal": normal_meta, "attack": attack_meta},
        "arrays": {
            "normal_v1_features.npy": {"shape": [EXPECTED_NORMAL_ROWS, EXPECTED_FEATURES], "dtype": "float32"},
            "attack_v0_features.npy": {"shape": [EXPECTED_ATTACK_ROWS, EXPECTED_FEATURES], "dtype": "float32"},
            "attack_v0_labels.npy": {"shape": [EXPECTED_ATTACK_ROWS], "dtype": "int8"},
            "attack_v0_timestamps.npy": {"shape": [EXPECTED_ATTACK_ROWS], "dtype": "int64"},
        },
        "test_segment_boundaries": segment_manifest["test_segments"],
        "raw_xlsx_modified": False,
        "full_source_sha_recomputed": False,
        "training": False,
    }
    (cache_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    load_cache(cache_dir, config)
    result = {"status": "created", "cache_dir": str(cache_dir), "manifest": manifest, "training": False}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(resolve_repo_path(ROOT, args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
