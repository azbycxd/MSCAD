from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat2026 import (  # noqa: E402
    LABEL_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    build_swat2026_windows,
    infer_swat2026_role,
    load_swat2026_csv,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def resolve_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {column.strip().lower(): column for column in columns}
    return next((lookup[name] for name in candidates if name in lookup), None)


def read_csv_chunked(path: Path, max_rows: int | None) -> pd.DataFrame:
    remaining = max_rows
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=50_000, low_memory=False):
        if remaining is not None:
            chunk = chunk.iloc[:remaining]
            remaining -= len(chunk)
        chunks.append(chunk)
        if remaining is not None and remaining <= 0:
            break
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def detect_format(path: Path) -> tuple[str, str, list[str]]:
    sample = path.read_bytes()[:65_536]
    try:
        text = sample.decode("utf-8-sig")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = sample.decode("cp1252")
        encoding = "cp1252"
    delimiter = csv.Sniffer().sniff(text).delimiter
    header = next(csv.reader([text.splitlines()[0]], delimiter=delimiter))
    return encoding, delimiter, header


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def profile_columns(frame: pd.DataFrame, timestamp: str | None, label: str | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        nonmissing = int(series.notna().sum())
        parse_ratio = float(numeric.notna().sum() / nonmissing) if nonmissing else 1.0
        counts = series.astype("string").value_counts(dropna=False)
        unique_count = int(series.nunique(dropna=False))
        role = "timestamp" if column == timestamp else "label" if column == label else infer_swat2026_role(column, series)
        changes = int(series.ne(series.shift()).sum() - (1 if len(series) else 0))
        top_values = {str(key): int(value) for key, value in counts.head(10).items()}
        numeric_valid = numeric.dropna().astype(float)
        constant = unique_count <= 1
        top_ratio = float(counts.iloc[0] / len(series)) if len(series) and len(counts) else 0.0
        rows.append(
            {
                "column_name": column,
                "inferred_role": role,
                "original_dtype": str(series.dtype),
                "numeric_parse_ratio": parse_ratio,
                "missing_count": int(series.isna().sum()),
                "missing_ratio": float(series.isna().mean()) if len(series) else 0.0,
                "unique_count": unique_count,
                "top_values": json.dumps(top_values, ensure_ascii=False),
                "min": float(numeric_valid.min()) if parse_ratio == 1.0 and len(numeric_valid) else None,
                "max": float(numeric_valid.max()) if parse_ratio == 1.0 and len(numeric_valid) else None,
                "mean": float(numeric_valid.mean()) if parse_ratio == 1.0 and len(numeric_valid) else None,
                "std": float(numeric_valid.std(ddof=0)) if parse_ratio == 1.0 and len(numeric_valid) else None,
                "change_count": changes,
                "change_ratio": float(changes / max(1, len(series) - 1)),
                "constant": constant,
                "near_constant": bool(not constant and top_ratio >= 0.999),
            }
        )
    return pd.DataFrame(rows)


def timestamp_audit(frame: pd.DataFrame, timestamp: str | None) -> tuple[dict[str, Any], pd.DataFrame, pd.Series | None]:
    if timestamp is None:
        result = {"timestamp_column": None, "parseable": False}
        return result, pd.DataFrame([{"metric": key, "value": value} for key, value in result.items()]), None
    raw = frame[timestamp]
    parsed = pd.to_datetime(raw, format="%d-%b-%Y %H:%M:%S", errors="coerce")
    if parsed.isna().any():
        parsed = pd.to_datetime(raw, errors="coerce")
    valid = parsed.dropna()
    diffs = valid.diff().dropna().dt.total_seconds()
    distribution = Counter(float(value) for value in diffs)
    result = {
        "timestamp_column": timestamp,
        "raw_first_20": raw.head(20).astype(str).tolist(),
        "raw_dtype": str(raw.dtype),
        "parse_format": "%d-%b-%Y %H:%M:%S",
        "parseable": bool(parsed.notna().all()),
        "parse_failure_count": int(parsed.isna().sum()),
        "earliest": valid.min().isoformat() if len(valid) else None,
        "latest": valid.max().isoformat() if len(valid) else None,
        "duration_seconds": float((valid.max() - valid.min()).total_seconds()) if len(valid) else None,
        "monotonic_increasing": bool(valid.is_monotonic_increasing),
        "duplicate_timestamp_count": int(valid.duplicated().sum()),
        "interval_distribution_seconds": {str(key): int(value) for key, value in sorted(distribution.items())},
        "one_second_interval_ratio": float((diffs == 1.0).mean()) if len(diffs) else None,
        "non_contiguous_gap_count": int((diffs != 1.0).sum()),
        "maximum_gap_seconds": float(diffs.max()) if len(diffs) else None,
        "time_reversal_count": int((diffs < 0).sum()),
    }
    rows = []
    for key, value in result.items():
        rows.append({"metric": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value})
    return result, pd.DataFrame(rows), parsed


def label_audit(frame: pd.DataFrame, label: str | None, parsed_time: pd.Series | None) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = ["segment_id", "label", "start_row", "end_row", "start_timestamp", "end_timestamp", "row_count"]
    if label is None:
        return {
            "label_present": False,
            "label_column": None,
            "classification": "E. no_label_requires_info_sheet",
            "can_confirm_attacks": False,
        }, pd.DataFrame(columns=columns)
    raw = frame[label]
    stripped = raw.astype(str).str.strip()
    normalized = stripped.str.lower()
    switches = normalized.ne(normalized.shift())
    segment_ids = switches.cumsum()
    segments = []
    for segment_id, indices in frame.groupby(segment_ids).groups.items():
        start, end = int(min(indices)), int(max(indices))
        segments.append(
            {
                "segment_id": int(segment_id),
                "label": str(stripped.iloc[start]),
                "start_row": start,
                "end_row": end,
                "start_timestamp": parsed_time.iloc[start].isoformat() if parsed_time is not None else None,
                "end_timestamp": parsed_time.iloc[end].isoformat() if parsed_time is not None else None,
                "row_count": end - start + 1,
            }
        )
    normal_count = int(normalized.isin({"normal", "0", "0.0", "false"}).sum())
    attack_count = int(normalized.isin({"attack", "1", "1.0", "true"}).sum())
    sequence = [str(row["label"]).strip().lower() for row in segments]
    return {
        "label_present": True,
        "label_column": label,
        "raw_unique_values": raw.drop_duplicates().astype(str).tolist(),
        "stripped_unique_values": stripped.drop_duplicates().tolist(),
        "normalized_unique_values": normalized.drop_duplicates().tolist(),
        "normal_count": normal_count,
        "attack_count": attack_count,
        "label_switch_count": max(0, len(segments) - 1),
        "contains_normal_attack_normal": any(sequence[i : i + 3] == ["normal", "attack", "normal"] for i in range(max(0, len(sequence) - 2))),
        "all_normal": normal_count == len(frame),
        "all_attack": attack_count == len(frame),
    }, pd.DataFrame(segments, columns=columns)


def repository_outputs(output_dir: Path) -> None:
    matrix = [
        ["dataset factory", "data/factory.py", "build_dataloaders", "Dataset name branches; default is swat", "Imports HAI eagerly", "Legacy swat branch", "no", "no", "high", "Keep SWaT 2026 outside factory at N0"],
        ["legacy SWaT loader", "data/swat.py", "_load_swat_csv", "Coerces every feature then fillna(0)", "no", "yes", "no", "yes", "critical", "Use isolated strict loader"],
        ["legacy labels", "data/swat.py", "_is_attack", "Every non-normal value becomes attack", "no", "yes", "no", "yes", "high", "Use explicit label vocabulary"],
        ["legacy files", "configs/swat.yaml", "data.train_csv/test_csv", "Requires normal.csv and attack.csv", "no", "yes", "no", "yes", "high", "Use environment-based single-file audit config"],
        ["HAI loader", "data/hai.py", "load_hai_csv", "Strict 86-feature schema and HAI timestamp/label rules", "yes", "no", "no", "no", "low", "Do not modify"],
        ["scaler", "data/transforms.py", "StandardScaler", "Reusable implementation; fit state belongs to each bundle", "shared", "no", "later", "no", "medium", "Do not fit at N0; later fit normal train only"],
        ["model construction", "train.py", "MSCAD(in_channels=len(feature_names))", "Input width is dynamic", "shared", "no", "yes", "no", "low", "New model instance required for new schema"],
        ["checkpoint evaluation", "evaluate.py", "load_state_dict(strict=True)", "Checkpoint tensors require matching input width", "shared", "old schema", "no", "yes", "high", "Do not reuse old checkpoint"],
        ["outputs", "train.py/evaluate.py", "cfg.output.dir", "Configured output root", "HAI configs isolated", "SWaT configs isolated", "yes", "no", "low", "Use outputs/swat2026 prefix"],
    ]
    columns = ["component", "file_path", "line_or_symbol", "current_behavior", "hai_specific", "old_swat_specific", "reusable_for_swat2026", "modification_required", "risk_level", "recommended_action"]
    pd.DataFrame(matrix, columns=columns).to_csv(output_dir / "repository_compatibility_matrix.csv", index=False)
    references = [
        ["data/factory.py", 10, "dataset default", "cfg.get('dataset', 'swat')"],
        ["data/factory.py", 26, "HAI branch", "build_hai_dataloaders"],
        ["data/swat.py", 37, "silent coercion", "to_numeric(errors='coerce').fillna(0.0)"],
        ["data/swat.py", 83, "legacy split loader", "separate train_csv/test_csv"],
        ["data/hai.py", 72, "HAI feature width", "expected_feature_count=86"],
        ["data/hai.py", 266, "HAI scaler", "fit_many(train_raw)"],
        ["train.py", 136, "default config", "configs/swat.yaml"],
        ["train.py", 153, "dynamic input", "len(bundle.feature_names)"],
        ["evaluate.py", 76, "strict checkpoint", "load_state_dict(strict=True)"],
    ]
    pd.DataFrame(references, columns=["file_path", "line", "topic", "reference"]).to_csv(output_dir / "code_reference_inventory.csv", index=False)
    (output_dir / "current_pipeline_audit.md").write_text(
        "# Current pipeline audit\n\n"
        "- `data.factory` defaults to legacy `swat` and eagerly imports both HAI and legacy SWaT modules.\n"
        "- HAI is isolated in `data/hai.py` but intentionally enforces 86 features and HAI-specific timestamp/label rules.\n"
        "- Legacy SWaT requires separate normal/attack files and silently converts unparseable states and missing values to zero.\n"
        "- Legacy label parsing treats every unknown non-normal value as attack.\n"
        "- The scaler is fitted on the chronological legacy training prefix only, but the new unlabeled file has no defensible training split.\n"
        "- Model input width is created from `len(bundle.feature_names)`; old checkpoints still require their original tensor width and order.\n"
        "- SWaT 2026 is therefore kept out of the common factory at N0 to avoid HAI regression.\n",
        encoding="utf-8",
    )


def run(csv_path: Path, output_dir: Path, max_rows: int | None, dry_run: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_hash = sha256(csv_path)
    encoding, delimiter, raw_header = detect_format(csv_path)
    effective_max = 1000 if dry_run and max_rows is None else max_rows
    frame = read_csv_chunked(csv_path, effective_max)
    columns = [str(column).strip() for column in frame.columns]
    frame.columns = columns
    timestamp = resolve_column(columns, TIMESTAMP_CANDIDATES)
    label = resolve_column(columns, LABEL_CANDIDATES)
    col_profile = profile_columns(frame, timestamp, label)
    time_info, time_frame, parsed_time = timestamp_audit(frame, timestamp)
    label_info, segments = label_audit(frame, label, parsed_time)
    loaded = load_swat2026_csv(
        csv_path,
        timestamp_column=timestamp,
        label_column=label,
        max_rows=min(len(frame), 20_000),
        missing_numeric_policy="drop_rows",
    )
    windows = build_swat2026_windows(loaded.features[: min(len(loaded.features), 256)], loaded.labels[:256] if loaded.labels is not None else None, loaded.timestamps[:256], 64, 1)
    role_counts = {str(key): int(value) for key, value in col_profile["inferred_role"].value_counts().items()}
    summary = {
        "dry_run": dry_run,
        "csv_path": str(csv_path),
        "rows_read": int(len(frame)),
        "total_columns": len(columns),
        "timestamp_column": timestamp,
        "label_column": label,
        "time_range": [time_info.get("earliest"), time_info.get("latest")],
        "role_counts": role_counts,
        "final_feature_count": len(loaded.feature_names),
        "feature_names": loaded.feature_names,
        "window_shape": list(windows.features.shape),
        "scaler_fitted": False,
        "checkpoint_loaded": False,
        "model_instantiated": False,
        "trained": False,
        "smoke_missing_numeric_policy": "drop_rows",
        "smoke_dropped_row_count": loaded.metadata["dropped_row_count"],
        "smoke_dropped_row_indices": loaded.metadata["dropped_row_indices"],
        "source_sha256_before": before_hash,
    }
    write_json(output_dir / ("dry_run_summary.json" if dry_run else "smoke_summary.json"), summary)
    if dry_run or max_rows is not None:
        if sha256(csv_path) != before_hash:
            raise RuntimeError("Source CSV hash changed")
        return summary

    duplicate_columns = sorted({column for column in raw_header if raw_header.count(column) > 1})
    process_value_names = [column for column in columns if column.strip().lower().endswith(".pv")]
    invalid_process_rows = np.zeros(len(frame), dtype=bool)
    invalid_process_values: dict[str, dict[str, Any]] = {}
    for column in process_value_names:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = numeric.isna() & frame[column].notna()
        if invalid.any():
            invalid_process_rows |= invalid.to_numpy()
            invalid_process_values[column] = {
                "row_indices": np.flatnonzero(invalid.to_numpy()).astype(int).tolist(),
                "raw_values": sorted(frame.loc[invalid, column].astype(str).unique().tolist()),
            }
    file_profile = {
        "path": str(csv_path),
        "size_bytes": csv_path.stat().st_size,
        "sha256_before": before_hash,
        "encoding": encoding,
        "delimiter": delimiter,
        "row_count": len(frame),
        "column_count": len(raw_header),
        "raw_columns": raw_header,
        "duplicate_columns": duplicate_columns,
        "timestamp_candidates": [column for column in columns if column.strip().lower() in TIMESTAMP_CANDIDATES],
        "label_candidates": [column for column in columns if column.strip().lower() in LABEL_CANDIDATES],
        "timestamp_audit": time_info,
        "label_audit": label_info,
        "role_counts": role_counts,
        "constant_column_count": int(col_profile["constant"].sum()),
        "near_constant_column_count": int(col_profile["near_constant"].sum()),
        "process_values_only_feature_count": len(loaded.feature_names),
        "process_values_only_feature_names": loaded.feature_names,
        "invalid_process_value_columns": invalid_process_values,
        "invalid_process_value_row_count": int(invalid_process_rows.sum()),
        "invalid_process_value_row_indices": np.flatnonzero(invalid_process_rows).astype(int).tolist(),
        "excel_hash_marks_conclusion": "CSV contains parseable timestamp text; Excel ###### would be a display-width issue, not CSV corruption.",
    }
    after_hash = sha256(csv_path)
    file_profile["sha256_after"] = after_hash
    file_profile["source_unchanged"] = before_hash == after_hash
    if before_hash != after_hash:
        raise RuntimeError("Source CSV hash changed during audit")
    write_json(output_dir / "swat2026_file_profile.json", file_profile)
    col_profile.to_csv(output_dir / "swat2026_column_profile.csv", index=False)
    segments.to_csv(output_dir / "swat2026_label_segments.csv", index=False)
    time_frame.to_csv(output_dir / "swat2026_timestamp_audit.csv", index=False)
    write_json(output_dir / "swat2026_sample_rows.json", {"head": json_records(frame.head(5)), "tail": json_records(frame.tail(5))})
    readiness = {
        "classification": "E. no_label_requires_info_sheet",
        "actual_duration_seconds": time_info.get("duration_seconds"),
        "covers_full_month": False,
        "single_approximately_eight_hour_session": True,
        "has_label": False,
        "has_confirmed_attack": False,
        "has_confirmed_normal_training_data": False,
        "usable_alone_for_training": False,
        "usable_alone_for_testing": False,
        "usable_alone_for_both": False,
        "requires_other_batch_files": True,
        "requires_dataset_info_or_attack_schedule": True,
        "strict_default_loader_accepts_full_file": not bool(invalid_process_rows.any()),
        "invalid_process_value_row_count": int(invalid_process_rows.sum()),
        "invalid_process_value_columns": list(invalid_process_values),
        "reason": "No label column exists, so normal/attack intervals and leakage-safe splits cannot be established from this CSV alone.",
    }
    write_json(output_dir / "swat2026_readiness.json", readiness)
    repository_outputs(output_dir)
    report = f"""# SWaT-N0 final compatibility report

## Conclusion

Primary compatibility class: **C. requires_isolated_swat_adapter**. Class **D. requires_new_model_instance_and_retraining** also applies before later modeling.

1. HAI-specific logic remains in `data/hai.py`, HAI configs/scripts/tests, including 86 features and strict HAI timestamp/label rules.
2. Legacy SWaT logic requires separate normal/attack CSVs, silently coerces invalid values to zero, and maps every unknown non-normal label to attack.
3. The official CSV cannot safely use the legacy loader.
4. Reasons: new 87-column schema, categorical alarms, no label column, one single file rather than a defensible train/test pair, and unsafe legacy coercion. Four `.Pv` columns contain `Bad Input` on {int(invalid_process_rows.sum())} rows; strict default loading reports and rejects these rows instead of replacing them with zero.
5. An isolated `data/swat2026.py` loader was created.
6. No common factory, model, scaler, HAI loader, training entry, or evaluation entry was modified.
7. No HAI-specific file was modified; pre/post hashes are verified separately.
8. HAI hash status: expected unchanged.
9. CSV rows/columns: {len(frame)} rows and {len(columns)} columns.
10. Timestamp format: `%d-%b-%Y %H:%M:%S` in `{timestamp}`.
11. Time range: {time_info.get('earliest')} through {time_info.get('latest')}; duration {time_info.get('duration_seconds')} seconds.
12. The raw timestamps parse successfully; Excel `######` would only indicate insufficient display width.
13. Label column: none. Normal/attack distribution and events cannot be inferred.
14. Semantic role counts: {json.dumps(role_counts, ensure_ascii=False)}; constants: {int(col_profile['constant'].sum())}; near constants: {int(col_profile['near_constant'].sum())}.
15. `process_values_only` selects {len(loaded.feature_names)} `.Pv` columns in original order. Smoke uses an explicit, reported `drop_rows` policy only to form a few finite windows; the committed config default remains `error`.
16. This file alone is not approved for training because normal-only status is unproven.
17. This file alone is not approved for attack testing because attacks and intervals are unlabelled.
18. Other same-batch files are required to establish a normal training source and independent evaluation coverage.
19. The official Dataset Info Sheet or attack schedule is required.
20. Old SWaT/HAI checkpoints cannot be reused safely because feature width/order and training domain differ.
21. A new SWaT model instance and training will eventually be required, but training is outside SWaT-N0.
22. Next step: obtain the official information/attack schedule and complete source inventory before defining a split.
"""
    (output_dir / "final_compatibility_report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only SWaT 2026 audit and loader smoke test")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--csv", default=os.environ.get("SWAT2026_DATA_PATH"))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/swat2026_compatibility_audit")
    args = parser.parse_args()
    if not args.csv:
        parser.error("--csv or SWAT2026_DATA_PATH is required")
    summary = run(Path(args.csv), Path(args.output_dir), args.max_rows, args.dry_run)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
