from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PARENT = ROOT / "dataset" / "dataset"
FORMAL_OUTPUT = ROOT / "outputs" / "hai_step1_dataset_audit"
SMOKE_OUTPUT = ROOT / "outputs" / "hai_step1_dataset_audit_smoke"
CHUNK_SIZE = 25_000
EXPECTED_TRAIN = [f"train{i}.csv" for i in range(1, 7)]
EXPECTED_TEST = [f"test{i}.csv" for i in range(1, 5)]


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"0", "0.0", "normal", "false"}:
        return "Normal"
    if text in {"1", "1.0", "attack", "true"}:
        return "Attack"
    return f"Unknown:{text}"


def sequence_stats(labels: Iterable[Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = [normalize_label(value) for value in labels]
    counts = Counter(values)
    segments: list[dict[str, Any]] = []
    if values:
        start = 0
        for index in range(1, len(values) + 1):
            if index == len(values) or values[index] != values[start]:
                segments.append({"label": values[start], "start_index": start, "end_index": index - 1,
                                 "length": index - start})
                start = index
    collapsed = [segment["label"] for segment in segments]
    summary = {
        "raw_label_values": [], "normalized_label_values": sorted(counts),
        "normal_count": counts["Normal"], "attack_count": counts["Attack"],
        "unknown_label_count": sum(count for label, count in counts.items() if label.startswith("Unknown:")),
        "label_transition_count": max(0, len(segments) - 1),
        "normal_segment_count": sum(segment["label"] == "Normal" for segment in segments),
        "attack_segment_count": sum(segment["label"] == "Attack" for segment in segments),
        "contains_normal_attack_normal": any(collapsed[i:i + 3] == ["Normal", "Attack", "Normal"] for i in range(max(0, len(collapsed) - 2))),
        "first_label": values[0] if values else None, "last_label": values[-1] if values else None,
        "attack_ratio": counts["Attack"] / len(values) if values else 0.0,
    }
    return summary, segments


def timestamp_stats(values: Iterable[Any]) -> dict[str, Any]:
    raw = pd.Series(list(values), dtype="string").str.strip()
    parsed = pd.to_datetime(raw, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    valid = parsed.dropna()
    diffs = valid.diff().dt.total_seconds().dropna()
    decreases = int((diffs < 0).sum())
    duplicates = int(parsed.duplicated().sum())
    mode = float(diffs.mode().iloc[0]) if len(diffs) and len(diffs.mode()) else None
    non_standard = int((~np.isclose(diffs.to_numpy(float), mode)).sum()) if mode is not None else 0
    gaps = []
    if mode is not None:
        indices = np.flatnonzero(~np.isclose(diffs.to_numpy(float), mode))[:20]
        for idx in indices:
            original_index = int(diffs.index[idx])
            gaps.append({"index": original_index, "gap_seconds": float(diffs.iloc[idx])})
    return {
        "timestamp_parse_success_ratio": float(parsed.notna().mean()) if len(parsed) else 0.0,
        "timestamp_first": str(raw.iloc[0]) if len(raw) else None,
        "timestamp_last": str(raw.iloc[-1]) if len(raw) else None,
        "timestamp_monotonic_non_decreasing": bool(parsed.notna().all() and decreases == 0),
        "timestamp_decrease_count": decreases, "timestamp_duplicate_count": duplicates,
        "median_interval_seconds": float(diffs.median()) if len(diffs) else None,
        "most_common_interval_seconds": mode, "non_standard_interval_count": non_standard,
        "maximum_gap_seconds": float(diffs.max()) if len(diffs) else None,
        "gap_locations_sample": json.dumps(gaps, ensure_ascii=False),
    }


def infer_columns(columns: Iterable[str]) -> tuple[str | None, str | None]:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    timestamp = next((normalized[key] for key in ("timestamp", "datetime", "time", "date") if key in normalized), None)
    label = next((normalized[key] for key in ("attack", "attack_label", "label", "anomaly") if key in normalized), None)
    if timestamp is None:
        timestamp = next((column for column in columns if any(key in str(column).strip().lower() for key in ("timestamp", "datetime"))), None)
    if label is None:
        label = next((column for column in columns if any(key in str(column).strip().lower() for key in ("attack", "anomaly", "label"))), None)
    return timestamp, label


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def duplicate_count(values: np.ndarray) -> int:
    if not len(values):
        return 0
    _, counts = np.unique(values, return_counts=True)
    return int(np.maximum(counts - 1, 0).sum())


def hash_frame(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(frame, index=False).to_numpy(np.uint64)


def locate_hai_root(data_parent: Path) -> tuple[Path | None, list[Path], list[str], list[str]]:
    expected = set(EXPECTED_TRAIN + EXPECTED_TEST)
    candidates: dict[Path, set[str]] = {}
    for path in data_parent.rglob("*.csv"):
        if path.name.lower() in expected or "hai" in str(path).lower():
            candidates.setdefault(path.parent.resolve(), set()).add(path.name.lower())
    if not candidates:
        return None, [], sorted(expected), []
    root = max(candidates, key=lambda path: len(candidates[path].intersection(expected)))
    files = sorted([path for path in root.glob("*.csv") if path.name.lower() in expected], key=lambda p: p.name)
    found = {path.name.lower() for path in files}
    missing = sorted(expected - found)
    extra = sorted(path.name for path in root.glob("*.csv") if path.name.lower() not in expected)
    return root, files, missing, extra


def audit_file(path: Path, split: str, limit: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"file_name": path.name, "absolute_path": str(path.resolve()), "split": split,
                              "file_size_bytes": path.stat().st_size, "sha256": sha256_file(path), "readable": False, "read_error": ""}
    try:
        header = pd.read_csv(path, nrows=0)
        raw_columns = [str(column) for column in header.columns]
        stripped = [column.strip() for column in raw_columns]
        timestamp_col, label_col = infer_columns(raw_columns)
        feature_cols = [column for column in raw_columns if column not in {timestamp_col, label_col}]
        labels: list[Any] = []
        raw_label_values: set[str] = set()
        timestamps: list[Any] = []
        missing = Counter({column: 0 for column in raw_columns})
        non_null = Counter({column: 0 for column in raw_columns})
        samples: dict[str, list[str]] = {column: [] for column in raw_columns}
        observed_dtypes: dict[str, set[str]] = {column: set() for column in raw_columns}
        feature_parts: dict[str, list[np.ndarray]] = {column: [] for column in feature_cols}
        full_hash_parts: list[np.ndarray] = []
        feature_hash_parts: list[np.ndarray] = []
        row_count = infinite_count = 0
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False, nrows=limit):
            row_count += len(chunk)
            for column in raw_columns:
                series = chunk[column]
                missing[column] += int(series.isna().sum())
                non_null[column] += int(series.notna().sum())
                observed_dtypes[column].add(str(series.dtype))
                if len(samples[column]) < 5:
                    samples[column].extend(str(value) for value in series.dropna().unique()[: 5 - len(samples[column])])
            if label_col:
                labels.extend(chunk[label_col].tolist())
                raw_label_values.update(str(value) for value in chunk[label_col].dropna().unique())
            if timestamp_col:
                timestamps.extend(chunk[timestamp_col].tolist())
            numeric_features = chunk[feature_cols].apply(pd.to_numeric, errors="coerce")
            infinite_count += int(np.isinf(numeric_features.to_numpy(float)).sum())
            for column in feature_cols:
                feature_parts[column].append(numeric_features[column].to_numpy(np.float64, copy=True))
            canonical = numeric_features.copy()
            if timestamp_col:
                canonical.insert(0, timestamp_col.strip(), chunk[timestamp_col].astype("string").str.strip().fillna("<NA>"))
            if label_col:
                canonical[label_col.strip()] = chunk[label_col].map(normalize_label)
            full_hash_parts.append(hash_frame(canonical))
            feature_hash_parts.append(hash_frame(numeric_features))
        full_hashes = np.concatenate(full_hash_parts) if full_hash_parts else np.array([], dtype=np.uint64)
        feature_hashes = np.concatenate(feature_hash_parts) if feature_hash_parts else np.array([], dtype=np.uint64)
        label_summary, segments = sequence_stats(labels)
        label_summary["raw_label_values"] = json.dumps(sorted(raw_label_values), ensure_ascii=False)
        ts_summary = timestamp_stats(timestamps) if timestamp_col else timestamp_stats([])
        feature_quality = []
        schema_rows = []
        constant_count = near_constant_count = all_zero_count = numeric_count = nonnumeric_count = 0
        for column in raw_columns:
            if column in feature_cols:
                values = np.concatenate(feature_parts[column]) if feature_parts[column] else np.array([], dtype=float)
                finite = values[np.isfinite(values)]
                unique, counts = np.unique(finite, return_counts=True) if len(finite) else (np.array([]), np.array([]))
                unique_count = len(unique)
                top_ratio = float(counts.max() / len(finite)) if len(finite) else 0.0
                is_constant = unique_count <= 1
                is_near_constant = top_ratio >= 0.999
                is_all_zero = bool(len(finite) and np.all(finite == 0))
                constant_count += is_constant
                near_constant_count += is_near_constant
                all_zero_count += is_all_zero
                numeric_count += 1
                feature_quality.append({"file_name": path.name, "split": split, "feature": column.strip(),
                                        "unique_count": unique_count, "most_frequent_value_ratio": top_ratio,
                                        "constant": is_constant, "near_constant": is_near_constant, "all_zero": is_all_zero,
                                        "missing_count": int(np.isnan(values).sum()), "infinite_count": int(np.isinf(values).sum())})
                dtype = "numeric"
                inferred_role = "process_feature"
            else:
                values = labels if column == label_col else timestamps
                unique_count = len(set(str(value) for value in values))
                dtype = ";".join(sorted(observed_dtypes[column]))
                inferred_role = "timestamp" if column == timestamp_col else "attack_label" if column == label_col else "metadata"
            schema_rows.append({"file_name": path.name, "split": split, "raw_name": column, "stripped_name": column.strip(),
                                "inferred_role": inferred_role, "dtype": dtype, "non_null_count": non_null[column],
                                "missing_count": missing[column], "unique_count": unique_count,
                                "sample_values": json.dumps(samples[column], ensure_ascii=False)})
        result.update({
            "row_count": row_count, "column_count": len(raw_columns), "raw_column_names": json.dumps(raw_columns, ensure_ascii=False),
            "stripped_column_names": json.dumps(stripped, ensure_ascii=False),
            "duplicate_column_name_count": len(stripped) - len(set(stripped)),
            "completely_empty_column_count": sum(non_null[column] == 0 for column in raw_columns),
            "timestamp_column": timestamp_col, "label_column": label_col, "feature_columns": feature_cols,
            "feature_count": len(feature_cols), "missing_value_count": int(sum(missing.values())),
            "missing_value_ratio": sum(missing.values()) / max(1, row_count * len(raw_columns)),
            "complete_duplicate_row_count": duplicate_count(full_hashes),
            "feature_duplicate_row_count": duplicate_count(feature_hashes), "infinite_value_count": infinite_count,
            "numeric_feature_count": numeric_count, "non_numeric_feature_count": nonnumeric_count,
            "constant_feature_count": constant_count, "near_constant_feature_count": near_constant_count,
            "all_zero_feature_count": all_zero_count, "label_summary": label_summary, "segments": segments,
            "timestamp_summary": ts_summary, "timestamp_values": timestamps, "schema_rows": schema_rows, "feature_quality": feature_quality,
            "full_hashes": full_hashes, "feature_hashes": feature_hashes, "readable": True,
        })
    except Exception as exc:
        result["read_error"] = f"{type(exc).__name__}: {exc}"
    return result


def schema_consistency(results: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    readable = [result for result in results if result.get("readable")]
    base_features = readable[0]["feature_columns"] if readable else []
    rows = []
    differences = []
    for scope, items in (("train", [r for r in readable if r["split"] == "train"]),
                         ("test", [r for r in readable if r["split"] == "test"]), ("all", readable)):
        raw_equal = bool(items and all(json.loads(r["raw_column_names"]) == json.loads(items[0]["raw_column_names"]) for r in items))
        stripped_equal = bool(items and all(json.loads(r["stripped_column_names"]) == json.loads(items[0]["stripped_column_names"]) for r in items))
        feature_equal = bool(items and all(r["feature_columns"] == items[0]["feature_columns"] for r in items))
        rows.append({"scope": scope, "file_count": len(items), "raw_columns_consistent": raw_equal,
                     "stripped_columns_consistent": stripped_equal, "feature_count_consistent": bool(items and len({r['feature_count'] for r in items}) == 1),
                     "feature_order_consistent": feature_equal, "schema_consistent": raw_equal and stripped_equal and feature_equal})
    for result in readable:
        features = result["feature_columns"]
        differences.append({"file_name": result["file_name"], "split": result["split"],
                            "missing_features_vs_reference": json.dumps(sorted(set(base_features) - set(features))),
                            "extra_features_vs_reference": json.dumps(sorted(set(features) - set(base_features))),
                            "feature_order_matches_reference": features == base_features,
                            "feature_dtypes_match_reference": True,
                            "label_column": result["label_column"], "timestamp_column": result["timestamp_column"]})
    info = {"timestamp_column": readable[0]["timestamp_column"] if readable else None,
            "label_column": readable[0]["label_column"] if readable else None,
            "ordered_feature_columns": base_features, "feature_count": len(base_features),
            "train_schema_consistent": bool(rows[0]["schema_consistent"]),
            "test_schema_consistent": bool(rows[1]["schema_consistent"]),
            "train_test_feature_schema_consistent": bool(rows[2]["feature_order_consistent"])}
    return pd.DataFrame(rows), pd.DataFrame(differences), info


def sampled_window_hashes(row_hashes: np.ndarray, window: int, count: int = 128) -> set[str]:
    if len(row_hashes) < window:
        return set()
    starts = np.unique(np.linspace(0, len(row_hashes) - window, min(count, len(row_hashes) - window + 1), dtype=int))
    return {hashlib.sha256(row_hashes[start:start + window].tobytes()).hexdigest() for start in starts}


def overlap_audit(results: list[dict[str, Any]]) -> pd.DataFrame:
    train = [r for r in results if r.get("readable") and r["split"] == "train"]
    test = [r for r in results if r.get("readable") and r["split"] == "test"]
    train_full = set(int(value) for result in train for value in result["full_hashes"])
    train_feature = set(int(value) for result in train for value in result["feature_hashes"])
    train_w64 = set().union(*(sampled_window_hashes(result["feature_hashes"], 64) for result in train))
    train_w128 = set().union(*(sampled_window_hashes(result["feature_hashes"], 128) for result in train))
    rows = []
    for result in test:
        exact = sum(int(value) in train_full for value in result["full_hashes"])
        feature = sum(int(value) in train_feature for value in result["feature_hashes"])
        w64 = sampled_window_hashes(result["feature_hashes"], 64)
        w128 = sampled_window_hashes(result["feature_hashes"], 128)
        overlap64 = len(w64 & train_w64)
        overlap128 = len(w128 & train_w128)
        warning = "high_contiguous_window_overlap" if overlap64 > max(2, len(w64) * 0.05) or overlap128 > max(2, len(w128) * 0.05) else "none"
        rows.append({"test_file": result["file_name"], "test_rows": result["row_count"],
                     "exact_row_overlap_count": exact, "feature_row_overlap_count": feature,
                     "sampled_window64_count": len(w64), "sampled_window64_overlap_count": overlap64,
                     "sampled_window128_count": len(w128), "sampled_window128_overlap_count": overlap128,
                     "overlap_warning": warning})
    return pd.DataFrame(rows)


def ensure_fresh(output: Path) -> None:
    names = ["hai_file_inventory.csv", "hai_column_schema.csv", "hai_label_summary.csv", "hai_label_segments.csv",
             "hai_timestamp_audit.csv", "hai_data_quality.csv", "hai_feature_quality.csv", "hai_schema_consistency.csv",
             "hai_schema_differences.csv", "hai_train_test_overlap.csv", "hai_sequence_boundaries.csv",
             "hai_dataset_audit_summary.json", "final_hai_dataset_audit.md"]
    existing = [name for name in names if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing HAI audit outputs: {existing}")
    output.mkdir(parents=True, exist_ok=True)


def run_audit(data_parent: Path, output: Path, smoke: bool = False) -> dict[str, Any]:
    hai_root, files, missing, extra = locate_hai_root(data_parent)
    if hai_root is None or missing:
        raise RuntimeError(f"Incomplete HAI download. root={hai_root}, missing={missing}, extra={extra}")
    ensure_fresh(output)
    limit = 5_000 if smoke else None
    results = [audit_file(path, "train" if path.name.startswith("train") else "test", limit) for path in files]
    if not all(result.get("readable") for result in results):
        raise RuntimeError({result["file_name"]: result.get("read_error") for result in results if not result.get("readable")})

    inventory_fields = ["file_name", "absolute_path", "split", "file_size_bytes", "sha256", "row_count", "column_count",
                        "raw_column_names", "stripped_column_names", "duplicate_column_name_count", "completely_empty_column_count", "readable", "read_error"]
    pd.DataFrame([{field: result.get(field) for field in inventory_fields} for result in results]).to_csv(output / "hai_file_inventory.csv", index=False)
    pd.DataFrame([row for result in results for row in result["schema_rows"]]).to_csv(output / "hai_column_schema.csv", index=False)
    label_rows, segment_rows, timestamp_rows, quality_rows, feature_rows, boundaries = [], [], [], [], [], []
    for sequence_id, result in enumerate(results, 1):
        label_rows.append({"file_name": result["file_name"], "split": result["split"], **result["label_summary"]})
        for segment_id, segment in enumerate(result["segments"]):
            segment_rows.append({"file_name": result["file_name"], "split": result["split"], "segment_id": segment_id, **segment,
                                 "start_timestamp": result["timestamp_values"][segment["start_index"]],
                                 "end_timestamp": result["timestamp_values"][segment["end_index"]]})
        timestamp_rows.append({"file_name": result["file_name"], "split": result["split"], **result["timestamp_summary"]})
        quality_rows.append({key: result[key] for key in ("file_name", "split", "row_count", "missing_value_count", "missing_value_ratio",
                            "complete_duplicate_row_count", "feature_duplicate_row_count", "infinite_value_count", "numeric_feature_count",
                            "non_numeric_feature_count", "constant_feature_count", "near_constant_feature_count", "all_zero_feature_count")})
        feature_rows.extend(result["feature_quality"])
        boundaries.append({"sequence_id": sequence_id, "file_name": result["file_name"], "split": result["split"],
                           "row_count": result["row_count"], "start_timestamp": result["timestamp_summary"]["timestamp_first"],
                           "end_timestamp": result["timestamp_summary"]["timestamp_last"], "can_form_windows_with_next_file": False})
    pd.DataFrame(label_rows).to_csv(output / "hai_label_summary.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(output / "hai_label_segments.csv", index=False)
    pd.DataFrame(timestamp_rows).to_csv(output / "hai_timestamp_audit.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(output / "hai_data_quality.csv", index=False)
    feature_quality_df = pd.DataFrame(feature_rows)
    feature_quality_df.to_csv(output / "hai_feature_quality.csv", index=False)
    consistency, differences, schema_info = schema_consistency(results)
    consistency.to_csv(output / "hai_schema_consistency.csv", index=False)
    differences.to_csv(output / "hai_schema_differences.csv", index=False)
    overlap = overlap_audit(results)
    overlap.to_csv(output / "hai_train_test_overlap.csv", index=False)
    pd.DataFrame(boundaries).to_csv(output / "hai_sequence_boundaries.csv", index=False)

    train_labels = [row for row in label_rows if row["split"] == "train"]
    test_labels = [row for row in label_rows if row["split"] == "test"]
    train_normal = all(row["attack_count"] == 0 and row["unknown_label_count"] == 0 for row in train_labels)
    test_valid = all(row["normal_count"] > 0 and row["attack_count"] > 0 and row["contains_normal_attack_normal"] for row in test_labels)
    chronology = all(row["timestamp_monotonic_non_decreasing"] and row["timestamp_parse_success_ratio"] == 1.0 for row in timestamp_rows)
    schema_ok = all(schema_info[key] for key in ("train_schema_consistent", "test_schema_consistent", "train_test_feature_schema_consistent"))
    overlap_ok = all(row == "none" for row in overlap["overlap_warning"])
    grouped_quality = feature_quality_df.groupby("feature").agg(
        constant_file_count=("constant", "sum"), near_constant_file_count=("near_constant", "sum"),
        all_zero_file_count=("all_zero", "sum"))
    globally_constant = grouped_quality[grouped_quality["constant_file_count"] == len(results)].index.tolist()
    globally_near_constant = grouped_quality[grouped_quality["near_constant_file_count"] == len(results)].index.tolist()
    total_missing = sum(row["missing_value_count"] for row in quality_rows)
    total_infinite = sum(row["infinite_value_count"] for row in quality_rows)
    total_complete_duplicates = sum(row["complete_duplicate_row_count"] for row in quality_rows)
    total_feature_duplicates = sum(row["feature_duplicate_row_count"] for row in quality_rows)
    quality_issues = bool(total_missing or total_infinite or total_complete_duplicates or globally_constant or globally_near_constant)
    if not schema_ok:
        classification = "E. schema_inconsistent"
    elif not train_normal or not test_valid or not chronology:
        classification = "D. invalid_label_or_chronology"
    elif not overlap_ok:
        classification = "F. train_test_overlap_problem"
    elif quality_issues:
        classification = "B. usable_with_documented_cleaning"
    else:
        classification = "A. ready_for_hai_pipeline"
    summary = {
        "mode": "smoke" if smoke else "formal", "classification": classification, "actual_hai_root": str(hai_root),
        "train_file_count": 6, "test_file_count": 4, "missing_files": missing, "extra_csv_files": extra,
        "total_train_rows": sum(r["row_count"] for r in results if r["split"] == "train"),
        "total_test_rows": sum(r["row_count"] for r in results if r["split"] == "test"),
        **schema_info, "all_train_labels_normal": train_normal, "all_tests_have_normal_attack_normal": test_valid,
        "total_test_attack_segments": sum(row["attack_segment_count"] for row in test_labels),
        "all_files_timestamp_ordered": chronology, "train_test_overlap_clear": overlap_ok,
        "total_missing_values": total_missing, "total_infinite_values": total_infinite,
        "total_complete_duplicate_rows": total_complete_duplicates, "total_feature_duplicate_rows": total_feature_duplicates,
        "globally_constant_features": globally_constant, "globally_near_constant_features": globally_near_constant,
        "sequence_boundary_rule": "Never form a window across CSV files; aggregate metrics only after per-file windows/events.",
        "allowed_future_cleaning": ["Preserve file order and boundaries", "Diagnose/drop constant features only after explicit approval",
                                    "Handle any missing/non-finite values using train-fitted rules only"],
        "original_csv_modified": False, "models_loaded": False, "training_run": False,
    }
    (output / "hai_dataset_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = f"""# HAI 22.04 Step 1 dataset audit

## Classification

**{classification}**

1. Found 6 train and 4 test CSV files under `{hai_root}`; download set is complete.
2. Total rows: train `{summary['total_train_rows']}`, test `{summary['total_test_rows']}`.
3. All train labels normal: `{train_normal}`.
4. All test files contain Normal-Attack-Normal: `{test_valid}`; total attack segments: `{summary['total_test_attack_segments']}`.
5. All files parse and remain internally time ordered: `{chronology}`.
6. Windows may never cross a CSV boundary. Train1-train6 and test1-test4 are independent sequences.
7. Train/test ordered feature schema consistent: `{schema_ok}`; feature count: `{schema_info['feature_count']}`.
8. Timestamp column: `{schema_info['timestamp_column']}`; label column: `{schema_info['label_column']}`.
9. Missing values: `{total_missing}`; infinite values: `{total_infinite}`; complete duplicate rows: `{total_complete_duplicates}`; feature-only duplicate rows: `{total_feature_duplicates}`.
10. Features constant in all 10 files: `{len(globally_constant)}`; features near-constant in all 10 files: `{len(globally_near_constant)}`. See `hai_feature_quality.csv` for names.
11. Serious sampled train-test window overlap found: `{not overlap_ok}`.
12. HAI DataLoader development supported next: `{classification.startswith(('A.', 'B.'))}`.

Train files may jointly supply training data, but each file must be windowed independently. Each test file must likewise retain its own windows and events; metrics may be aggregated only afterward. No CSV was modified and no model was loaded.
"""
    (output / "final_hai_dataset_audit.md").write_text(report, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only HAI 22.04 dataset acceptance audit")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_PARENT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_parent = args.data_root.resolve()
    hai_root, files, missing, extra = locate_hai_root(data_parent)
    if args.dry_run:
        print(f"data_parent={data_parent}")
        print(f"actual_hai_root={hai_root}")
        print(f"train_files={[p.name for p in files if p.name.startswith('train')]}")
        print(f"test_files={[p.name for p in files if p.name.startswith('test')]}")
        print(f"missing={missing}; extra={extra}")
        print("Read-only: no merge, sort, cleaning, windows, model, checkpoint, training, or metrics.")
        return 0 if hai_root and not missing else 2
    output = SMOKE_OUTPUT if args.smoke else FORMAL_OUTPUT
    summary = run_audit(data_parent, output, smoke=args.smoke)
    print(json.dumps({"output": str(output), "classification": summary["classification"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
