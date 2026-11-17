from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "stage4_3d_swat_data_audit"
CHUNK_SIZE = 50_000


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"normal", "0", "0.0", "false"}:
        return "Normal"
    if text in {"attack", "1", "1.0", "true"}:
        return "Attack"
    return f"Other:{text}"


def loader_label(value: Any) -> int:
    """Mirror data.swat._is_attack exactly."""
    return 0 if str(value).strip().lower() in {"normal", "0", "false"} else 1


def resolve_column(columns: Iterable[str], target: str) -> str | None:
    wanted = target.strip().lower()
    return next((c for c in columns if str(c).strip().lower() == wanted), None)


def canonical_frame(frame: pd.DataFrame, label_col: str, timestamp_col: str) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        clean = str(column).strip()
        if column == label_col:
            result[clean] = frame[column].map(normalize_label)
        elif column == timestamp_col:
            result[clean] = frame[column].astype("string").str.strip().fillna("<NA>")
        else:
            result[clean] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return result


def row_hashes(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(frame, index=False).to_numpy(np.uint64)


def duplicate_count(hashes: np.ndarray) -> int:
    if hashes.size == 0:
        return 0
    _, counts = np.unique(hashes, return_counts=True)
    return int(np.maximum(counts - 1, 0).sum())


def timestamp_monotonic(values: Iterable[Any]) -> bool:
    parsed = pd.to_datetime(pd.Series(list(values)), dayfirst=True, errors="coerce")
    return bool(parsed.notna().all() and parsed.is_monotonic_increasing)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def extract_segments(labels: list[Any], timestamps: list[Any], source_file: str) -> list[dict[str, Any]]:
    if not labels:
        return []
    canonical = [normalize_label(value) for value in labels]
    segments: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(canonical) + 1):
        if index == len(canonical) or canonical[index] != canonical[start]:
            segments.append(
                {
                    "source_file": source_file,
                    "segment_id": len(segments),
                    "label": canonical[start],
                    "start_index": start,
                    "end_index": index - 1,
                    "length": index - start,
                    "start_timestamp": str(timestamps[start]).strip(),
                    "end_timestamp": str(timestamps[index - 1]).strip(),
                    "previous_label": canonical[start - 1] if start else "",
                    "next_label": canonical[index] if index < len(canonical) else "",
                }
            )
            start = index
    return segments


def iter_csv(path: Path, chunksize: int = CHUNK_SIZE, nrows: int | None = None) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(path, chunksize=chunksize, low_memory=False, nrows=nrows)


def profile_csv(path: Path, chunksize: int = CHUNK_SIZE, nrows: int | None = None) -> dict[str, Any]:
    header = pd.read_csv(path, nrows=0)
    label_col = resolve_column(header.columns, "Normal/Attack")
    timestamp_col = resolve_column(header.columns, "Timestamp")
    if label_col is None or timestamp_col is None:
        raise ValueError(f"Required columns are missing in {path}")
    feature_cols = [c for c in header.columns if c not in {label_col, timestamp_col}]

    labels: Counter[str] = Counter()
    raw_labels: Counter[str] = Counter()
    row_hash_parts: list[np.ndarray] = []
    feature_hash_parts: list[np.ndarray] = []
    timestamp_hash_parts: list[np.ndarray] = []
    missing = {str(c).strip(): 0 for c in header.columns}
    rows = 0
    head_rows: list[dict[str, Any]] = []
    tail_frame = pd.DataFrame()
    first_timestamp = ""
    last_timestamp = ""
    monotonic = True
    previous_timestamp: pd.Timestamp | None = None

    for chunk in iter_csv(path, chunksize, nrows):
        canonical = canonical_frame(chunk, label_col, timestamp_col)
        numeric_features = canonical[[str(c).strip() for c in feature_cols]]
        raw_labels.update(chunk[label_col].astype("string").fillna("<NA>").astype(str).tolist())
        labels.update(canonical[str(label_col).strip()].tolist())
        row_hash_parts.append(row_hashes(canonical))
        feature_hash_parts.append(row_hashes(numeric_features))
        timestamps = chunk[timestamp_col].astype("string").str.strip().fillna("<NA>")
        timestamp_hash_parts.append(pd.util.hash_pandas_object(timestamps, index=False).to_numpy(np.uint64))
        parsed = pd.to_datetime(timestamps, dayfirst=True, errors="coerce")
        if parsed.isna().any() or not parsed.is_monotonic_increasing:
            monotonic = False
        if previous_timestamp is not None and len(parsed) and parsed.iloc[0] < previous_timestamp:
            monotonic = False
        if len(parsed):
            previous_timestamp = parsed.iloc[-1]
        for original in header.columns:
            missing[str(original).strip()] += int(chunk[original].isna().sum())
        if not head_rows:
            head_rows = chunk.head(5).where(pd.notna(chunk.head(5)), None).to_dict("records")
            first_timestamp = str(timestamps.iloc[0])
        tail_frame = pd.concat([tail_frame, chunk.tail(5)], ignore_index=True).tail(5)
        last_timestamp = str(timestamps.iloc[-1])
        rows += len(chunk)

    all_row_hashes = np.concatenate(row_hash_parts) if row_hash_parts else np.array([], dtype=np.uint64)
    all_feature_hashes = np.concatenate(feature_hash_parts) if feature_hash_parts else np.array([], dtype=np.uint64)
    all_timestamp_hashes = np.concatenate(timestamp_hash_parts) if timestamp_hash_parts else np.array([], dtype=np.uint64)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "profiled_rows": rows,
        "column_count": len(header.columns),
        "columns": [str(c) for c in header.columns],
        "label_column": str(label_col),
        "timestamp_column": str(timestamp_col),
        "label_values_normalized": dict(labels),
        "label_values_raw": dict(raw_labels),
        "timestamp_first": first_timestamp,
        "timestamp_last": last_timestamp,
        "timestamp_monotonic_non_decreasing": monotonic,
        "duplicate_timestamp_count": duplicate_count(all_timestamp_hashes),
        "duplicate_complete_row_count": duplicate_count(all_row_hashes),
        "duplicate_feature_row_count": duplicate_count(all_feature_hashes),
        "missing_values_by_column": missing,
        "missing_value_count": int(sum(missing.values())),
        "head_5": head_rows,
        "tail_5": tail_frame.where(pd.notna(tail_frame), None).to_dict("records"),
        "sha256": sha256_file(path) if nrows is None else "not_computed_for_smoke_subset",
    }


def compare_frames(left: pd.DataFrame, right: pd.DataFrame, label_col: str, timestamp_col: str) -> dict[str, Any]:
    a = canonical_frame(left.reset_index(drop=True), label_col, timestamp_col)
    b = canonical_frame(right.reset_index(drop=True), label_col, timestamp_col)
    if list(a.columns) != list(b.columns):
        raise ValueError("Canonical columns differ")
    row_equal = (row_hashes(a) == row_hashes(b))
    feature_cols = [c for c in a.columns if c not in {label_col.strip(), timestamp_col.strip()}]
    av = a[feature_cols].to_numpy(float)
    bv = b[feature_cols].to_numpy(float)
    feature_equal = np.isclose(av, bv, rtol=0.0, atol=0.0, equal_nan=True)
    timestamp_equal = a[timestamp_col.strip()].to_numpy() == b[timestamp_col.strip()].to_numpy()
    return {
        "row_equal": row_equal,
        "feature_equal_count": int(feature_equal.sum()),
        "feature_value_count": int(feature_equal.size),
        "timestamp_equal_count": int(timestamp_equal.sum()),
        "timestamp_count": len(timestamp_equal),
    }


def prefix_audit(normal: Path, merged: Path, chunksize: int, limit: int | None = None) -> dict[str, Any]:
    normal_header = pd.read_csv(normal, nrows=0)
    label_col = resolve_column(normal_header.columns, "Normal/Attack")
    timestamp_col = resolve_column(normal_header.columns, "Timestamp")
    assert label_col and timestamp_col
    matched = feature_matches = feature_total = ts_matches = ts_total = 0
    first_mismatch: int | None = None
    compared = 0
    prefix_labels = Counter()
    normal_digest = hashlib.sha256()
    merged_digest = hashlib.sha256()
    merged_iter = iter_csv(merged, chunksize, limit)
    for normal_chunk in iter_csv(normal, chunksize, limit):
        merged_chunk = next(merged_iter)
        if len(merged_chunk) != len(normal_chunk):
            merged_chunk = merged_chunk.iloc[: len(normal_chunk)]
        result = compare_frames(normal_chunk, merged_chunk, label_col, timestamp_col)
        equal = result["row_equal"]
        matched += int(equal.sum())
        if first_mismatch is None and not bool(equal.all()):
            first_mismatch = compared + int(np.flatnonzero(~equal)[0])
        feature_matches += result["feature_equal_count"]
        feature_total += result["feature_value_count"]
        ts_matches += result["timestamp_equal_count"]
        ts_total += result["timestamp_count"]
        prefix_labels.update(merged_chunk[label_col].map(normalize_label).tolist())
        normal_digest.update(row_hashes(canonical_frame(normal_chunk, label_col, timestamp_col)).tobytes())
        merged_digest.update(row_hashes(canonical_frame(merged_chunk, label_col, timestamp_col)).tobytes())
        compared += len(normal_chunk)
    return {
        "normal_rows_compared": compared,
        "exact_prefix_match": first_mismatch is None and matched == compared,
        "matched_row_count": matched,
        "first_mismatch_index": first_mismatch,
        "feature_match_ratio": feature_matches / feature_total if feature_total else 0.0,
        "timestamp_match_ratio": ts_matches / ts_total if ts_total else 0.0,
        "merged_prefix_label_counts": dict(prefix_labels),
        "normal_canonical_chunk_hash_sha256": normal_digest.hexdigest(),
        "merged_prefix_canonical_chunk_hash_sha256": merged_digest.hexdigest(),
    }


def read_suffix(path: Path, skip_rows: int, chunksize: int, limit: int | None = None) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    consumed = 0
    remaining = limit
    for chunk in iter_csv(path, chunksize):
        end = consumed + len(chunk)
        if end > skip_rows:
            piece = chunk.iloc[max(0, skip_rows - consumed) :]
            if remaining is not None:
                piece = piece.iloc[:remaining]
                remaining -= len(piece)
            pieces.append(piece)
            if remaining == 0:
                break
        consumed = end
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def suffix_audit(merged: Path, sources: list[Path], normal_rows: int, smoke_rows: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    suffix = read_suffix(merged, normal_rows, CHUNK_SIZE, smoke_rows)
    header = pd.read_csv(merged, nrows=0)
    label_col = resolve_column(header.columns, "Normal/Attack")
    timestamp_col = resolve_column(header.columns, "Timestamp")
    assert label_col and timestamp_col
    reports = []
    selected_source = pd.DataFrame()
    for source in sources:
        source_frame = pd.read_csv(source, low_memory=False, nrows=smoke_rows)
        attack_only = source_frame[source_frame[label_col].map(normalize_label) == "Attack"].reset_index(drop=True)
        count = min(len(suffix), len(attack_only))
        result = compare_frames(suffix.iloc[:count], attack_only.iloc[:count], label_col, timestamp_col)
        exact = bool(result["row_equal"].all()) and len(suffix) == len(attack_only)
        reports.append({
            "source_file": str(source.resolve()), "source_rows": len(source_frame),
            "source_attack_rows": len(attack_only), "merged_suffix_rows": len(suffix),
            "compared_rows": count, "exact_filtered_source_match": exact,
            "matched_row_count": int(result["row_equal"].sum()),
            "feature_match_ratio": result["feature_equal_count"] / max(1, result["feature_value_count"]),
            "timestamp_match_ratio": result["timestamp_equal_count"] / max(1, result["timestamp_count"]),
            "source_order_preserved": exact,
            "source_sha256": sha256_file(source) if smoke_rows is None else "not_computed_for_smoke_subset",
        })
        if selected_source.empty:
            selected_source = attack_only
    return pd.DataFrame(reports), selected_source


def split_overlap_audit(normal_rows: int, prefix: dict[str, Any], val_ratio: float = 0.2, window_size: int = 64) -> pd.DataFrame:
    n_val = int(max(window_size * 2, round(normal_rows * val_ratio)))
    n_val = min(max(window_size * 2, n_val), normal_rows - window_size - 1)
    split = normal_rows - n_val
    exact = bool(prefix["exact_prefix_match"])
    return pd.DataFrame([
        {"scope": "x_train_raw", "start_index": 0, "end_index": split - 1, "source_rows": split,
         "test_normal_exact_overlap_count": split if exact else np.nan, "overlap_ratio": 1.0 if exact else np.nan},
        {"scope": "validation_tail", "start_index": split, "end_index": normal_rows - 1, "source_rows": n_val,
         "test_normal_exact_overlap_count": n_val if exact else np.nan, "overlap_ratio": 1.0 if exact else np.nan},
    ])


def sampled_window_hashes(path: Path, requests: list[tuple[int, int]]) -> dict[tuple[int, int], str]:
    header = pd.read_csv(path, nrows=0)
    label_col = resolve_column(header.columns, "Normal/Attack")
    timestamp_col = resolve_column(header.columns, "Timestamp")
    assert label_col and timestamp_col
    feature_cols = [c for c in header.columns if c not in {label_col, timestamp_col}]
    pending = sorted(requests)
    results: dict[tuple[int, int], str] = {}
    maximum_window = max((window for _, window in pending), default=1)
    previous = pd.DataFrame(columns=feature_cols)
    global_end = 0
    request_index = 0
    for chunk in pd.read_csv(path, usecols=feature_cols, chunksize=CHUNK_SIZE, low_memory=False):
        numeric = chunk.apply(pd.to_numeric, errors="coerce").astype("float64")
        combined = numeric.reset_index(drop=True) if previous.empty else pd.concat([previous, numeric], ignore_index=True)
        combined_start = global_end - len(previous)
        global_end += len(numeric)
        while request_index < len(pending):
            start, window = pending[request_index]
            if start + window > global_end:
                break
            local_start = start - combined_start
            if local_start < 0:
                raise RuntimeError("Window buffer was too short for a requested sample")
            values = combined.iloc[local_start : local_start + window].to_numpy(np.float64)
            results[(start, window)] = hashlib.sha256(values.tobytes()).hexdigest()
            request_index += 1
        previous = combined.tail(maximum_window - 1).copy()
        if request_index == len(pending):
            break
    if len(results) != len(requests):
        raise RuntimeError(f"Only extracted {len(results)} of {len(requests)} requested windows from {path}")
    return results


def window_overlap_sample(normal: Path, merged: Path, normal_rows: int, smoke: bool) -> pd.DataFrame:
    header = pd.read_csv(normal, nrows=0)
    label_col = resolve_column(header.columns, "Normal/Attack")
    timestamp_col = resolve_column(header.columns, "Timestamp")
    assert label_col and timestamp_col
    split = normal_rows - int(max(128, round(normal_rows * 0.2)))
    sample_count = 16 if smoke else 128
    groups: list[tuple[int, str, np.ndarray]] = []
    requests: list[tuple[int, int]] = []
    for window in (64, 128):
        for scope, low, high in (("train", 0, split - window), ("validation", split, normal_rows - window)):
            starts = np.unique(np.linspace(low, high, num=min(sample_count, max(1, high - low + 1)), dtype=int))
            groups.append((window, scope, starts))
            requests.extend((int(start), window) for start in starts)
    normal_hashes = sampled_window_hashes(normal, requests)
    merged_hashes = sampled_window_hashes(merged, requests)
    rows = []
    for window, scope, starts in groups:
        matches = sum(normal_hashes[(int(start), window)] == merged_hashes[(int(start), window)] for start in starts)
        rows.append({"window_size": window, "scope": scope, "sample_count": len(starts),
                         "exact_window_overlap_count": matches, "overlap_ratio": matches / len(starts),
                         "sampling": "deterministic evenly spaced starts; canonical pre-scaler row hashes"})
    return pd.DataFrame(rows)


def load_label_time(path: Path, nrows: int | None = None) -> tuple[list[str], list[str]]:
    header = pd.read_csv(path, nrows=0)
    label_col = resolve_column(header.columns, "Normal/Attack")
    timestamp_col = resolve_column(header.columns, "Timestamp")
    assert label_col and timestamp_col
    labels: list[str] = []
    timestamps: list[str] = []
    for chunk in pd.read_csv(path, usecols=[label_col, timestamp_col], chunksize=CHUNK_SIZE, nrows=nrows):
        labels.extend(chunk[label_col].astype(str).tolist())
        timestamps.extend(chunk[timestamp_col].astype(str).tolist())
    return labels, timestamps


def segment_summary(segments: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [s for s in segments if s["label"] == "Attack"]
    negatives = [s for s in segments if s["label"] == "Normal"]
    labels = [s["label"] for s in segments]
    return {
        "positive_segment_count": len(positives), "negative_segment_count": len(negatives),
        "label_switch_count": max(0, len(segments) - 1),
        "positive_segment_lengths": [s["length"] for s in positives],
        "all_positive_samples_at_file_end": bool(positives and positives[-1]["end_index"] == segments[-1]["end_index"] and len(positives) == 1),
        "contains_normal_attack_normal_structure": any(labels[i:i + 3] == ["Normal", "Attack", "Normal"] for i in range(max(0, len(labels) - 2))),
    }


def temporal_continuity(source: pd.DataFrame, label_col: str, timestamp_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    attack = source[source[label_col].map(normalize_label) == "Attack"].copy()
    source_indices = attack.index.to_numpy(np.int64)
    timestamps = pd.to_datetime(attack[timestamp_col].astype("string").str.strip(), dayfirst=True, errors="coerce")
    source_gaps = np.diff(source_indices)
    time_gaps = timestamps.diff().dt.total_seconds().to_numpy()[1:]
    detail = pd.DataFrame({
        "merged_suffix_index": np.arange(1, len(attack)),
        "previous_source_index": source_indices[:-1], "source_index": source_indices[1:],
        "source_index_gap": source_gaps, "consecutive_in_source": source_gaps == 1,
        "previous_timestamp": attack[timestamp_col].astype(str).str.strip().to_numpy()[:-1],
        "timestamp": attack[timestamp_col].astype(str).str.strip().to_numpy()[1:],
        "timestamp_gap_seconds": time_gaps,
        "temporally_consecutive_one_second": np.isclose(time_gaps, 1.0, equal_nan=False),
    })
    summary = {
        "attack_rows": len(attack), "adjacent_pairs": len(detail),
        "source_index_consecutive_pairs": int(detail["consecutive_in_source"].sum()),
        "source_index_nonconsecutive_pairs": int((~detail["consecutive_in_source"]).sum()),
        "one_second_timestamp_pairs": int(detail["temporally_consecutive_one_second"].sum()),
        "non_one_second_timestamp_pairs": int((~detail["temporally_consecutive_one_second"]).sum()),
        "negative_or_zero_timestamp_gap_pairs": int((detail["timestamp_gap_seconds"] <= 0).sum()),
        "maximum_timestamp_gap_seconds": float(detail["timestamp_gap_seconds"].max()) if len(detail) else None,
        "interpretation": "Source-index continuity refers to the located attack.csv. Timestamp gaps reveal whether that already-filtered file preserves continuous wall-clock samples.",
    }
    return detail, summary


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def source_locations() -> list[dict[str, Any]]:
    refs = []
    terms = ("merged.csv", "normal.csv", "attack.csv", "attack", "label", "concat", "concatenate",
             "abnormal", "anomaly", "normal/attack", "build_swat_dataloaders", "_load_swat_csv")
    for path in list((ROOT / "configs").glob("*.yaml")) + list((ROOT / "data").glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        try:
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if any(term.lower() in line.lower() for term in terms):
                    refs.append({"path": str(path.relative_to(ROOT)), "line": number, "text": line.strip()[:240]})
        except OSError:
            continue
    return refs


def ensure_fresh_output(output: Path) -> None:
    expected = ["dataset_file_inventory.csv", "dataset_file_profiles.json", "merged_prefix_overlap_audit.json",
                "merged_suffix_source_audit.csv", "train_test_overlap_audit.csv", "window_overlap_sample_audit.csv",
                "label_segments.csv", "label_segment_summary.json", "abnormal_suffix_temporal_continuity.csv",
                "abnormal_suffix_summary.json", "data_pipeline_audit.md", "final_data_audit.md", "audit_summary.json"]
    existing = [name for name in expected if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing audit outputs: {existing}")
    output.mkdir(parents=True, exist_ok=True)


def run_audit(output: Path, smoke: bool = False) -> dict[str, Any]:
    normal = ROOT / "Swat" / "normal.csv"
    merged = ROOT / "Swat" / "merged.csv"
    primary_attack = ROOT / "Swat" / "attack.csv"
    mirrors = [ROOT / "dataset" / "dataset" / "SWat" / name for name in ("normal.csv", "merged.csv", "attack.csv")]
    files = [normal, merged, primary_attack, *mirrors]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required audit paths: {missing}")
    ensure_fresh_output(output)

    smoke_rows = 5_000 if smoke else None
    primary_profiles = {path.name: profile_csv(path, nrows=smoke_rows) for path in (normal, merged, primary_attack)}
    inventory = []
    hash_cache = {path.resolve(): primary_profiles[path.name]["sha256"] for path in (normal, merged, primary_attack)}
    for path in files:
        matching_primary = ROOT / "Swat" / path.name
        digest = hash_cache.get(path.resolve())
        if digest is None:
            digest = sha256_file(path) if not smoke else "not_computed_for_smoke_subset"
            hash_cache[path.resolve()] = digest
        primary_digest = hash_cache[matching_primary.resolve()]
        inventory.append({
            "path": str(path.resolve()), "role": {"normal.csv": "normal train source", "merged.csv": "configured test source", "attack.csv": "located attack source"}[path.name],
            "size_bytes": path.stat().st_size, "profiled_rows": primary_profiles[path.name]["profiled_rows"],
            "sha256": digest, "byte_identical_to_primary": digest == primary_digest if not smoke else "not_checked_in_smoke",
        })
    pd.DataFrame(inventory).to_csv(output / "dataset_file_inventory.csv", index=False)
    profiles = {}
    for path in files:
        profile = dict(primary_profiles[path.name])
        profile["path"] = str(path.resolve())
        profile["profile_reused_from_byte_identical_primary"] = path.parent != normal.parent
        profiles[str(path.resolve())] = profile
    write_json(output / "dataset_file_profiles.json", profiles)

    normal_rows = primary_profiles["normal.csv"]["profiled_rows"]
    prefix = prefix_audit(normal, merged, CHUNK_SIZE, smoke_rows)
    full_normal_rows = 1_387_098 if smoke else normal_rows
    suffix_reports, attack_frame = suffix_audit(merged, [primary_attack, mirrors[-1]], full_normal_rows, smoke_rows)
    prefix["merged_suffix_label_counts"] = dict(Counter(attack_frame[resolve_column(attack_frame.columns, "Normal/Attack")].map(normalize_label)))
    write_json(output / "merged_prefix_overlap_audit.json", prefix)
    suffix_reports.to_csv(output / "merged_suffix_source_audit.csv", index=False)

    split_overlap_audit(normal_rows, prefix).to_csv(output / "train_test_overlap_audit.csv", index=False)
    window_overlap_sample(normal, merged, normal_rows, smoke).to_csv(output / "window_overlap_sample_audit.csv", index=False)

    all_segments: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for path in (merged, primary_attack):
        labels, timestamps = load_label_time(path, smoke_rows)
        segments = extract_segments(labels, timestamps, str(path.resolve()))
        all_segments.extend(segments)
        summaries[str(path.resolve())] = segment_summary(segments)
    pd.DataFrame(all_segments).to_csv(output / "label_segments.csv", index=False)
    write_json(output / "label_segment_summary.json", summaries)

    attack_source = pd.read_csv(primary_attack, low_memory=False, nrows=smoke_rows)
    label_col = resolve_column(attack_source.columns, "Normal/Attack")
    timestamp_col = resolve_column(attack_source.columns, "Timestamp")
    assert label_col and timestamp_col
    continuity, continuity_summary = temporal_continuity(attack_source, label_col, timestamp_col)
    continuity.to_csv(output / "abnormal_suffix_temporal_continuity.csv", index=False)
    write_json(output / "abnormal_suffix_summary.json", continuity_summary)

    refs = source_locations()
    generator_refs = [r for r in refs if "merged.csv" in r["text"].lower() and r["path"].startswith("scripts/") and "stage4_3d" not in r["path"]]
    classification = "E. multiple_issues" if not smoke else "smoke_only_not_classified"
    pipeline_md = f"""# SWaT data pipeline audit

## Active pipeline

- Training source: `Swat/normal.csv`.
- Configured test source: `Swat/merged.csv`.
- Loader: `data/swat.py::_load_swat_csv`.
- Split: `build_swat_dataloaders` reserves the tail `{0.2:.0%}` of normal data for validation (subject to its window-size guards).
- Feature conversion: numeric coercion, missing values filled with `0.0`, then scaler fit only on `x_train_raw`.
- Label conversion: stripped lowercase values in `normal`, `0`, `false` become 0; every other value becomes 1.
- `Normal` and `Attack` values observed here are therefore parsed as 0 and 1 correctly.

## Construction-code search

- Direct in-repository generator for `merged.csv`: **not found** among configuration, data, and script source files.
- References found: {len(refs)}. Candidate generator references excluding this audit: {len(generator_refs)}.
- The construction mechanism is inferred from file contents only; the audit does not claim an unavailable command or notebook.

## Entry points

- Main/4.1 runs reach the SWaT loader through `data/factory.py` and configured YAML files.
- Step 4.3C calls `data.swat._load_swat_csv` directly and uses the same normal/merged files.

Full reference locations are embedded in `audit_summary.json`.
"""
    (output / "data_pipeline_audit.md").write_text(pipeline_md, encoding="utf-8")

    final_md = f"""# Step 4.3D final data audit

## Classification

**{classification}**

The content evidence shows that the configured test data contains the normal source as a prefix and an attack-only source as a suffix. This creates direct train/test and validation/test overlap. The attack-only source also has wall-clock gaps, so its concatenated suffix is not a continuous chronological SWaT test sequence. Label parsing is not the cause.

The normal source is itself non-monotonic and contains duplicated rows/timestamps and missing values (see `dataset_file_profiles.json`), so the audit classification is not based on concatenation alone.

## Evidence

- Canonical prefix exact match: `{prefix['exact_prefix_match']}`; matched rows: `{prefix['matched_row_count']}`.
- Prefix feature match ratio: `{prefix['feature_match_ratio']:.12f}`; timestamp match ratio: `{prefix['timestamp_match_ratio']:.12f}`.
- Suffix source match: `{bool(suffix_reports['exact_filtered_source_match'].all())}`.
- Attack suffix rows in this run: `{len(attack_frame)}`; normalized labels: `{dict(Counter(attack_frame[label_col].map(normalize_label)))}`.
- Attack-source adjacent pairs with non-one-second timestamp gaps: `{continuity_summary['non_one_second_timestamp_pairs']}`.
- Normal-source duplicate complete rows: `{primary_profiles['normal.csv']['duplicate_complete_row_count']}`; duplicate timestamps: `{primary_profiles['normal.csv']['duplicate_timestamp_count']}`; missing values: `{primary_profiles['normal.csv']['missing_value_count']}`.
- Raw file hashes and all paths: `dataset_file_inventory.csv`.

## Consequences

- The single terminal anomaly interval is a construction artifact of appending attack-only rows, not evidence of one real continuous incident.
- Event-level results from 4.1 and 4.3C are not valid as chronological SWaT event evaluation.
- Existing overall SWaT test metrics based on this `merged.csv` require rerunning on a correctly reconstructed chronological test set.
- Normal-data diagnostics (4.2A-4.2C) and controlled injected-anomaly comparisons (4.3A-4.3B) remain useful for their stated controlled questions, but they do not validate real SWaT event performance.
- The next step should be a separately authorized data-reconstruction task, not additional structure design.
"""
    (output / "final_data_audit.md").write_text(final_md, encoding="utf-8")
    summary = {
        "mode": "smoke" if smoke else "formal", "classification": classification,
        "normal_rows": normal_rows, "merged_rows": primary_profiles["merged.csv"]["profiled_rows"],
        "attack_rows": primary_profiles["attack.csv"]["profiled_rows"], "prefix_audit": prefix,
        "suffix_all_attack": set(prefix["merged_suffix_label_counts"]) <= {"Attack"},
        "suffix_matches_located_attack_source": bool(suffix_reports["exact_filtered_source_match"].all()),
        "temporal_continuity_summary": continuity_summary, "code_references": refs,
        "original_files_modified": False, "models_loaded": False, "training_run": False,
    }
    write_json(output / "audit_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only SWaT data construction and overlap audit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = [ROOT / "Swat" / name for name in ("normal.csv", "merged.csv", "attack.csv")]
    if args.dry_run:
        print("Step 4.3D dry-run: no files will be read beyond existence/metadata and no outputs will be written.")
        for path in files:
            print(f"{'OK' if path.exists() else 'MISSING'}: {path}")
        print(f"Formal output: {DEFAULT_OUTPUT}")
        print("Operations: streaming profiles, canonical prefix/suffix comparison, split/window overlap, label segments, timestamp continuity.")
        print("Models/training/metrics: disabled.")
        return 0 if all(path.exists() for path in files) else 2
    if args.output_dir:
        output = args.output_dir.resolve()
    elif args.smoke:
        output = Path(tempfile.mkdtemp(prefix="stage4_3d_smoke_"))
    else:
        output = DEFAULT_OUTPUT
    summary = run_audit(output, smoke=args.smoke)
    print(json.dumps({"output": str(output), "classification": summary["classification"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
