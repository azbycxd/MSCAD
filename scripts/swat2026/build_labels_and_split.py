from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.swat2026 import load_swat2026_csv
from scripts.swat2026.extract_and_review_info_sheet import (
    KNOWN_SCHEDULE_IMAGE_SHA256,
    TRANSCRIPTION,
)


EXPECTED_DATA_SHA256 = "d1dd05034b393a6b201f39e86ac3b933000551d26680c5d91c5ff8db0559cf72"
NORMAL_START = pd.Timestamp("2026-03-11 09:00:00")
NORMAL_END = pd.Timestamp("2026-03-11 13:00:00")
TEST_START = NORMAL_END
TEST_END = pd.Timestamp("2026-03-11 17:00:00")
OUTSIDE_END = pd.Timestamp("2026-03-11 17:01:00")
TIMESTAMP_COLUMN_CANDIDATES = {"t_stamp", "timestamp", "datetime", "time", "date"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def timestamp_column(columns: list[str]) -> str:
    lookup = {str(column).strip().lower(): str(column).strip() for column in columns}
    value = next((lookup[name] for name in TIMESTAMP_COLUMN_CANDIDATES if name in lookup), None)
    if value is None:
        raise ValueError("Timestamp column not found")
    return value


def parse_timestamps(frame: pd.DataFrame) -> pd.Series:
    column = timestamp_column(frame.columns.astype(str).str.strip().tolist())
    parsed = pd.to_datetime(frame[column], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Unparseable timestamps: {int(parsed.isna().sum())}")
    if parsed.duplicated().any() or not parsed.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and monotonic")
    differences = parsed.diff().dropna().dt.total_seconds()
    if not differences.eq(1.0).all():
        raise ValueError("11-Mar source is not a complete one-second timeline")
    return parsed


def select_model_features(data_path: Path) -> list[str]:
    header_contract = load_swat2026_csv(data_path, feature_mode="process_values_only", max_rows=0)
    names = header_contract.feature_names
    if len(names) != 31 or not all(name.strip().lower().endswith(".pv") for name in names):
        raise ValueError(f"Expected 31 stable .Pv features, got {len(names)}")
    return names


def load_approved_schedule(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_hash = sha256_file(path)
    candidate = pd.read_csv(path, dtype=str, keep_default_na=False)
    issues: list[str] = []
    if len(candidate) != 11:
        issues.append(f"expected_11_rows_got_{len(candidate)}")
    expected = {attack_id: (name, start, end, target) for attack_id, name, start, end, target in TRANSCRIPTION}
    seen: set[str] = set()
    approved_rows = []
    for _, row in candidate.iterrows():
        attack_id = row.get("attack_id", "").strip()
        if attack_id in seen:
            issues.append(f"duplicate_attack_id_{attack_id}")
        seen.add(attack_id)
        expected_values = expected.get(attack_id)
        actual = (row.get("attack_name", ""), row.get("start_time", ""), row.get("end_time", ""), row.get("target", ""))
        if expected_values is None or actual != expected_values:
            issues.append(f"candidate_mismatch_attack_{attack_id}")
        if row.get("date", "") != "2026-03-11":
            issues.append(f"invalid_date_attack_{attack_id}")
        if row.get("source_image_sha256", "").lower() != KNOWN_SCHEDULE_IMAGE_SHA256:
            issues.append(f"image_sha_mismatch_attack_{attack_id}")
        try:
            start = pd.Timestamp(f"{row['date']} {row['start_time']}")
            end = pd.Timestamp(f"{row['date']} {row['end_time']}")
        except (KeyError, ValueError):
            issues.append(f"unparseable_time_attack_{attack_id}")
            continue
        if not TEST_START <= start < end <= TEST_END:
            issues.append(f"outside_attack_phase_attack_{attack_id}")
        approved = row.to_dict()
        approved.update({
            "description": row.get("attack_name", ""), "description_source": "copied_from_attack_name",
            "date_source": "document_context", "timezone": "local_clock_unspecified",
            "human_review_status": "approved", "boundary_policy": "left_closed_right_open",
            "approval_source": "explicit_user_review_for_SWaT_N2",
        })
        approved_rows.append(approved)
    if seen != set(expected):
        issues.append("attack_id_set_mismatch")
    approved_frame = pd.DataFrame(approved_rows)
    if not approved_frame.empty:
        starts = pd.to_datetime(approved_frame["date"] + " " + approved_frame["start_time"])
        ends = pd.to_datetime(approved_frame["date"] + " " + approved_frame["end_time"])
        order = np.argsort(starts.to_numpy())
        starts, ends = starts.iloc[order].reset_index(drop=True), ends.iloc[order].reset_index(drop=True)
        for index in range(len(starts) - 1):
            if starts.iloc[index + 1] < ends.iloc[index]:
                issues.append(f"left_open_overlap_{index + 1}_{index + 2}")
    validation = {
        "candidate_sha256": source_hash, "row_count": len(candidate), "expected_image_sha256": KNOWN_SCHEDULE_IMAGE_SHA256,
        "issues": sorted(set(issues)), "all_rows_human_approved": not issues and len(candidate) == 11,
        "boundary_policy": "left_closed_right_open", "timezone_policy": "local_clock_unspecified",
    }
    return approved_frame, validation


def build_point_labels(timestamps: pd.Series, schedule: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame({"timestamp": timestamps.dt.strftime("%Y-%m-%dT%H:%M:%S")})
    phase = np.select(
        [(timestamps >= NORMAL_START) & (timestamps < NORMAL_END),
         (timestamps >= TEST_START) & (timestamps < TEST_END),
         (timestamps >= TEST_END) & (timestamps < OUTSIDE_END)],
        ["normal_phase", "attack_experiment_phase", "outside_declared_phase"], default="outside_source_contract")
    labels = np.zeros(len(timestamps), dtype=np.int8)
    attack_ids = np.full(len(timestamps), "", dtype=object)
    attack_names = np.full(len(timestamps), "", dtype=object)
    targets = np.full(len(timestamps), "", dtype=object)
    assignment_count = np.zeros(len(timestamps), dtype=np.int8)
    for _, attack in schedule.iterrows():
        start = pd.Timestamp(f"{attack['date']} {attack['start_time']}")
        end = pd.Timestamp(f"{attack['date']} {attack['end_time']}")
        mask = ((timestamps >= start) & (timestamps < end)).to_numpy()
        assignment_count += mask.astype(np.int8)
        labels[mask] = 1
        attack_ids[mask] = str(attack["attack_id"])
        attack_names[mask] = attack["attack_name"]
        targets[mask] = attack["target"]
    if (assignment_count > 1).any():
        raise ValueError("A timestamp was assigned to multiple attacks")
    result["phase"] = phase
    result["point_label"] = labels
    result["attack_id"] = attack_ids
    result["attack_name"] = attack_names
    result["target"] = targets
    result["evaluation_eligible"] = phase == "attack_experiment_phase"
    result["label_source"] = np.where(labels == 1, "official_A12_human_approved_attack_schedule",
                                       "official_A12_phase_and_approved_schedule")
    result["boundary_policy"] = "left_closed_right_open"
    result["timezone_policy"] = "local_clock_unspecified"
    return result


def validate_label_counts(labels: pd.DataFrame) -> dict[str, Any]:
    test = labels[labels["phase"].eq("attack_experiment_phase")]
    normal_phase_points = int(labels["phase"].eq("normal_phase").sum())
    outside_points = int(labels["phase"].eq("outside_declared_phase").sum())
    attack_points = int(test["point_label"].sum())
    test_points = len(test)
    test_normal = test_points - attack_points
    counts_ok = (normal_phase_points, test_points, outside_points, attack_points, test_normal) == (14400, 14400, 60, 3120, 11280)
    return {
        "normal_phase_points": normal_phase_points, "test_phase_points": test_points,
        "outside_declared_phase_points": outside_points, "attack_point_count": attack_points,
        "test_normal_point_count": test_normal, "attack_prevalence": attack_points / test_points if test_points else None,
        "expected_counts_passed": counts_ok,
    }


def region_masks(labels: pd.DataFrame) -> dict[str, np.ndarray]:
    phase = labels["phase"].to_numpy()
    point = labels["point_label"].to_numpy(dtype=int)
    return {
        "full_session": np.ones(len(labels), dtype=bool),
        "normal_phase": phase == "normal_phase",
        "attack_experiment_phase": phase == "attack_experiment_phase",
        "attack_intervals": point == 1,
        "attack_gap_normal": (phase == "attack_experiment_phase") & (point == 0),
        "outside_declared_phase": phase == "outside_declared_phase",
    }


def audit_model_features(
    frame: pd.DataFrame, labels: pd.DataFrame, feature_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    masks = region_masks(labels)
    invalid_by_feature: dict[str, np.ndarray] = {}
    issue_by_feature: dict[str, np.ndarray] = {}
    numeric_by_feature: dict[str, pd.Series] = {}
    quality_rows: list[dict[str, Any]] = []
    invalid_cells: list[dict[str, Any]] = []
    all_87_bad_input_count = 0
    for column in frame.columns:
        raw = frame[column].astype("string")
        all_87_bad_input_count += int(raw.str.strip().str.lower().eq("bad input").fillna(False).sum())
    for feature in feature_names:
        raw = frame[feature].astype("string")
        stripped = raw.str.strip()
        bad = stripped.str.lower().eq("bad input").fillna(False).to_numpy()
        empty = (raw.isna() | stripped.eq("")).fillna(True).to_numpy()
        numeric = pd.to_numeric(stripped, errors="coerce")
        numeric_values = numeric.to_numpy(dtype=float)
        pos_inf = np.isposinf(numeric_values)
        neg_inf = np.isneginf(numeric_values)
        non_numeric = numeric.isna().to_numpy() & ~bad & ~empty
        invalid = bad | empty | non_numeric | pos_inf | neg_inf
        issues = np.full(len(frame), "", dtype=object)
        for mask, name in [(bad, "bad_input"), (empty, "empty"), (non_numeric, "non_numeric"),
                           (pos_inf, "positive_infinity"), (neg_inf, "negative_infinity")]:
            issues[mask] = np.where(issues[mask] == "", name, issues[mask] + ";" + name)
        invalid_by_feature[feature] = invalid
        issue_by_feature[feature] = issues
        numeric_by_feature[feature] = numeric
        for region, region_mask in masks.items():
            valid_values = numeric_values[region_mask & ~invalid]
            unique_count = int(pd.Series(valid_values).nunique(dropna=True)) if len(valid_values) else 0
            dominant_ratio = float(pd.Series(valid_values).value_counts(normalize=True).iloc[0]) if len(valid_values) else None
            quality_rows.append({
                "region": region, "feature": feature, "row_count": int(region_mask.sum()),
                "bad_input_count": int((bad & region_mask).sum()), "empty_count": int((empty & region_mask).sum()),
                "non_numeric_count": int((non_numeric & region_mask).sum()),
                "positive_infinity_count": int((pos_inf & region_mask).sum()),
                "negative_infinity_count": int((neg_inf & region_mask).sum()),
                "invalid_cell_count": int((invalid & region_mask).sum()), "valid_numeric_count": int(len(valid_values)),
                "unique_valid_value_count": unique_count, "constant": bool(len(valid_values) and unique_count == 1),
                "near_constant": bool(len(valid_values) and unique_count > 1 and dominant_ratio is not None and dominant_ratio >= 0.99),
                "dominant_value_ratio": dominant_ratio,
            })
        for index in np.flatnonzero(invalid):
            invalid_cells.append({
                "timestamp": labels.iloc[index]["timestamp"], "feature": feature,
                "raw_value": "" if pd.isna(frame.iloc[index][feature]) else str(frame.iloc[index][feature]),
                "invalid_reason": issues[index], "phase": labels.iloc[index]["phase"],
                "point_label": int(labels.iloc[index]["point_label"]),
                "evaluation_eligible": bool(labels.iloc[index]["evaluation_eligible"]),
            })
    invalid_matrix = np.column_stack([invalid_by_feature[name] for name in feature_names])
    invalid_counts = invalid_matrix.sum(axis=1).astype(int)
    invalid_names = [";".join(name for name, value in zip(feature_names, row) if value) for row in invalid_matrix]
    validity = labels[["timestamp", "phase", "point_label", "evaluation_eligible"]].copy()
    validity.insert(1, "invalid_feature_count", invalid_counts)
    validity.insert(2, "invalid_feature_names", invalid_names)
    validity.insert(3, "all_model_features_valid", invalid_counts == 0)
    invalid_rows = validity[validity["invalid_feature_count"].gt(0)].copy()
    aggregate_rows = []
    for region, mask in masks.items():
        region_quality = [row for row in quality_rows if row["region"] == region]
        aggregate_rows.append({
            "region": region, "feature": "__ALL_31_MODEL_FEATURES__", "row_count": int(mask.sum()),
            "bad_input_count": sum(row["bad_input_count"] for row in region_quality),
            "empty_count": sum(row["empty_count"] for row in region_quality),
            "non_numeric_count": sum(row["non_numeric_count"] for row in region_quality),
            "positive_infinity_count": sum(row["positive_infinity_count"] for row in region_quality),
            "negative_infinity_count": sum(row["negative_infinity_count"] for row in region_quality),
            "invalid_cell_count": sum(row["invalid_cell_count"] for row in region_quality),
            "valid_numeric_count": sum(row["valid_numeric_count"] for row in region_quality),
            "unique_valid_value_count": "", "constant": "", "near_constant": "", "dominant_value_ratio": "",
        })
    quality = pd.DataFrame(quality_rows + aggregate_rows)
    details = {
        "all_87_columns_bad_input_count": all_87_bad_input_count,
        "model_31_features_bad_input_count": int(sum(value.sum() for value in [
            frame[name].astype("string").str.strip().str.lower().eq("bad input").fillna(False).to_numpy() for name in feature_names])),
        "model_invalid_cell_count": int(invalid_matrix.sum()),
        "model_invalid_timestamp_count": int((invalid_counts > 0).sum()),
        "feature_count": len(feature_names), "timeline_row_count": len(frame), "timeline_preserved": True,
    }
    return quality, pd.DataFrame(invalid_cells), invalid_rows, validity, details


def count_windows(validity: np.ndarray, window_size: int) -> dict[str, int]:
    total = max(0, len(validity) - window_size + 1)
    if total == 0:
        return {"candidate_window_count": 0, "valid_window_count": 0, "invalid_window_count": 0}
    invalid_per_window = np.convolve((~validity).astype(int), np.ones(window_size, dtype=int), mode="valid")
    valid = int((invalid_per_window == 0).sum())
    return {"candidate_window_count": total, "valid_window_count": valid, "invalid_window_count": total - valid}


def build_split(timestamps: pd.Series, validity: pd.DataFrame, max_window_size: int, train_ratio: float) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    context_start = TEST_START - pd.Timedelta(seconds=max_window_size - 1)
    candidate_mask = (timestamps >= NORMAL_START) & (timestamps < context_start)
    candidate_indices = np.flatnonzero(candidate_mask.to_numpy())
    train_count = math.floor(len(candidate_indices) * train_ratio)
    train_indices, validation_indices = candidate_indices[:train_count], candidate_indices[train_count:]
    context_indices = np.flatnonzero(((timestamps >= context_start) & (timestamps < TEST_START)).to_numpy())
    test_indices = np.flatnonzero(((timestamps >= TEST_START) & (timestamps < TEST_END)).to_numpy())
    outside_indices = np.flatnonzero(((timestamps >= TEST_END) & (timestamps < OUTSIDE_END)).to_numpy())
    sets = {"train": set(train_indices), "validation": set(validation_indices), "test_context": set(context_indices),
            "test_evaluation": set(test_indices), "outside_declared_phase": set(outside_indices)}
    overlap_pairs = []
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if sets[left] & sets[right]:
                overlap_pairs.append(f"{left}:{right}")
    all_valid = validity["all_model_features_valid"].to_numpy(dtype=bool)
    train_windows = count_windows(all_valid[train_indices], max_window_size)
    validation_windows = count_windows(all_valid[validation_indices], max_window_size)
    test_input_indices = np.concatenate([context_indices, test_indices])
    test_invalid_per_window = np.convolve((~all_valid[test_input_indices]).astype(int), np.ones(max_window_size, dtype=int), mode="valid")
    test_windows = {"candidate_window_count": len(test_indices), "valid_window_count": int((test_invalid_per_window == 0).sum()),
                    "invalid_window_count": int((test_invalid_per_window > 0).sum())}
    def range_row(role: str, indices: np.ndarray, evaluation: bool, context: bool = False) -> dict[str, Any]:
        start = timestamps.iloc[indices[0]] if len(indices) else None
        end_exclusive = timestamps.iloc[indices[-1]] + pd.Timedelta(seconds=1) if len(indices) else None
        return {"split_role": role, "start_time_inclusive": start.isoformat() if start is not None else "",
                "end_time_exclusive": end_exclusive.isoformat() if end_exclusive is not None else "", "raw_point_count": len(indices),
                "evaluation_eligible": evaluation, "context_only": context, "source_file": "11-Mar-2026_0900_1700.csv"}
    ranges = pd.DataFrame([
        range_row("train", train_indices, False), range_row("validation", validation_indices, False),
        range_row("test_context", context_indices, False, True), range_row("test_evaluation", test_indices, True),
        range_row("outside_declared_phase", outside_indices, False),
    ])
    window_rows = []
    for role, value in [("train", train_windows), ("validation", validation_windows), ("test_evaluation", test_windows)]:
        window_rows.append({"split_role": role, "max_window_size": max_window_size, **value,
                            "invalid_window_policy": "exclude_later" if role != "test_evaluation" else "retain_mask_for_stratified_reporting"})
    sufficient = train_windows["valid_window_count"] > 0 and validation_windows["valid_window_count"] > 0
    leakage = {
        "status": "A. leakage_free" if not overlap_pairs and sufficient else ("B. overlap_detected" if overlap_pairs else "C. insufficient_valid_normal_windows"),
        "raw_timestamp_overlap_pairs": overlap_pairs, "same_raw_row_multiple_splits": bool(overlap_pairs),
        "train_windows_cross_boundary": False, "validation_windows_cross_boundary": False,
        "test_context_in_train_or_validation": bool((sets["test_context"] & sets["train"]) or (sets["test_context"] & sets["validation"])),
        "windows_cross_csv": False, "included_source_files": ["11-Mar-2026_0900_1700.csv"],
        "excluded_unresolved_sessions": ["19-Feb-2026", "20-Feb-2026"],
        "scaler_fitted": False, "model_trained": False, "checkpoint_loaded": False,
        "test_labels_used_for_feature_selection": False, "sufficient_valid_normal_windows": sufficient,
    }
    manifest = {
        "strategy": "chronological_no_shuffle", "chronological_train_ratio": train_ratio,
        "max_window_size": max_window_size, "test_context_seconds": max_window_size - 1,
        "train_raw_point_count": len(train_indices), "validation_raw_point_count": len(validation_indices),
        "test_context_point_count": len(context_indices), "test_evaluation_point_count": len(test_indices),
        "scaler_fit_scope_future": "train_only", "scaler_fitted": False, "model_input_arrays_generated": False,
        "validation_can_influence_train": False, "cross_file_windows_allowed": False,
        "unresolved_sessions_included": False,
    }
    return manifest, ranges, pd.DataFrame(window_rows), leakage


def attack_event_summary(schedule: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = schedule.copy()
    ordered["_start"] = pd.to_datetime(ordered["date"] + " " + ordered["start_time"])
    ordered["_end"] = pd.to_datetime(ordered["date"] + " " + ordered["end_time"])
    ordered = ordered.sort_values("_start").reset_index(drop=True)
    for index, row in ordered.iterrows():
        next_start = ordered.iloc[index + 1]["_start"] if index + 1 < len(ordered) else TEST_END
        rows.append({"attack_id": row["attack_id"], "attack_name": row["attack_name"], "target": row["target"],
                     "start_time": row["_start"].isoformat(), "end_time": row["_end"].isoformat(),
                     "attack_point_count_left_open": int((row["_end"] - row["_start"]).total_seconds()),
                     "normal_gap_after_seconds": max(0, int((next_start - row["_end"]).total_seconds())),
                     "shared_boundary_with_next": bool(next_start == row["_end"] and index + 1 < len(ordered))})
    return pd.DataFrame(rows)


def run(data_path: Path, schedule_path: Path, output_dir: Path, boundary_policy: str, max_window_size: int, dry_run: bool = False) -> dict[str, Any]:
    if boundary_policy != "left_closed_right_open":
        raise ValueError("N2 human approval authorizes only left_closed_right_open")
    data_hash_before, schedule_hash_before = sha256_file(data_path), sha256_file(schedule_path)
    if data_hash_before != EXPECTED_DATA_SHA256:
        raise ValueError("Official 11-Mar source SHA-256 differs from the N1 audit")
    approved, schedule_validation = load_approved_schedule(schedule_path)
    if not schedule_validation["all_rows_human_approved"]:
        raise ValueError("Approved schedule validation failed: " + ";".join(schedule_validation["issues"]))
    frame = pd.read_csv(data_path, low_memory=False)
    frame.columns = frame.columns.astype(str).str.strip()
    timestamps = parse_timestamps(frame)
    feature_names = select_model_features(data_path)
    labels = build_point_labels(timestamps, approved)
    label_counts = validate_label_counts(labels)
    if not label_counts["expected_counts_passed"]:
        raise ValueError(f"Label count contract failed: {label_counts}")
    quality, invalid_cells, invalid_rows, validity, feature_details = audit_model_features(frame, labels, feature_names)
    split_manifest, split_ranges, window_summary, leakage = build_split(timestamps, validity, max_window_size, 0.8)
    events = attack_event_summary(approved)
    label_summary = {
        **label_counts, "attack_event_count": len(approved),
        "positive_normal_gap_count": int(events["normal_gap_after_seconds"].iloc[:-1].gt(0).sum()),
        "trailing_normal_gap_seconds": int(events.iloc[-1]["normal_gap_after_seconds"]),
        "first_attack_start": events.iloc[0]["start_time"], "last_attack_end": events.iloc[-1]["end_time"],
        "shared_boundary_count": int(events["shared_boundary_with_next"].sum()),
        "boundary_policy": boundary_policy, "timezone_policy": "local_clock_unspecified",
    }
    sources_unchanged = data_hash_before == sha256_file(data_path) and schedule_hash_before == sha256_file(schedule_path)
    valid_windows = leakage["sufficient_valid_normal_windows"]
    readiness = "A. ready_for_model_preparation"
    if not sources_unchanged:
        readiness = "F. source_data_changed"
    elif not schedule_validation["all_rows_human_approved"] or not label_counts["expected_counts_passed"]:
        readiness = "D. schedule_validation_failed"
    elif leakage["status"] == "B. overlap_detected":
        readiness = "E. leakage_detected"
    elif not valid_windows:
        readiness = "C. insufficient_valid_normal_windows"
    elif feature_details["model_invalid_cell_count"] and validity.empty:
        readiness = "B. labels_ready_feature_quality_unresolved"
    readiness_payload = {
        "classification": readiness, "source_sha256_matches_n1": data_hash_before == EXPECTED_DATA_SHA256,
        "schedule_rows_approved": len(approved), "label_counts_passed": label_counts["expected_counts_passed"],
        "leakage_status": leakage["status"], "sufficient_valid_normal_windows": valid_windows,
        "invalid_mask_generated": not dry_run, "model_trained": False, "scaler_fitted": False,
        "checkpoint_loaded": False, "ready_for_training_execution": False,
        "note": "A means artifacts are ready for a separately authorized model-preparation stage; no training is approved here.",
    }
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        approved.to_csv(output_dir / "approved_attack_schedule.csv", index=False, encoding="utf-8-sig")
        labels.to_csv(output_dir / "point_labels_11mar.csv", index=False, encoding="utf-8-sig")
        quality.to_csv(output_dir / "model_feature_quality_summary.csv", index=False, encoding="utf-8-sig")
        invalid_cells.to_csv(output_dir / "model_feature_invalid_cells.csv", index=False, encoding="utf-8-sig")
        invalid_rows.to_csv(output_dir / "model_feature_invalid_rows.csv", index=False, encoding="utf-8-sig")
        validity.to_csv(output_dir / "point_validity_11mar.csv", index=False, encoding="utf-8-sig")
        write_json(output_dir / "model_feature_list.json", {
            "selection_rule": "data.swat2026 process_values_only: normalized column name endswith .pv",
            "feature_count": len(feature_names), "feature_names": feature_names, **feature_details,
            "source_sha256": data_hash_before, "values_modified": False,
        })
        write_json(output_dir / "split_manifest.json", split_manifest)
        split_ranges.to_csv(output_dir / "split_ranges.csv", index=False, encoding="utf-8-sig")
        window_summary.to_csv(output_dir / "window_eligibility_summary.csv", index=False, encoding="utf-8-sig")
        write_json(output_dir / "leakage_audit.json", leakage)
        write_json(output_dir / "label_summary.json", label_summary)
        events.to_csv(output_dir / "attack_event_summary.csv", index=False, encoding="utf-8-sig")
        write_json(output_dir / "n2_readiness.json", readiness_payload)
        write_json(output_dir / "source_integrity_manifest.json", {
            "data_path": str(data_path), "data_sha256": data_hash_before,
            "schedule_path": str(schedule_path), "schedule_sha256": schedule_hash_before,
            "sources_unchanged": sources_unchanged, "official_sources_modified": False,
        })
    summary = {
        "dry_run": dry_run, "approved_attack_count": len(approved), **label_counts,
        "model_feature_count": len(feature_names), **feature_details,
        "train_valid_windows": int(window_summary.loc[window_summary.split_role.eq("train"), "valid_window_count"].iloc[0]),
        "validation_valid_windows": int(window_summary.loc[window_summary.split_role.eq("validation"), "valid_window_count"].iloc[0]),
        "test_valid_windows": int(window_summary.loc[window_summary.split_role.eq("test_evaluation"), "valid_window_count"].iloc[0]),
        "leakage_status": leakage["status"], "n2_readiness": readiness, "model_trained": False, "scaler_fitted": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build human-approved SWaT A.12 labels, validity mask, and leakage-free split")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--schedule-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--boundary-policy", default="left_closed_right_open")
    parser.add_argument("--max-window-size", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(Path(args.data_path), Path(args.schedule_path), Path(args.output_dir), args.boundary_policy, args.max_window_size, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
