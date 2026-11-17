from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


EXPECTED_SOURCE_SHA256 = "d1dd05034b393a6b201f39e86ac3b933000551d26680c5d91c5ff8db0559cf72"
TRAIN_START = pd.Timestamp("2026-03-11 09:00:00")
TRAIN_END = pd.Timestamp("2026-03-11 12:10:18")
VALIDATION_START = TRAIN_END
VALIDATION_END = pd.Timestamp("2026-03-11 12:57:53")
TEST_CONTEXT_START = VALIDATION_END
TEST_START = pd.Timestamp("2026-03-11 13:00:00")
TEST_END = pd.Timestamp("2026-03-11 17:00:00")
OUTSIDE_END = pd.Timestamp("2026-03-11 17:01:00")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


@dataclass
class PreparedSWaTData:
    source_path: Path
    source_sha256: str
    feature_names: list[str]
    timestamps: pd.DatetimeIndex
    raw_features: np.ndarray
    scaled_features: np.ndarray
    valid_points: np.ndarray
    labels: pd.DataFrame
    scaler: StandardScaler
    scaler_metadata: dict[str, Any]
    feature_statistics: pd.DataFrame
    window_index: pd.DataFrame

    def valid_window_rows(self, split: str) -> pd.DataFrame:
        rows = self.window_index[self.window_index["split"].eq(split) & self.window_index["valid"]]
        return rows.reset_index(drop=True)


class SWaTWindowDataset(Dataset):
    """Lazy window view; full window tensors are never persisted to disk."""

    def __init__(self, prepared: PreparedSWaTData, split: str) -> None:
        self.features = prepared.scaled_features
        self.rows = prepared.valid_window_rows(split)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[index]
        values = self.features[int(row.start_index) : int(row.end_index) + 1]
        tensor = torch.from_numpy(np.asarray(values, dtype=np.float32).copy())
        if not torch.isfinite(tensor).all():
            raise ValueError("A supposedly valid SWaT window contains NaN or Inf")
        return tensor, tensor


def _parse_source(data_path: Path, feature_names: list[str]) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    frame = pd.read_csv(data_path, low_memory=False)
    frame.columns = frame.columns.astype(str).str.strip()
    missing = [name for name in feature_names if name not in frame]
    if missing:
        raise ValueError(f"N2 feature columns are missing: {missing}")
    timestamp_column = next((name for name in frame if name.strip().lower() in {"t_stamp", "timestamp", "datetime", "time", "date"}), None)
    if timestamp_column is None:
        raise ValueError("Timestamp column not found")
    timestamps = pd.to_datetime(frame[timestamp_column], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Source timestamps are incomplete, duplicated, or unordered")
    if not timestamps.diff().dropna().dt.total_seconds().eq(1.0).all():
        raise ValueError("Source timeline is not exactly one-second resolution")
    numeric = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    valid = np.isfinite(values).all(axis=1)
    return pd.DatetimeIndex(timestamps), values, valid


def fit_train_only_scaler(
    raw_features: np.ndarray,
    timestamps: pd.DatetimeIndex,
    valid_points: np.ndarray,
    feature_names: list[str],
    source_sha256: str,
) -> tuple[StandardScaler, dict[str, Any], pd.DataFrame]:
    train_mask = (timestamps >= TRAIN_START) & (timestamps < TRAIN_END)
    fit_mask = train_mask & valid_points
    fit_values = raw_features[fit_mask]
    if int(train_mask.sum()) != 11_418 or len(fit_values) != 11_410:
        raise ValueError(f"Scaler row contract failed: train={int(train_mask.sum())}, valid={len(fit_values)}")
    scaler = StandardScaler().fit(fit_values)
    raw_std = np.std(fit_values, axis=0, ddof=0)
    zero = raw_std == 0.0
    near_zero = (~zero) & (raw_std <= 1e-12)
    if not np.all(scaler.scale_[zero] == 1.0):
        raise ValueError("StandardScaler did not safely map zero variance scales to one")
    metadata = {
        "scaler_type": "sklearn.preprocessing.StandardScaler",
        "feature_order": feature_names,
        "fit_split": "train_only_unique_raw_points",
        "fit_start": TRAIN_START.isoformat(),
        "fit_end": TRAIN_END.isoformat(),
        "total_train_raw_points": int(train_mask.sum()),
        "valid_fit_rows": int(fit_mask.sum()),
        "excluded_fit_rows": int(train_mask.sum() - fit_mask.sum()),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "zero_variance_features": [name for name, flag in zip(feature_names, zero) if flag],
        "near_zero_variance_features": [name for name, flag in zip(feature_names, near_zero) if flag],
        "source_csv_sha256": source_sha256,
        "labels_used": False,
        "validation_used": False,
        "test_used": False,
        "overlapping_windows_used_for_fit": False,
    }
    rows = []
    for index, name in enumerate(feature_names):
        column = fit_values[:, index]
        rows.append({
            "feature": name, "count": len(column), "min": float(np.min(column)), "max": float(np.max(column)),
            "mean": float(np.mean(column)), "std": float(np.std(column, ddof=0)),
            "median": float(np.median(column)), "q01": float(np.quantile(column, 0.01)),
            "q99": float(np.quantile(column, 0.99)), "unique_count": int(np.unique(column).size),
            "zero_variance": bool(zero[index]), "near_zero_variance": bool(near_zero[index]),
        })
    return scaler, metadata, pd.DataFrame(rows)


def _window_rows(
    timestamps: pd.DatetimeIndex,
    valid_points: np.ndarray,
    labels: pd.DataFrame,
    window_size: int,
) -> pd.DataFrame:
    label_lookup = labels.set_index(pd.to_datetime(labels["timestamp"]))
    rows: list[dict[str, Any]] = []

    def append_region(split: str, indices: np.ndarray, evaluation: bool) -> None:
        if evaluation:
            end_indices = indices
        else:
            end_indices = indices[window_size - 1 :]
        for window_id, end_index in enumerate(end_indices):
            start_index = int(end_index) - window_size + 1
            is_valid = bool(valid_points[start_index : int(end_index) + 1].all())
            stamp = timestamps[int(end_index)]
            label = label_lookup.loc[stamp] if evaluation else None
            rows.append({
                "split": split, "window_id": window_id, "start_index": start_index, "end_index": int(end_index),
                "window_start": timestamps[start_index].isoformat(), "window_end": stamp.isoformat(),
                "score_timestamp": stamp.isoformat(), "valid": is_valid,
                "exclusion_reason": "" if is_valid else "contains_invalid_model_feature",
                "point_label": int(label["point_label"]) if evaluation else "",
                "attack_id": str(label["attack_id"]) if evaluation and pd.notna(label["attack_id"]) else "",
                "phase": str(label["phase"]) if evaluation else "",
                "evaluation_eligible": bool(label["evaluation_eligible"]) if evaluation else False,
            })

    train = np.flatnonzero(np.asarray((timestamps >= TRAIN_START) & (timestamps < TRAIN_END)))
    validation = np.flatnonzero(np.asarray((timestamps >= VALIDATION_START) & (timestamps < VALIDATION_END)))
    test = np.flatnonzero(np.asarray((timestamps >= TEST_START) & (timestamps < TEST_END)))
    append_region("train", train, False)
    append_region("validation", validation, False)
    append_region("test", test, True)
    return pd.DataFrame(rows)


def prepare_swat_training_data(
    data_path: str | Path,
    n2_dir: str | Path,
    output_dir: str | Path | None,
    window_size: int = 128,
) -> PreparedSWaTData:
    data_path, n2_dir = Path(data_path), Path(n2_dir)
    source_hash = sha256_file(data_path)
    feature_contract = _read_json(n2_dir / "model_feature_list.json")
    readiness = _read_json(n2_dir / "n2_readiness.json")
    leakage = _read_json(n2_dir / "leakage_audit.json")
    if source_hash != EXPECTED_SOURCE_SHA256 or source_hash != feature_contract["source_sha256"]:
        raise ValueError("Official source SHA-256 differs from the N1/N2 contract")
    if readiness["classification"] != "A. ready_for_model_preparation" or leakage["status"] != "A. leakage_free":
        raise ValueError("N2 readiness/leakage contract is not approved")
    feature_names = list(feature_contract["feature_names"])
    if len(feature_names) != 31 or not all(name.lower().endswith(".pv") for name in feature_names):
        raise ValueError("Expected exactly 31 stable .Pv features from N2")
    timestamps, raw_features, valid_points = _parse_source(data_path, feature_names)
    labels = pd.read_csv(n2_dir / "point_labels_11mar.csv", encoding="utf-8-sig", dtype={"attack_id": str})
    if len(labels) != len(timestamps) or not pd.to_datetime(labels["timestamp"]).equals(pd.Series(timestamps)):
        raise ValueError("N2 point labels do not align one-to-one with source timestamps")
    if int((~valid_points).sum()) != 8:
        raise ValueError(f"Expected 8 invalid model-feature timestamps, got {int((~valid_points).sum())}")
    non_train_invalid = (~valid_points) & ~((timestamps >= TRAIN_START) & (timestamps < TRAIN_END))
    if non_train_invalid.any():
        raise ValueError("Validation/test/outside contains invalid model features")
    scaler, scaler_metadata, feature_statistics = fit_train_only_scaler(
        raw_features, timestamps, valid_points, feature_names, source_hash
    )
    scaled = (raw_features - scaler.mean_) / scaler.scale_
    window_index = _window_rows(timestamps, valid_points, labels, window_size)
    counts = window_index.groupby("split")["valid"].sum().astype(int).to_dict()
    candidates = window_index.groupby("split").size().astype(int).to_dict()
    if candidates != {"test": 14_400, "train": 11_291, "validation": 2_728}:
        raise ValueError(f"Window candidate count contract failed: {candidates}")
    if counts != {"test": 14_400, "train": 11_156, "validation": 2_728}:
        raise ValueError(f"Valid window count contract failed: {counts}")
    prepared = PreparedSWaTData(
        data_path, source_hash, feature_names, timestamps, raw_features, scaled.astype(np.float32), valid_points,
        labels, scaler, scaler_metadata, feature_statistics, window_index,
    )
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / "scaler_metadata.json", scaler_metadata)
        feature_statistics.to_csv(output / "train_feature_statistics.csv", index=False)
        window_index.to_csv(output / "window_index_manifest.csv", index=False)
        _write_json(output / "prepared_data_manifest.json", {
            "source_csv": str(data_path), "source_csv_sha256": source_hash,
            "n2_dir": str(n2_dir), "n2_readiness": readiness["classification"],
            "leakage_audit": leakage["status"], "window_size": window_size,
            "feature_count": len(feature_names), "feature_order": feature_names,
            "train_raw_points": 11_418, "scaler_fit_rows": 11_410,
            "train_candidate_windows": candidates["train"], "train_valid_windows": counts["train"],
            "validation_windows": counts["validation"], "test_evaluation_windows": counts["test"],
            "test_context_points": 127, "outside_points_used": 0,
            "arrays_persisted": False, "invalid_values_interpolated": False,
        })
    return prepared
