"""Minimal frozen-cache loader for the official SWaT 2015 Sprint S1 protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from data.transforms import StandardScaler


EXPECTED_FEATURES = 51
EXPECTED_NORMAL_ROWS = 495_000
EXPECTED_ATTACK_ROWS = 449_919
TRAIN_RAW_ROWS = 396_000
VALIDATION_RAW_ROWS = 99_000
WINDOW = 64
STRIDE = 1
EXACT_LABEL_MAP = {"Normal": 0, "Attack": 1, "A ttack": 1}
LEGACY_NAMES = {"normal.csv", "merged.csv"}
CACHE_FORMAT_VERSION = 1


class OfficialSWaTError(RuntimeError):
    """Raised when the frozen official protocol would be violated."""


@dataclass(frozen=True)
class CacheArrays:
    normal_features: np.ndarray
    attack_features: np.ndarray
    attack_labels: np.ndarray
    attack_timestamps: np.ndarray
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SplitPlan:
    train_raw_start: int
    train_raw_end_exclusive: int
    validation_raw_start: int
    validation_raw_end_exclusive: int
    train_window_starts: np.ndarray
    validation_window_starts: np.ndarray
    test_window_starts_by_segment: tuple[np.ndarray, ...]
    test_segment_raw_bounds: tuple[tuple[int, int], ...]


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = (repo_root / path).resolve() if not path.is_absolute() else path.resolve()
    return resolved


def reject_legacy_path(path: Path) -> None:
    if path.name.casefold() in LEGACY_NAMES or any(part.casefold() == "swat" for part in path.parts):
        raise OfficialSWaTError(f"Legacy SWaT path is forbidden: {path}")


def canonicalize_point_label(raw_label: Any) -> np.int8:
    if not isinstance(raw_label, str) or raw_label not in EXACT_LABEL_MAP:
        raise OfficialSWaTError(f"unknown exact SWaT Physical label: {raw_label!r}")
    return np.int8(EXACT_LABEL_MAP[raw_label])


def canonicalize_point_labels(raw_labels: Sequence[Any]) -> np.ndarray:
    return np.asarray([canonicalize_point_label(value) for value in raw_labels], dtype=np.int8)


def protocol_fingerprint(config: dict[str, Any]) -> str:
    frozen = {
        "seed": config["seed"],
        "data": config["data"],
        "scaler": config["scaler"],
        "model": config["model"],
        "training": config["training"],
        "scoring": config["scoring"],
        "v3a": config["v3a"],
    }
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_manifest(cache_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    reject_legacy_path(cache_dir)
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.is_file():
        raise OfficialSWaTError(f"cache manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cache_format_version") != CACHE_FORMAT_VERSION:
        raise OfficialSWaTError("cache format version mismatch")
    if manifest.get("protocol_fingerprint") != protocol_fingerprint(config):
        raise OfficialSWaTError("cache protocol fingerprint mismatch")
    if manifest.get("column_order") is None or len(manifest["column_order"]) != EXPECTED_FEATURES:
        raise OfficialSWaTError("cache does not contain exactly 51 ordered features")
    return manifest


def load_normal_cache(cache_dir: Path, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Load only Normal v1 so model fitting cannot inspect Attack labels."""
    cache_dir = cache_dir.resolve()
    manifest = _load_manifest(cache_dir, config)
    normal = np.load(cache_dir / "normal_v1_features.npy", mmap_mode="r")
    if normal.shape != (EXPECTED_NORMAL_ROWS, EXPECTED_FEATURES) or normal.dtype != np.dtype("float32"):
        raise OfficialSWaTError(f"Normal cache contract mismatch: {normal.shape}/{normal.dtype}")
    return normal, manifest


def load_attack_cache(
    cache_dir: Path, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load frozen Attack inputs/labels/timestamps after model and thresholds are fixed."""
    cache_dir = cache_dir.resolve()
    manifest = _load_manifest(cache_dir, config)
    features = np.load(cache_dir / "attack_v0_features.npy", mmap_mode="r")
    labels = np.load(cache_dir / "attack_v0_labels.npy", mmap_mode="r")
    timestamps = np.load(cache_dir / "attack_v0_timestamps.npy", mmap_mode="r")
    expected = (
        (features, (EXPECTED_ATTACK_ROWS, EXPECTED_FEATURES), np.dtype("float32"), "features"),
        (labels, (EXPECTED_ATTACK_ROWS,), np.dtype("int8"), "labels"),
        (timestamps, (EXPECTED_ATTACK_ROWS,), np.dtype("int64"), "timestamps"),
    )
    for array, shape, dtype, name in expected:
        if array.shape != shape or array.dtype != dtype:
            raise OfficialSWaTError(f"Attack cache {name} contract mismatch: {array.shape}/{array.dtype}")
    if set(np.unique(labels).tolist()) - {0, 1}:
        raise OfficialSWaTError("canonical attack labels are not binary")
    return features, labels, timestamps, manifest


def load_cache(cache_dir: Path, config: dict[str, Any]) -> CacheArrays:
    normal, manifest = load_normal_cache(cache_dir, config)
    attack, labels, timestamps, _ = load_attack_cache(cache_dir, config)
    return CacheArrays(normal, attack, labels, timestamps, manifest)


def fit_scaler_on_declared_train(
    normal_features: np.ndarray, *, train_raw_rows: int, expected_total_rows: int, expected_features: int
) -> tuple[StandardScaler, dict[str, Any]]:
    if normal_features.shape != (expected_total_rows, expected_features):
        raise OfficialSWaTError(f"unexpected Normal shape: {normal_features.shape}")
    scaler = StandardScaler().fit(np.asarray(normal_features[:train_raw_rows], dtype=np.float32))
    if scaler.mean_ is None or scaler.raw_std_ is None or scaler.effective_scale_ is None:
        raise OfficialSWaTError("scaler fit failed")
    zero_variance = np.flatnonzero(scaler.raw_std_.reshape(-1) < scaler.eps)
    if not np.all(scaler.effective_scale_.reshape(-1)[zero_variance] == 1.0):
        raise OfficialSWaTError("zero-variance scale policy was violated")
    metadata = {
        "fit_scope": f"normal_v1_raw_rows_[0,{train_raw_rows})",
        "fit_row_count": train_raw_rows,
        "validation_rows_seen": 0,
        "attack_rows_seen": 0,
        "feature_count": expected_features,
        "zero_variance_feature_indices": zero_variance.astype(int).tolist(),
        "zero_variance_effective_scale": 1.0,
    }
    return scaler, metadata


def fit_train_only_scaler(normal_features: np.ndarray, train_raw_rows: int = TRAIN_RAW_ROWS) -> tuple[StandardScaler, dict[str, Any]]:
    if train_raw_rows != TRAIN_RAW_ROWS:
        raise OfficialSWaTError("Sprint S1 scaler must fit exactly the first 396000 raw Normal v1 rows")
    return fit_scaler_on_declared_train(
        normal_features,
        train_raw_rows=train_raw_rows,
        expected_total_rows=EXPECTED_NORMAL_ROWS,
        expected_features=EXPECTED_FEATURES,
    )


def save_scaler(path: Path, scaler: StandardScaler, metadata: dict[str, Any]) -> None:
    assert scaler.mean_ is not None and scaler.raw_std_ is not None and scaler.effective_scale_ is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=scaler.mean_.astype(np.float32),
        raw_std=scaler.raw_std_.astype(np.float32),
        effective_scale=scaler.effective_scale_.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_scaler(path: Path) -> tuple[StandardScaler, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        scaler = StandardScaler()
        scaler.mean_ = payload["mean"]
        scaler.raw_std_ = payload["raw_std"]
        scaler.effective_scale_ = payload["effective_scale"]
        scaler.std_ = scaler.effective_scale_
        metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata.get("fit_scope") != "normal_v1_raw_rows_[0,396000)" or metadata.get("attack_rows_seen") != 0:
        raise OfficialSWaTError("saved scaler was not fitted on the frozen Train-only scope")
    return scaler, metadata


def make_window_starts(raw_start: int, raw_end_exclusive: int, window: int = WINDOW, stride: int = STRIDE) -> np.ndarray:
    if window != WINDOW or stride != STRIDE:
        raise OfficialSWaTError("Sprint S1 fixes window=64 and stride=1")
    if raw_start < 0 or raw_end_exclusive - raw_start < window:
        raise OfficialSWaTError("raw segment is too short for the frozen window")
    return np.arange(raw_start, raw_end_exclusive - window + 1, stride, dtype=np.int64)


def naive_timestamp_to_int64(value: str) -> int:
    datetime.fromisoformat(value)
    return int(np.datetime64(value, "s").astype(np.int64))


def locate_test_segment_bounds(timestamps: np.ndarray, segment_manifest: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    if segment_manifest.get("gap_policy") != "hard_session_boundary":
        raise OfficialSWaTError("T2.5 hard session boundary is missing")
    bounds: list[tuple[int, int]] = []
    for segment in segment_manifest["test_segments"]:
        start_value = naive_timestamp_to_int64(segment["start"])
        end_value = naive_timestamp_to_int64(segment["end_inclusive"])
        start_hits = np.flatnonzero(timestamps == start_value)
        end_hits = np.flatnonzero(timestamps == end_value)
        if len(start_hits) != 1 or len(end_hits) != 1:
            raise OfficialSWaTError(f"segment boundary timestamp not unique: {segment['segment_id']}")
        start = int(start_hits[0])
        end_exclusive = int(end_hits[0]) + 1
        if end_exclusive - start != int(segment["row_count"]):
            raise OfficialSWaTError(f"segment row count mismatch: {segment['segment_id']}")
        segment_timestamps = np.asarray(timestamps[start:end_exclusive], dtype=np.int64)
        if len(segment_timestamps) > 1 and not np.all(np.diff(segment_timestamps) == 1):
            raise OfficialSWaTError(f"non-contiguous timestamp inside segment: {segment['segment_id']}")
        bounds.append((start, end_exclusive))
    for left, right in zip(bounds, bounds[1:]):
        # Segment rows are adjacent in the workbook even though timestamps have
        # a hard gap; equality is therefore the valid non-overlapping case.
        if left[1] > right[0]:
            raise OfficialSWaTError("test segments overlap")
    return tuple(bounds)


def build_split_plan(timestamps: np.ndarray, segment_manifest: dict[str, Any]) -> SplitPlan:
    if len(timestamps) != EXPECTED_ATTACK_ROWS:
        raise OfficialSWaTError("unexpected Attack timestamp count")
    bounds = locate_test_segment_bounds(timestamps, segment_manifest)
    test_starts = tuple(make_window_starts(start, end) for start, end in bounds)
    plan = SplitPlan(
        train_raw_start=0,
        train_raw_end_exclusive=TRAIN_RAW_ROWS,
        validation_raw_start=TRAIN_RAW_ROWS,
        validation_raw_end_exclusive=EXPECTED_NORMAL_ROWS,
        train_window_starts=make_window_starts(0, TRAIN_RAW_ROWS),
        validation_window_starts=make_window_starts(TRAIN_RAW_ROWS, EXPECTED_NORMAL_ROWS),
        test_window_starts_by_segment=test_starts,
        test_segment_raw_bounds=bounds,
    )
    if len(plan.train_window_starts) != TRAIN_RAW_ROWS - WINDOW + 1:
        raise OfficialSWaTError("Train window count mismatch")
    if len(plan.validation_window_starts) != VALIDATION_RAW_ROWS - WINDOW + 1:
        raise OfficialSWaTError("Validation window count mismatch")
    if plan.train_window_starts[-1] + WINDOW > TRAIN_RAW_ROWS:
        raise OfficialSWaTError("Train window crosses into Validation")
    if plan.validation_window_starts[0] != TRAIN_RAW_ROWS:
        raise OfficialSWaTError("Validation windows do not start independently at the split")
    return plan


def gather_scaled_windows(
    features: np.ndarray,
    starts: np.ndarray,
    scaler: StandardScaler,
    window: int = WINDOW,
) -> np.ndarray:
    if scaler.mean_ is None or scaler.effective_scale_ is None:
        raise OfficialSWaTError("scaler is not fitted")
    row_indices = starts[:, None] + np.arange(window, dtype=np.int64)[None, :]
    raw = np.asarray(features[row_indices], dtype=np.float32)
    scaled = (raw - scaler.mean_.astype(np.float32)) / scaler.effective_scale_.astype(np.float32)
    if not np.isfinite(scaled).all():
        raise OfficialSWaTError("non-finite scaled window")
    return np.ascontiguousarray(scaled, dtype=np.float32)


def endpoint_labels(labels: np.ndarray, starts: np.ndarray, window: int = WINDOW) -> np.ndarray:
    endpoints = starts + window - 1
    if len(endpoints) and int(endpoints[-1]) >= len(labels):
        raise OfficialSWaTError("endpoint label index is outside the source segment")
    return np.asarray(labels[endpoints], dtype=np.int8)


def endpoint_timestamps(timestamps: np.ndarray, starts: np.ndarray, window: int = WINDOW) -> np.ndarray:
    return np.asarray(timestamps[starts + window - 1], dtype=np.int64)


def iter_window_batches(
    features: np.ndarray,
    starts: np.ndarray,
    scaler: StandardScaler,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    order = np.arange(len(starts), dtype=np.int64)
    if shuffle:
        order = np.random.default_rng(seed).permutation(order)
    for offset in range(0, len(order), batch_size):
        selected_starts = starts[order[offset : offset + batch_size]]
        yield gather_scaled_windows(features, selected_starts, scaler), selected_starts


def validate_config_contract(config: dict[str, Any]) -> None:
    data = config["data"]
    expected = {
        "normal_raw_rows": EXPECTED_NORMAL_ROWS,
        "input_channels": EXPECTED_FEATURES,
        "train_raw_rows": TRAIN_RAW_ROWS,
        "validation_raw_rows": VALIDATION_RAW_ROWS,
        "window": WINDOW,
        "stride": STRIDE,
    }
    for key, value in expected.items():
        if int(data[key]) != value:
            raise OfficialSWaTError(f"frozen config mismatch: {key}={data[key]!r}, expected {value}")
    if data["point_label_mapping"] != EXACT_LABEL_MAP or data["unknown_label_policy"] != "hard_error":
        raise OfficialSWaTError("point-label policy differs from T2.5")
    if data["cross_split_windows"] or data["cross_gap_windows"] or data["startup_trim_after_official_v1"]:
        raise OfficialSWaTError("frozen split/gap/Normal-v1 policy was weakened")
