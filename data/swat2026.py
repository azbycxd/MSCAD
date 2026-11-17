from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np
import pandas as pd


TIMESTAMP_CANDIDATES = ("t_stamp", "timestamp", "datetime", "time", "date")
LABEL_CANDIDATES = ("label", "attack", "attack_label", "anomaly", "normal/attack")


@dataclass(frozen=True)
class SWaT2026Data:
    features: np.ndarray
    labels: np.ndarray | None
    timestamps: np.ndarray
    feature_names: list[str]
    feature_roles: dict[str, str]
    categorical_mappings: dict[str, dict[str, float]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SWaT2026Windows:
    features: np.ndarray
    labels: np.ndarray | None
    timestamps: np.ndarray
    start_indices: np.ndarray
    end_indices: np.ndarray


def _normalized(name: Any) -> str:
    return str(name).strip().lower()


def _resolve_column(columns: list[str], requested: str | None, candidates: tuple[str, ...]) -> str | None:
    lookup = {_normalized(column): column for column in columns}
    if requested is not None:
        resolved = lookup.get(_normalized(requested))
        if resolved is None:
            raise ValueError(f"Column {requested!r} was not found")
        return resolved
    return next((lookup[name] for name in candidates if name in lookup), None)


def infer_swat2026_role(column: str, series: pd.Series | None = None) -> str:
    name = _normalized(column)
    if name in TIMESTAMP_CANDIDATES:
        return "timestamp"
    if name in LABEL_CANDIDATES:
        return "label"
    if name.endswith(".alarm") or name.endswith(".al"):
        return "alarm_state"
    if name.endswith("_state"):
        return "process_state"
    if name.endswith(".status") or name.endswith(".stat"):
        return "process_state"
    if name.endswith(".pv") or name.endswith(".speed"):
        return "continuous_numeric"
    if series is not None:
        numeric = pd.to_numeric(series, errors="coerce")
        parse_ratio = float(numeric.notna().mean()) if len(series) else 1.0
        if parse_ratio == 1.0:
            values = set(numeric.dropna().unique().tolist())
            return "binary_numeric" if values.issubset({0, 1, 0.0, 1.0}) else "continuous_numeric"
        values = {_normalized(value) for value in series.dropna().unique()}
        if values and values.issubset({"active", "inactive"}):
            return "categorical_state"
    return "unknown"


def _parse_labels(series: pd.Series) -> np.ndarray:
    mapping = {"normal": 0, "0": 0, "0.0": 0, "false": 0, "attack": 1, "1": 1, "1.0": 1, "true": 1}
    normalized = series.map(_normalized)
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"Unsupported SWaT label values: {unknown}")
    return normalized.map(mapping).to_numpy(np.int64)


def _encode_categorical(
    series: pd.Series,
    column: str,
    configured: Mapping[str, Mapping[Any, float]],
    unknown_policy: str,
) -> tuple[np.ndarray, dict[str, float]]:
    raw_values = series.map(lambda value: str(value).strip() if pd.notna(value) else value)
    mapping = {str(key).strip(): float(value) for key, value in configured.get(column, {}).items()}
    observed = {value for value in raw_values.dropna().unique().tolist()}
    if not mapping and observed and observed.issubset({"Active", "Inactive"}):
        mapping = {"Active": 1.0, "Inactive": 0.0}
    unknown = sorted(str(value) for value in observed if value not in mapping)
    if raw_values.isna().any():
        unknown.append("<missing>")
    if unknown:
        message = f"Unknown categorical values in {column}: {unknown}"
        if unknown_policy == "error":
            raise ValueError(message)
        if unknown_policy != "warn":
            raise ValueError(f"Unsupported unknown_category_policy: {unknown_policy!r}")
        warnings.warn(message + "; encoding as -1", RuntimeWarning, stacklevel=2)
    encoded = raw_values.map(mapping).fillna(-1.0 if unknown_policy == "warn" else np.nan).to_numpy(np.float32)
    return encoded, mapping


def load_swat2026_csv(
    csv_path: str | Path,
    timestamp_column: str | None = None,
    label_column: str | None = None,
    feature_mode: str = "process_values_only",
    include_status_columns: bool = False,
    include_alarm_columns: bool = False,
    categorical_mappings: Mapping[str, Mapping[Any, float]] | None = None,
    unknown_category_policy: str = "error",
    missing_numeric_policy: str = "error",
    max_rows: int | None = None,
) -> SWaT2026Data:
    path = Path(csv_path)
    frame = pd.read_csv(path, nrows=max_rows, low_memory=False)
    raw_columns = [str(column) for column in frame.columns]
    columns = [column.strip() for column in raw_columns]
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate columns after trimming whitespace")
    frame.columns = columns
    timestamp = _resolve_column(columns, timestamp_column, TIMESTAMP_CANDIDATES)
    label = _resolve_column(columns, label_column, LABEL_CANDIDATES)
    if timestamp is None:
        raise ValueError(f"No timestamp column found; candidates={TIMESTAMP_CANDIDATES}")
    parsed_time = pd.to_datetime(frame[timestamp], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    if parsed_time.isna().any():
        parsed_time = pd.to_datetime(frame[timestamp], errors="coerce")
    if parsed_time.isna().any():
        raise ValueError(f"Unparseable timestamps in {timestamp}")

    roles = {column: infer_swat2026_role(column, frame[column]) for column in columns}
    candidates = [column for column in columns if column not in {timestamp, label}]
    if feature_mode == "process_values_only":
        selected = [column for column in candidates if _normalized(column).endswith(".pv")]
    elif feature_mode == "all_numeric":
        selected = [column for column in candidates if roles[column] in {"continuous_numeric", "binary_numeric"}]
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode!r}")
    if include_status_columns:
        selected.extend(column for column in candidates if roles[column] == "process_state" and column not in selected)
    if include_alarm_columns:
        selected.extend(column for column in candidates if roles[column] == "alarm_state" and column not in selected)
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate feature columns selected")
    if not selected:
        raise ValueError("No SWaT 2026 feature columns selected")

    configured = categorical_mappings or {}
    arrays: list[np.ndarray] = []
    valid_rows = np.ones(len(frame), dtype=bool)
    invalid_numeric_rows: dict[str, list[int]] = {}
    used_mappings: dict[str, dict[str, float]] = {}
    for column in selected:
        series = frame[column]
        if roles[column] in {"continuous_numeric", "binary_numeric", "process_state"}:
            numeric = pd.to_numeric(series, errors="coerce")
            invalid = numeric.isna() | ~np.isfinite(numeric.fillna(0.0).to_numpy(dtype=float))
            if invalid.any():
                bad_indices = np.flatnonzero(invalid.to_numpy()).astype(int).tolist()
                invalid_numeric_rows[column] = bad_indices
                if missing_numeric_policy == "error":
                    bad_values = series.iloc[bad_indices[:5]].astype(str).tolist()
                    raise ValueError(f"Missing numeric or non-numeric values in {column}: {bad_values}")
                if missing_numeric_policy != "drop_rows":
                    raise ValueError(f"Unsupported missing_numeric_policy: {missing_numeric_policy!r}")
                valid_rows &= ~invalid.to_numpy()
            arrays.append(numeric.to_numpy(np.float32))
        else:
            encoded, mapping = _encode_categorical(series, column, configured, unknown_category_policy)
            arrays.append(encoded)
            used_mappings[column] = mapping
    features = np.column_stack(arrays).astype(np.float32, copy=False)
    dropped_indices = np.flatnonzero(~valid_rows).astype(int).tolist()
    if dropped_indices:
        warnings.warn(
            f"Explicit drop_rows policy removed {len(dropped_indices)} rows containing invalid numeric process values",
            RuntimeWarning,
            stacklevel=2,
        )
        features = features[valid_rows]
        parsed_time = parsed_time[valid_rows]
    if not np.isfinite(features).all():
        raise ValueError("SWaT 2026 features contain NaN or Inf")
    labels = _parse_labels(frame[label]) if label is not None else None
    if labels is not None and dropped_indices:
        labels = labels[valid_rows]
    metadata = {
        "csv_path": str(path),
        "row_count": int(len(frame)),
        "timestamp_column": timestamp,
        "label_column": label,
        "feature_mode": feature_mode,
        "include_status_columns": bool(include_status_columns),
        "include_alarm_columns": bool(include_alarm_columns),
        "scaler_fitted": False,
        "checkpoint_loaded": False,
        "missing_numeric_policy": missing_numeric_policy,
        "dropped_row_count": len(dropped_indices),
        "dropped_row_indices": dropped_indices,
        "invalid_numeric_rows_by_column": invalid_numeric_rows,
    }
    return SWaT2026Data(
        features=features,
        labels=labels,
        timestamps=parsed_time.to_numpy(dtype="datetime64[ns]"),
        feature_names=selected,
        feature_roles={column: roles[column] for column in selected},
        categorical_mappings=used_mappings,
        metadata=metadata,
    )


def build_swat2026_windows(
    features: np.ndarray,
    labels: np.ndarray | None,
    timestamps: np.ndarray,
    window_size: int,
    stride: int,
) -> SWaT2026Windows:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if features.ndim != 2 or len(features) != len(timestamps) or (labels is not None and len(labels) != len(features)):
        raise ValueError("Feature, label, and timestamp lengths must align")
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or Inf")
    if len(features) < window_size:
        raise ValueError("Input length is smaller than window_size")
    starts = np.arange(0, len(features) - window_size + 1, stride, dtype=np.int64)
    ends = starts + window_size - 1
    windows = np.stack([features[start : start + window_size] for start in starts]).astype(np.float32, copy=False)
    window_labels = labels[ends].astype(np.int64, copy=False) if labels is not None else None
    return SWaT2026Windows(windows, window_labels, np.asarray(timestamps)[ends], starts, ends)
