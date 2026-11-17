from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from data.transforms import StandardScaler
from utils.seed import seed_worker


EXPECTED_TRAIN_FILES = [f"train{i}.csv" for i in range(1, 7)]
EXPECTED_TEST_FILES = [f"test{i}.csv" for i in range(1, 5)]
TIMESTAMP_CANDIDATES = ("timestamp", "datetime", "time", "date")
LABEL_CANDIDATES = ("attack", "attack_label", "label", "anomaly")


def _resolve_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    return next((normalized[name] for name in candidates if name in normalized), None)


def _parse_label(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"0", "0.0", "normal", "false"}:
        return 0
    if text in {"1", "1.0", "attack", "true"}:
        return 1
    raise ValueError(f"Unsupported HAI label value: {value!r}")


@dataclass
class HAISequence:
    sequence_id: str
    file_name: str
    split: str
    x: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    original_row_offset: int = 0

    def __post_init__(self) -> None:
        if self.x.ndim != 2 or len(self.x) != len(self.y) or len(self.x) != len(self.timestamps):
            raise ValueError("HAISequence x/y/timestamps lengths must match and x must be 2D.")


def load_hai_csv(
    path: str | Path,
    expected_features: list[str] | None = None,
    expected_feature_count: int = 86,
    max_rows: int | None = None,
) -> tuple[HAISequence, list[str]]:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    raw_columns = [str(column) for column in frame.columns]
    stripped = [column.strip() for column in raw_columns]
    if len(stripped) != len(set(stripped)):
        raise ValueError(f"Duplicate columns after strip in {csv_path}")
    rename = dict(zip(raw_columns, stripped))
    frame = frame.rename(columns=rename)
    timestamp_col = _resolve_column(frame.columns, TIMESTAMP_CANDIDATES)
    label_col = _resolve_column(frame.columns, LABEL_CANDIDATES)
    if timestamp_col is None or label_col is None:
        raise ValueError(f"Could not resolve timestamp/label columns in {csv_path}")
    feature_names = [column for column in frame.columns if column not in {timestamp_col, label_col}]
    if len(feature_names) != expected_feature_count:
        raise ValueError(f"Expected {expected_feature_count} HAI features, got {len(feature_names)} in {csv_path}")
    if expected_features is not None and feature_names != expected_features:
        raise ValueError(f"Feature schema/order changed in {csv_path}")
    timestamps = pd.to_datetime(frame[timestamp_col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"Unparseable timestamp in {csv_path}")
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError(f"Timestamps are not strictly increasing in {csv_path}")
    y = frame[label_col].map(_parse_label).to_numpy(np.int64)
    numeric = frame[feature_names].apply(pd.to_numeric, errors="raise")
    x = numeric.to_numpy(np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite feature value in {csv_path}")
    sequence = HAISequence(
        sequence_id=csv_path.stem,
        file_name=csv_path.name,
        split="train" if csv_path.stem.startswith("train") else "test",
        x=x,
        y=y,
        timestamps=timestamps.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(str),
    )
    return sequence, feature_names


class MultiSequenceWindowDataset(Dataset):
    def __init__(
        self,
        sequences: list[HAISequence],
        window_size: int = 64,
        stride: int = 1,
        label_mode: str = "end",
        return_metadata: bool = False,
    ) -> None:
        if label_mode != "end":
            raise ValueError("HAI online alignment requires label_mode='end'.")
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive.")
        self.sequences = sequences
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.label_mode = label_mode
        self.return_metadata = bool(return_metadata)
        self.window_counts = np.asarray(
            [max(0, (len(sequence.x) - self.window_size) // self.stride + 1) for sequence in sequences], dtype=np.int64
        )
        self.cumulative_offsets = np.concatenate(([0], np.cumsum(self.window_counts))).astype(np.int64)

    def __len__(self) -> int:
        return int(self.cumulative_offsets[-1])

    def locate_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sequence_index = bisect_right(self.cumulative_offsets, index) - 1
        local_window_index = index - int(self.cumulative_offsets[sequence_index])
        return sequence_index, local_window_index

    def __getitem__(self, index: int):
        sequence_index, local_window_index = self.locate_index(index)
        sequence = self.sequences[sequence_index]
        start = local_window_index * self.stride
        end = start + self.window_size - 1
        x = torch.from_numpy(sequence.x[start : end + 1])
        y = torch.tensor(int(sequence.y[end]), dtype=torch.long)
        if not self.return_metadata:
            return x, y
        metadata = {
            "sequence_id": sequence.sequence_id,
            "file_name": sequence.file_name,
            "start_row": sequence.original_row_offset + start,
            "end_row": sequence.original_row_offset + end,
            "timestamp": str(sequence.timestamps[end]),
            "split": sequence.split,
            "original_label": int(sequence.y[end]),
        }
        return x, y, metadata


@dataclass
class HAIDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loaders: dict[str, DataLoader]
    test_loader: DataLoader
    scaler: StandardScaler
    feature_names: list[str]
    train_sequences: list[HAISequence]
    val_sequences: list[HAISequence]
    test_sequences: list[HAISequence]
    split_manifest: list[dict[str, Any]]
    attack_events: list[dict[str, Any]]
    feature_activity: list[dict[str, Any]]

    @property
    def x_train_norm(self) -> np.ndarray:
        return np.concatenate([sequence.x for sequence in self.train_sequences], axis=0)

    @property
    def x_val_norm(self) -> np.ndarray:
        return np.concatenate([sequence.x for sequence in self.val_sequences], axis=0)


def extract_attack_events(sequences: list[HAISequence]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    global_id = 0
    for sequence in sequences:
        starts = np.flatnonzero((sequence.y == 1) & np.r_[True, sequence.y[:-1] == 0])
        ends = np.flatnonzero((sequence.y == 1) & np.r_[sequence.y[1:] == 0, True])
        for event_id, (start, end) in enumerate(zip(starts, ends), 1):
            global_id += 1
            events.append({
                "sequence_id": sequence.sequence_id, "file_name": sequence.file_name,
                "event_id_in_file": event_id, "global_event_id": global_id,
                "start_row": int(start), "end_row": int(end),
                "start_timestamp": str(sequence.timestamps[start]), "end_timestamp": str(sequence.timestamps[end]),
                "duration_points": int(end - start + 1),
            })
    return events


def _feature_activity(train_raw: list[HAISequence], feature_names: list[str], eps: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(feature_names):
        arrays = [sequence.x[:, feature_index] for sequence in train_raw]
        combined = np.concatenate(arrays)
        unique, counts = np.unique(combined, return_counts=True)
        global_std = float(combined.astype(np.float64).std())
        top_ratio = float(counts.max() / len(combined))
        for sequence, values in zip(train_raw, arrays):
            seq_unique, seq_counts = np.unique(values, return_counts=True)
            rows.append({
                "feature": feature, "scope": sequence.sequence_id, "raw_std": float(values.astype(np.float64).std()),
                "most_frequent_ratio": float(seq_counts.max() / len(values)), "active": len(seq_unique) > 1,
                "constant": len(seq_unique) == 1, "near_constant": float(seq_counts.max() / len(values)) >= 0.999,
            })
        rows.append({
            "feature": feature, "scope": "global_train_fit", "raw_std": global_std,
            "most_frequent_ratio": top_ratio, "active": global_std >= eps and len(unique) > 1,
            "constant": global_std < eps or len(unique) == 1,
            "near_constant": global_std >= eps and top_ratio >= 0.999,
        })
    return rows


def build_hai_dataloaders(
    data_root: str | Path,
    window_size: int = 64,
    stride_train: int = 1,
    stride_test: int = 1,
    validation_ratio: float = 0.20,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    return_metadata: bool = False,
    max_rows_per_file: int | None = None,
) -> HAIDataBundle:
    root = Path(data_root)
    paths = [root / name for name in EXPECTED_TRAIN_FILES + EXPECTED_TEST_FILES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing HAI files: {missing}")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")
    loaded: list[HAISequence] = []
    feature_names: list[str] | None = None
    for path in paths:
        sequence, names = load_hai_csv(path, expected_features=feature_names, max_rows=max_rows_per_file)
        feature_names = names if feature_names is None else feature_names
        loaded.append(sequence)
    assert feature_names is not None
    train_full = loaded[:6]
    test_raw = loaded[6:]
    train_raw: list[HAISequence] = []
    val_raw: list[HAISequence] = []
    manifest: list[dict[str, Any]] = []
    for sequence in train_full:
        split_index = int(len(sequence.x) * (1.0 - validation_ratio))
        if split_index < window_size or len(sequence.x) - split_index < window_size:
            raise ValueError(f"Sequence {sequence.file_name} is too short for independent train/validation windows.")
        train_sequence = HAISequence(sequence.sequence_id, sequence.file_name, "train", sequence.x[:split_index],
                                     sequence.y[:split_index], sequence.timestamps[:split_index], 0)
        val_sequence = HAISequence(sequence.sequence_id, sequence.file_name, "validation", sequence.x[split_index:],
                                   sequence.y[split_index:], sequence.timestamps[split_index:], split_index)
        train_raw.append(train_sequence)
        val_raw.append(val_sequence)
        manifest.append({
            "file_name": sequence.file_name, "original_rows": len(sequence.x), "train_rows": split_index,
            "validation_rows": len(sequence.x) - split_index, "split_index": split_index,
            "train_start_timestamp": str(train_sequence.timestamps[0]), "train_end_timestamp": str(train_sequence.timestamps[-1]),
            "validation_start_timestamp": str(val_sequence.timestamps[0]), "validation_end_timestamp": str(val_sequence.timestamps[-1]),
        })
    scaler = StandardScaler(eps=1e-8).fit_many([sequence.x for sequence in train_raw])
    activity = _feature_activity(train_raw, feature_names, scaler.eps)

    def normalized(sequence: HAISequence) -> HAISequence:
        x = scaler.transform(sequence.x).astype(np.float32, copy=False)
        if not np.isfinite(x).all():
            raise ValueError(f"Scaler produced non-finite values for {sequence.file_name}/{sequence.split}")
        return HAISequence(sequence.sequence_id, sequence.file_name, sequence.split, x, sequence.y,
                           sequence.timestamps, sequence.original_row_offset)

    train_sequences = [normalized(sequence) for sequence in train_raw]
    val_sequences = [normalized(sequence) for sequence in val_raw]
    test_sequences = [normalized(sequence) for sequence in test_raw]
    train_dataset = MultiSequenceWindowDataset(train_sequences, window_size, stride_train, return_metadata=return_metadata)
    val_dataset = MultiSequenceWindowDataset(val_sequences, window_size, stride_test, return_metadata=return_metadata)
    test_datasets = {sequence.sequence_id: MultiSequenceWindowDataset([sequence], window_size, stride_test,
                                                                      return_metadata=return_metadata) for sequence in test_sequences}
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator,
                              worker_init_fn=seed_worker, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            worker_init_fn=seed_worker, pin_memory=pin_memory, drop_last=False)
    test_loaders = {name: DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                     worker_init_fn=seed_worker, pin_memory=pin_memory, drop_last=False) for name, dataset in test_datasets.items()}
    aggregate_test_dataset = ConcatDataset(list(test_datasets.values()))
    test_loader = DataLoader(aggregate_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             worker_init_fn=seed_worker, pin_memory=pin_memory, drop_last=False)
    return HAIDataBundle(
        train_loader=train_loader, val_loader=val_loader, test_loaders=test_loaders, test_loader=test_loader,
        scaler=scaler, feature_names=feature_names, train_sequences=train_sequences, val_sequences=val_sequences,
        test_sequences=test_sequences, split_manifest=manifest, attack_events=extract_attack_events(test_raw),
        feature_activity=activity,
    )
