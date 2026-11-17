from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import ast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.transforms import StandardScaler


def _load_array(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() == ".npy":
        x = np.load(str(p))
    elif p.suffix.lower() in {".csv", ".txt"}:
        x = pd.read_csv(str(p), header=None).to_numpy()
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array [T,D], got shape={x.shape} from {path}")
    return x.astype(np.float32)


def _load_label(path: str, length: int) -> np.ndarray:
    y = _load_array(path)
    if y.shape[1] > 1:
        # If point labels are given as multi-column flags, collapse to any-anomaly.
        y = (y.max(axis=1, keepdims=True) > 0).astype(np.int64)
    y = y.reshape(-1)
    if len(y) != length:
        raise ValueError(f"Label length mismatch: len(y)={len(y)} vs length={length}, file={path}")
    return y.astype(np.int64)


def _resolve_omni_channel(base_dir: str, channel: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = Path(base_dir)
    tr = b / "train" / f"{channel}.npy"
    te = b / "test" / f"{channel}.npy"
    yl = b / "test_label" / f"{channel}.npy"
    x_train = _load_array(str(tr))
    x_test = _load_array(str(te))
    if yl.exists():
        y_test = _load_label(str(yl), len(x_test))
    else:
        raise FileNotFoundError(f"Missing test label file: {yl}")
    return x_train, x_test, y_test


def _label_from_csv(
    csv_path: str,
    channel: str,
    length: int,
    spacecraft: str | None = None,
) -> np.ndarray:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Label CSV not found: {csv_path}")
    df = pd.read_csv(str(p))
    if "chan_id" not in df.columns or "anomaly_sequences" not in df.columns:
        raise ValueError("labeled_anomalies.csv must contain chan_id and anomaly_sequences columns.")
    sub = df[df["chan_id"].astype(str) == str(channel)]
    if spacecraft is not None and "spacecraft" in df.columns:
        sub = sub[sub["spacecraft"].astype(str) == str(spacecraft)]
    if len(sub) == 0:
        raise ValueError(f"Channel {channel} not found in {csv_path}")
    row = sub.iloc[0]
    raw = str(row["anomaly_sequences"])
    seqs = ast.literal_eval(raw) if raw.strip() else []
    y = np.zeros(length, dtype=np.int64)
    for seg in seqs:
        if not isinstance(seg, (list, tuple)) or len(seg) != 2:
            continue
        s = max(0, int(seg[0]))
        e = min(length - 1, int(seg[1]))
        if e >= s:
            y[s : e + 1] = 1
    return y


class SMAPWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, window_size: int, stride: int) -> None:
        self.x = x
        self.y = y
        self.window_size = int(window_size)
        self.stride = int(stride)
        if len(x) < self.window_size:
            raise ValueError("Input length is smaller than window_size.")
        self.starts = list(range(0, len(x) - self.window_size + 1, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        e = s + self.window_size
        xw = self.x[s:e]
        yw = int(self.y[s:e].max())
        return torch.from_numpy(xw), torch.tensor(yw, dtype=torch.long)


@dataclass
class SMAPBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler: StandardScaler
    feature_names: List[str]
    x_train_norm: np.ndarray
    x_val_norm: np.ndarray


def build_smap_dataloaders(
    train_path: str | None = None,
    test_path: str | None = None,
    test_label_path: str | None = None,
    omni_dir: str | None = None,
    channel: str | None = None,
    labeled_anomalies_csv: str | None = None,
    spacecraft: str | None = "SMAP",
    window_size: int = 64,
    stride_train: int = 1,
    stride_test: int = 1,
    val_ratio: float = 0.2,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> SMAPBundle:
    if omni_dir is not None and channel is not None:
        try:
            x_train, x_test, y_test = _resolve_omni_channel(omni_dir, channel)
        except FileNotFoundError:
            # NASA SMAP original format often ships with labeled_anomalies.csv instead of point label .npy files.
            b = Path(omni_dir)
            tr = b / "train" / f"{channel}.npy"
            te = b / "test" / f"{channel}.npy"
            x_train = _load_array(str(tr))
            x_test = _load_array(str(te))
            if labeled_anomalies_csv is None:
                raise ValueError(
                    "test_label/*.npy not found. Please provide data.labeled_anomalies_csv to build point labels."
                )
            y_test = _label_from_csv(labeled_anomalies_csv, channel=channel, length=len(x_test), spacecraft=spacecraft)
    else:
        if train_path is None or test_path is None or test_label_path is None:
            raise ValueError("For SMAP, provide either (omni_dir + channel) or (train_path + test_path + test_label_path).")
        x_train = _load_array(train_path)
        x_test = _load_array(test_path)
        y_test = _load_label(test_label_path, len(x_test))

    y_train = np.zeros(len(x_train), dtype=np.int64)
    n = len(x_train)
    n_val = int(max(window_size * 2, round(n * float(val_ratio))))
    n_val = min(max(window_size * 2, n_val), n - window_size - 1)
    split = n - n_val
    x_train_raw, y_train_raw = x_train[:split], y_train[:split]
    x_val_raw, y_val_raw = x_train[split:], y_train[split:]

    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train_raw)
    x_val_norm = scaler.transform(x_val_raw)
    x_test_norm = scaler.transform(x_test)

    train_ds = SMAPWindowDataset(x_train_norm, y_train_raw, window_size=window_size, stride=stride_train)
    val_ds = SMAPWindowDataset(x_val_norm, y_val_raw, window_size=window_size, stride=stride_test)
    test_ds = SMAPWindowDataset(x_test_norm, y_test, window_size=window_size, stride=stride_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    feature_names = [f"f{i}" for i in range(x_train.shape[1])]
    return SMAPBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        feature_names=feature_names,
        x_train_norm=x_train_norm,
        x_val_norm=x_val_norm,
    )
