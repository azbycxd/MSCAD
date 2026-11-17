from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from data.transforms import StandardScaler


def _is_attack(v: str) -> int:
    s = str(v).strip().lower()
    return 0 if s in {"normal", "0", "false"} else 1


def _norm_col_name(name: str) -> str:
    return str(name).strip().lower()


def _resolve_col(df: pd.DataFrame, target: str) -> str | None:
    t = _norm_col_name(target)
    for c in df.columns:
        if _norm_col_name(c) == t:
            return c
    return None


def _load_swat_csv(path: str, label_col: str, timestamp_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path)
    label_col_real = _resolve_col(df, label_col)
    ts_col_real = _resolve_col(df, timestamp_col)
    exclude = {c for c in [label_col_real, ts_col_real] if c is not None}
    cols = [c for c in df.columns if c not in exclude]
    x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    if label_col_real is not None:
        y = df[label_col_real].apply(_is_attack).to_numpy(np.int64)
    else:
        y = np.zeros(len(df), dtype=np.int64)
    return x, y, cols


class SWaTWindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window_size: int,
        stride: int,
    ) -> None:
        self.x = x
        self.y = y
        self.window_size = int(window_size)
        self.stride = int(stride)
        if len(x) < window_size:
            raise ValueError("Input length is smaller than window_size.")
        self.starts = list(range(0, len(x) - window_size + 1, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        e = s + self.window_size
        xw = self.x[s:e]
        yw = int(self.y[s:e].max())
        return torch.from_numpy(xw), torch.tensor(yw, dtype=torch.long)


@dataclass
class SWaTBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler: StandardScaler
    feature_names: List[str]
    x_train_norm: np.ndarray
    x_val_norm: np.ndarray


def build_swat_dataloaders(
    train_csv: str,
    test_csv: str,
    label_col: str,
    timestamp_col: str,
    window_size: int,
    stride_train: int,
    stride_test: int,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
) -> SWaTBundle:
    x_train, y_train, feature_names = _load_swat_csv(train_csv, label_col, timestamp_col)
    x_test, y_test, _ = _load_swat_csv(test_csv, label_col, timestamp_col)

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

    train_ds = SWaTWindowDataset(x_train_norm, y_train_raw, window_size=window_size, stride=stride_train)
    val_ds = SWaTWindowDataset(x_val_norm, y_val_raw, window_size=window_size, stride=stride_test)
    test_ds = SWaTWindowDataset(x_test_norm, y_test, window_size=window_size, stride=stride_test)

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
    return SWaTBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        feature_names=feature_names,
        x_train_norm=x_train_norm,
        x_val_norm=x_val_norm,
    )
