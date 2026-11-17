from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.transforms import StandardScaler


def _build_feature_names(n: int) -> List[str]:
    return [f"f{i}" for i in range(n)]


def _window_starts_by_group(groups: np.ndarray, window_size: int, stride: int) -> List[int]:
    starts: List[int] = []
    n = len(groups)
    if n < window_size:
        return starts
    seg_start = 0
    while seg_start < n:
        g = groups[seg_start]
        seg_end = seg_start + 1
        while seg_end < n and groups[seg_end] == g:
            seg_end += 1
        seg_len = seg_end - seg_start
        if seg_len >= window_size:
            starts.extend(range(seg_start, seg_end - window_size + 1, stride))
        seg_start = seg_end
    return starts


class CMAPSSWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray, window_size: int, stride: int) -> None:
        self.x = x
        self.y = y
        self.groups = groups
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.starts = _window_starts_by_group(self.groups, self.window_size, self.stride)
        if len(self.starts) <= 0:
            raise ValueError("No valid windows found. Check window_size/stride or grouped sequence lengths.")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        e = s + self.window_size
        xw = self.x[s:e]
        yw = int(self.y[s:e].max())
        return torch.from_numpy(xw), torch.tensor(yw, dtype=torch.long)


@dataclass
class CMAPSSBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler: StandardScaler
    feature_names: List[str]
    x_train_norm: np.ndarray
    x_val_norm: np.ndarray


def _decode_feature_names(raw: np.ndarray | None, fallback_dim: int) -> List[str]:
    if raw is None:
        return _build_feature_names(fallback_dim)
    vals = []
    for x in raw.tolist():
        if isinstance(x, bytes):
            vals.append(x.decode("utf-8"))
        else:
            vals.append(str(x))
    if len(vals) != fallback_dim:
        return _build_feature_names(fallback_dim)
    return vals


def _calc_rul_from_train(df: pd.DataFrame) -> np.ndarray:
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    rul = max_cycle - df["cycle"]
    return rul.to_numpy(np.int64)


def _calc_rul_from_test(df: pd.DataFrame, rul_file: str) -> np.ndarray:
    rul_tail = pd.read_csv(rul_file, sep=r"\s+", header=None, engine="python")
    rul_tail = rul_tail.dropna(axis=1, how="all")
    rul_end = rul_tail.iloc[:, 0].to_numpy(np.int64)
    unit_ids = np.sort(df["unit"].unique())
    if len(rul_end) != len(unit_ids):
        raise ValueError(f"RUL file length mismatch: len(RUL)={len(rul_end)} vs units={len(unit_ids)}")
    max_cycle = df.groupby("unit")["cycle"].transform("max").to_numpy(np.int64)
    unit_to_rul_end = {int(u): int(r) for u, r in zip(unit_ids, rul_end)}
    this_rul_end = df["unit"].map(unit_to_rul_end).to_numpy(np.int64)
    rul = this_rul_end + (max_cycle - df["cycle"].to_numpy(np.int64))
    return rul


def _make_anomaly_label(rul: np.ndarray, mode: str, last_k: int, rul_threshold: int) -> np.ndarray:
    m = str(mode).lower()
    if m == "last_k":
        # Failure cycle is RUL=0, so last K cycles means RUL in [0, K-1].
        return (rul < int(last_k)).astype(np.int64)
    if m == "rul_threshold":
        return (rul <= int(rul_threshold)).astype(np.int64)
    raise ValueError(f"Unsupported CMAPSS label_mode: {mode}")


def _load_fd001_as_arrays(
    train_path: str,
    test_path: str,
    rul_path: str,
    label_mode: str,
    last_k_cycles: int,
    rul_threshold: int,
    use_ops: bool,
    use_sensors: bool,
    near_const_var_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

    def read_fd(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        df = df.dropna(axis=1, how="all")
        if df.shape[1] < len(cols):
            raise ValueError(f"FD file has too few columns: {path}, got={df.shape[1]}")
        df = df.iloc[:, : len(cols)].copy()
        df.columns = cols
        return df

    tr = read_fd(train_path)
    te = read_fd(test_path)

    feat_cols: List[str] = []
    if use_ops:
        feat_cols.extend(["op1", "op2", "op3"])
    if use_sensors:
        feat_cols.extend([f"s{i}" for i in range(1, 22)])
    if len(feat_cols) <= 0:
        raise ValueError("At least one of use_ops/use_sensors must be True.")

    rul_tr = _calc_rul_from_train(tr)
    rul_te = _calc_rul_from_test(te, rul_path)
    y_tr_anom = _make_anomaly_label(rul_tr, label_mode, last_k_cycles, rul_threshold)
    y_te = _make_anomaly_label(rul_te, label_mode, last_k_cycles, rul_threshold)

    x_tr_all = tr[feat_cols].to_numpy(np.float32)
    x_te_all = te[feat_cols].to_numpy(np.float32)
    unit_tr_all = tr["unit"].to_numpy(np.int64)
    unit_te = te["unit"].to_numpy(np.int64)

    keep_train = (y_tr_anom == 0)
    x_tr = x_tr_all[keep_train]
    unit_tr = unit_tr_all[keep_train]
    y_tr = np.zeros(len(x_tr), dtype=np.int64)

    # Remove near-constant features based on normal-only training split.
    v = np.var(x_tr, axis=0)
    keep_feat = v > float(near_const_var_threshold)
    if not np.any(keep_feat):
        keep_feat = np.ones_like(v, dtype=bool)
    x_tr = x_tr[:, keep_feat]
    x_te = x_te_all[:, keep_feat]
    feat_cols = [c for c, k in zip(feat_cols, keep_feat.tolist()) if k]

    return x_tr, y_tr, unit_tr, x_te, y_te, unit_te, feat_cols


def _load_npz_as_arrays(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"CMAPSS npz not found: {npz_path}")
    z = np.load(str(p), allow_pickle=True)
    req = ["train_x", "train_y", "train_unit", "test_x", "test_y", "test_unit"]
    miss = [k for k in req if k not in z]
    if miss:
        raise ValueError(f"CMAPSS npz missing keys: {miss}, required={req}")
    x_tr = np.asarray(z["train_x"], dtype=np.float32)
    y_tr = np.asarray(z["train_y"], dtype=np.int64).reshape(-1)
    u_tr = np.asarray(z["train_unit"], dtype=np.int64).reshape(-1)
    x_te = np.asarray(z["test_x"], dtype=np.float32)
    y_te = np.asarray(z["test_y"], dtype=np.int64).reshape(-1)
    u_te = np.asarray(z["test_unit"], dtype=np.int64).reshape(-1)
    if len(x_tr) != len(y_tr) or len(x_tr) != len(u_tr):
        raise ValueError("train_x/train_y/train_unit length mismatch in CMAPSS npz.")
    if len(x_te) != len(y_te) or len(x_te) != len(u_te):
        raise ValueError("test_x/test_y/test_unit length mismatch in CMAPSS npz.")
    feature_names = _decode_feature_names(z["feature_names"] if "feature_names" in z else None, x_tr.shape[1])
    return x_tr, y_tr, u_tr, x_te, y_te, u_te, feature_names


def build_cmapss_dataloaders(
    source: str,
    window_size: int,
    stride_train: int,
    stride_test: int,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    npz_path: str | None = None,
    train_path: str | None = None,
    test_path: str | None = None,
    rul_path: str | None = None,
    label_mode: str = "last_k",
    last_k_cycles: int = 30,
    rul_threshold: int = 30,
    use_ops: bool = True,
    use_sensors: bool = True,
    near_const_var_threshold: float = 1e-8,
) -> CMAPSSBundle:
    src = str(source).lower()
    if src == "npz":
        if npz_path is None:
            raise ValueError("CMAPSS source=npz requires data.npz_path.")
        x_train_all, y_train_all, unit_train_all, x_test, y_test, unit_test, feature_names = _load_npz_as_arrays(npz_path)
    elif src == "fd001_txt":
        if train_path is None or test_path is None or rul_path is None:
            raise ValueError("CMAPSS source=fd001_txt requires train_path, test_path, rul_path.")
        (
            x_train_all,
            y_train_all,
            unit_train_all,
            x_test,
            y_test,
            unit_test,
            feature_names,
        ) = _load_fd001_as_arrays(
            train_path=train_path,
            test_path=test_path,
            rul_path=rul_path,
            label_mode=label_mode,
            last_k_cycles=int(last_k_cycles),
            rul_threshold=int(rul_threshold),
            use_ops=bool(use_ops),
            use_sensors=bool(use_sensors),
            near_const_var_threshold=float(near_const_var_threshold),
        )
    else:
        raise ValueError(f"Unsupported CMAPSS source: {source}")

    n = len(x_train_all)
    if n < (window_size * 4 + 2):
        raise ValueError(
            f"CMAPSS train length ({n}) is too small for window_size={window_size}. "
            "Please reduce window_size or provide more training rows."
        )
    n_val = int(max(window_size * 2, round(n * float(val_ratio))))
    n_val = min(max(window_size * 2, n_val), n - window_size - 1)
    split = n - n_val

    x_train_raw = x_train_all[:split]
    y_train_raw = y_train_all[:split]
    unit_train_raw = unit_train_all[:split]

    x_val_raw = x_train_all[split:]
    y_val_raw = y_train_all[split:]
    unit_val_raw = unit_train_all[split:]

    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train_raw)
    x_val_norm = scaler.transform(x_val_raw)
    x_test_norm = scaler.transform(x_test)

    train_ds = CMAPSSWindowDataset(
        x=x_train_norm,
        y=y_train_raw,
        groups=unit_train_raw,
        window_size=window_size,
        stride=stride_train,
    )
    val_ds = CMAPSSWindowDataset(
        x=x_val_norm,
        y=y_val_raw,
        groups=unit_val_raw,
        window_size=window_size,
        stride=stride_test,
    )
    test_ds = CMAPSSWindowDataset(
        x=x_test_norm,
        y=y_test,
        groups=unit_test,
        window_size=window_size,
        stride=stride_test,
    )

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

    return CMAPSSBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        feature_names=feature_names,
        x_train_norm=x_train_norm,
        x_val_norm=x_val_norm,
    )
