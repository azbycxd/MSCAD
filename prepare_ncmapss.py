from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import h5py
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "prepare_ncmapss.py requires h5py. "
        "Use a Python env that has h5py (for example D:/soft/anaconda3/python.exe)."
    ) from e

from utils.config_io import save_json


def _decode_str_array(x: np.ndarray) -> List[str]:
    out = []
    for v in x.tolist():
        if isinstance(v, bytes):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out


def _downsample_indices_by_unit(unit_ids: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return np.arange(len(unit_ids), dtype=np.int64)
    keep = []
    n = len(unit_ids)
    s = 0
    while s < n:
        u = unit_ids[s]
        e = s + 1
        while e < n and unit_ids[e] == u:
            e += 1
        idx = np.arange(s, e, factor, dtype=np.int64)
        keep.append(idx)
        s = e
    return np.concatenate(keep, axis=0) if keep else np.zeros((0,), dtype=np.int64)


def _max_cycle_by_unit(a: np.ndarray) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for u in np.unique(a[:, 0]):
        ui = int(u)
        out[ui] = int(a[a[:, 0] == u, 1].max())
    return out


def _label_last_k(a: np.ndarray, max_cycle_map: Dict[int, int], k: int) -> np.ndarray:
    y = np.zeros((len(a),), dtype=np.int64)
    for i in range(len(a)):
        u = int(a[i, 0])
        c = int(a[i, 1])
        if c >= (max_cycle_map[u] - k + 1):
            y[i] = 1
    return y


def _to_cycle_level(
    x: np.ndarray,
    a: np.ndarray,
    y_rul: np.ndarray,
    y_anom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unit = a[:, 0].astype(np.int64)
    cycle = a[:, 1].astype(np.int64)
    keys = np.stack([unit, cycle], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    g = len(uniq)
    d = x.shape[1]
    x_out = np.zeros((g, d), dtype=np.float32)
    rul_out = np.zeros((g,), dtype=np.float32)
    anom_out = np.zeros((g,), dtype=np.int64)
    cnt = np.zeros((g,), dtype=np.float32)
    for i in range(len(x)):
        j = inv[i]
        x_out[j] += x[i]
        rul_out[j] += float(y_rul[i])
        anom_out[j] = max(anom_out[j], int(y_anom[i]))
        cnt[j] += 1.0
    cnt = np.maximum(cnt, 1.0)
    x_out = x_out / cnt[:, None]
    rul_out = rul_out / cnt
    return x_out, uniq[:, 0].astype(np.int64), uniq[:, 1].astype(np.int64), anom_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", default="dataset/dataset/CMAPSS/N-CMAPSS_DS02-006.h5")
    parser.add_argument("--out-npz", default="dataset/dataset/CMAPSS/processed/ncmapss_ds02_006_lastk30.npz")
    parser.add_argument("--out-meta", default="dataset/dataset/CMAPSS/processed/ncmapss_ds02_006_lastk30_meta.json")
    parser.add_argument("--feature-groups", default="W,X_s", help="Comma-separated among W,X_s,X_v,T")
    parser.add_argument("--label-mode", default="last_k", choices=["last_k", "rul_threshold"])
    parser.add_argument("--last-k-cycles", type=int, default=30)
    parser.add_argument("--rul-threshold", type=int, default=30)
    parser.add_argument("--downsample-factor", type=int, default=20)
    parser.add_argument("--aggregate-cycle", action="store_true")
    parser.add_argument("--near-const-var-threshold", type=float, default=1e-8)
    args = parser.parse_args()

    groups = [g.strip() for g in args.feature_groups.split(",") if g.strip()]
    valid = {"W", "X_s", "X_v", "T"}
    bad = [g for g in groups if g not in valid]
    if bad:
        raise ValueError(f"Unsupported feature groups: {bad}. Valid={sorted(valid)}")
    if len(groups) <= 0:
        raise ValueError("No feature groups selected.")

    p = Path(args.h5_path)
    if not p.exists():
        raise FileNotFoundError(f"N-CMAPSS file not found: {args.h5_path}")

    with h5py.File(str(p), "r") as f:
        a_dev_full = np.asarray(f["A_dev"][:, :2], dtype=np.int64)
        a_test_full = np.asarray(f["A_test"][:, :2], dtype=np.int64)
        y_dev_rul_full = np.asarray(f["Y_dev"][:, 0], dtype=np.int64)
        y_test_rul_full = np.asarray(f["Y_test"][:, 0], dtype=np.int64)

        idx_dev = _downsample_indices_by_unit(a_dev_full[:, 0], int(args.downsample_factor))
        idx_test = _downsample_indices_by_unit(a_test_full[:, 0], int(args.downsample_factor))

        a_dev = a_dev_full[idx_dev]
        a_test = a_test_full[idx_test]
        y_dev_rul = y_dev_rul_full[idx_dev]
        y_test_rul = y_test_rul_full[idx_test]

        x_dev_blocks = []
        x_test_blocks = []
        feat_names: List[str] = []
        for g in groups:
            x_dev_blocks.append(np.asarray(f[f"{g}_dev"][idx_dev], dtype=np.float32))
            x_test_blocks.append(np.asarray(f[f"{g}_test"][idx_test], dtype=np.float32))
            v = _decode_str_array(np.asarray(f[f"{g}_var"][...]))
            feat_names.extend([f"{g}:{name}" for name in v])

    x_dev = np.concatenate(x_dev_blocks, axis=1).astype(np.float32)
    x_test = np.concatenate(x_test_blocks, axis=1).astype(np.float32)

    if args.label_mode == "last_k":
        y_dev_anom = _label_last_k(a_dev, _max_cycle_by_unit(a_dev_full), int(args.last_k_cycles))
        y_test_anom = _label_last_k(a_test, _max_cycle_by_unit(a_test_full), int(args.last_k_cycles))
    else:
        y_dev_anom = (y_dev_rul <= int(args.rul_threshold)).astype(np.int64)
        y_test_anom = (y_test_rul <= int(args.rul_threshold)).astype(np.int64)

    if args.aggregate_cycle:
        x_dev, unit_dev, cycle_dev, y_dev_anom = _to_cycle_level(x_dev, a_dev, y_dev_rul, y_dev_anom)
        x_test, unit_test, cycle_test, y_test_anom = _to_cycle_level(x_test, a_test, y_test_rul, y_test_anom)
    else:
        unit_dev, cycle_dev = a_dev[:, 0].astype(np.int64), a_dev[:, 1].astype(np.int64)
        unit_test, cycle_test = a_test[:, 0].astype(np.int64), a_test[:, 1].astype(np.int64)

    keep_train = (y_dev_anom == 0)
    train_x = x_dev[keep_train]
    train_unit = unit_dev[keep_train]
    train_cycle = cycle_dev[keep_train]
    train_y = np.zeros((len(train_x),), dtype=np.int64)

    # Feature pruning on normal-only train split.
    v = np.var(train_x, axis=0)
    keep_feat = v > float(args.near_const_var_threshold)
    if not np.any(keep_feat):
        keep_feat = np.ones_like(v, dtype=bool)
    train_x = train_x[:, keep_feat]
    test_x = x_test[:, keep_feat]
    feat_names = [n for n, k in zip(feat_names, keep_feat.tolist()) if k]

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_npz),
        train_x=train_x.astype(np.float32),
        train_y=train_y.astype(np.int64),
        train_unit=train_unit.astype(np.int64),
        train_cycle=train_cycle.astype(np.int64),
        test_x=test_x.astype(np.float32),
        test_y=y_test_anom.astype(np.int64),
        test_unit=unit_test.astype(np.int64),
        test_cycle=cycle_test.astype(np.int64),
        feature_names=np.asarray(feat_names, dtype=object),
    )

    meta = {
        "h5_path": str(p),
        "feature_groups": groups,
        "label_mode": args.label_mode,
        "last_k_cycles": int(args.last_k_cycles),
        "rul_threshold": int(args.rul_threshold),
        "downsample_factor": int(args.downsample_factor),
        "aggregate_cycle": bool(args.aggregate_cycle),
        "near_const_var_threshold": float(args.near_const_var_threshold),
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "train_units": int(len(np.unique(train_unit))),
        "test_units": int(len(np.unique(unit_test))),
        "train_pos": int(train_y.sum()),
        "test_pos": int(y_test_anom.sum()),
        "n_features": int(train_x.shape[1]),
    }
    save_json(args.out_meta, meta)
    print(f"Saved: {out_npz}")
    print(f"Saved: {args.out_meta}")
    print(meta)


if __name__ == "__main__":
    main()
