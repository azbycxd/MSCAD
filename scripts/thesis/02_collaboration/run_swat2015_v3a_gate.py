"""Run the frozen normal-only V3-A incremental cross-channel information gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat2015_official import (  # noqa: E402
    EXPECTED_FEATURES,
    EXPECTED_NORMAL_ROWS,
    TRAIN_RAW_ROWS,
    OfficialSWaTError,
    load_normal_cache,
    load_scaler,
    resolve_repo_path,
    validate_config_contract,
)
from utils.config_io import load_config  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def build_full_history(values: np.ndarray, lags: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return only t-k inputs and the aligned current t targets."""
    if values.ndim != 2 or not lags or any(int(lag) <= 0 for lag in lags):
        raise ValueError("lagged history requires a 2D array and positive lags")
    max_lag = max(int(lag) for lag in lags)
    if len(values) <= max_lag:
        raise ValueError("not enough rows for the requested lag history")
    history = np.concatenate(
        [values[max_lag - int(lag) : len(values) - int(lag)] for lag in lags], axis=1
    ).astype(np.float64, copy=False)
    current = np.asarray(values[max_lag:], dtype=np.float64)
    return history, current


def self_history_columns(target_index: int, feature_count: int, lag_count: int) -> np.ndarray:
    if not 0 <= target_index < feature_count:
        raise ValueError("target index is out of range")
    return np.asarray([lag_index * feature_count + target_index for lag_index in range(lag_count)], dtype=np.int64)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    error = np.asarray(y_pred - y_true, dtype=np.float64)
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error**2, axis=0))
    centered = np.asarray(y_true, dtype=np.float64) - np.mean(y_true, axis=0, dtype=np.float64)
    denominator = np.sum(centered**2, axis=0)
    r2 = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    valid = denominator > 0
    r2[valid] = 1.0 - np.sum(error[:, valid] ** 2, axis=0) / denominator[valid]
    return mae, rmse, r2


def relative_gain(self_mae: np.ndarray, full_mae: np.ndarray, epsilon: float) -> np.ndarray:
    return (np.asarray(self_mae) - np.asarray(full_mae)) / np.maximum(np.asarray(self_mae), epsilon)


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(str(config_path))
    validate_config_contract(config)
    v3a = config["v3a"]
    lags = [int(value) for value in v3a["lags"]]
    alpha = float(v3a["alpha"])
    floor = float(v3a["numerical_floor"])
    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    a0_output = resolve_repo_path(ROOT, config["paths"]["a0_output"])
    output_dir = resolve_repo_path(ROOT, config["paths"]["v3a_output"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite an existing V3-A output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # This gate deliberately never opens Attack features, labels, timestamps, or A0 test scores.
    normal, manifest = load_normal_cache(cache_dir, config)
    scaler, scaler_metadata = load_scaler(a0_output / "scaler.npz")
    if scaler_metadata.get("fit_row_count") != TRAIN_RAW_ROWS or scaler_metadata.get("attack_rows_seen") != 0:
        raise OfficialSWaTError("V3-A did not receive the frozen A0 Train-only scaler")
    train = scaler.transform(np.asarray(normal[:TRAIN_RAW_ROWS], dtype=np.float32)).astype(np.float32, copy=False)
    validation = scaler.transform(np.asarray(normal[TRAIN_RAW_ROWS:EXPECTED_NORMAL_ROWS], dtype=np.float32)).astype(
        np.float32, copy=False
    )
    train_history, train_current = build_full_history(train, lags)
    validation_history, validation_current = build_full_history(validation, lags)
    expected_full_features = EXPECTED_FEATURES * len(lags)
    if train_history.shape[1] != int(v3a["full_feature_count"]) or train_history.shape[1] != expected_full_features:
        raise OfficialSWaTError("V3-A full history must contain exactly 51 x 5 = 255 past features")

    full_model = Ridge(alpha=alpha)
    full_model.fit(train_history, train_current)
    full_prediction = np.asarray(full_model.predict(validation_history), dtype=np.float64)
    full_mae, full_rmse, full_r2 = regression_metrics(validation_current, full_prediction)

    self_mae = np.empty(EXPECTED_FEATURES, dtype=np.float64)
    self_rmse = np.empty(EXPECTED_FEATURES, dtype=np.float64)
    self_r2 = np.empty(EXPECTED_FEATURES, dtype=np.float64)
    self_coefficients = np.empty((EXPECTED_FEATURES, len(lags)), dtype=np.float64)
    self_intercepts = np.empty(EXPECTED_FEATURES, dtype=np.float64)
    for target_index in range(EXPECTED_FEATURES):
        columns = self_history_columns(target_index, EXPECTED_FEATURES, len(lags))
        self_model = Ridge(alpha=alpha)
        self_model.fit(train_history[:, columns], train_current[:, target_index])
        prediction = np.asarray(self_model.predict(validation_history[:, columns]), dtype=np.float64)[:, None]
        mae, rmse, r2 = regression_metrics(validation_current[:, target_index : target_index + 1], prediction)
        self_mae[target_index] = mae[0]
        self_rmse[target_index] = rmse[0]
        self_r2[target_index] = r2[0]
        self_coefficients[target_index] = np.asarray(self_model.coef_, dtype=np.float64)
        self_intercepts[target_index] = float(self_model.intercept_)

    gain = relative_gain(self_mae, full_mae, floor)
    assert scaler.raw_std_ is not None
    train_std = np.asarray(scaler.raw_std_, dtype=np.float64).reshape(-1)
    eligible = (train_std > floor) & (self_mae > floor)
    target_names = list(manifest["column_order"])
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(target_names):
        rows.append(
            {
                "target": target,
                "self_mae": float(self_mae[index]),
                "self_rmse": float(self_rmse[index]),
                "full_mae": float(full_mae[index]),
                "full_rmse": float(full_rmse[index]),
                "relative_gain": float(gain[index]),
                "self_r2": None if not np.isfinite(self_r2[index]) else float(self_r2[index]),
                "full_r2": None if not np.isfinite(full_r2[index]) else float(full_r2[index]),
                "train_std": float(train_std[index]),
                "eligible": bool(eligible[index]),
                "incremental_collaboration_target": bool(eligible[index] and gain[index] >= 0.10),
            }
        )
    eligible_gain = gain[eligible]
    count_05 = int(np.sum(eligible_gain >= 0.05))
    count_10 = int(np.sum(eligible_gain >= 0.10))
    count_20 = int(np.sum(eligible_gain >= 0.20))
    if count_10 >= int(v3a["minimum_promising_targets"]):
        status = "PROMISING_INCREMENTAL_CROSS_CHANNEL_SIGNAL"
    elif count_10 >= 1:
        status = "NARROW_INCREMENTAL_SIGNAL"
    else:
        status = "NO_INCREMENTAL_SIGNAL"
    ranked = sorted((row for row in rows if row["eligible"]), key=lambda row: row["relative_gain"], reverse=True)
    summary = {
        "experiment_id": "V3-A",
        "scientific_status": "VALID_DIAGNOSTIC",
        "status": status,
        "scope": "Normal v1 Train and Validation only; Attack data unopened",
        "seed": int(config["seed"]),
        "lags": lags,
        "alpha": alpha,
        "solver_dtype": "float64",
        "solver_dtype_reason": "avoid the ill-conditioned float32 normal-equation warning; protocol and alpha unchanged",
        "train_rows_after_lag": len(train_current),
        "validation_rows_after_lag": len(validation_current),
        "full_feature_count": train_history.shape[1],
        "eligible_target_count": int(eligible.sum()),
        "gain_ge_5_percent_count": count_05,
        "gain_ge_10_percent_count": count_10,
        "gain_ge_20_percent_count": count_20,
        "median_gain": float(np.median(eligible_gain)) if len(eligible_gain) else 0.0,
        "max_gain": float(np.max(eligible_gain)) if len(eligible_gain) else 0.0,
        "top_10_gain_targets": [
            {"target": row["target"], "relative_gain": row["relative_gain"]} for row in ranked[:10]
        ],
        "attack_data_read": False,
        "hyperparameter_tuning": False,
        "anomaly_detection": False,
    }
    write_json(output_dir / "config_resolved.json", config)
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "target_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / "ridge_coefficients.npz",
        full_coef=np.asarray(full_model.coef_),
        full_intercept=np.asarray(full_model.intercept_),
        self_coef=self_coefficients,
        self_intercept=self_intercepts,
        lags=np.asarray(lags, dtype=np.int64),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(resolve_repo_path(ROOT, args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
