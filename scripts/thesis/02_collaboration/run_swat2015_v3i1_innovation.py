"""V3-I1: discover dynamic collaboration among Normal-only innovations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat2015_official import (  # noqa: E402
    EXPECTED_NORMAL_ROWS,
    TRAIN_RAW_ROWS,
    OfficialSWaTError,
    load_normal_cache,
    load_scaler,
    resolve_repo_path,
    validate_config_contract,
)
from utils.config_io import load_config  # noqa: E402


LAGS = np.asarray([1, 5, 10, 20, 30], dtype=np.int64)
ALPHA = 1.0
FIT_STRIDE = 4
TOP_PREDICTOR_COUNT = 5
FEATURES_PER_CHANNEL = len(LAGS)
GAIN_THRESHOLD = 0.10
HALF_GAIN_THRESHOLD = 0.05
R2_THRESHOLD = 0.10


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def normal_v1_profiles(audit: dict[str, Any]) -> list[dict[str, Any]]:
    def visit(value: Any) -> list[dict[str, Any]] | None:
        if isinstance(value, dict):
            if value.get("file_name") == "SWaT_Dataset_Normal_v1.xlsx":
                profiles = value.get("variable_profiles") or value.get("schema", {}).get("variable_profiles")
                if profiles:
                    return profiles
            for child in value.values():
                found = visit(child)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = visit(child)
                if found is not None:
                    return found
        return None

    profiles = visit(audit)
    if profiles is None or len(profiles) != 51:
        raise OfficialSWaTError("Normal v1 audit must contain exactly 51 variable profiles")
    return profiles


def classify_channels(audit: dict[str, Any], feature_names: Sequence[str]) -> tuple[list[int], list[int], list[int]]:
    profiles = normal_v1_profiles(audit)
    by_name = {row["name"]: row for row in profiles}
    if set(by_name) != set(feature_names):
        raise OfficialSWaTError("Normal v1 audit columns do not match cached Normal columns")
    continuous, discrete, constant = [], [], []
    for index, name in enumerate(feature_names):
        row = by_name[name]
        kind = row["observed_value_kind"]
        if bool(row["constant"]) or kind == "constant_numeric":
            constant.append(index)
        elif kind == "continuous_numeric":
            continuous.append(index)
        elif kind == "discrete_numeric":
            discrete.append(index)
        else:
            raise OfficialSWaTError(f"unsupported audited value kind for {name}: {kind}")
    if (len(continuous), len(discrete), len(constant)) != (25, 15, 11):
        raise OfficialSWaTError("Normal v1 audited channel counts changed from frozen 25/15/11")
    return continuous, discrete, constant


def own_history(values: np.ndarray, times: np.ndarray, channel: int) -> np.ndarray:
    return np.column_stack([values[times - lag, channel] for lag in LAGS]).astype(np.float64, copy=False)


def ridge_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    mean_x = x.mean(axis=0); mean_y = float(y.mean())
    xc = x - mean_x; yc = y - mean_y
    system = xc.T @ xc
    system.flat[:: len(mean_x) + 1] += ALPHA
    coefficient = np.linalg.solve(system, xc.T @ yc)
    intercept = float(mean_y - mean_x @ coefficient)
    if not np.isfinite(coefficient).all() or not np.isfinite(intercept):
        raise OfficialSWaTError("non-finite Ridge solution")
    return coefficient, intercept


def innovations(values: np.ndarray, channel: int, coefficient: np.ndarray, intercept: float) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    times = np.arange(int(LAGS.max()), len(values), dtype=np.int64)
    prediction = own_history(values, times, channel) @ coefficient + intercept
    result[times] = np.asarray(values[times, channel], dtype=np.float64) - prediction
    if not np.isfinite(result[times]).all():
        raise OfficialSWaTError("non-finite frozen-self innovation")
    return result


def discrete_transitions(values: np.ndarray, channels: Sequence[int]) -> np.ndarray:
    result = np.full((len(values), len(channels)), np.nan, dtype=np.float64)
    result[1:] = np.asarray(values[1:, channels], dtype=np.float64) - np.asarray(values[:-1, channels], dtype=np.float64)
    return result


def dynamic_history(continuous_innovations: np.ndarray, transitions: np.ndarray, times: np.ndarray) -> np.ndarray:
    dynamic = np.column_stack((continuous_innovations, transitions))
    result = np.concatenate([dynamic[times - lag] for lag in LAGS], axis=1).astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise OfficialSWaTError("non-finite dynamic history")
    return result


def target_candidate_columns(target_continuous_position: int, dynamic_channel_count: int) -> np.ndarray:
    result = np.asarray([
        lag_position * dynamic_channel_count + channel
        for lag_position in range(len(LAGS))
        for channel in range(dynamic_channel_count)
        if channel != target_continuous_position
    ], dtype=np.int64)
    if np.any(result % dynamic_channel_count == target_continuous_position):
        raise OfficialSWaTError("cross model contains target innovation history")
    return result


def selected_columns(channels: Sequence[int], dynamic_channel_count: int) -> np.ndarray:
    if len(channels) != TOP_PREDICTOR_COUNT:
        raise OfficialSWaTError("innovation model must select exactly Top5 channels")
    return np.asarray([
        lag_position * dynamic_channel_count + channel
        for lag_position in range(len(LAGS)) for channel in channels
    ], dtype=np.int64)


def choose_top5(coefficient: np.ndarray, columns: np.ndarray, dynamic_channel_count: int, excluded: int) -> np.ndarray:
    importance = np.full(dynamic_channel_count, -np.inf, dtype=np.float64)
    for channel in range(dynamic_channel_count):
        if channel != excluded:
            importance[channel] = np.sum(np.abs(coefficient[columns % dynamic_channel_count == channel]))
    ranked = sorted((i for i in range(dynamic_channel_count) if i != excluded), key=lambda i: (-importance[i], i))
    return np.asarray(ranked[:TOP_PREDICTOR_COUNT], dtype=np.int64)


def regression_stats(y: np.ndarray, prediction: np.ndarray, baseline: float) -> tuple[float, float, float, float]:
    baseline_mae = float(np.mean(np.abs(y - baseline)))
    cross_mae = float(np.mean(np.abs(y - prediction)))
    gain = float((baseline_mae - cross_mae) / baseline_mae) if baseline_mae > 0 else 0.0
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum((y - prediction) ** 2) / denominator) if denominator > 0 else 0.0
    return baseline_mae, cross_mae, gain, r2


def stable_target(gain: float, gain_first: float, gain_second: float, r2: float) -> bool:
    return gain >= GAIN_THRESHOLD and gain_first >= HALF_GAIN_THRESHOLD and gain_second >= HALF_GAIN_THRESHOLD and r2 >= R2_THRESHOLD


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(str(config_path)); validate_config_contract(config)
    if config["v3a"]["lags"] != LAGS.tolist() or float(config["v3a"]["alpha"]) != ALPHA:
        raise OfficialSWaTError("V3-I1 requires frozen lags and Ridge alpha=1")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing V3-I1 output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    normal, cache_manifest = load_normal_cache(resolve_repo_path(ROOT, config["paths"]["cache_dir"]), config)
    scaler, scaler_meta = load_scaler(resolve_repo_path(ROOT, config["paths"]["a0_output"]) / "scaler.npz")
    if scaler_meta["fit_row_count"] != TRAIN_RAW_ROWS or scaler_meta["attack_rows_seen"] != 0:
        raise OfficialSWaTError("V3-I1 requires frozen Train-only scaler")
    feature_names = list(cache_manifest["column_order"])
    audit_path = resolve_repo_path(ROOT, config["paths"]["t2_audit_json"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    continuous, discrete, constant = classify_channels(audit, feature_names)
    train = scaler.transform(np.asarray(normal[:TRAIN_RAW_ROWS], dtype=np.float32)).astype(np.float32, copy=False)
    validation = scaler.transform(np.asarray(normal[TRAIN_RAW_ROWS:EXPECTED_NORMAL_ROWS], dtype=np.float32)).astype(np.float32, copy=False)

    self_fit_times = np.arange(int(LAGS.max()), len(train), FIT_STRIDE, dtype=np.int64)
    train_innov = np.full((len(train), len(continuous)), np.nan, dtype=np.float64)
    val_innov = np.full((len(validation), len(continuous)), np.nan, dtype=np.float64)
    self_models: dict[str, Any] = {}
    for position, channel in enumerate(continuous):
        coefficient, intercept = ridge_fit(own_history(train, self_fit_times, channel), train[self_fit_times, channel])
        train_innov[:, position] = innovations(train, channel, coefficient, intercept)
        val_innov[:, position] = innovations(validation, channel, coefficient, intercept)
        self_models[feature_names[channel]] = {"own_history_only": True, "lags": LAGS.tolist(), "alpha": ALPHA, "coefficient": coefficient.tolist(), "intercept": intercept}

    train_delta = discrete_transitions(train, discrete); val_delta = discrete_transitions(validation, discrete)
    min_time = 2 * int(LAGS.max())
    train_fit_times = np.arange(min_time, len(train), FIT_STRIDE, dtype=np.int64)
    val_times = np.arange(min_time, len(validation), dtype=np.int64)
    x_train = dynamic_history(train_innov, train_delta, train_fit_times)
    x_val = dynamic_history(val_innov, val_delta, val_times)
    dynamic_names = [feature_names[i] for i in continuous] + [feature_names[i] for i in discrete]
    dynamic_types = ["continuous_innovation"] * len(continuous) + ["discrete_transition"] * len(discrete)
    dynamic_count = len(dynamic_names)
    raw_r1: dict[str, float] = {}
    with (ROOT / "outputs" / "thesis_v3r_cross_only" / "r1_target_metrics.csv").open(encoding="utf-8") as handle:
        raw_r1 = {row["target"]: float(row["overall_r2"]) for row in csv.DictReader(handle) if row["overall_r2"]}

    rows: list[dict[str, Any]] = []; cross_models: dict[str, Any] = {}
    split = len(val_times) // 2
    for target_position, target_channel in enumerate(continuous):
        target = feature_names[target_channel]
        candidates = target_candidate_columns(target_position, dynamic_count)
        y_fit = train_innov[train_fit_times, target_position]
        initial_coef, _ = ridge_fit(x_train[:, candidates], y_fit)
        predictors = choose_top5(initial_coef, candidates, dynamic_count, target_position)
        columns = selected_columns(predictors, dynamic_count)
        coefficient, intercept = ridge_fit(x_train[:, columns], y_fit)
        prediction = x_val[:, columns] @ coefficient + intercept
        y_val = val_innov[val_times, target_position]
        baseline = float(np.median(train_innov[min_time:, target_position]))
        baseline_mae, cross_mae, gain, r2 = regression_stats(y_val, prediction, baseline)
        _, _, gain_first, r2_first = regression_stats(y_val[:split], prediction[:split], baseline)
        _, _, gain_second, r2_second = regression_stats(y_val[split:], prediction[split:], baseline)
        stable = stable_target(gain, gain_first, gain_second, r2)
        predictor_names = [dynamic_names[i] for i in predictors]
        predictor_types = [dynamic_types[i] for i in predictors]
        row = {
            "target": target, "raw_r1_r2": raw_r1[target], "baseline_mae": baseline_mae,
            "cross_mae": cross_mae, "relative_innovation_gain": gain, "innovation_r2": r2,
            "gain_first": gain_first, "r2_first": r2_first, "gain_second": gain_second,
            "r2_second": r2_second, "top5_predictors": ";".join(predictor_names),
            "top5_predictor_types": ";".join(predictor_types), "stable": stable,
        }
        if not all(np.isfinite(float(row[key])) for key in ("raw_r1_r2", "baseline_mae", "cross_mae", "relative_innovation_gain", "innovation_r2", "gain_first", "r2_first", "gain_second", "r2_second")):
            raise OfficialSWaTError("non-finite V3-I1 metric")
        rows.append(row)
        cross_models[target] = {
            "target_continuous_position": target_position, "target_self_innovation_excluded": True,
            "predictor_selection_split": "Normal Train only", "candidate_feature_source": "other continuous innovations and discrete transitions only",
            "top5_channels": predictor_names, "top5_types": predictor_types, "selected_columns": columns.tolist(),
            "lags": LAGS.tolist(), "alpha": ALPHA, "coefficient": coefficient.tolist(), "intercept": intercept,
        }

    stable_rows = [row for row in rows if row["stable"]]
    if len(stable_rows) >= 3:
        status = "INNOVATION_COLLABORATION_SUPPORTED"
    elif stable_rows:
        status = "INNOVATION_COLLABORATION_NARROW"
    else:
        status = "INNOVATION_COLLABORATION_NOT_SUPPORTED"
    summary = {
        "status": status, "continuous_target_count": len(continuous), "discrete_predictor_count": len(discrete),
        "constant_excluded_count": len(constant), "stable_target_count": len(stable_rows),
        "stable_targets": [row["target"] for row in stable_rows],
        "median_stable_gain": float(np.median([row["relative_innovation_gain"] for row in stable_rows])) if stable_rows else None,
        "median_stable_innovation_r2": float(np.median([row["innovation_r2"] for row in stable_rows])) if stable_rows else None,
        "normal_source": "Official SWaT 2015 Normal v1 only", "train_rows": TRAIN_RAW_ROWS,
        "validation_rows": EXPECTED_NORMAL_ROWS - TRAIN_RAW_ROWS, "self_fit_stride": FIT_STRIDE,
        "lags": LAGS.tolist(), "alpha": ALPHA, "top_predictor_channels": TOP_PREDICTOR_COUNT,
        "stable_rule": "gain>=0.10 and gain_first>=0.05 and gain_second>=0.05 and innovation_r2>=0.10",
        "attack_data_read": False, "synthetic_anomaly": False, "parameter_search": False,
        "fusion": False, "test_driven_selection": False,
    }
    with (output_dir / "v3i1_target_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(output_dir / "v3i1_predictor_manifest.json", {
        "schema_source": str(audit_path), "classification_source_file": "SWaT_Dataset_Normal_v1.xlsx",
        "continuous_targets": [feature_names[i] for i in continuous], "discrete_predictors": [feature_names[i] for i in discrete],
        "constant_excluded": [feature_names[i] for i in constant], "self_models": self_models, "cross_models": cross_models,
    })
    write_json(output_dir / "v3i1_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False)); return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/thesis/swat2015_midterm_seed1.yaml")
    parser.add_argument("--output", default="outputs/thesis_v3i_innovation")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); run(resolve_repo_path(ROOT, args.config), resolve_repo_path(ROOT, args.output)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
