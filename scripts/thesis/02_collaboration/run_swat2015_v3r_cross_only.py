"""V3 Repair: selective cross-only dependency discovery and controlled validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

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
from models.backbone_ae import BackboneAE  # noqa: E402
from utils.config_io import load_config  # noqa: E402


LAGS = np.asarray([1, 5, 10, 20, 30], dtype=np.int64)
ALPHA = 1.0
FIT_STRIDE = 4
TOP_PREDICTOR_COUNT = 5
INITIAL_FEATURE_COUNT = 250
SELECTED_FEATURE_COUNT = 25
NUMERICAL_FLOOR = 1e-8
SCALE_FLOOR = 1e-6
SHIFT_SECONDS = 60
DONOR_MIN_SEPARATION = 300
MAX_SAMPLES = 2000
WINDOW = 64


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def history_matrix(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.concatenate([values[times - lag] for lag in LAGS], axis=1).astype(np.float64, copy=False)


def cross_only_columns(target_index: int) -> np.ndarray:
    columns = [lag_index * EXPECTED_FEATURES + channel for lag_index in range(len(LAGS)) for channel in range(EXPECTED_FEATURES) if channel != target_index]
    result = np.asarray(columns, dtype=np.int64)
    if len(result) != INITIAL_FEATURE_COUNT or any(result % EXPECTED_FEATURES == target_index):
        raise OfficialSWaTError("cross-only design contains target self-history")
    return result


def selected_columns(channels: Sequence[int]) -> np.ndarray:
    result = np.asarray([lag_index * EXPECTED_FEATURES + channel for lag_index in range(len(LAGS)) for channel in channels], dtype=np.int64)
    if len(channels) != TOP_PREDICTOR_COUNT or len(result) != SELECTED_FEATURE_COUNT:
        raise OfficialSWaTError("selective cross-only design must contain 5 channels x 5 lags")
    return result


def centered_products(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_x = np.mean(x, axis=0, dtype=np.float64)
    mean_y = np.mean(y, axis=0, dtype=np.float64)
    gram = x.T @ x - len(x) * np.outer(mean_x, mean_x)
    cross = x.T @ y - len(x) * np.outer(mean_x, mean_y)
    return gram, cross, mean_x, mean_y


def ridge_from_products(
    gram: np.ndarray,
    cross: np.ndarray,
    mean_x: np.ndarray,
    mean_y: np.ndarray,
    columns: np.ndarray,
    target_index: int,
) -> tuple[np.ndarray, float]:
    system = gram[np.ix_(columns, columns)].copy()
    system.flat[:: len(columns) + 1] += ALPHA
    coefficient = np.linalg.solve(system, cross[columns, target_index])
    intercept = float(mean_y[target_index] - mean_x[columns] @ coefficient)
    if not np.isfinite(coefficient).all() or not np.isfinite(intercept):
        raise OfficialSWaTError("non-finite cross-only Ridge solution")
    return coefficient, intercept


def predictor_importance(coefficient: np.ndarray, columns: np.ndarray, target_index: int) -> np.ndarray:
    importance = np.full(EXPECTED_FEATURES, -np.inf, dtype=np.float64)
    for channel in range(EXPECTED_FEATURES):
        if channel == target_index:
            continue
        mask = columns % EXPECTED_FEATURES == channel
        importance[channel] = float(np.sum(np.abs(coefficient[mask])))
    return importance


def choose_top_predictors(importance: np.ndarray, target_index: int) -> np.ndarray:
    candidates = [index for index in range(EXPECTED_FEATURES) if index != target_index]
    ranked = sorted(candidates, key=lambda index: (-importance[index], index))
    return np.asarray(ranked[:TOP_PREDICTOR_COUNT], dtype=np.int64)


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float | None]:
    error = np.asarray(prediction - y, dtype=np.float64)
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = None if denominator <= 0 else float(1.0 - np.sum(error**2) / denominator)
    return mae, rmse, r2


def stable_target(overall: float | None, first: float | None, second: float | None, train_std: float) -> bool:
    return bool(
        overall is not None and first is not None and second is not None
        and overall >= 0.30 and first >= 0.20 and second >= 0.20 and train_std > NUMERICAL_FLOOR
    )


def split_validation_times(row_count: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.arange(int(LAGS.max()), row_count, dtype=np.int64)
    split = len(valid) // 2
    calibration, evaluation = valid[:split], valid[split:]
    if calibration[-1] >= evaluation[0]:
        raise OfficialSWaTError("Validation halves overlap")
    return calibration, evaluation


def deterministic_even_sample(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return np.asarray(values, dtype=np.int64)
    return np.asarray(values[np.linspace(0, len(values) - 1, maximum, dtype=np.int64)], dtype=np.int64)


def cyclic_donors(universe: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    result = np.empty(len(sample_times), dtype=np.int64)
    size = len(universe)
    for row, current in enumerate(sample_times):
        current_index = int(np.searchsorted(universe, current))
        candidate_index = (current_index + size // 2) % size
        for step in range(size):
            donor = int(universe[(candidate_index + step) % size])
            if abs(donor - int(current)) >= DONOR_MIN_SEPARATION:
                result[row] = donor
                break
        else:
            raise OfficialSWaTError("no deterministic donor satisfies 300-second separation")
    return result


def cross_residual(
    values: np.ndarray,
    times: np.ndarray,
    target_index: int,
    columns: np.ndarray,
    coefficient: np.ndarray,
    intercept: float,
    donor_times: np.ndarray | None = None,
) -> np.ndarray:
    source_times = times if donor_times is None else donor_times
    inputs = history_matrix(values, source_times)[:, columns]
    residual = np.abs(values[times, target_index].astype(np.float64) - (inputs @ coefficient + intercept))
    if not np.isfinite(residual).all():
        raise OfficialSWaTError("non-finite selective cross-only residual")
    return residual


def fit_residual_reference(residual: np.ndarray, *, source_split: str) -> tuple[float, float]:
    if source_split != "calibration_normal":
        raise OfficialSWaTError("cross-only residual reference requires calibration_normal")
    center = float(np.median(residual))
    scale = max(1.4826 * float(np.median(np.abs(residual - center))), SCALE_FLOOR)
    return center, scale


def residual_score(residual: np.ndarray, center: float, scale: float) -> np.ndarray:
    result = np.maximum(0.0, (residual - center) / scale)
    if not np.isfinite(result).all():
        raise OfficialSWaTError("non-finite selective cross-only score")
    return result


def calibration_q99(score: np.ndarray, *, source_split: str) -> float:
    if source_split != "calibration_normal":
        raise OfficialSWaTError("q99 requires calibration_normal only")
    return float(np.quantile(score, 0.99))


def hybrid_windows(
    values: np.ndarray,
    times: np.ndarray,
    donor_times: np.ndarray,
    predictor_channels: np.ndarray,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-WINDOW + 1, 1, dtype=np.int64)
    clean = np.asarray(values[times[:, None] + offsets], dtype=np.float32)
    donor = np.asarray(values[donor_times[:, None] + offsets], dtype=np.float32)
    corrupt = clean.copy()
    corrupt[:, :, predictor_channels] = donor[:, :, predictor_channels]
    if not np.array_equal(clean[:, :, target_index], corrupt[:, :, target_index]):
        raise OfficialSWaTError("controlled corruption changed target history")
    untouched = np.asarray([i for i in range(EXPECTED_FEATURES) if i not in set(predictor_channels.tolist())])
    if not np.array_equal(clean[:, :, untouched], corrupt[:, :, untouched]):
        raise OfficialSWaTError("controlled corruption changed a non-predictor channel")
    return clean, corrupt


def build_a0(config: dict[str, Any], checkpoint: dict[str, Any], device: torch.device) -> BackboneAE:
    model_cfg = config["model"]
    model = BackboneAE(int(model_cfg["in_channels"]), int(model_cfg["hidden_channels"]), int(model_cfg["latent_channels"]), float(model_cfg["dropout"])).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    return model


def a0_scores(model: torch.nn.Module, windows: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(windows), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(windows[offset:offset + batch_size])).to(device)
            chunks.append(torch.abs(model(x)[:, -1] - x[:, -1]).mean(dim=1).cpu().numpy())
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise OfficialSWaTError("non-finite A0 comparator score")
    return result


def rank_metrics(clean: np.ndarray, corrupt: np.ndarray) -> tuple[float, float]:
    labels = np.concatenate([np.zeros(len(clean), dtype=np.int8), np.ones(len(corrupt), dtype=np.int8)])
    scores = np.concatenate([clean, corrupt])
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(str(config_path)); validate_config_contract(config)
    if config["v3a"]["lags"] != LAGS.tolist() or float(config["v3a"]["alpha"]) != ALPHA:
        raise OfficialSWaTError("Repair must preserve five lags and Ridge alpha=1")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing V3-R output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    normal, manifest = load_normal_cache(resolve_repo_path(ROOT, config["paths"]["cache_dir"]), config)
    scaler, scaler_meta = load_scaler(resolve_repo_path(ROOT, config["paths"]["a0_output"]) / "scaler.npz")
    if scaler_meta["fit_row_count"] != TRAIN_RAW_ROWS or scaler_meta["attack_rows_seen"] != 0:
        raise OfficialSWaTError("Repair requires A0 Train-only scaler")
    train = scaler.transform(np.asarray(normal[:TRAIN_RAW_ROWS], dtype=np.float32)).astype(np.float32, copy=False)
    validation = scaler.transform(np.asarray(normal[TRAIN_RAW_ROWS:EXPECTED_NORMAL_ROWS], dtype=np.float32)).astype(np.float32, copy=False)
    train_times = np.arange(int(LAGS.max()), len(train), FIT_STRIDE, dtype=np.int64)
    train_history = history_matrix(train, train_times)
    train_current = np.asarray(train[train_times], dtype=np.float64)
    gram, cross, mean_x, mean_y = centered_products(train_history, train_current)
    del train_history, train_current, train
    validation_times = np.arange(int(LAGS.max()), len(validation), dtype=np.int64)
    validation_history = history_matrix(validation, validation_times)
    validation_current = np.asarray(validation[validation_times], dtype=np.float64)
    half = len(validation_times) // 2
    feature_names = list(manifest["column_order"])
    assert scaler.raw_std_ is not None
    train_std = np.asarray(scaler.raw_std_).reshape(-1)
    r1_rows: list[dict[str, Any]] = []
    predictor_models: dict[str, Any] = {}
    for target_index, target in enumerate(feature_names):
        initial_columns = cross_only_columns(target_index)
        initial_coef, _ = ridge_from_products(gram, cross, mean_x, mean_y, initial_columns, target_index)
        importance = predictor_importance(initial_coef, initial_columns, target_index)
        predictors = choose_top_predictors(importance, target_index)
        sparse_columns = selected_columns(predictors)
        sparse_coef, sparse_intercept = ridge_from_products(gram, cross, mean_x, mean_y, sparse_columns, target_index)
        prediction = validation_history[:, sparse_columns] @ sparse_coef + sparse_intercept
        mae, rmse, overall_r2 = regression_metrics(validation_current[:, target_index], prediction)
        _, _, first_r2 = regression_metrics(validation_current[:half, target_index], prediction[:half])
        _, _, second_r2 = regression_metrics(validation_current[half:, target_index], prediction[half:])
        eligible = stable_target(overall_r2, first_r2, second_r2, float(train_std[target_index]))
        important_lags = []
        for channel in predictors:
            channel_positions = np.flatnonzero(sparse_columns % EXPECTED_FEATURES == channel)
            best_position = channel_positions[np.argmax(np.abs(sparse_coef[channel_positions]))]
            important_lags.append(int(LAGS[best_position // EXPECTED_FEATURES]))
        row = {
            "target": target, "overall_r2": overall_r2, "r2_first": first_r2, "r2_second": second_r2,
            "mae": mae, "rmse": rmse,
            "top5_predictors": ";".join(feature_names[index] for index in predictors),
            "important_lags": ";".join(map(str, important_lags)), "train_std": float(train_std[target_index]),
            "cross_only_stable_target": eligible,
        }
        r1_rows.append(row)
        predictor_models[target] = {
            "target_index": target_index, "predictor_indices": predictors.tolist(),
            "predictor_channels": [feature_names[index] for index in predictors], "important_lags": important_lags,
            "selected_columns": sparse_columns.tolist(), "coefficient": sparse_coef.tolist(), "intercept": sparse_intercept,
        }
    eligible_rows = [row for row in r1_rows if row["cross_only_stable_target"]]
    if len(eligible_rows) >= 3:
        r1_status = "CROSS_ONLY_DEPENDENCY_SUPPORTED"
    elif len(eligible_rows) >= 1:
        r1_status = "CROSS_ONLY_DEPENDENCY_NARROW"
    else:
        r1_status = "CROSS_ONLY_DEPENDENCY_NOT_SUPPORTED"
    r1_summary = {
        "status": r1_status, "eligible_target_count": len(eligible_rows),
        "eligible_target_names": [row["target"] for row in eligible_rows],
        "median_eligible_r2": float(np.median([row["overall_r2"] for row in eligible_rows])) if eligible_rows else None,
        "fit_stride": FIT_STRIDE, "train_fit_endpoint_count": len(train_times), "validation_stride": 1,
        "lags": LAGS.tolist(), "alpha": ALPHA, "initial_feature_count": INITIAL_FEATURE_COUNT,
        "top_predictor_channels": TOP_PREDICTOR_COUNT, "selected_feature_count": SELECTED_FEATURE_COUNT,
        "eligibility": "overall R2>=.30 and first/second R2>=.20 and Train variance>floor",
        "attack_data_read": False, "target_outcome_driven_selection": False,
    }
    with (output_dir / "r1_target_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(r1_rows[0])); writer.writeheader(); writer.writerows(r1_rows)
    write_json(output_dir / "r1_predictor_manifest.json", predictor_models)
    write_json(output_dir / "r1_summary.json", r1_summary)
    if r1_status != "CROSS_ONLY_DEPENDENCY_SUPPORTED":
        print(json.dumps({"r1": r1_summary, "r2": "NOT_RUN_BY_GATE"}, indent=2)); return r1_summary

    calibration_times, evaluation_times = split_validation_times(len(validation))
    evaluation_start = int(evaluation_times[0])
    legal_samples = evaluation_times[evaluation_times >= evaluation_start + SHIFT_SECONDS + WINDOW - 1]
    sample_times = deterministic_even_sample(legal_samples, MAX_SAMPLES)
    donor_universe = evaluation_times[evaluation_times >= evaluation_start + WINDOW - 1]
    shift_donors = sample_times - SHIFT_SECONDS
    swap_donors = cyclic_donors(donor_universe, sample_times)
    if int(shift_donors.min()) - WINDOW + 1 < evaluation_start or np.any(np.abs(swap_donors - sample_times) < DONOR_MIN_SEPARATION):
        raise OfficialSWaTError("R2 donor boundary/separation violation")
    checkpoint_path = resolve_repo_path(ROOT, config["paths"]["a0_output"]) / "best.pt"
    checkpoint_hash_before = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["best_epoch"] != 13 or not torch.cuda.is_available():
        raise OfficialSWaTError("R2 requires frozen epoch-13 A0 and CUDA")
    device = torch.device("cuda"); a0_model = build_a0(config, checkpoint, device)
    if sum(p.numel() for p in a0_model.parameters()) != 11730:
        raise OfficialSWaTError("A0 comparator changed")
    r2_rows: list[dict[str, Any]] = []
    for eligible in eligible_rows:
        target = eligible["target"]; model = predictor_models[target]
        target_index = int(model["target_index"]); predictors = np.asarray(model["predictor_indices"], dtype=np.int64)
        columns = np.asarray(model["selected_columns"], dtype=np.int64); coefficient = np.asarray(model["coefficient"])
        intercept = float(model["intercept"])
        calibration_residual = cross_residual(validation, calibration_times, target_index, columns, coefficient, intercept)
        center, scale = fit_residual_reference(calibration_residual, source_split="calibration_normal")
        calibration_score = residual_score(calibration_residual, center, scale)
        threshold = calibration_q99(calibration_score, source_split="calibration_normal")
        clean_residual = cross_residual(validation, sample_times, target_index, columns, coefficient, intercept)
        clean_score = residual_score(clean_residual, center, scale); normal_fpr = float(np.mean(clean_score > threshold))
        for injection, donors in (("TemporalShift60", shift_donors), ("DeterministicContextSwap", swap_donors)):
            corrupt_residual = cross_residual(validation, sample_times, target_index, columns, coefficient, intercept, donors)
            corrupt_score = residual_score(corrupt_residual, center, scale)
            clean_windows, corrupt_windows = hybrid_windows(validation, sample_times, donors, predictors, target_index)
            clean_a0 = a0_scores(a0_model, clean_windows, device, int(config["training"]["batch_size"]))
            corrupt_a0 = a0_scores(a0_model, corrupt_windows, device, int(config["training"]["batch_size"]))
            v3_auroc, v3_aupr = rank_metrics(clean_score, corrupt_score); a0_auroc, a0_aupr = rank_metrics(clean_a0, corrupt_a0)
            r2_rows.append({
                "target": target, "injection": injection, "sample_count_per_class": len(sample_times),
                "v3_auroc": v3_auroc, "v3_aupr": v3_aupr, "a0_auroc": a0_auroc, "a0_aupr": a0_aupr,
                "q99_threshold": threshold, "corruption_recall": float(np.mean(corrupt_score > threshold)),
                "normal_fpr": normal_fpr, "median_clean_residual": float(np.median(clean_residual)),
                "median_corrupted_residual": float(np.median(corrupt_residual)),
            })
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_hash_before:
        raise OfficialSWaTError("A0 checkpoint changed during R2")
    v3_aurocs = np.asarray([row["v3_auroc"] for row in r2_rows]); cells = len(r2_rows)
    beats = sum(row["v3_auroc"] > row["a0_auroc"] + 0.05 for row in r2_rows)
    median_recall = float(np.median([row["corruption_recall"] for row in r2_rows])); median_fpr = float(np.median([row["normal_fpr"] for row in r2_rows]))
    gates = {"A_median_v3_auroc_ge_080": bool(np.median(v3_aurocs) >= .80), "B_beats_a0_plus_005_ge_75pct": bool(beats >= np.ceil(.75 * cells)), "C_median_recall_ge_050": median_recall >= .50, "D_median_fpr_le_003": median_fpr <= .03}
    if all(gates.values()): status = "SELECTIVE_CROSS_ONLY_MECHANISM_SUPPORTED"
    elif float(np.median(v3_aurocs)) >= .70: status = "SELECTIVE_CROSS_ONLY_PARTIAL"
    else: status = "SELECTIVE_CROSS_ONLY_NOT_SUPPORTED"
    def med(injection: str, key: str) -> float:
        return float(np.median([row[key] for row in r2_rows if row["injection"] == injection]))
    r2_summary = {
        "status": status, "eligible_targets_frozen_from_r1": [row["target"] for row in eligible_rows], "cell_count": cells,
        "temporal_shift": {"median_v3_auroc": med("TemporalShift60", "v3_auroc"), "median_v3_aupr": med("TemporalShift60", "v3_aupr"), "median_a0_auroc": med("TemporalShift60", "a0_auroc")},
        "context_swap": {"median_v3_auroc": med("DeterministicContextSwap", "v3_auroc"), "median_v3_aupr": med("DeterministicContextSwap", "v3_aupr"), "median_a0_auroc": med("DeterministicContextSwap", "a0_auroc")},
        "overall": {"median_v3_auroc": float(np.median(v3_aurocs)), "median_v3_aupr": float(np.median([row["v3_aupr"] for row in r2_rows])), "v3_beats_a0_plus_005_cells": beats, "total_cells": cells, "median_corruption_recall": median_recall, "median_normal_fpr": median_fpr},
        "gates": gates, "normal_only": True, "attack_data_read": False, "a0_retraining": False,
        "parameter_search": False, "fusion": False, "test_driven_tuning": False,
    }
    with (output_dir / "r2_target_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(r2_rows[0])); writer.writeheader(); writer.writerows(r2_rows)
    write_json(output_dir / "r2_summary.json", r2_summary)
    write_json(output_dir / "r2_injection_manifest.json", {
        "source": "Official SWaT 2015 Normal v1 Validation only", "sample_count": len(sample_times),
        "sampling": "deterministic evenly spaced", "shift_seconds": SHIFT_SECONDS,
        "swap_rule": "sorted evaluation times; cyclic offset floor(N/2); scan forward until separation>=300",
        "minimum_swap_separation": int(np.min(np.abs(swap_donors - sample_times))),
        "sample_times_sha256": hashlib.sha256(sample_times.tobytes()).hexdigest(), "target_unchanged": True,
        "only_top5_predictors_replaced": True, "attack_data_read": False,
    })
    print(json.dumps({"r1": r1_summary, "r2": r2_summary}, indent=2, ensure_ascii=False)); return r2_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/thesis/swat2015_midterm_seed1.yaml")
    parser.add_argument("--output", default="outputs/thesis_v3r_cross_only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); run(resolve_repo_path(ROOT, args.config), resolve_repo_path(ROOT, args.output)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
