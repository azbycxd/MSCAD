"""Score frozen A0 with Train-MAD channel calibration and fixed top-6 aggregation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat2015_official import (  # noqa: E402
    EXPECTED_FEATURES,
    EXPECTED_NORMAL_ROWS,
    TRAIN_RAW_ROWS,
    WINDOW,
    OfficialSWaTError,
    build_split_plan,
    endpoint_labels,
    endpoint_timestamps,
    iter_window_batches,
    load_attack_cache,
    load_normal_cache,
    load_scaler,
    make_window_starts,
    resolve_repo_path,
    validate_config_contract,
)
from models.backbone_ae import BackboneAE  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.metrics import binary_metrics  # noqa: E402


TOP_K = 6
CHANNEL_COUNT = 51
MAD_FACTOR = 1.4826
SCALE_FLOOR = 1e-6
THRESHOLD_QUANTILE = 0.99


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def build_model(config: dict[str, Any]) -> BackboneAE:
    model = config["model"]
    return BackboneAE(
        in_channels=int(model["in_channels"]),
        hidden_channels=int(model["hidden_channels"]),
        latent_channels=int(model["latent_channels"]),
        dropout=float(model["dropout"]),
    )


def endpoint_residuals(
    model: torch.nn.Module,
    features: np.ndarray,
    starts: np.ndarray,
    scaler: Any,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for windows, _ in iter_window_batches(features, starts, scaler, batch_size, shuffle=False, seed=0):
            x = torch.from_numpy(windows).to(device)
            residual = torch.abs(model(x)[:, -1, :] - x[:, -1, :])
            chunks.append(residual.cpu().numpy())
    result = np.concatenate(chunks).astype(np.float32, copy=False)
    if result.shape != (len(starts), EXPECTED_FEATURES) or not np.isfinite(result).all():
        raise OfficialSWaTError("endpoint residual matrix has wrong shape or non-finite values")
    return result


def fit_channel_calibration(
    endpoint_residual: np.ndarray, *, source_split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source_split != "normal_train":
        raise OfficialSWaTError("channel calibration may only fit Normal Train endpoint residual")
    if endpoint_residual.ndim != 2 or endpoint_residual.shape[1] != CHANNEL_COUNT:
        raise OfficialSWaTError("channel calibration requires 51 residual columns")
    if not np.isfinite(endpoint_residual).all():
        raise OfficialSWaTError("channel calibration received NaN/Inf")
    median = np.median(endpoint_residual, axis=0)
    mad = np.median(np.abs(endpoint_residual - median), axis=0)
    scale = np.maximum(MAD_FACTOR * mad, SCALE_FLOOR)
    if median.shape != (CHANNEL_COUNT,) or scale.shape != (CHANNEL_COUNT,) or np.any(scale < SCALE_FLOOR):
        raise OfficialSWaTError("invalid channel calibration statistics")
    return median.astype(np.float64), mad.astype(np.float64), scale.astype(np.float64)


def calibrate_residual(endpoint_residual: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = np.maximum(0.0, (np.asarray(endpoint_residual, dtype=np.float64) - median) / scale)
    if z.shape[1] != CHANNEL_COUNT or not np.isfinite(z).all():
        raise OfficialSWaTError("calibrated residual contains NaN/Inf or wrong channel count")
    return z


def top6_score(z: np.ndarray) -> np.ndarray:
    if z.ndim != 2 or z.shape[1] != CHANNEL_COUNT:
        raise OfficialSWaTError("top-6 score requires exactly 51 calibrated channels")
    return np.mean(np.partition(z, -TOP_K, axis=1)[:, -TOP_K:], axis=1)


def normalized_mean_score(z: np.ndarray) -> np.ndarray:
    if z.ndim != 2 or z.shape[1] != CHANNEL_COUNT:
        raise OfficialSWaTError("normalized-mean score requires exactly 51 calibrated channels")
    return np.mean(z, axis=1)


def validation_q99(scores: np.ndarray, *, source_split: str) -> float:
    if source_split != "normal_validation":
        raise OfficialSWaTError("threshold may only come from Normal Validation scores")
    if not np.isfinite(scores).all():
        raise OfficialSWaTError("threshold source contains NaN/Inf")
    return float(np.quantile(scores, THRESHOLD_QUANTILE))


def assert_same_a0_endpoints(
    timestamp: np.ndarray, labels: np.ndarray, a0_timestamp: np.ndarray, a0_labels: np.ndarray
) -> None:
    if not np.array_equal(timestamp, a0_timestamp) or not np.array_equal(labels, a0_labels):
        raise OfficialSWaTError("S2 Test endpoints differ from frozen S1 A0 endpoints")


def evaluate(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    if not np.isfinite(score).all() or not math.isfinite(threshold):
        raise OfficialSWaTError("non-finite evaluation input")
    prediction = (score > threshold).astype(np.int8)
    result = binary_metrics(y_true, prediction, score)
    tn = int(((y_true == 0) & (prediction == 0)).sum())
    fp = int(((y_true == 0) & (prediction == 1)).sum())
    fn = int(((y_true == 1) & (prediction == 0)).sum())
    tp = int(((y_true == 1) & (prediction == 1)).sum())
    result.update({"fpr": fp / max(tn + fp, 1), "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return result


def score_summary(scores: np.ndarray) -> dict[str, float]:
    return {
        "count": int(len(scores)),
        "min": float(np.min(scores)),
        "median": float(np.median(scores)),
        "q95": float(np.quantile(scores, 0.95)),
        "q99": float(np.quantile(scores, 0.99)),
        "max": float(np.max(scores)),
    }


def choose_status(primary: dict[str, float], a0: dict[str, float]) -> str:
    deltas = {key: primary[key] - a0[key] for key in ("precision", "recall", "f1", "aupr", "auroc", "fpr")}
    strong = (
        deltas["fpr"] <= -0.10
        and deltas["precision"] > 0
        and deltas["f1"] > 0
        and deltas["aupr"] >= -0.02
        and deltas["auroc"] >= -0.02
    )
    if strong:
        return "STRONG_RECON_BASELINE_CANDIDATE"
    if primary["aupr"] >= 0.70 and primary["auroc"] >= 0.80:
        return "THRESHOLD_TRANSFER_REMAINS_PRIMARY_LIMITATION"
    return "ROBUST_TOPK_SCORING_NOT_SUPPORTED"


def run(s2_config_path: Path) -> dict[str, Any]:
    s2_config = load_config(str(s2_config_path))
    if int(s2_config["primary"]["k"]) != TOP_K or float(s2_config["threshold_quantile"]) != THRESHOLD_QUANTILE:
        raise OfficialSWaTError("S2 fixes k=6 and Validation q99")
    if s2_config["retraining"] or s2_config["test_driven_tuning"] or s2_config["point_adjustment"]:
        raise OfficialSWaTError("S2 forbids retraining, Test-driven tuning, and point adjustment")
    s1_config_path = resolve_repo_path(ROOT, s2_config["s1_config"])
    s1_config = load_config(str(s1_config_path))
    validate_config_contract(s1_config)
    output_dir = resolve_repo_path(ROOT, s2_config["output"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing S2 output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    a0_output = resolve_repo_path(ROOT, s2_config["a0_output"])
    a0_metrics = json.loads((a0_output / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(a0_output / "best.pt", map_location="cpu", weights_only=False)
    if checkpoint["best_epoch"] != 13 or a0_metrics["parameter_count"] != 11730:
        raise OfficialSWaTError("S2 did not receive the frozen epoch-13, 11730-parameter A0 checkpoint")
    if checkpoint["selection_scope"] != "Normal v1 validation reconstruction loss only; Attack unopened":
        raise OfficialSWaTError("A0 checkpoint selection provenance is unexpected")
    if not torch.cuda.is_available():
        raise OfficialSWaTError("S2 requires the same available CUDA runtime used by A0")
    device = torch.device("cuda")
    model = build_model(s1_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    if sum(parameter.numel() for parameter in model.parameters()) != 11730:
        raise OfficialSWaTError("A0 parameter count changed")

    cache_dir = resolve_repo_path(ROOT, s1_config["paths"]["cache_dir"])
    normal, cache_manifest = load_normal_cache(cache_dir, s1_config)
    scaler, scaler_metadata = load_scaler(a0_output / "scaler.npz")
    if scaler_metadata["attack_rows_seen"] != 0 or scaler_metadata["fit_row_count"] != TRAIN_RAW_ROWS:
        raise OfficialSWaTError("S2 scaler is not the frozen Train-only A0 scaler")
    batch_size = int(s1_config["training"]["batch_size"])
    train_starts = make_window_starts(0, TRAIN_RAW_ROWS)
    validation_starts = make_window_starts(TRAIN_RAW_ROWS, EXPECTED_NORMAL_ROWS)
    train_residual = endpoint_residuals(model, normal, train_starts, scaler, batch_size, device)
    median, mad, scale = fit_channel_calibration(train_residual, source_split="normal_train")
    del train_residual
    validation_residual = endpoint_residuals(model, normal, validation_starts, scaler, batch_size, device)
    validation_z = calibrate_residual(validation_residual, median, scale)
    validation_top6 = top6_score(validation_z)
    validation_mean = normalized_mean_score(validation_z)
    top6_threshold = validation_q99(validation_top6, source_split="normal_validation")
    mean_threshold = validation_q99(validation_mean, source_split="normal_validation")

    # Attack is opened only after Train calibration and both Validation thresholds are fixed.
    attack, attack_labels, attack_timestamps, _ = load_attack_cache(cache_dir, s1_config)
    segment_manifest = json.loads(
        resolve_repo_path(ROOT, s1_config["paths"]["t25_segment_manifest"]).read_text(encoding="utf-8")
    )
    plan = build_split_plan(attack_timestamps, segment_manifest)
    residual_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    timestamp_chunks: list[np.ndarray] = []
    for starts, bounds in zip(plan.test_window_starts_by_segment, plan.test_segment_raw_bounds):
        if int(starts[0]) != bounds[0] or int(starts[-1] + WINDOW) != bounds[1]:
            raise OfficialSWaTError("S2 window crosses a frozen Test segment boundary")
        residual_chunks.append(endpoint_residuals(model, attack, starts, scaler, batch_size, device))
        label_chunks.append(endpoint_labels(attack_labels, starts))
        timestamp_chunks.append(endpoint_timestamps(attack_timestamps, starts))
    test_residual = np.concatenate(residual_chunks)
    y_true = np.concatenate(label_chunks)
    timestamp = np.concatenate(timestamp_chunks)
    with np.load(a0_output / "test_scores.npz", allow_pickle=False) as a0_scores:
        assert_same_a0_endpoints(timestamp, y_true, a0_scores["endpoint_timestamp"], a0_scores["label"])
    test_z = calibrate_residual(test_residual, median, scale)
    test_top6 = top6_score(test_z)
    test_mean = normalized_mean_score(test_z)
    primary = evaluate(y_true, test_top6, top6_threshold)
    secondary = evaluate(y_true, test_mean, mean_threshold)
    a0 = a0_metrics["primary_endpoint_mae"]
    delta = {key: float(primary[key] - a0[key]) for key in ("precision", "recall", "f1", "aupr", "auroc", "fpr")}
    status = choose_status(primary, a0)
    metrics = {
        "experiment_id": "S15-A0+",
        "scientific_status": "VALID_DIAGNOSTIC / PRELIMINARY",
        "status": status,
        "a0_checkpoint_epoch": 13,
        "parameter_count": 11730,
        "test_evaluation_point_count": len(y_true),
        "top_k": TOP_K,
        "calibration_scope": "Normal Train endpoint residual only",
        "primary_threshold_source": "Normal Validation top6 q99",
        "primary_threshold": top6_threshold,
        "secondary_threshold_source": "Normal Validation normalized_mean q99",
        "secondary_threshold": mean_threshold,
        "a0_primary": a0,
        "a0plus_primary_top6": primary,
        "secondary_normalized_mean": secondary,
        "delta_a0plus_minus_a0": delta,
        "status_policy": {
            "strong": "FPR drop >=0.10, Precision/F1 improve, and AUPR/AUROC each drop <=0.02",
            "threshold_transfer_limitation": "strong rule false but AUPR>=0.70 and AUROC>=0.80",
            "otherwise": "ROBUST_TOPK_SCORING_NOT_SUPPORTED",
        },
        "cache_protocol_fingerprint": cache_manifest["protocol_fingerprint"],
        "retraining": False,
        "threshold_search": False,
        "k_search": False,
        "test_driven_tuning": False,
        "point_adjustment": False,
    }
    with (output_dir / "channel_calibration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["channel", "median", "mad", "scale", "fit_split"])
        writer.writeheader()
        for channel, center, raw_mad, channel_scale in zip(
            cache_manifest["column_order"], median, mad, scale
        ):
            writer.writerow(
                {
                    "channel": channel,
                    "median": float(center),
                    "mad": float(raw_mad),
                    "scale": float(channel_scale),
                    "fit_split": "Normal Train endpoint residual only",
                }
            )
    write_json(output_dir / "metrics.json", metrics)
    write_json(
        output_dir / "validation_score_summary.json",
        {
            "source": "Normal Validation only",
            "primary_top6": score_summary(validation_top6),
            "secondary_normalized_mean": score_summary(validation_mean),
            "thresholds_frozen_before_attack_open": True,
        },
    )
    write_json(
        output_dir / "test_score_summary.json",
        {
            "endpoint_count": len(y_true),
            "endpoints_identical_to_a0": True,
            "primary_top6": score_summary(test_top6),
            "secondary_normalized_mean": score_summary(test_mean),
        },
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    return metrics


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
