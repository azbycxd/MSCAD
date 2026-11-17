"""Run the frozen single-seed official SWaT 2015 reconstruction baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
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
    VALIDATION_RAW_ROWS,
    WINDOW,
    OfficialSWaTError,
    build_split_plan,
    endpoint_labels,
    endpoint_timestamps,
    fit_train_only_scaler,
    iter_window_batches,
    load_attack_cache,
    load_normal_cache,
    make_window_starts,
    resolve_repo_path,
    save_scaler,
    validate_config_contract,
)
from losses.recon import ReconstructionLoss  # noqa: E402
from models.backbone_ae import BackboneAE  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.metrics import binary_metrics  # noqa: E402
from utils.seed import set_seed  # noqa: E402


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


def average_loss(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    features: np.ndarray,
    starts: np.ndarray,
    scaler: Any,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for windows, _ in iter_window_batches(features, starts, scaler, batch_size, shuffle=False, seed=0):
            x = torch.from_numpy(windows).to(device)
            loss = criterion(model(x), x)
            if not torch.isfinite(loss):
                raise OfficialSWaTError("non-finite validation reconstruction loss")
            total += float(loss.item()) * len(windows)
            count += len(windows)
    return total / count


def score_windows(
    model: torch.nn.Module,
    features: np.ndarray,
    starts: np.ndarray,
    scaler: Any,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    endpoint_chunks: list[np.ndarray] = []
    full_chunks: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for windows, _ in iter_window_batches(features, starts, scaler, batch_size, shuffle=False, seed=0):
            x = torch.from_numpy(windows).to(device)
            error = torch.abs(model(x) - x)
            endpoint_chunks.append(error[:, -1, :].mean(dim=1).cpu().numpy())
            full_chunks.append(error.mean(dim=(1, 2)).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    endpoint = np.concatenate(endpoint_chunks).astype(np.float32, copy=False)
    full = np.concatenate(full_chunks).astype(np.float32, copy=False)
    if len(endpoint) != len(starts) or not np.isfinite(endpoint).all() or not np.isfinite(full).all():
        raise OfficialSWaTError("score output is incomplete or non-finite")
    return endpoint, full, elapsed


def evaluate_score(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    if not np.isfinite(scores).all() or not math.isfinite(threshold):
        raise OfficialSWaTError("evaluation received a non-finite score or threshold")
    prediction = (scores > threshold).astype(np.int8)
    result = binary_metrics(y_true, prediction, scores)
    tn = int(((y_true == 0) & (prediction == 0)).sum())
    fp = int(((y_true == 0) & (prediction == 1)).sum())
    fn = int(((y_true == 1) & (prediction == 0)).sum())
    tp = int(((y_true == 1) & (prediction == 1)).sum())
    result.update({"fpr": fp / max(tn + fp, 1), "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return result


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(str(config_path))
    validate_config_contract(config)
    seed = int(config["seed"])
    set_seed(seed)
    if config["device"] != "cuda" or not torch.cuda.is_available():
        raise OfficialSWaTError("the frozen A0 run requires the available CUDA p310 runtime")
    device = torch.device("cuda")
    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    output_dir = resolve_repo_path(ROOT, config["paths"]["a0_output"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite an existing A0 output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Until the checkpoint and q99 thresholds are frozen, only Normal v1 is opened.
    normal, cache_manifest = load_normal_cache(cache_dir, config)
    scaler, scaler_metadata = fit_train_only_scaler(normal)
    save_scaler(output_dir / "scaler.npz", scaler, scaler_metadata)
    train_starts = make_window_starts(0, TRAIN_RAW_ROWS)
    validation_starts = make_window_starts(TRAIN_RAW_ROWS, EXPECTED_NORMAL_ROWS)

    model = build_model(config).to(device)
    criterion = ReconstructionLoss()
    training = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    batch_size = int(training["batch_size"])
    patience = int(training["early_stopping_patience"])
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "best.pt"

    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for windows, _ in iter_window_batches(
            normal, train_starts, scaler, batch_size, shuffle=True, seed=seed + epoch
        ):
            x = torch.from_numpy(windows).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), x)
            if not torch.isfinite(loss):
                raise OfficialSWaTError("non-finite training reconstruction loss")
            loss.backward()
            optimizer.step()
            train_total += float(loss.item()) * len(windows)
            train_count += len(windows)
        train_loss = train_total / train_count
        validation_loss = average_loss(
            model, criterion, normal, validation_starts, scaler, batch_size, device
        )
        improved = validation_loss < best_loss
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "improved": improved}
        )
        print(json.dumps(history[-1]), flush=True)
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "selection_scope": "Normal v1 validation reconstruction loss only; Attack unopened",
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    train_endpoint, train_full, _ = score_windows(
        model, normal, train_starts, scaler, batch_size, device
    )
    validation_endpoint, validation_full, _ = score_windows(
        model, normal, validation_starts, scaler, batch_size, device
    )
    quantile = float(config["scoring"]["threshold_quantile"])
    endpoint_threshold = float(np.quantile(validation_endpoint, quantile))
    full_threshold = float(np.quantile(validation_full, quantile))
    thresholds = {
        "selection_scope": "Normal v1 validation only; Attack labels not opened",
        "quantile": quantile,
        "comparator": ">",
        "endpoint_mae": endpoint_threshold,
        "full_window_mae": full_threshold,
    }
    write_json(output_dir / "thresholds.json", thresholds)

    # Attack data is first opened after model/checkpoint/threshold selection is complete.
    attack, attack_labels, attack_timestamps, _ = load_attack_cache(cache_dir, config)
    segment_manifest_path = resolve_repo_path(ROOT, config["paths"]["t25_segment_manifest"])
    segment_manifest = json.loads(segment_manifest_path.read_text(encoding="utf-8"))
    split_plan = build_split_plan(attack_timestamps, segment_manifest)
    test_endpoint_chunks: list[np.ndarray] = []
    test_full_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    timestamp_chunks: list[np.ndarray] = []
    segment_reports: list[dict[str, Any]] = []
    inference_seconds = 0.0
    for index, starts in enumerate(split_plan.test_window_starts_by_segment):
        endpoint, full, elapsed = score_windows(model, attack, starts, scaler, batch_size, device)
        labels = endpoint_labels(attack_labels, starts)
        timestamps = endpoint_timestamps(attack_timestamps, starts)
        test_endpoint_chunks.append(endpoint)
        test_full_chunks.append(full)
        label_chunks.append(labels)
        timestamp_chunks.append(timestamps)
        inference_seconds += elapsed
        raw_start, raw_end = split_plan.test_segment_raw_bounds[index]
        if int(starts[0]) != raw_start or int(starts[-1] + WINDOW) != raw_end:
            raise OfficialSWaTError("test window escaped its frozen segment")
        segment_reports.append(
            {
                "segment_id": segment_manifest["test_segments"][index]["segment_id"],
                "raw_bounds": [raw_start, raw_end],
                "window_count": len(starts),
                "excluded_leading_raw_points": WINDOW - 1,
            }
        )
    test_endpoint = np.concatenate(test_endpoint_chunks)
    test_full = np.concatenate(test_full_chunks)
    y_true = np.concatenate(label_chunks)
    endpoint_time = np.concatenate(timestamp_chunks)
    expected_points = sum(int(segment["row_count"]) - WINDOW + 1 for segment in segment_manifest["test_segments"])
    if len(y_true) != expected_points:
        raise OfficialSWaTError("test point count differs from two independently windowed frozen segments")

    endpoint_metrics = evaluate_score(y_true, test_endpoint, endpoint_threshold)
    full_metrics = evaluate_score(y_true, test_full, full_threshold)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics = {
        "experiment_id": "S15-A0",
        "status": "PRELIMINARY_OFFICIAL_SWAT2015",
        "dataset": config["data"]["dataset"],
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "checkpoint_selection": training["checkpoint_selection"],
        "thresholds": thresholds,
        "primary_endpoint_mae": endpoint_metrics,
        "secondary_full_window_mae": full_metrics,
        "test_evaluation_point_count": len(y_true),
        "test_anomaly_count": int(y_true.sum()),
        "test_anomaly_prevalence": float(y_true.mean()),
        "parameter_count": parameter_count,
        "inference_seconds": inference_seconds,
        "inference_ms_per_window": 1000.0 * inference_seconds / len(y_true),
        "point_adjustment": False,
        "test_driven_tuning": False,
        "segments": segment_reports,
    }
    score_summary = {
        "endpoint_mae": {
            "train_median": float(np.median(train_endpoint)),
            "validation_median": float(np.median(validation_endpoint)),
            "test_median": float(np.median(test_endpoint)),
        },
        "full_window_mae": {
            "train_median": float(np.median(train_full)),
            "validation_median": float(np.median(validation_full)),
            "test_median": float(np.median(test_full)),
        },
    }
    write_json(output_dir / "config_resolved.json", config)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "score_summary.json", score_summary)
    write_json(
        output_dir / "model_summary.json",
        {
            "class": config["model"]["class"],
            "parameter_count": parameter_count,
            "input_shape": ["B", WINDOW, EXPECTED_FEATURES],
            "output_shape": ["B", WINDOW, EXPECTED_FEATURES],
            "cache_protocol_fingerprint": cache_manifest["protocol_fingerprint"],
        },
    )
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss", "improved"])
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(
        output_dir / "test_scores.npz",
        endpoint_timestamp=endpoint_time,
        label=y_true,
        endpoint_mae=test_endpoint,
        full_window_mae=test_full,
    )
    print(json.dumps(metrics, indent=2), flush=True)
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
