from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat2026_training import (  # noqa: E402
    SWaTWindowDataset,
    TEST_END,
    TEST_START,
    PreparedSWaTData,
    prepare_swat_training_data,
    sha256_file,
)
from models.mscad import MSCAD  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.metrics import binary_metrics  # noqa: E402
from utils.project_paths import resolve_data_path, resolve_output_path, resolve_repo_path  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def resolve_path(value: str) -> Path:
    return resolve_repo_path(value)


def build_device(requested: str) -> torch.device:
    return torch.device("cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu")


def make_model(cfg: dict[str, Any], device: torch.device) -> MSCAD:
    model_cfg = cfg["model"]
    if int(model_cfg["input_channels"]) != 31:
        raise ValueError("Batch-M1 requires exactly 31 input channels")
    if model_cfg.get("init_checkpoint") is not None:
        raise ValueError("Batch-M1 forbids old checkpoint initialization")
    if not model_cfg.get("rec_only", False) or model_cfg.get("use_structure_loss", True):
        raise ValueError("Batch-M1 is reconstruction-only")
    return MSCAD(
        in_channels=31,
        hidden_channels=int(model_cfg["hidden_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
        dropout=float(model_cfg["dropout"]),
        cov_type="corr",
        short_window=16,
        backbone_variant=str(model_cfg.get("backbone_variant", "base")),
    ).to(device)


def parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def reconstruction_score(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Existing project definition: full-window MAE, averaged over time and channels."""
    return (x_hat - x).abs().mean(dim=(1, 2))


def configure_reproducibility(seed: int) -> dict[str, Any]:
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    return {
        "seed": seed, "python_seed": seed, "numpy_seed": seed, "torch_cpu_seed": seed,
        "torch_cuda_seed": seed, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "pandas_version": pd.__version__, "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    }


def build_loaders(prepared: PreparedSWaTData, cfg: dict[str, Any]) -> dict[str, DataLoader]:
    batch_size = int(cfg["data"]["batch_size"])
    seed = int(cfg["seed"])
    generator = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(SWaTWindowDataset(prepared, "train"), batch_size=batch_size, shuffle=True,
                            generator=generator, num_workers=0, drop_last=False),
        "train_score": DataLoader(SWaTWindowDataset(prepared, "train"), batch_size=batch_size, shuffle=False,
                                  num_workers=0, drop_last=False),
        "validation": DataLoader(SWaTWindowDataset(prepared, "validation"), batch_size=batch_size, shuffle=False,
                                 num_workers=0, drop_last=False),
        "test": DataLoader(SWaTWindowDataset(prepared, "test"), batch_size=batch_size, shuffle=False,
                           num_workers=0, drop_last=False),
    }


def audit_model(model: MSCAD, cfg: dict[str, Any], device: torch.device, output: Path) -> dict[str, Any]:
    model.train()
    single = torch.randn(1, 128, 31, device=device)
    small = torch.randn(4, 128, 31, device=device)
    single_output = model(single)
    output_batch = model(small)
    loss = model.rec_loss_fn(output_batch, small)
    if not torch.isfinite(loss):
        raise FloatingPointError("Model audit loss is non-finite")
    model.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    backward_ok = bool(gradients and all(value is not None and torch.isfinite(value).all() for value in gradients))
    if single_output.shape != single.shape or output_batch.shape != small.shape or not backward_ok:
        raise ValueError("31-channel model forward/backward shape audit failed")
    total, trainable = parameter_counts(model)
    hai = MSCAD(in_channels=86, hidden_channels=int(cfg["model"]["hidden_channels"]),
                latent_channels=int(cfg["model"]["latent_channels"]), dropout=float(cfg["model"]["dropout"]),
                backbone_variant=str(cfg["model"]["backbone_variant"]))
    hai_total, _ = parameter_counts(hai)
    payload = {
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "backbone_class": f"{type(model.backbone).__module__}.{type(model.backbone).__name__}",
        "input_layout": "[batch,time,channels]", "input_shape": list(small.shape),
        "output_shape": list(output_batch.shape), "input_channels": 31, "window_size": 128,
        "trainable_parameters": trainable, "total_parameters": total,
        "estimated_parameter_bytes": total * 4, "hai_86_channel_parameter_count": hai_total,
        "parameter_difference_from_hai": total - hai_total,
        "architecture_summary": str(model), "old_checkpoint_loaded": False,
        "forward_pass_ok": True, "backward_pass_ok": backward_ok,
        "finite_loss": True, "finite_gradients": backward_ok,
        "input_output_shape_equal": list(output_batch.shape) == list(small.shape),
    }
    write_json(output / "model_audit.json", payload)
    model.zero_grad(set_to_none=True)
    return payload


def train_epoch(model: MSCAD, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total, samples = 0.0, 0
    for x, target in loader:
        x, target = x.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.rec_loss_fn(model(x), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; formal training is blocked")
        loss.backward()
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise FloatingPointError("Non-finite gradient; formal training is blocked")
        optimizer.step()
        total += float(loss.detach()) * len(x)
        samples += len(x)
    return total / max(samples, 1)


@torch.no_grad()
def validation_loss(model: MSCAD, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, samples = 0.0, 0
    for x, target in loader:
        x, target = x.to(device), target.to(device)
        loss = model.rec_loss_fn(model(x), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss; formal training is blocked")
        total += float(loss) * len(x)
        samples += len(x)
    return total / max(samples, 1)


def train_model(
    model: MSCAD,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    checkpoint_path: Path,
    epochs_override: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    training = cfg["training"]
    if str(training["optimizer"]).lower() != "adamw":
        raise ValueError("Only the inherited AdamW optimizer is approved")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    epochs = int(epochs_override or training["max_epochs"])
    patience = int(training["early_stopping_patience"])
    best_loss, best_epoch, counter = float("inf"), 0, 0
    rows = []
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_value = train_epoch(model, train_loader, optimizer, device)
        validation_value = validation_loss(model, validation_loader, device)
        finite = bool(np.isfinite([train_value, validation_value]).all())
        if not finite:
            raise FloatingPointError("Non-finite epoch result; formal training is blocked")
        improved = validation_value < best_loss
        if improved:
            best_loss, best_epoch, counter = validation_value, epoch, 0
            torch.save({
                "model_state": model.state_dict(), "config": cfg, "seed": int(cfg["seed"]),
                "best_epoch": epoch, "best_validation_loss": validation_value,
                "selection_scope": "validation reconstruction loss only", "test_used": False,
                "old_checkpoint_loaded": False,
            }, checkpoint_path)
        else:
            counter += 1
        rows.append({
            "epoch": epoch, "train_loss": train_value, "validation_loss": validation_value,
            "improved": improved, "early_stop_counter": counter, "finite": finite,
            "elapsed_seconds": time.perf_counter() - started,
        })
        print(f"epoch={epoch} train_loss={train_value:.8f} validation_loss={validation_value:.8f} improved={improved}")
        if counter >= patience:
            break
    history = pd.DataFrame(rows)
    summary = {
        "epochs_run": len(history), "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "final_train_loss": float(history.iloc[-1]["train_loss"]), "early_stopped": len(history) < epochs,
        "test_used_during_training": False,
    }
    return history, summary


@torch.no_grad()
def collect_scores(model: MSCAD, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    values = []
    for x, _ in loader:
        x = x.to(device)
        score = reconstruction_score(x, model(x))
        if not torch.isfinite(score).all():
            raise FloatingPointError("Non-finite reconstruction score")
        values.append(score.cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def threshold_from_validation(validation_scores: np.ndarray, quantile: float = 0.99) -> float:
    if not np.isfinite(validation_scores).all():
        raise ValueError("Validation scores must be finite")
    return float(np.quantile(validation_scores, quantile))


def point_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (scores > threshold).astype(np.int8)
    base = binary_metrics(labels, prediction, scores)
    tp = int(((labels == 1) & (prediction == 1)).sum())
    tn = int(((labels == 0) & (prediction == 0)).sum())
    fp = int(((labels == 0) & (prediction == 1)).sum())
    fn = int(((labels == 1) & (prediction == 0)).sum())
    return {
        **base, "specificity": tn / max(tn + fp, 1), "fpr": fp / max(fp + tn, 1),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "predicted_positive_points": int(prediction.sum()), "point_adjustment": False,
    }


def event_evaluation(score_frame: pd.DataFrame, schedule: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for _, event in schedule.sort_values("attack_id", key=lambda value: value.astype(int)).iterrows():
        start = pd.Timestamp(f"{event['date']} {event['start_time']}")
        end = pd.Timestamp(f"{event['date']} {event['end_time']}")
        subset = score_frame[(score_frame["timestamp"] >= start) & (score_frame["timestamp"] < end)]
        alarms = subset["rec_score"].to_numpy() > threshold
        detected = bool(alarms.any())
        first = subset.iloc[int(np.flatnonzero(alarms)[0])]["timestamp"] if detected else pd.NaT
        rows.append({
            "attack_id": int(event["attack_id"]), "attack_name": event["attack_name"],
            "start_time": start.isoformat(), "end_time_exclusive": end.isoformat(),
            "attack_points": len(subset), "detected": detected,
            "first_detection_timestamp": first.isoformat() if detected else "",
            "first_detection_delay_seconds": float((first - start).total_seconds()) if detected else "",
            "alarm_points": int(alarms.sum()), "point_recall": float(alarms.mean()) if len(alarms) else 0.0,
        })
    return pd.DataFrame(rows)


def false_positive_intervals(score_frame: pd.DataFrame, schedule: pd.DataFrame, threshold: float) -> pd.DataFrame:
    normal = score_frame[score_frame["point_label"].eq(0)].copy()
    groups = normal["timestamp"].diff().dt.total_seconds().ne(1).cumsum()
    first_attack = pd.Timestamp(f"{schedule.iloc[0]['date']} {schedule.iloc[0]['start_time']}")
    last_end = pd.Timestamp(f"{schedule.iloc[-1]['date']} {schedule.iloc[-1]['end_time']}")
    rows = []
    for _, frame in normal.groupby(groups):
        start, end = frame.iloc[0]["timestamp"], frame.iloc[-1]["timestamp"] + pd.Timedelta(seconds=1)
        if end <= first_attack:
            kind = "pre_attack_normal"
        elif start >= last_end:
            kind = "trailing_normal"
        else:
            kind = "inter_attack_normal"
        alarms = frame["rec_score"].to_numpy() > threshold
        rows.append({
            "interval_type": kind, "start_time": start.isoformat(), "end_time_exclusive": end.isoformat(),
            "normal_points": len(frame), "false_positive_points": int(alarms.sum()),
            "false_positive_rate": float(alarms.mean()),
        })
    return pd.DataFrame(rows)


@torch.no_grad()
def benchmark(model: MSCAD, device: torch.device, cfg: dict[str, Any]) -> dict[str, Any]:
    model.eval()
    warmup = int(cfg["efficiency"]["warmup_iterations"])
    iterations = int(cfg["efficiency"]["measurement_iterations"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    results = {}
    for batch in (1, 128):
        sample = torch.randn(batch, 128, 31, device=device)
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        results[f"batch_{batch}_mean_forward_ms"] = (time.perf_counter() - started) * 1000 / iterations
    results.update({
        "device": str(device), "warmup_iterations": warmup, "measurement_iterations": iterations,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else "not_applicable",
    })
    return results


def _resolved_config(cfg: dict[str, Any], device: torch.device, reproducibility: dict[str, Any]) -> dict[str, Any]:
    resolved = json.loads(json.dumps(cfg))
    resolved["resolved_device"] = str(device)
    resolved["reproducibility"] = reproducibility
    return resolved


def run(cfg: dict[str, Any], dry_run: bool = False, smoke_epochs: int | None = None) -> dict[str, Any]:
    if int(cfg["seed"]) != 1:
        raise ValueError("Batch-M1 authorizes seed=1 only")
    if dry_run and smoke_epochs is not None:
        raise ValueError("dry-run and smoke mode are mutually exclusive")
    mode = "dry_run" if dry_run else ("smoke" if smoke_epochs is not None else "formal")
    base_output = resolve_output_path(cfg["output"]["dir"])
    output = base_output / mode if mode != "formal" else base_output
    output.mkdir(parents=True, exist_ok=True)
    reproducibility = configure_reproducibility(int(cfg["seed"]))
    device = build_device(str(cfg["device"]))
    reproducibility["resolved_device"] = str(device)
    reproducibility["gpu_model"] = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    write_json(output / "resolved_config.yaml", _resolved_config(cfg, device, reproducibility))
    prepared = prepare_swat_training_data(
        resolve_data_path(cfg["data"]["data_path"]), resolve_output_path(cfg["data"]["n2_dir"]), output,
        int(cfg["data"]["window_size"]),
    )
    loaders = build_loaders(prepared, cfg)
    model = make_model(cfg, device)
    model_audit = audit_model(model, cfg, device, output)
    if dry_run:
        result = {"mode": mode, "dry_run_passed": True, "model_audit": model_audit,
                  "train_windows": len(loaders["train"].dataset), "validation_windows": len(loaders["validation"].dataset),
                  "test_windows": len(loaders["test"].dataset), "formal_checkpoint_saved": False}
        write_json(output / "dry_run_summary.json", result)
        print(json.dumps(result, indent=2))
        return result

    checkpoint = output / "checkpoints" / ("smoke_best_seed1.pt" if mode == "smoke" else "best_seed1.pt")
    history, training = train_model(model, loaders["train"], loaders["validation"], cfg, device, checkpoint, smoke_epochs)
    history.to_csv(output / "training_history.csv", index=False)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if saved.get("test_used") is not False or saved.get("old_checkpoint_loaded") is not False:
        raise ValueError("Checkpoint provenance audit failed")
    model.load_state_dict(saved["model_state"], strict=True)
    if mode == "smoke":
        reloaded_loss = validation_loss(model, loaders["validation"], device)
        if not np.isfinite(reloaded_loss):
            raise FloatingPointError("Smoke checkpoint reload produced non-finite loss")
        summary = {**training, "mode": mode, "checkpoint_reload_ok": True,
                   "checkpoint_path": str(checkpoint), "formal_evaluation_run": False}
        write_json(output / "smoke_summary.json", summary)
        print(json.dumps(summary, indent=2))
        return summary

    # Model selection is complete here. Test data and labels have not entered training or threshold selection.
    train_scores = collect_scores(model, loaders["train_score"], device)
    validation_scores = collect_scores(model, loaders["validation"], device)
    threshold = threshold_from_validation(validation_scores, float(cfg["scoring"]["validation_quantile"]))
    test_scores = collect_scores(model, loaders["test"], device)
    train_rows = prepared.valid_window_rows("train")
    validation_rows = prepared.valid_window_rows("validation")
    test_rows = prepared.valid_window_rows("test")
    pd.DataFrame({"timestamp": train_rows["score_timestamp"], "rec_score": train_scores}).to_csv(output / "rec_scores_train.csv", index=False)
    pd.DataFrame({"timestamp": validation_rows["score_timestamp"], "rec_score": validation_scores}).to_csv(output / "rec_scores_validation.csv", index=False)
    score_frame = pd.DataFrame({
        "timestamp": pd.to_datetime(test_rows["score_timestamp"]), "rec_score": test_scores,
        "point_label": test_rows["point_label"].astype(int).to_numpy(), "attack_id": test_rows["attack_id"].to_numpy(),
        "phase": test_rows["phase"].to_numpy(), "evaluation_eligible": test_rows["evaluation_eligible"].astype(bool).to_numpy(),
    })
    score_frame.to_csv(output / "rec_scores_test.csv", index=False)
    labels = score_frame["point_label"].to_numpy(dtype=int)
    metrics = point_metrics(labels, test_scores, threshold)
    metrics.update({
        "threshold": threshold, "threshold_source": "validation_scores_q99",
        "threshold_quantile": float(cfg["scoring"]["validation_quantile"]),
        "threshold_fixed_before_test_scoring": True,
        "primary_score_name": cfg["scoring"]["primary_score_name"],
        "score_formula": "mean(abs(X_hat-X), axes=(time,channels))",
        "score_scope": "full_window", "channel_aggregation": "arithmetic_mean",
        "timestamp_alignment": "window_end", "point_adjustment": False,
        "test_points": len(labels), "attack_points": int(labels.sum()), "normal_points": int((labels == 0).sum()),
    })
    write_json(output / "baseline_metrics.json", metrics)
    schedule = pd.read_csv(resolve_path(cfg["data"]["n2_dir"]) / "approved_attack_schedule.csv", encoding="utf-8-sig")
    events = event_evaluation(score_frame, schedule, threshold)
    events.to_csv(output / "event_detection_summary.csv", index=False)
    intervals = false_positive_intervals(score_frame, schedule, threshold)
    intervals.to_csv(output / "false_positive_interval_summary.csv", index=False)
    efficiency = benchmark(model, device, cfg)
    total, trainable = parameter_counts(model)
    efficiency.update({
        "total_parameters": total, "trainable_parameters": trainable,
        "checkpoint_size_bytes": checkpoint.stat().st_size, "checkpoint_sha256": sha256_file(checkpoint),
    })
    write_json(output / "efficiency_summary.json", efficiency)
    training.update({
        "seed": int(cfg["seed"]), "device": str(device), "train_window_count": len(loaders["train"].dataset),
        "validation_window_count": len(loaders["validation"].dataset),
        "scaler_fit_row_count": prepared.scaler_metadata["valid_fit_rows"],
        "checkpoint_sha256": efficiency["checkpoint_sha256"], "checkpoint_size_bytes": checkpoint.stat().st_size,
        "test_used_during_training": False,
    })
    write_json(output / "training_summary.json", training)
    conditions = {
        "source_hash_matches": prepared.source_sha256 == "d1dd05034b393a6b201f39e86ac3b933000551d26680c5d91c5ff8db0559cf72",
        "n2_contract_matches": True, "scaler_fit_rows_11410": prepared.scaler_metadata["valid_fit_rows"] == 11_410,
        "train_windows_11156": len(loaders["train"].dataset) == 11_156,
        "validation_windows_2728": len(loaders["validation"].dataset) == 2_728,
        "test_windows_14400": len(loaders["test"].dataset) == 14_400,
        "model_forward_backward_ok": model_audit["forward_pass_ok"] and model_audit["backward_pass_ok"],
        "training_finite": bool(history["finite"].all()), "best_checkpoint_generated": checkpoint.exists(),
        "test_used_after_model_selection": True, "metrics_generated": True, "event_report_generated": len(events) == 11,
        "structure_score_computed": False, "fusion_score_computed": False, "multiple_seeds_run": False,
    }
    classification = "A. single_seed_rec_baseline_complete" if all(
        value for key, value in conditions.items() if key not in {"structure_score_computed", "fusion_score_computed", "multiple_seeds_run"}
    ) and not any(conditions[key] for key in {"structure_score_computed", "fusion_score_computed", "multiple_seeds_run"}) else "F. evaluation_failed"
    readiness = {"classification": classification, "conditions": conditions, "performance_affects_readiness": False}
    write_json(output / "batch_m1_readiness.json", readiness)
    result = {"mode": mode, "readiness": classification, "training": training, "metrics": metrics,
              "events_detected": int(events["detected"].sum()), "efficiency": efficiency}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SWaT A.12 Batch-M1 single-seed reconstruction baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-epochs", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke_epochs is not None and args.smoke_epochs <= 0:
        raise ValueError("--smoke-epochs must be positive")
    run(load_config(args.config), dry_run=args.dry_run, smoke_epochs=args.smoke_epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
