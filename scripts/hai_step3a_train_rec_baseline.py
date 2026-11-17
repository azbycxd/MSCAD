from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hai import MultiSequenceWindowDataset, build_hai_dataloaders  # noqa: E402
from models.mscad import MSCAD  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.metrics import binary_metrics  # noqa: E402
from utils.pot import POTThreshold  # noqa: E402
from utils.seed import set_seed  # noqa: E402


OUTPUT_FILES = [
    "config_resolved.json", "model_summary.json", "training_history.csv", "best.pt",
    "train_score_stats.json", "threshold_summary.json", "test_point_metrics_by_file.csv",
    "test_point_metrics_summary.csv", "test_event_diagnostics.csv", "test_event_summary.csv",
    "test_score_arrays.npz", "efficiency_metrics.json", "figures/train_validation_rec_loss.png",
    "final_hai_rec_baseline_diagnosis.md",
]


def build_device(name: str) -> torch.device:
    return torch.device("cuda" if name == "cuda" and torch.cuda.is_available() else "cpu")


def ensure_fresh(output: Path) -> None:
    existing = [name for name in OUTPUT_FILES if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite HAI Step 3A outputs: {existing}")
    (output / "figures").mkdir(parents=True, exist_ok=True)


def reconstruction_score(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return (x_hat - x).abs().mean(dim=(1, 2))


def make_model(cfg: dict, feature_count: int, device: torch.device) -> MSCAD:
    model_cfg = cfg["model"]
    configured = int(model_cfg["input_channels"])
    if configured != feature_count:
        raise ValueError(f"Configured input_channels={configured}, data feature_count={feature_count}")
    if not bool(model_cfg.get("rec_only", False)) or bool(model_cfg.get("use_structure_loss", True)):
        raise ValueError("HAI Step 3A requires rec_only=true and use_structure_loss=false")
    return MSCAD(
        in_channels=feature_count,
        hidden_channels=int(model_cfg["hidden_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
        dropout=float(model_cfg["dropout"]),
        cov_type="corr",
        short_window=int(cfg["data"].get("short_window", 16)),
        backbone_variant=str(model_cfg.get("backbone_variant", "base")),
    ).to(device)


def parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def limited_loader(loader: DataLoader, maximum_windows: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = loader.dataset
    count = min(maximum_windows, len(dataset))
    indices = np.unique(np.linspace(0, len(dataset) - 1, count, dtype=int)).tolist()
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Subset(dataset, indices), batch_size=loader.batch_size, shuffle=shuffle,
                      generator=generator if shuffle else None, num_workers=0, pin_memory=False, drop_last=False)


def train_epoch(model: MSCAD, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    samples = 0
    max_grad_norm = 0.0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        x_hat = model(x)
        loss = model.rec_loss_fn(x_hat, x)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite reconstruction loss during training")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite gradient norm during training")
        optimizer.step()
        batch = x.size(0)
        total_loss += float(loss.detach()) * batch
        samples += batch
        max_grad_norm = max(max_grad_norm, float(grad_norm))
    return total_loss / max(1, samples), max_grad_norm


@torch.no_grad()
def validation_loss(model: MSCAD, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    samples = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        loss = model.rec_loss_fn(model(x), x)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation reconstruction loss")
        total += float(loss) * x.size(0)
        samples += x.size(0)
    return total / max(1, samples)


def fit_normal_thresholds(train_scores: np.ndarray, validation_scores: np.ndarray, cfg: dict) -> list[dict[str, Any]]:
    if not np.isfinite(train_scores).all() or not np.isfinite(validation_scores).all():
        raise ValueError("Threshold fitting scores must be finite")
    threshold_cfg = cfg["thresholds"]
    rows: list[dict[str, Any]] = []
    q = float(threshold_cfg["pot_q"])
    target = float(threshold_cfg["pot_target_prob"])
    try:
        pot = POTThreshold(q=q, target_prob=target).fit(train_scores)
        if pot.tau is None or not np.isfinite(pot.tau):
            raise RuntimeError(f"POT returned invalid tau={pot.tau}")
        excess_count = int((train_scores > float(pot.u)).sum())
        rows.append({"threshold_method": "train_only_pot", "threshold_value": float(pot.tau),
                     "fitting_sample_count": len(train_scores), "fitting_scope": "train windows only",
                     "status": "ok", "pot_q": q, "pot_target_prob": target, "pot_u": float(pot.u),
                     "pot_excess_count": excess_count, "failure_reason": ""})
    except Exception as exc:
        rows.append({"threshold_method": "train_only_pot", "threshold_value": None,
                     "fitting_sample_count": len(train_scores), "fitting_scope": "train windows only",
                     "status": "failed", "pot_q": q, "pot_target_prob": target, "pot_u": None,
                     "pot_excess_count": None, "failure_reason": f"{type(exc).__name__}: {exc}"})
    val_q = float(threshold_cfg["validation_quantile"])
    rows.append({"threshold_method": "normal_validation_q99", "threshold_value": float(np.quantile(validation_scores, val_q)),
                 "fitting_sample_count": len(validation_scores), "fitting_scope": "normal validation windows only",
                 "status": "ok", "validation_quantile": val_q, "failure_reason": ""})
    return rows


@torch.no_grad()
def collect_scores(model: MSCAD, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    scores, labels, timestamps, rows, sequence_ids = [], [], [], [], []
    for batch in loader:
        x, y = batch[:2]
        x = x.to(device, non_blocking=True)
        score = reconstruction_score(x, model(x))
        scores.append(score.cpu().numpy())
        labels.append(y.numpy())
        if len(batch) == 3:
            metadata = batch[2]
            timestamps.extend(str(value) for value in metadata["timestamp"])
            sequence_ids.extend(str(value) for value in metadata["sequence_id"])
            end_rows = metadata["end_row"]
            rows.extend(int(value) for value in (end_rows.tolist() if torch.is_tensor(end_rows) else end_rows))
    return {
        "score": np.concatenate(scores).astype(np.float32),
        "label": np.concatenate(labels).astype(np.int64),
        "timestamp": np.asarray(timestamps, dtype=str), "end_row": np.asarray(rows, dtype=np.int64),
        "sequence_id": np.asarray(sequence_ids, dtype=str),
    }


def metric_row(file_name: str, method: str, values: dict[str, np.ndarray], threshold: float) -> dict[str, Any]:
    labels = values["label"]
    scores = values["score"]
    prediction = (scores > threshold).astype(np.int64)
    metrics = binary_metrics(labels, prediction, scores)
    return {
        "file_name": file_name, "threshold_method": method, **metrics,
        "anomaly_point_count": int((labels == 1).sum()), "normal_point_count": int((labels == 0).sum()),
        "predicted_alarm_count": int(prediction.sum()),
        "false_positive_count": int(((labels == 0) & (prediction == 1)).sum()),
        "false_negative_count": int(((labels == 1) & (prediction == 0)).sum()),
    }


def summarize_point_metrics(by_file: pd.DataFrame, values_by_file: dict[str, dict[str, np.ndarray]], thresholds: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    metric_names = ["precision", "recall", "f1", "aupr", "auroc"]
    for threshold in thresholds:
        if threshold["status"] != "ok":
            continue
        method = threshold["threshold_method"]
        subset = by_file[by_file.threshold_method == method]
        rows.append({"threshold_method": method, "summary_scope": "macro",
                     **{name: float(subset[name].mean()) for name in metric_names},
                     "anomaly_point_count": int(subset.anomaly_point_count.sum()),
                     "normal_point_count": int(subset.normal_point_count.sum()),
                     "predicted_alarm_count": int(subset.predicted_alarm_count.sum()),
                     "false_positive_count": int(subset.false_positive_count.sum()),
                     "false_negative_count": int(subset.false_negative_count.sum())})
        labels = np.concatenate([values["label"] for values in values_by_file.values()])
        scores = np.concatenate([values["score"] for values in values_by_file.values()])
        prediction = (scores > float(threshold["threshold_value"])).astype(np.int64)
        metrics = binary_metrics(labels, prediction, scores)
        rows.append({"threshold_method": method, "summary_scope": "micro", **metrics,
                     "anomaly_point_count": int((labels == 1).sum()), "normal_point_count": int((labels == 0).sum()),
                     "predicted_alarm_count": int(prediction.sum()),
                     "false_positive_count": int(((labels == 0) & (prediction == 1)).sum()),
                     "false_negative_count": int(((labels == 1) & (prediction == 0)).sum())})
    return pd.DataFrame(rows)


def event_diagnostics(events: list[dict[str, Any]], values_by_file: dict[str, dict[str, np.ndarray]], thresholds: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        if threshold["status"] != "ok":
            continue
        tau = float(threshold["threshold_value"])
        for event in events:
            values = values_by_file[event["file_name"]]
            mask = (values["end_row"] >= event["start_row"]) & (values["end_row"] <= event["end_row"])
            event_scores = values["score"][mask]
            event_rows = values["end_row"][mask]
            event_times = values["timestamp"][mask]
            alarms = event_scores > tau
            detected = bool(alarms.any())
            first_index = int(np.flatnonzero(alarms)[0]) if detected else -1
            rows.append({
                "threshold_method": threshold["threshold_method"], "sequence_id": event["sequence_id"],
                "file_name": event["file_name"], "event_id": event["event_id_in_file"],
                "global_event_id": event["global_event_id"], "start_timestamp": event["start_timestamp"],
                "end_timestamp": event["end_timestamp"], "duration": event["duration_points"], "detected": detected,
                "first_alarm_timestamp": str(event_times[first_index]) if detected else "",
                "first_alarm_delay_points": int(event_rows[first_index] - event["start_row"]) if detected else None,
                "point_recall_in_event": float(alarms.mean()) if len(alarms) else 0.0,
                "peak_rec_score": float(event_scores.max()) if len(event_scores) else None,
                "mean_rec_score": float(event_scores.mean()) if len(event_scores) else None,
            })
    return pd.DataFrame(rows)


def summarize_events(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, file_name), frame in diagnostics.groupby(["threshold_method", "file_name"]):
        delays = frame.loc[frame.detected, "first_alarm_delay_points"].dropna().astype(float)
        rows.append({"threshold_method": method, "summary_scope": file_name,
                     "total_event_count": len(frame), "detected_event_count": int(frame.detected.sum()),
                     "event_recall": float(frame.detected.mean()), "fully_missed_event_count": int((~frame.detected).sum()),
                     "median_detection_delay": float(delays.median()) if len(delays) else None,
                     "mean_detection_delay": float(delays.mean()) if len(delays) else None})
    for method, frame in diagnostics.groupby("threshold_method"):
        delays = frame.loc[frame.detected, "first_alarm_delay_points"].dropna().astype(float)
        rows.append({"threshold_method": method, "summary_scope": "all_tests",
                     "total_event_count": len(frame), "detected_event_count": int(frame.detected.sum()),
                     "event_recall": float(frame.detected.mean()), "fully_missed_event_count": int((~frame.detected).sum()),
                     "median_detection_delay": float(delays.median()) if len(delays) else None,
                     "mean_detection_delay": float(delays.mean()) if len(delays) else None})
    return pd.DataFrame(rows)


def benchmark(model: MSCAD, batch: torch.Tensor, device: torch.device, test_loaders: dict[str, DataLoader]) -> dict[str, Any]:
    model.eval()
    x = batch.to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - start) * 1000.0 / 100
        x_hat = model(x)
        for _ in range(10):
            _ = reconstruction_score(x, x_hat)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = reconstruction_score(x, x_hat)
        if device.type == "cuda":
            torch.cuda.synchronize()
        scoring_ms = (time.perf_counter() - start) * 1000.0 / 100
    start = time.perf_counter()
    windows = 0
    with torch.no_grad():
        for loader in test_loaders.values():
            for xb, _ in loader:
                xb = xb.to(device, non_blocking=True)
                _ = reconstruction_score(xb, model(xb))
                windows += len(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_cpu = None
    try:
        import psutil
        info = psutil.Process(os.getpid()).memory_info()
        peak_cpu = float(getattr(info, "peak_wset", info.rss) / (1024 ** 2))
    except Exception:
        pass
    return {
        "model_forward_time_ms_per_batch": forward_ms,
        "model_forward_time_ms_per_window": forward_ms / len(x),
        "rec_scoring_time_ms_per_batch": scoring_ms,
        "rec_scoring_time_ms_per_window": scoring_ms / len(x),
        "end_to_end_test_seconds": elapsed,
        "end_to_end_test_windows_per_second": windows / elapsed,
        "peak_cpu_memory_mb": peak_cpu,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else None,
        "device": str(device), "batch_size": len(x), "input_shape": list(x.shape),
        "timing_mode": "model.eval(), torch.no_grad(), 10 warmup, 100 timed runs; loader time excluded from pure timings",
    }


def metadata_loader(sequence, window_size: int, batch_size: int, indices: np.ndarray | None = None) -> DataLoader:
    dataset = MultiSequenceWindowDataset([sequence], window_size=window_size, stride=1, return_metadata=True)
    selected = Subset(dataset, indices.tolist()) if indices is not None else dataset
    return DataLoader(selected, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)


def run(cfg: dict, output: Path, smoke: bool = False) -> dict[str, Any]:
    ensure_fresh(output)
    seed = int(cfg["seed"])
    if int(cfg["training"]["seed"]) != seed:
        raise ValueError("Top-level and training seeds must match")
    set_seed(seed)
    if cfg["training"].get("init_checkpoint") is not None:
        raise ValueError("SWaT or other checkpoint initialization is forbidden")
    device = build_device(str(cfg.get("device", "cuda")))
    data_cfg = cfg["data"]
    max_rows = 25_000 if smoke else None
    bundle = build_hai_dataloaders(
        data_root=data_cfg["data_root"], window_size=int(data_cfg["window_size"]),
        stride_train=int(data_cfg["stride_train"]), stride_test=int(data_cfg["stride_test"]),
        validation_ratio=float(data_cfg["val_ratio"]), batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]), pin_memory=device.type == "cuda", seed=seed,
        max_rows_per_file=max_rows,
    )
    if len(bundle.feature_names) != 86:
        raise ValueError("HAI Step 3A requires all 86 features")
    model = make_model(cfg, len(bundle.feature_names), device)
    total_params, trainable_params = parameter_counts(model)
    resolved = dict(cfg)
    resolved["resolved_output_dir"] = str(output.resolve())
    resolved["resolved_device"] = str(device)
    resolved["reproducibility"] = {
        "python_random_seed": seed, "numpy_seed": seed, "torch_cpu_seed": seed,
        "torch_cuda_seed": seed, "dataloader_generator_seed": seed,
        "worker_seed": "torch.initial_seed() modulo 2**32",
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    (output / "config_resolved.json").write_text(json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8")
    sample = next(iter(bundle.train_loader))[0].to(device)
    with torch.no_grad():
        sample_output = model(sample)
    model_summary = {
        "model": "MSCAD BackboneAE rec-only", "backbone_variant": cfg["model"]["backbone_variant"],
        "input_layout": "[batch,time,channels]", "sample_input_shape": list(sample.shape),
        "sample_output_shape": list(sample_output.shape), "input_channels_configurable": True,
        "hardcoded_51_channels": False, "window_size_configurable": True,
        "encoder": ["DepthwiseSeparableConv1d(86->64,k=3)", "DepthwiseSeparableConv1d(64->32,k=3)"],
        "decoder": ["DepthwiseSeparableConv1d(32->64,k=3)", "Conv1d(64->86,k=1)"],
        "loss": "SmoothL1Loss(x_hat, x), reduction=mean", "loss_code": "losses/recon.py::ReconstructionLoss",
        "rec_score": "mean(abs(X-X_hat), dim=(time,channels))", "rec_score_code": "models/mscad.py::score_components",
        "structure_loss_computed": False, "short_score_computed": False, "long_score_computed": False,
        "total_parameter_count": total_params, "trainable_parameter_count": trainable_params,
    }
    (output / "model_summary.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")

    train_loader = limited_loader(bundle.train_loader, 1_024, True, seed) if smoke else bundle.train_loader
    val_loader = limited_loader(bundle.val_loader, 1_024, False, seed) if smoke else bundle.val_loader
    epochs = min(2, int(cfg["training"]["epochs"])) if smoke else int(cfg["training"]["epochs"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]),
                                  weight_decay=float(cfg["training"]["weight_decay"]))
    best_val = float("inf")
    best_epoch = -1
    early_counter = 0
    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss, max_grad = train_epoch(model, train_loader, optimizer, device)
        val_loss = validation_loss(model, val_loader, device)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            early_counter = 0
            torch.save({"model_state": model.state_dict(), "cfg": cfg, "feature_names": bundle.feature_names,
                        "best_epoch": epoch, "best_val_rec_loss": val_loss, "seed": seed,
                        "selection_scope": "normal validation reconstruction loss only"}, output / "best.pt")
        else:
            early_counter += 1
        history.append({"epoch": epoch, "train_rec_loss": train_loss, "val_rec_loss": val_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.perf_counter() - started,
                        "improved": improved, "early_stop_counter": early_counter, "max_gradient_norm": max_grad,
                        "finite": bool(np.isfinite([train_loss, val_loss, max_grad]).all())})
        print(f"epoch={epoch} train_rec={train_loss:.6f} val_rec={val_loss:.6f} improved={improved} grad={max_grad:.4f}")
        if bool(cfg["training"].get("early_stopping", True)) and early_counter >= int(cfg["training"]["patience"]):
            break
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output / "training_history.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.plot(history_frame.epoch, history_frame.train_rec_loss, label="train rec")
    plt.plot(history_frame.epoch, history_frame.val_rec_loss, label="validation rec")
    plt.xlabel("epoch"); plt.ylabel("SmoothL1 reconstruction loss"); plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(output / "figures" / "train_validation_rec_loss.png", dpi=160); plt.close()

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    score_train_loader = limited_loader(DataLoader(bundle.train_loader.dataset, batch_size=data_cfg["batch_size"], shuffle=False), 4_096, False, seed) if smoke else DataLoader(bundle.train_loader.dataset, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=0)
    score_val_loader = limited_loader(DataLoader(bundle.val_loader.dataset, batch_size=data_cfg["batch_size"], shuffle=False), 4_096, False, seed) if smoke else DataLoader(bundle.val_loader.dataset, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=0)
    train_values = collect_scores(model, score_train_loader, device)
    val_values = collect_scores(model, score_val_loader, device)
    thresholds = fit_normal_thresholds(train_values["score"], val_values["score"], cfg)
    (output / "threshold_summary.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    score_stats = {"train": {"count": len(train_values["score"]), "mean": float(train_values["score"].mean()),
                              "std": float(train_values["score"].std()), "min": float(train_values["score"].min()),
                              "max": float(train_values["score"].max())},
                   "validation": {"count": len(val_values["score"]), "mean": float(val_values["score"].mean()),
                                  "std": float(val_values["score"].std()), "min": float(val_values["score"].min()),
                                  "max": float(val_values["score"].max())}}
    (output / "train_score_stats.json").write_text(json.dumps(score_stats, indent=2), encoding="utf-8")

    values_by_file: dict[str, dict[str, np.ndarray]] = {}
    evaluation_loaders: dict[str, DataLoader] = {}
    for sequence in bundle.test_sequences:
        full_dataset = MultiSequenceWindowDataset([sequence], int(data_cfg["window_size"]), 1, return_metadata=True)
        indices = None
        if smoke:
            indices = np.unique(np.linspace(0, len(full_dataset) - 1, min(4_096, len(full_dataset)), dtype=int))
        loader = metadata_loader(sequence, int(data_cfg["window_size"]), int(data_cfg["batch_size"]), indices)
        evaluation_loaders[sequence.sequence_id] = DataLoader(loader.dataset, batch_size=data_cfg["batch_size"], shuffle=False)
        values_by_file[sequence.file_name] = collect_scores(model, loader, device)
    point_rows = []
    for threshold in thresholds:
        if threshold["status"] != "ok":
            continue
        for file_name, values in values_by_file.items():
            point_rows.append(metric_row(file_name, threshold["threshold_method"], values, float(threshold["threshold_value"])))
    point_by_file = pd.DataFrame(point_rows)
    point_by_file.to_csv(output / "test_point_metrics_by_file.csv", index=False)
    point_summary = summarize_point_metrics(point_by_file, values_by_file, thresholds)
    point_summary.to_csv(output / "test_point_metrics_summary.csv", index=False)
    event_frame = event_diagnostics(bundle.attack_events, values_by_file, thresholds)
    event_frame.to_csv(output / "test_event_diagnostics.csv", index=False)
    event_summary = summarize_events(event_frame) if len(event_frame) else pd.DataFrame(columns=["threshold_method", "summary_scope"])
    event_summary.to_csv(output / "test_event_summary.csv", index=False)

    arrays: dict[str, np.ndarray] = {"train_score": train_values["score"], "validation_score": val_values["score"]}
    for file_name, values in values_by_file.items():
        key = Path(file_name).stem
        for name, value in values.items():
            arrays[f"{key}_{name}"] = value
        for threshold in thresholds:
            if threshold["status"] == "ok":
                arrays[f"{key}_pred_{threshold['threshold_method']}"] = (values["score"] > float(threshold["threshold_value"])).astype(np.int64)
    np.savez_compressed(output / "test_score_arrays.npz", **arrays)

    timing_loaders = {name: loader for name, loader in bundle.test_loaders.items()}
    efficiency = benchmark(model, next(iter(bundle.test_loaders["test1"]))[0], device, timing_loaders)
    efficiency.update({"total_parameter_count": total_params, "trainable_parameter_count": trainable_params,
                       "checkpoint_size_bytes": (output / "best.pt").stat().st_size,
                       "checkpoint_size_mb": (output / "best.pt").stat().st_size / (1024 ** 2)})
    (output / "efficiency_metrics.json").write_text(json.dumps(efficiency, indent=2), encoding="utf-8")
    severe_overfit = bool(history_frame.iloc[-1].val_rec_loss > history_frame.iloc[-1].train_rec_loss * 2.0)
    diagnosis = {
        "smoke": smoke, "seed": seed, "best_epoch": best_epoch, "best_val_rec_loss": best_val,
        "final_train_rec_loss": float(history_frame.iloc[-1].train_rec_loss),
        "non_finite_detected": not bool(history_frame.finite.all()),
        "gradient_explosion_detected": bool((history_frame.max_gradient_norm > 1e3).any()),
        "severe_overfit_warning": severe_overfit,
        "ready_for_multiseed": bool(not smoke and history_frame.finite.all() and not severe_overfit),
    }
    report = f"""# HAI Step 3A rec-only baseline diagnosis

- Mode: `{'smoke' if smoke else 'formal'}`; seed: `{seed}`; device: `{device}`.
- Model: existing MSCAD base BackboneAE, configurable 86 channels, window 64; no layer hardcodes 51.
- Architecture: DSConv 86->64->32 encoder, DSConv 32->64 plus 1x1 Conv 64->86 decoder.
- Objective: existing SmoothL1 reconstruction loss only (`losses/recon.py`); no structure loss or structure score computed.
- Model selection: normal validation reconstruction loss only. Best epoch: `{best_epoch}`, best validation loss: `{best_val:.8f}`.
- Thresholds: train-only POT and normal-validation q99; no test label is used for fitting or selection.
- Test files are evaluated independently and events never cross file boundaries.
- Parameters: `{total_params}` total / `{trainable_params}` trainable; checkpoint: `{efficiency['checkpoint_size_mb']:.4f}` MB.
- Non-finite loss/score: `{diagnosis['non_finite_detected']}`; gradient explosion: `{diagnosis['gradient_explosion_detected']}`; severe overfit warning: `{severe_overfit}`.
- Ready for seed 2/3 multiseed training: `{diagnosis['ready_for_multiseed']}`.

See CSV/JSON artifacts for per-threshold point metrics, all 58 event diagnostics, thresholds, timings, and score arrays. Smoke metrics are pipeline checks only and are not formal evidence.
"""
    (output / "final_hai_rec_baseline_diagnosis.md").write_text(report, encoding="utf-8")
    return diagnosis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAI rec-only baseline training and evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    output = ROOT / (cfg["output"]["smoke_dir"] if args.smoke else cfg["output"]["dir"])
    if args.dry_run:
        print(f"config={Path(args.config).resolve()}")
        print(f"data_root={cfg['data']['data_root']}")
        print(f"output={output}")
        print("input=[batch,64,86]; output=same shape; channels configurable; no hardcoded 51 in model")
        print("rec-only=SmoothL1Loss; rec_score=mean(abs(X-X_hat)); structure losses/scores disabled")
        print("thresholds=train-only POT + normal-validation q99; test labels excluded from fitting/model selection")
        print(f"seed={cfg['seed']}; init_checkpoint=None; no SWaT checkpoint; four test files/events remain separate")
        return 0
    diagnosis = run(cfg, output, smoke=args.smoke)
    print(json.dumps({"output": str(output), **diagnosis}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
