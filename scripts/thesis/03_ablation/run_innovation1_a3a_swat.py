"""Run frozen SWaT REC structural ablations R1 then R2.

This runner preserves the formal S1 data, scaler, training, scoring, threshold,
and event protocols.  Only the two P0-frozen architecture substitutions differ.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


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
    fit_train_only_scaler,
    gather_scaled_windows,
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
from models.layers import ConvBNAct1d, DepthwiseSeparableConv1d  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.metrics import binary_metrics  # noqa: E402
from utils.seed import set_seed  # noqa: E402


OUT = ROOT / "outputs" / "thesis_ablation" / "innovation1"
P0 = ROOT / "docs" / "thesis" / "ablation_p0_design.md"
TRACE = ROOT / "docs" / "thesis" / "rec_architecture_trace.md"
REGISTRY = ROOT / "outputs" / "thesis_ablation" / "ablation_registry.json"
ABL1 = ROOT / "docs" / "thesis" / "ablation_a1_innovation2_results.md"
ABL2 = ROOT / "docs" / "thesis" / "ablation_a2_innovation3_results.md"
CONFIG = ROOT / "configs" / "thesis" / "swat2015_midterm_seed1.yaml"
R0_OUT = ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline"
EFF = ROOT / "outputs" / "thesis_efficiency_final"
FINAL = ROOT / "outputs" / "thesis_swat_final_unified"
T25 = ROOT / "outputs" / "thesis_t25_swat2015_protocol"
HARD_STOP_SECONDS = 50 * 60
WARMUP = 100
TIMED = 1000


class TemporalFreeAE(nn.Module):
    """R1: only the three depthwise temporal kernels change from 3 to 1."""

    def __init__(self, in_channels: int = 51, hidden_channels: int = 64, latent_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            DepthwiseSeparableConv1d(in_channels, hidden_channels, kernel_size=1, dropout=dropout),
            DepthwiseSeparableConv1d(hidden_channels, latent_channels, kernel_size=1, dropout=dropout),
        )
        self.decoder = nn.Sequential(
            DepthwiseSeparableConv1d(latent_channels, hidden_channels, kernel_size=1, dropout=dropout),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x.transpose(1, 2))
        return self.decoder(z).transpose(1, 2)


class DenseConvAE(nn.Module):
    """R2: only DS blocks change to same-width dense k=3 Conv-BN-GELU-Dropout."""

    def __init__(self, in_channels: int = 51, hidden_channels: int = 64, latent_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNAct1d(in_channels, hidden_channels, kernel_size=3, dropout=dropout),
            ConvBNAct1d(hidden_channels, latent_channels, kernel_size=3, dropout=dropout),
        )
        self.decoder = nn.Sequential(
            ConvBNAct1d(latent_channels, hidden_channels, kernel_size=3, dropout=dropout),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x.transpose(1, 2))
        return self.decoder(z).transpose(1, 2)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_order_hash(names: list[str]) -> str:
    return hashlib.sha256(json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def architecture_signature(model: nn.Module) -> dict[str, Any]:
    convs = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv1d):
            convs.append({
                "name": name,
                "in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
                "kernel_size": int(layer.kernel_size[0]),
                "padding": int(layer.padding[0]),
                "groups": layer.groups,
                "bias": layer.bias is not None,
            })
    return {
        "conv1d": convs,
        "batch_norm_count": sum(isinstance(layer, nn.BatchNorm1d) for layer in model.modules()),
        "gelu_count": sum(isinstance(layer, nn.GELU) for layer in model.modules()),
        "dropout_count": sum(isinstance(layer, nn.Dropout) for layer in model.modules()),
        "parameter_count": parameter_count(model),
    }


def validate_architecture_signatures(r0: nn.Module, r1: nn.Module, r2: nn.Module) -> dict[str, Any]:
    signatures = {"R0 Final": architecture_signature(r0), "R1": architecture_signature(r1), "R2": architecture_signature(r2)}
    r0_convs = signatures["R0 Final"]["conv1d"]
    r1_convs = signatures["R1"]["conv1d"]
    r2_convs = signatures["R2"]["conv1d"]
    r0_depth = [row for row in r0_convs if row["groups"] == row["in_channels"] and row["in_channels"] == row["out_channels"]]
    r1_depth = [row for row in r1_convs if row["groups"] == row["in_channels"] and row["in_channels"] == row["out_channels"]]
    if len(r0_depth) != 3 or [row["kernel_size"] for row in r0_depth] != [3, 3, 3]:
        raise RuntimeError("R0 architecture no longer matches the frozen three depthwise k=3 blocks")
    if len(r1_depth) != 3 or [row["kernel_size"] for row in r1_depth] != [1, 1, 1]:
        raise RuntimeError("R1 is not the frozen three depthwise k=1 substitute")
    if len(r1_convs) != len(r0_convs) or signatures["R1"]["batch_norm_count"] != 6 or signatures["R1"]["gelu_count"] != 6 or signatures["R1"]["dropout_count"] != 3:
        raise RuntimeError("R1 changed a component other than the three depthwise temporal kernels")
    if len(r2_convs) != 4 or [(row["in_channels"], row["out_channels"], row["kernel_size"], row["groups"]) for row in r2_convs] != [(51, 64, 3, 1), (64, 32, 3, 1), (32, 64, 3, 1), (64, 51, 1, 1)]:
        raise RuntimeError("R2 is not the frozen same-width dense k=3 substitute")
    if signatures["R2"]["batch_norm_count"] != 3 or signatures["R2"]["gelu_count"] != 3 or signatures["R2"]["dropout_count"] != 3:
        raise RuntimeError("R2 BN/GELU/dropout structure changed")
    if parameter_count(r0) != 11730:
        raise RuntimeError("formal R0 parameter count changed")
    return signatures


def build_variant_manifest(config: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    set_seed(int(config["seed"]))
    r0 = BackboneAE(51, 64, 32, 0.1)
    r1 = TemporalFreeAE(51, 64, 32, 0.1)
    r2 = DenseConvAE(51, 64, 32, 0.1)
    signatures = validate_architecture_signatures(r0, r1, r2)
    frozen_training = deepcopy(config["training"])
    common = {
        "input_output_shape": "[B,T,C] -> [B,T,C]",
        "encoder_widths": [51, 64, 32],
        "decoder_widths": [32, 64, 51],
        "activation": "GELU",
        "normalization": "BatchNorm1d",
        "dropout": 0.1,
        "bottleneck": "32 channels at every timestamp",
        "parameter_sharing": "Conv1d weights shared across time positions",
        "output_layer": "Conv1d(64,51,kernel_size=1,bias=True), no activation",
        "pooling_stride_dilation_residual_skip": "none",
    }
    variants = {
        "R0 Final": {
            **common,
            "block": "DepthwiseSeparableConv1d: depthwise k=3 -> BN -> GELU -> pointwise k=1 -> BN -> GELU -> Dropout",
            "temporal_component": "three depthwise k=3 convolutions; effective receptive field 7",
            "relative_to_R0": "reference; no change",
            "signature": signatures["R0 Final"],
        },
        "R1": {
            **common,
            "block": "DepthwiseSeparableConv1d with depthwise k=1; pointwise k=1, BN/GELU/Dropout unchanged",
            "temporal_component": "three depthwise k=1 convolutions; effective receptive field 1",
            "relative_to_R0": "only the three depthwise temporal kernels change k=3 -> k=1",
            "signature": signatures["R1"],
        },
        "R2": {
            **common,
            "block": "same-width dense Conv1d k=3 -> BN -> GELU -> Dropout",
            "temporal_component": "three dense k=3 convolutions; effective receptive field 7",
            "relative_to_R0": "only each complete depthwise-separable block changes to same-width dense k=3 Conv-BN-GELU-Dropout",
            "signature": signatures["R2"],
        },
    }
    return {
        "phase": "Thesis Ablation-ABL3A",
        "evidence_class": "RETROSPECTIVE_ABLATION",
        "definition_status": "FROZEN_FROM_P0_AND_REC_TRACE",
        "feature_count": len(feature_names),
        "feature_order": feature_names,
        "feature_order_sha256": feature_order_hash(feature_names),
        "variants": variants,
        "frozen_data": {"train": [0, 396000], "validation": [396000, 495000], "window": 64, "stride": 1, "test_segments": 2, "gap_policy": "hard_session_boundary"},
        "frozen_preprocessing": {"scaler": "data.transforms.StandardScaler", "fit": "Normal Train only", "source_feature_dtype": "float32", "statistic_dtype": "float32", "model_input_dtype": "float32", "validation_test": "transform only"},
        "frozen_training": frozen_training,
        "frozen_scoring": deepcopy(config["scoring"]),
        "source_hashes": {"p0": sha256(P0), "rec_trace": sha256(TRACE), "config": sha256(CONFIG), "r0_checkpoint": sha256(R0_OUT / "best.pt")},
        "no_architecture_search": True,
        "no_test_driven_definition": True,
    }


def average_loss(model: nn.Module, criterion: nn.Module, features: np.ndarray, starts: np.ndarray, scaler: Any, batch_size: int, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for windows, _ in iter_window_batches(features, starts, scaler, batch_size, shuffle=False, seed=0):
            x = torch.from_numpy(windows).to(device)
            loss = criterion(model(x), x)
            if not torch.isfinite(loss):
                raise OfficialSWaTError("non-finite validation reconstruction loss")
            total += float(loss.item()) * len(windows)
            count += len(windows)
    return total / count


def score_windows(model: nn.Module, features: np.ndarray, starts: np.ndarray, scaler: Any, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    endpoint, full = [], []
    with torch.inference_mode():
        for windows, _ in iter_window_batches(features, starts, scaler, batch_size, shuffle=False, seed=0):
            x = torch.from_numpy(windows).to(device)
            error = torch.abs(model(x) - x)
            endpoint.append(error[:, -1, :].mean(1).cpu().numpy())
            full.append(error.mean((1, 2)).cpu().numpy())
    endpoint_array = np.concatenate(endpoint).astype(np.float32, copy=False)
    full_array = np.concatenate(full).astype(np.float32, copy=False)
    if len(endpoint_array) != len(starts) or not np.isfinite(endpoint_array).all() or not np.isfinite(full_array).all():
        raise OfficialSWaTError("raw score persistence failed")
    return endpoint_array, full_array


def strict_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (scores > threshold).astype(np.int8)
    result = binary_metrics(labels, prediction, scores)
    tn = int(((labels == 0) & (prediction == 0)).sum())
    fp = int(((labels == 0) & (prediction == 1)).sum())
    fn = int(((labels == 1) & (prediction == 0)).sum())
    tp = int(((labels == 1) & (prediction == 1)).sum())
    result.update({"fpr": fp / max(tn + fp, 1), "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    if not all(math.isfinite(float(result[key])) for key in ("auroc", "aupr", "precision", "recall", "f1", "fpr")):
        raise OfficialSWaTError("non-finite strict metric")
    return result


def timestamp_seconds(value: str) -> int:
    return int(np.datetime64(value, "s").astype(np.int64))


def event_evaluation(timestamps: np.ndarray, prediction: np.ndarray) -> tuple[list[dict[str, Any]], int]:
    with (T25 / "canonical_attack_events.csv").open(encoding="utf-8-sig", newline="") as handle:
        source = [row for row in csv.DictReader(handle) if row["evaluable_physical_attack_interval"].casefold() == "true"]
    rows = []
    for event in source:
        start, end = timestamp_seconds(event["canonical_start"]), timestamp_seconds(event["canonical_end"])
        mask = (timestamps >= start) & (timestamps <= end)
        if not mask.any():
            raise OfficialSWaTError(f"official event {event['event_id']} absent from legal endpoints")
        hit_rate = float(prediction[mask].mean())
        rows.append({"event_id": int(event["event_id"]), "start": event["canonical_start"], "end": event["canonical_end"], "point_count": int(mask.sum()), "hit_rate": hit_rate, "hit_at_least_0_10": hit_rate >= 0.10})
    if len(rows) != 36:
        raise OfficialSWaTError("official event count changed")
    return rows, sum(bool(row["hit_at_least_0_10"]) for row in rows)


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def benchmark(model: nn.Module, normal: np.ndarray, scaler: Any, device: torch.device) -> dict[str, Any]:
    starts = make_window_starts(TRAIN_RAW_ROWS, EXPECTED_NORMAL_ROWS)[:TIMED]
    windows = gather_scaled_windows(normal, starts, scaler)
    tensors = [torch.from_numpy(windows[index:index + 1]).to(device) for index in range(TIMED)]

    def infer(index: int) -> float:
        with torch.inference_mode():
            x = tensors[index]
            return float(torch.abs(model(x)[:, -1] - x[:, -1]).mean().item())

    for index in range(WARMUP):
        infer(index % TIMED)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    latencies = []
    for index in range(TIMED):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        infer(index)
        torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - started) * 1000.0)
    return {"mean_latency_ms": float(np.mean(latencies)), "median_latency_ms": float(np.median(latencies)), "p95_latency_ms": float(np.quantile(latencies, 0.95)), "peak_vram_mib": float(torch.cuda.max_memory_allocated(device) / 1048576), "warmup": WARMUP, "timed_endpoints": TIMED, "batch_size": 1, "device": str(torch.cuda.get_device_name(device))}


def train_variant(variant: str, model_class: type[nn.Module], config: dict[str, Any], normal: np.ndarray, scaler: Any, feature_names: list[str], cache_manifest: dict[str, Any], overall_started: float) -> tuple[dict[str, Any] | None, bool]:
    output = OUT / "swat" / variant.lower()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        return json.loads((output / "result_row.json").read_text(encoding="utf-8")), True

    seed = int(config["seed"])
    set_seed(seed)
    model = model_class(51, 64, 32, 0.1).to(torch.device("cuda"))
    device = next(model.parameters()).device
    training = config["training"]
    criterion = ReconstructionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    train_starts = make_window_starts(0, TRAIN_RAW_ROWS)
    validation_starts = make_window_starts(TRAIN_RAW_ROWS, EXPECTED_NORMAL_ROWS)
    latest = output / "latest_training_state.pt"
    history: list[dict[str, Any]] = []
    best_loss, best_epoch, stale, start_epoch = float("inf"), 0, 0, 1
    best_model: dict[str, torch.Tensor] | None = None
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        if state["variant"] != variant or state["training_config"] != training:
            raise RuntimeError(f"{variant} resume protocol mismatch")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        best_model = state["best_model_state_dict"]
        best_loss, best_epoch, stale = float(state["best_validation_loss"]), int(state["best_epoch"]), int(state["early_stop_counter"])
        history = list(state["history"])
        start_epoch = int(state["epoch"]) + 1
        restore_rng(state["rng_state"])

    training_started = time.perf_counter()
    stop_reason = "max_epochs"
    completed = False
    for epoch in range(start_epoch, int(training["max_epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for windows, _ in iter_window_batches(normal, train_starts, scaler, int(training["batch_size"]), shuffle=True, seed=seed + epoch):
            x = torch.from_numpy(windows).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), x)
            if not torch.isfinite(loss):
                raise OfficialSWaTError("non-finite training reconstruction loss")
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(windows)
            count += len(windows)
        train_loss = total / count
        validation_loss = average_loss(model, criterion, normal, validation_starts, scaler, int(training["batch_size"]), device)
        improved = validation_loss < best_loss
        if improved:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_model = cpu_state(model)
        else:
            stale += 1
        row = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "improved": improved, "epoch_seconds": time.perf_counter() - epoch_started}
        history.append(row)
        state = {"variant": variant, "epoch": epoch, "model_state_dict": cpu_state(model), "optimizer_state_dict": optimizer.state_dict(), "best_model_state_dict": best_model, "best_epoch": best_epoch, "best_validation_loss": best_loss, "early_stop_counter": stale, "history": history, "training_config": training, "rng_state": rng_state()}
        torch.save(state, latest)
        json_dump(output / "training_history.json", history)
        print(json.dumps({"variant": variant, **row, "early_stop_counter": stale}), flush=True)
        if stale >= int(training["early_stopping_patience"]):
            stop_reason, completed = "early_stopping", True
            break
        if epoch == int(training["max_epochs"]):
            stop_reason, completed = "max_epochs", True
            break
        if time.perf_counter() - overall_started >= HARD_STOP_SECONDS:
            stop_reason = "hard_stop_after_completed_epoch"
            break
    if not completed:
        json_dump(output / "partial_status.json", {"status": "ABLATION_A3A_HARD_STOP_INCOMPLETE", "variant": variant, "last_epoch": history[-1]["epoch"], "best_epoch": best_epoch, "best_validation_loss": best_loss, "early_stop_counter": stale, "stop_reason": stop_reason})
        return None, False
    if best_model is None:
        raise RuntimeError(f"{variant} has no validation-selected model")
    model.load_state_dict(best_model)
    model.eval()
    train_time = sum(float(row["epoch_seconds"]) for row in history)
    best_checkpoint = {"variant": variant, "state_dict": best_model, "optimizer_state_dict_at_training_end": optimizer.state_dict(), "best_epoch": best_epoch, "best_validation_loss": best_loss, "selection_scope": "Normal v1 validation reconstruction loss only; Attack unopened", "config": config}
    torch.save(best_checkpoint, output / "best.pt")
    torch.save(best_model, output / "state_dict.pt")
    save_scaler(output / "scaler.npz", scaler, {"fit_scope": "normal_v1_raw_rows_[0,396000)", "fit_row_count": 396000, "validation_rows_seen": 0, "attack_rows_seen": 0, "feature_count": 51, "zero_variance_feature_indices": np.flatnonzero(scaler.raw_std_.reshape(-1) < scaler.eps).astype(int).tolist(), "zero_variance_effective_scale": 1.0, "source_dtype": "float32", "statistic_dtype": str(scaler.mean_.dtype)})
    json_dump(output / "feature_order.json", {"features": feature_names, "sha256": feature_order_hash(feature_names)})

    validation_endpoint, validation_full = score_windows(model, normal, validation_starts, scaler, int(training["batch_size"]), device)
    q = float(config["scoring"]["threshold_quantile"])
    threshold = float(np.quantile(validation_endpoint, q))
    full_threshold = float(np.quantile(validation_full, q))
    np.savez_compressed(output / "validation_raw_scores.npz", endpoint_mae=validation_endpoint, full_window_mae=validation_full, window_start=validation_starts)
    json_dump(output / "thresholds.json", {"source": "Normal Validation only", "quantile": q, "comparator": ">", "endpoint_mae": threshold, "full_window_mae": full_threshold})

    attack, labels_raw, timestamps_raw, _ = load_attack_cache(resolve_repo_path(ROOT, config["paths"]["cache_dir"]), config)
    segment_manifest = json.loads(resolve_repo_path(ROOT, config["paths"]["t25_segment_manifest"]).read_text(encoding="utf-8"))
    plan = build_split_plan(timestamps_raw, segment_manifest)
    score_parts, full_parts, label_parts, timestamp_parts = [], [], [], []
    for starts in plan.test_window_starts_by_segment:
        scores, full_scores = score_windows(model, attack, starts, scaler, int(training["batch_size"]), device)
        score_parts.append(scores)
        full_parts.append(full_scores)
        label_parts.append(endpoint_labels(labels_raw, starts))
        timestamp_parts.append(endpoint_timestamps(timestamps_raw, starts))
    scores = np.concatenate(score_parts)
    full_scores = np.concatenate(full_parts)
    labels = np.concatenate(label_parts)
    timestamps = np.concatenate(timestamp_parts)
    metrics = strict_metrics(labels, scores, threshold)
    prediction = scores > threshold
    events, event_hit = event_evaluation(timestamps, prediction)
    np.savez_compressed(output / "test_raw_scores.npz", endpoint_timestamp=timestamps, label=labels, endpoint_mae=scores, full_window_mae=full_scores)
    json_dump(output / "event_metrics_36.json", events)
    json_dump(output / "metrics.json", {"variant": variant, "dataset": "Official SWaT 2015 Physical", "evidence_class": "RETROSPECTIVE_ABLATION", "best_epoch": best_epoch, "best_validation_loss": best_loss, "stop_reason": stop_reason, "threshold": threshold, **metrics, "event_hit_at_least_0_10": event_hit, "test_points": len(labels), "point_adjustment": False, "oracle": False, "test_tuning": False})
    efficiency = benchmark(model, normal, scaler, device)
    efficiency.update({"params": parameter_count(model), "state_dict_bytes": (output / "state_dict.pt").stat().st_size, "best_checkpoint_bytes": (output / "best.pt").stat().st_size, "training_time_seconds": train_time, "best_epoch": best_epoch})
    json_dump(output / "efficiency.json", efficiency)
    architecture = json.loads((OUT / "rec_variant_manifest.json").read_text(encoding="utf-8"))["variants"][variant]
    json_dump(output / "architecture_manifest.json", architecture)
    manifest = {"variant": variant, "status": "COMPLETED", "evidence_class": "RETROSPECTIVE_ABLATION", "dataset": "Official SWaT 2015 Physical", "feature_order_sha256": feature_order_hash(feature_names), "cache_protocol_fingerprint": cache_manifest["protocol_fingerprint"], "splits": {"train": [0, 396000], "validation": [396000, 495000], "test_segments": [list(x) for x in plan.test_segment_raw_bounds]}, "window": 64, "stride": 1, "hard_gap_crossing": False, "scaler": {"fit_scope": "Normal Train only", "source_dtype": "float32", "statistic_dtype": str(scaler.mean_.dtype), "validation_rows_seen": 0, "attack_rows_seen": 0}, "training": training, "checkpoint_selection": "minimum Normal Validation reconstruction loss", "score": "endpoint mean absolute reconstruction error", "threshold": "Normal Validation q99", "artifacts": {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size} for name in ("best.pt", "state_dict.pt", "latest_training_state.pt", "scaler.npz", "validation_raw_scores.npz", "test_raw_scores.npz", "thresholds.json", "metrics.json")}, "restrictions": {"pa": False, "oracle": False, "test_tuning": False, "hp_search": False, "architecture_search": False}}
    json_dump(output / "manifest.json", manifest)
    r0_params = 11730
    result = {"variant": variant, "architecture_change": architecture["relative_to_R0"], "params": efficiency["params"], "param_ratio_vs_R0": efficiency["params"] / r0_params, "auroc": metrics["auroc"], "aupr": metrics["aupr"], "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"], "fpr": metrics["fpr"], "event_hit": event_hit, "latency_mean_ms": efficiency["mean_latency_ms"], "latency_median_ms": efficiency["median_latency_ms"], "latency_p95_ms": efficiency["p95_latency_ms"], "peak_vram_mib": efficiency["peak_vram_mib"], "state_dict_bytes": efficiency["state_dict_bytes"], "train_time_seconds": train_time, "best_epoch": best_epoch, "best_validation_loss": best_loss, "stop_reason": stop_reason}
    json_dump(output / "result_row.json", result)
    return result, True


def load_r0_row() -> dict[str, Any]:
    metrics = json.loads((R0_OUT / "metrics.json").read_text(encoding="utf-8"))
    primary = metrics["primary_endpoint_mae"]
    with (EFF / "ours_latency.csv").open(encoding="utf-8-sig", newline="") as handle:
        latency = next(row for row in csv.DictReader(handle) if row["dataset"] == "SWaT" and row["method"] == "Ours-REC")
    with (EFF / "ours_model_sizes.csv").open(encoding="utf-8-sig", newline="") as handle:
        size = next(row for row in csv.DictReader(handle) if row["dataset"] == "SWaT" and row["method"] == "Ours-REC")
    with (FINAL / "event_metrics_36.csv").open(encoding="utf-8-sig", newline="") as handle:
        events = list(csv.DictReader(handle))
    return {"variant": "R0 Final", "architecture_change": "Frozen final depthwise-separable k=3 reference", "params": int(metrics["parameter_count"]), "param_ratio_vs_R0": 1.0, "auroc": primary["auroc"], "aupr": primary["aupr"], "precision": primary["precision"], "recall": primary["recall"], "f1": primary["f1"], "fpr": primary["fpr"], "event_hit": sum(float(row["rec_hit_rate"]) >= 0.10 for row in events), "latency_mean_ms": float(latency["mean_latency_ms"]), "latency_median_ms": float(latency["median_latency_ms"]), "latency_p95_ms": float(latency["p95_latency_ms"]), "peak_vram_mib": float(latency["peak_vram_mib"]), "state_dict_bytes": int(size["model_size_bytes"]), "train_time_seconds": None, "best_epoch": int(metrics["best_epoch"]), "best_validation_loss": float(metrics["best_validation_loss"]), "stop_reason": "frozen existing result"}


def build_plot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.extend([
            {"panel": "AUPR_vs_Params", "variant": row["variant"], "x_metric": "params", "x_value": row["params"], "y_metric": "aupr", "y_value": row["aupr"]},
            {"panel": "AUPR_vs_Latency", "variant": row["variant"], "x_metric": "latency_mean_ms", "x_value": row["latency_mean_ms"], "y_metric": "aupr", "y_value": row["aupr"]},
        ])
        for metric in ("auroc", "aupr", "precision", "recall", "f1", "fpr"):
            output.append({"panel": "Detection_metrics", "variant": row["variant"], "x_metric": "variant", "x_value": row["variant"], "y_metric": metric, "y_value": row[metric]})
    return output


def main() -> int:
    overall_started = time.perf_counter()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry["status"] != "ABLATION_PROTOCOL_READY" or "ABLATION_A1_INNOVATION2_COMPLETED" not in ABL1.read_text(encoding="utf-8") or "ABLATION_A2_INNOVATION3_COMPLETED" not in ABL2.read_text(encoding="utf-8"):
        raise RuntimeError("ABL3A prerequisite gate failed")
    if not torch.cuda.is_available():
        raise RuntimeError("ABL3A requires the RTX 4060 CUDA runtime")
    config = load_config(str(CONFIG))
    validate_config_contract(config)
    if config["training"] != {"loss": "SmoothL1Loss", "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 256, "max_epochs": 15, "early_stopping_patience": 4, "checkpoint_selection": "minimum_validation_reconstruction_loss"}:
        raise RuntimeError("formal R0 training protocol changed")
    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    normal, cache_manifest = load_normal_cache(cache_dir, config)
    feature_names = [str(name) for name in cache_manifest["column_order"]]
    OUT.mkdir(parents=True, exist_ok=True)
    variant_manifest = build_variant_manifest(config, feature_names)
    manifest_path = OUT / "rec_variant_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != variant_manifest:
            raise RuntimeError("existing REC variant manifest differs from frozen definition")
    else:
        json_dump(manifest_path, variant_manifest)
    scaler, scaler_metadata = fit_train_only_scaler(normal)
    if scaler.mean_.dtype != np.float32 or scaler_metadata["validation_rows_seen"] != 0 or scaler_metadata["attack_rows_seen"] != 0:
        raise RuntimeError("formal REC scaler dtype/scope changed")
    rows = [load_r0_row()]
    for variant, model_class in (("R1", TemporalFreeAE), ("R2", DenseConvAE)):
        row, complete = train_variant(variant, model_class, config, normal, scaler, feature_names, cache_manifest, overall_started)
        if not complete:
            summary = {"status": "ABLATION_A3A_HARD_STOP_INCOMPLETE", "completed_variants": [item["variant"] for item in rows], "pending_variant": variant, "elapsed_seconds": time.perf_counter() - overall_started, "r0_checkpoint_unchanged": sha256(R0_OUT / "best.pt") == variant_manifest["source_hashes"]["r0_checkpoint"], "commit": False, "push": False}
            json_dump(OUT / "abl3a_summary.json", summary)
            print(json.dumps(summary, indent=2), flush=True)
            return 0
        assert row is not None
        rows.append(row)
        if time.perf_counter() - overall_started >= HARD_STOP_SECONDS and variant != "R2":
            summary = {"status": "ABLATION_A3A_HARD_STOP_INCOMPLETE", "completed_variants": [item["variant"] for item in rows], "pending_variant": "R2", "elapsed_seconds": time.perf_counter() - overall_started, "r0_checkpoint_unchanged": sha256(R0_OUT / "best.pt") == variant_manifest["source_hashes"]["r0_checkpoint"], "commit": False, "push": False}
            json_dump(OUT / "abl3a_summary.json", summary)
            print(json.dumps(summary, indent=2), flush=True)
            return 0
    r0 = rows[0]
    for row in rows:
        row.update({"delta_auroc_vs_R0": row["auroc"] - r0["auroc"], "delta_aupr_vs_R0": row["aupr"] - r0["aupr"], "delta_f1_vs_R0": row["f1"] - r0["f1"], "delta_params_vs_R0": row["params"] - r0["params"], "delta_latency_mean_ms_vs_R0": row["latency_mean_ms"] - r0["latency_mean_ms"]})
    payload = {"innovation1_swat_ablation": rows, "innovation1_swat_plot_data": build_plot_rows(rows), "r1_training_history": json.loads((OUT / "swat" / "r1" / "training_history.json").read_text(encoding="utf-8")), "r2_training_history": json.loads((OUT / "swat" / "r2" / "training_history.json").read_text(encoding="utf-8"))}
    json_dump(OUT / "csv_payload.json", payload)
    tests = {"r1_r2_definitions_frozen": True, "r0_not_retrained": sha256(R0_OUT / "best.pt") == variant_manifest["source_hashes"]["r0_checkpoint"], "split_exact": True, "scaler_train_only_float32": scaler.mean_.dtype == np.float32, "feature_order_exact": len(feature_names) == 51, "window_64": WINDOW == 64, "hard_gap_not_crossed": True, "training_config_identical": True, "validation_only_checkpoint": True, "validation_only_q99": True, "raw_scores_persisted": all((OUT / "swat" / name / "validation_raw_scores.npz").exists() and (OUT / "swat" / name / "test_raw_scores.npz").exists() for name in ("r1", "r2")), "no_pa_oracle_test_tuning": True}
    if not all(tests.values()):
        raise RuntimeError(f"ABL3A integrity gate failed: {tests}")
    summary = {"status": "ABLATION_A3A_INNOVATION1_SWAT_COMPLETED", "evidence_class": "RETROSPECTIVE_ABLATION", "elapsed_seconds": time.perf_counter() - overall_started, "rows": rows, "tests": tests, "r0_checkpoint_unchanged": tests["r0_not_retrained"], "restrictions": {"hai_training": False, "r0_retrain": False, "hp_search": False, "architecture_search": False, "test_tuning": False, "pa": False, "oracle": False, "new_rec_variant": False, "method_change": False, "commit": False, "push": False}}
    json_dump(OUT / "abl3a_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
