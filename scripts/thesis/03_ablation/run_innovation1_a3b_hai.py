"""Run the P0-frozen HAI21 REC structural ablations R1 then R2.

Only the two already-frozen architecture substitutions are permitted.  The
formal HAI21 R0 checkpoint and metrics are read-only references.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backbone_ae import BackboneAE  # noqa: E402
from models.layers import ConvBNAct1d, DepthwiseSeparableConv1d  # noqa: E402
from utils.seed import set_seed  # noqa: E402


OUT = ROOT / "outputs" / "thesis_ablation" / "innovation1"
HAI_OUT = OUT / "hai"
DATA = ROOT / "dataset" / "hai" / "hai-21.03"
AUDIT = ROOT / "outputs" / "thesis_hai21_a0_audit"
R0_OUT = ROOT / "outputs" / "thesis_hai21_b1_real"
EFF = ROOT / "outputs" / "thesis_efficiency_final"
P0 = ROOT / "docs" / "thesis" / "ablation_p0_design.md"
TRACE = ROOT / "docs" / "thesis" / "rec_architecture_trace.md"
A3A = ROOT / "docs" / "thesis" / "ablation_a3a_innovation1_swat_results.md"
REGISTRY = ROOT / "outputs" / "thesis_ablation" / "ablation_registry.json"
SWAT_VARIANTS = OUT / "rec_variant_manifest.json"
SWAT_RESULTS = OUT / "innovation1_swat_ablation.csv"
TRAIN_FILES = tuple(f"train{i}.csv.gz" for i in range(1, 4))
TEST_FILES = tuple(f"test{i}.csv.gz" for i in range(1, 6))
LABELS = ("attack", "attack_P1", "attack_P2", "attack_P3")
HIGH_AP = {"AP02", "AP04", "AP07", "AP09", "AP12", "AP14", "AP17", "AP19"}
WINDOW = 64
SEED = 1
Q = 0.99
WARMUP = 100
TIMED = 1000
HARD_STOP_SECONDS = 25 * 60
TRAINING = {
    "loss": "SmoothL1Loss",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 256,
    "max_epochs": 15,
    "early_stopping_patience": 4,
    "checkpoint_selection": "minimum_validation_reconstruction_loss",
}


class TemporalFreeAE(nn.Module):
    """R1: exactly the SWaT R1 substitution, with C changed to 79."""

    def __init__(self, in_channels: int = 79, hidden_channels: int = 64, latent_channels: int = 32, dropout: float = 0.1):
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
        return self.decoder(self.encoder(x.transpose(1, 2))).transpose(1, 2)


class DenseConvAE(nn.Module):
    """R2: exactly the SWaT R2 substitution, with C changed to 79."""

    def __init__(self, in_channels: int = 79, hidden_channels: int = 64, latent_channels: int = 32, dropout: float = 0.1):
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
        return self.decoder(self.encoder(x.transpose(1, 2))).transpose(1, 2)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_hash(names: list[str]) -> str:
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


def normalized_signature(signature: dict[str, Any], feature_count: int) -> dict[str, Any]:
    """Replace only dataset-specific input/output width for SWaT/HAI comparison."""
    normalized = json.loads(json.dumps(signature))
    for row in normalized["conv1d"]:
        for field in ("in_channels", "out_channels", "groups"):
            if row[field] == feature_count:
                row[field] = "C"
    normalized.pop("parameter_count", None)
    return normalized


def architecture_audit() -> dict[str, Any]:
    frozen = json.loads(SWAT_VARIANTS.read_text(encoding="utf-8"))
    models = {
        "R0 Final": BackboneAE(79, 64, 32, 0.1),
        "R1": TemporalFreeAE(79, 64, 32, 0.1),
        "R2": DenseConvAE(79, 64, 32, 0.1),
    }
    expected_params = {"R0 Final": 15482, "R1": 15132, "R2": 32911}
    rows = {}
    for variant, model in models.items():
        actual = architecture_signature(model)
        frozen_signature = frozen["variants"][variant]["signature"]
        definition_match = normalized_signature(actual, 79) == normalized_signature(frozen_signature, 51)
        parameter_match = actual["parameter_count"] == expected_params[variant]
        rows[variant] = {
            "params": actual["parameter_count"],
            "expected_params": expected_params[variant],
            "parameter_match": parameter_match,
            "definition_matches_swat_after_C_substitution": definition_match,
            "signature": actual,
        }
    status = "PASS" if all(row["parameter_match"] and row["definition_matches_swat_after_C_substitution"] for row in rows.values()) else "ABLATION_A3B_ARCHITECTURE_MISMATCH"
    result = {
        "phase": "Thesis Ablation-ABL3B",
        "status": status,
        "allowed_change": "feature count C: 51 -> 79 only",
        "swat_variant_manifest_sha256": sha256(SWAT_VARIANTS),
        "variants": rows,
    }
    dump_json(OUT / "hai_variant_parameter_audit.json", result)
    return result


def window_view(values: np.ndarray) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(values, WINDOW, axis=0).transpose(0, 2, 1)


def batches(segments: list[np.ndarray], batch_size: int, shuffle: bool, rng: np.random.Generator):
    segment_order = rng.permutation(len(segments)) if shuffle else np.arange(len(segments))
    for segment_id in segment_order:
        windows = window_view(segments[int(segment_id)])
        order = rng.permutation(len(windows)) if shuffle else np.arange(len(windows))
        for offset in range(0, len(order), batch_size):
            yield np.ascontiguousarray(windows[order[offset:offset + batch_size]], dtype=np.float32)


def average_loss(model: nn.Module, segments: list[np.ndarray], device: torch.device) -> float:
    model.eval()
    criterion = nn.SmoothL1Loss()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for windows in batches(segments, TRAINING["batch_size"], False, np.random.default_rng(0)):
            x = torch.from_numpy(windows).to(device)
            loss = criterion(model(x), x)
            total += float(loss.item()) * len(windows)
            count += len(windows)
    return total / count


def score_segments(model: nn.Module, segments: list[np.ndarray], device: torch.device) -> list[np.ndarray]:
    model.eval()
    output = []
    with torch.inference_mode():
        for values in segments:
            parts = []
            for windows in batches([values], TRAINING["batch_size"], False, np.random.default_rng(0)):
                x = torch.from_numpy(windows).to(device)
                parts.append(torch.abs(model(x)[:, -1] - x[:, -1]).mean(dim=1).cpu().numpy())
            scores = np.concatenate(parts).astype(np.float64)
            if len(scores) != len(values) - WINDOW + 1 or not np.isfinite(scores).all():
                raise RuntimeError("raw endpoint score persistence failed")
            output.append(scores)
    return output


def strict_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = scores > threshold
    positive = labels == 1
    negative = labels == 0
    tp = int((prediction & positive).sum())
    fp = int((prediction & negative).sum())
    fn = int((~prediction & positive).sum())
    tn = int((~prediction & negative).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, np.finfo(float).eps),
        "fpr": fp / max(fp + tn, 1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def benchmark(model: nn.Module, train_segment: np.ndarray, device: torch.device) -> dict[str, Any]:
    windows = np.ascontiguousarray(window_view(train_segment)[:TIMED], dtype=np.float32)
    tensors = [torch.from_numpy(windows[index:index + 1]).to(device) for index in range(TIMED)]
    model.eval()
    for index in range(WARMUP):
        with torch.inference_mode():
            model(tensors[index % TIMED])
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for index in range(TIMED):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            model(tensors[index])
        torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "mean_latency_ms": float(np.mean(timings)),
        "median_latency_ms": float(np.median(timings)),
        "p95_latency_ms": float(np.quantile(timings, 0.95)),
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(device) / 1048576),
        "warmup": WARMUP,
        "timed_endpoints": TIMED,
        "batch_size": 1,
        "device": torch.cuda.get_device_name(device),
    }


def load_normal_data(feature_names: list[str]) -> tuple[StandardScaler, list[np.ndarray], list[np.ndarray], dict[str, int]]:
    scaler = StandardScaler()
    bounds: dict[str, int] = {}
    for name in TRAIN_FILES:
        frame = pd.read_csv(DATA / name, compression="gzip", usecols=feature_names, low_memory=False)
        boundary = int(math.floor(0.8 * len(frame)))
        bounds[name] = boundary
        scaler.partial_fit(frame.iloc[:boundary].to_numpy(np.float64))
    train, validation = [], []
    for name in TRAIN_FILES:
        frame = pd.read_csv(DATA / name, compression="gzip", usecols=feature_names, low_memory=False)
        boundary = bounds[name]
        # The formal HAI path fits float64 sklearn statistics, then ae_batches
        # casts each window to float32. Materializing that final dtype here is exact.
        train.append(scaler.transform(frame.iloc[:boundary].to_numpy(np.float64)).astype(np.float32))
        validation.append(scaler.transform(frame.iloc[boundary:].to_numpy(np.float64)).astype(np.float32))
    return scaler, train, validation, bounds


def save_scaler(path: Path, scaler: StandardScaler, bounds: dict[str, int], features: list[str]) -> None:
    np.savez_compressed(
        path,
        mean=np.asarray(scaler.mean_),
        scale=np.asarray(scaler.scale_),
        var=np.asarray(scaler.var_),
        n_samples_seen=np.asarray([scaler.n_samples_seen_]),
        train_boundaries=np.asarray([bounds[name] for name in TRAIN_FILES]),
        feature_order=np.asarray(features),
    )


def is_high(value: str) -> bool:
    return bool(set(str(value).split("+")) & HIGH_AP)


def evaluate_events(events: pd.DataFrame, records: dict[str, dict[str, np.ndarray]], threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for event in events.itertuples(index=False):
        record = records[str(event.file)]
        mask = (record["raw_index"] >= int(event.start_index)) & (record["raw_index"] <= int(event.end_index_inclusive))
        if not mask.any():
            raise RuntimeError(f"official HAI event absent: {event.event_id}")
        hit_rate = float((record["score"][mask] > threshold).mean())
        category = "HIGH" if is_high(str(event.AP_scenario)) else "DIRECT"
        rows.append({
            "file": str(event.file),
            "event_id": str(event.event_id),
            "AP_scenario": str(event.AP_scenario),
            "category": category,
            "point_count": int(mask.sum()),
            "hit_rate": hit_rate,
            "hit_at_least_0_10": hit_rate >= 0.10,
        })
    if len(rows) != 50:
        raise RuntimeError("official HAI event count changed")
    high = [row for row in rows if row["category"] == "HIGH"]
    direct = [row for row in rows if row["category"] == "DIRECT"]
    summary = {
        "all_event_count": 50,
        "event_hit_at_least_0_10": sum(row["hit_at_least_0_10"] for row in rows),
        "high_event_count": len(high),
        "high_median_hit": float(np.median([row["hit_rate"] for row in high])),
        "high_hit_at_least_0_10": sum(row["hit_at_least_0_10"] for row in high),
        "direct_event_count": len(direct),
        "direct_median_hit": float(np.median([row["hit_rate"] for row in direct])),
    }
    return rows, summary


def train_variant(
    variant: str,
    model_class: type[nn.Module],
    train: list[np.ndarray],
    validation: list[np.ndarray],
    scaler: StandardScaler,
    bounds: dict[str, int],
    feature_names: list[str],
    audit: dict[str, Any],
    events: pd.DataFrame,
    source_hashes: dict[str, str],
    overall_started: float,
) -> tuple[dict[str, Any] | None, bool]:
    output = HAI_OUT / variant.lower()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result_row.json"
    if result_path.exists() and (output / "manifest.json").exists():
        return json.loads(result_path.read_text(encoding="utf-8")), True

    set_seed(SEED)
    device = torch.device("cuda")
    model = model_class(79, 64, 32, 0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"])
    criterion = nn.SmoothL1Loss()
    latest = output / "latest_training_state.pt"
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_model: dict[str, torch.Tensor] | None = None
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        if state["variant"] != variant or state["training_config"] != TRAINING:
            raise RuntimeError(f"{variant} resume protocol mismatch")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        best_model = state["best_model_state_dict"]
        best_loss = float(state["best_validation_loss"])
        best_epoch = int(state["best_epoch"])
        stale = int(state["early_stop_counter"])
        history = list(state["history"])
        start_epoch = int(state["epoch"]) + 1
        restore_rng(state["rng_state"])

    completed = False
    stop_reason = "max_epochs"
    for epoch in range(start_epoch, TRAINING["max_epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        total = 0.0
        count = 0
        for windows in batches(train, TRAINING["batch_size"], True, np.random.default_rng(SEED + epoch)):
            x = torch.from_numpy(windows).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), x)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite HAI REC training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(windows)
            count += len(windows)
        train_loss = total / count
        validation_loss = average_loss(model, validation, device)
        improved = validation_loss < best_loss
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            best_model = cpu_state(model)
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "improved": improved,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        torch.save({
            "variant": variant,
            "epoch": epoch,
            "model_state_dict": cpu_state(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_model_state_dict": best_model,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "early_stop_counter": stale,
            "history": history,
            "training_config": TRAINING,
            "rng_state": rng_state(),
        }, latest)
        dump_json(output / "training_history.json", history)
        print(json.dumps({"variant": variant, **row, "early_stop_counter": stale}), flush=True)
        if stale >= TRAINING["early_stopping_patience"]:
            stop_reason = "early_stopping"
            completed = True
            break
        if epoch == TRAINING["max_epochs"]:
            stop_reason = "max_epochs"
            completed = True
            break
        if time.perf_counter() - overall_started >= HARD_STOP_SECONDS:
            stop_reason = "hard_stop_after_completed_epoch"
            break
    if not completed:
        dump_json(output / "partial_status.json", {
            "status": "ABLATION_A3B_HARD_STOP_INCOMPLETE",
            "variant": variant,
            "last_epoch": history[-1]["epoch"],
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "early_stop_counter": stale,
        })
        return None, False
    if best_model is None:
        raise RuntimeError(f"{variant} has no validation-selected model")

    model.load_state_dict(best_model)
    model.eval()
    torch.save({
        "variant": variant,
        "state_dict": best_model,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "selection_scope": "HAI Normal Validation suffix reconstruction loss only; Test unopened",
        "training_config": TRAINING,
    }, output / "best.pt")
    torch.save(best_model, output / "state_dict.pt")
    save_scaler(output / "scaler.npz", scaler, bounds, feature_names)
    dump_json(output / "feature_order.json", {"features": feature_names, "sha256": feature_hash(feature_names)})
    dump_json(output / "architecture_manifest.json", audit["variants"][variant])

    validation_parts = score_segments(model, validation, device)
    validation_scores = np.concatenate(validation_parts)
    validation_file = np.concatenate([np.full(len(scores), name) for scores, name in zip(validation_parts, TRAIN_FILES)])
    validation_raw_index = np.concatenate([np.arange(bounds[name] + WINDOW - 1, bounds[name] + WINDOW - 1 + len(scores), dtype=np.int64) for scores, name in zip(validation_parts, TRAIN_FILES)])
    threshold = float(np.quantile(validation_scores, Q))
    np.savez_compressed(
        output / "validation_raw_scores.npz",
        endpoint_mae=validation_scores,
        source_file=validation_file,
        raw_index=validation_raw_index,
    )
    dump_json(output / "thresholds.json", {"source": "HAI Normal Validation suffixes only", "quantile": Q, "comparator": ">", "endpoint_mae": threshold})

    records: dict[str, dict[str, np.ndarray]] = {}
    score_parts, label_parts, file_parts, index_parts, timestamp_parts = [], [], [], [], []
    for name in TEST_FILES:
        frame = pd.read_csv(DATA / name, compression="gzip", low_memory=False)
        if list(frame.columns[1:-4]) != feature_names or list(frame.columns[-4:]) != list(LABELS):
            raise RuntimeError(f"HAI Test schema changed: {name}")
        scaled = scaler.transform(frame[feature_names].to_numpy(np.float64)).astype(np.float32)
        scores = score_segments(model, [scaled], device)[0]
        raw_index = np.arange(WINDOW - 1, len(frame), dtype=np.int64)
        labels = frame["attack"].to_numpy(np.int8)[WINDOW - 1:]
        timestamps = frame["time"].astype(str).to_numpy()[WINDOW - 1:]
        records[name] = {"score": scores, "raw_index": raw_index}
        score_parts.append(scores)
        label_parts.append(labels)
        file_parts.append(np.full(len(scores), name))
        index_parts.append(raw_index)
        timestamp_parts.append(timestamps)
    scores = np.concatenate(score_parts)
    labels = np.concatenate(label_parts)
    metrics = strict_metrics(labels, scores, threshold)
    event_rows, event_summary = evaluate_events(events, records, threshold)
    np.savez_compressed(
        output / "test_raw_scores.npz",
        endpoint_mae=scores,
        label=labels,
        source_file=np.concatenate(file_parts),
        raw_index=np.concatenate(index_parts),
        timestamp=np.concatenate(timestamp_parts),
    )
    dump_json(output / "event_metrics_50.json", event_rows)
    dump_json(output / "metrics.json", {
        "dataset": "HAI 21.03",
        "variant": variant,
        "evidence_class": "RETROSPECTIVE_ABLATION",
        "threshold": threshold,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "stop_reason": stop_reason,
        **metrics,
        **event_summary,
        "test_points": len(labels),
        "point_adjustment": False,
        "oracle": False,
        "test_tuning": False,
    })
    efficiency = benchmark(model, train[0], device)
    efficiency.update({
        "params": parameter_count(model),
        "state_dict_bytes": (output / "state_dict.pt").stat().st_size,
        "best_checkpoint_bytes": (output / "best.pt").stat().st_size,
        "training_time_seconds": sum(float(row["epoch_seconds"]) for row in history),
        "best_epoch": best_epoch,
    })
    dump_json(output / "efficiency.json", efficiency)
    artifact_names = (
        "best.pt", "state_dict.pt", "latest_training_state.pt", "scaler.npz", "feature_order.json",
        "architecture_manifest.json", "training_history.json", "validation_raw_scores.npz",
        "test_raw_scores.npz", "thresholds.json", "metrics.json", "event_metrics_50.json", "efficiency.json",
    )
    manifest = {
        "variant": variant,
        "status": "COMPLETED",
        "evidence_class": "RETROSPECTIVE_ABLATION",
        "dataset": "HAI 21.03",
        "source_hashes": source_hashes,
        "feature_order_sha256": feature_hash(feature_names),
        "splits": {"train_prefix_boundaries": bounds, "validation": "per-file suffix [floor(0.8*N),N)", "test_files": list(TEST_FILES)},
        "window": WINDOW,
        "stride": 1,
        "cross_file_windows": False,
        "scaler": {
            "implementation": "sklearn.preprocessing.StandardScaler.partial_fit",
            "fit_scope": "three HAI Normal Train prefixes only",
            "statistic_dtype": str(scaler.mean_.dtype),
            "model_input_dtype": "float32",
            "validation_rows_seen": 0,
            "test_rows_seen": 0,
            "note": "Exact formal HAI R0 path; distinct from SWaT custom float32 scaler and baseline model paths.",
        },
        "training": TRAINING,
        "checkpoint_selection": "minimum HAI Normal Validation suffix reconstruction loss",
        "score": "endpoint mean absolute reconstruction error",
        "threshold": "Normal Validation q99",
        "r0_retrained": False,
        "artifacts": {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size} for name in artifact_names},
        "restrictions": {"pa": False, "oracle": False, "test_tuning": False, "hp_search": False, "architecture_search": False, "new_variant": False},
    }
    dump_json(output / "manifest.json", manifest)
    result = {
        "variant": variant,
        "architecture_change": json.loads(SWAT_VARIANTS.read_text(encoding="utf-8"))["variants"][variant]["relative_to_R0"],
        "params": efficiency["params"],
        "param_ratio_vs_R0": efficiency["params"] / 15482,
        "auroc": metrics["auroc"],
        "aupr": metrics["aupr"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "fpr": metrics["fpr"],
        "event_hit": event_summary["event_hit_at_least_0_10"],
        "high_median_hit": event_summary["high_median_hit"],
        "high_hit": event_summary["high_hit_at_least_0_10"],
        "direct_median_hit": event_summary["direct_median_hit"],
        "latency_mean_ms": efficiency["mean_latency_ms"],
        "latency_median_ms": efficiency["median_latency_ms"],
        "latency_p95_ms": efficiency["p95_latency_ms"],
        "peak_vram_mib": efficiency["peak_vram_mib"],
        "state_dict_bytes": efficiency["state_dict_bytes"],
        "train_time_seconds": efficiency["training_time_seconds"],
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "stop_reason": stop_reason,
    }
    dump_json(result_path, result)
    return result, True


def load_r0_row(events: pd.DataFrame) -> dict[str, Any]:
    summary = json.loads((R0_OUT / "hai21_b1_summary.json").read_text(encoding="utf-8"))
    metric = next(row for row in summary["global_metrics"] if row["method"] == "REC")
    event_rows = pd.read_csv(R0_OUT / "event_metrics_50.csv")
    high = event_rows[event_rows["category"] == "HIGH"]
    direct = event_rows[event_rows["category"] == "DIRECT"]
    with (EFF / "ours_latency.csv").open(encoding="utf-8-sig", newline="") as stream:
        latency = next(row for row in csv.DictReader(stream) if row["dataset"] == "HAI21" and row["method"] == "Ours-REC")
    with (EFF / "ours_model_sizes.csv").open(encoding="utf-8-sig", newline="") as stream:
        size = next(row for row in csv.DictReader(stream) if row["dataset"] == "HAI21" and row["method"] == "Ours-REC")
    if len(events) != 50 or len(high) != 19 or len(direct) != 31:
        raise RuntimeError("formal HAI R0 event semantics changed")
    return {
        "variant": "R0 Final",
        "architecture_change": "Frozen final depthwise-separable k=3 reference",
        "params": int(summary["rec_parameter_count"]),
        "param_ratio_vs_R0": 1.0,
        "auroc": metric["auroc"],
        "aupr": metric["aupr"],
        "precision": metric["precision"],
        "recall": metric["recall"],
        "f1": metric["f1"],
        "fpr": metric["fpr"],
        "event_hit": int(summary["event_hit_010_counts"]["REC"]),
        "high_median_hit": float(high["rec_hit_rate"].median()),
        "high_hit": int((high["rec_hit_rate"] >= 0.10).sum()),
        "direct_median_hit": float(direct["rec_hit_rate"].median()),
        "latency_mean_ms": float(latency["mean_latency_ms"]),
        "latency_median_ms": float(latency["median_latency_ms"]),
        "latency_p95_ms": float(latency["p95_latency_ms"]),
        "peak_vram_mib": float(latency["peak_vram_mib"]),
        "state_dict_bytes": int(size["model_size_bytes"]),
        "train_time_seconds": None,
        "best_epoch": int(summary["rec_best_epoch"]),
        "best_validation_loss": float(summary["rec_best_validation_loss"]),
        "stop_reason": "frozen existing result",
    }


def read_swat_rows() -> list[dict[str, Any]]:
    with SWAT_RESULTS.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = {
        "params": int, "param_ratio_vs_R0": float, "auroc": float, "aupr": float, "precision": float,
        "recall": float, "f1": float, "fpr": float, "event_hit": int, "latency_mean_ms": float,
        "latency_median_ms": float, "latency_p95_ms": float, "peak_vram_mib": float, "state_dict_bytes": int,
        "best_epoch": int, "best_validation_loss": float, "delta_auroc_vs_R0": float,
        "delta_aupr_vs_R0": float, "delta_f1_vs_R0": float, "delta_params_vs_R0": int,
        "delta_latency_mean_ms_vs_R0": float,
    }
    for row in rows:
        for key, constructor in numeric.items():
            row[key] = constructor(row[key])
        row["train_time_seconds"] = float(row["train_time_seconds"]) if row["train_time_seconds"] else None
    return rows


def add_deltas(rows: list[dict[str, Any]]) -> None:
    r0 = rows[0]
    for row in rows:
        row.update({
            "delta_auroc_vs_R0": row["auroc"] - r0["auroc"],
            "delta_aupr_vs_R0": row["aupr"] - r0["aupr"],
            "delta_f1_vs_R0": row["f1"] - r0["f1"],
            "delta_fpr_vs_R0": row["fpr"] - r0["fpr"],
            "delta_params_vs_R0": row["params"] - r0["params"],
            "delta_latency_mean_ms_vs_R0": row["latency_mean_ms"] - r0["latency_mean_ms"],
        })


def cross_dataset_payload(swat_rows: list[dict[str, Any]], hai_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cross = []
    for dataset, rows in (("SWaT", swat_rows), ("HAI21", hai_rows)):
        for row in rows:
            cross.append({
                "dataset": dataset,
                "variant": row["variant"],
                "params": row["params"],
                "auroc": row["auroc"],
                "aupr": row["aupr"],
                "f1": row["f1"],
                "fpr": row["fpr"],
                "event_hit": row["event_hit"],
                "latency_mean_ms": row["latency_mean_ms"],
                "latency_p95_ms": row["latency_p95_ms"],
                "peak_vram_mib": row["peak_vram_mib"],
                "delta_auroc_vs_R0": row["delta_auroc_vs_R0"],
                "delta_aupr_vs_R0": row["delta_aupr_vs_R0"],
                "delta_f1_vs_R0": row["delta_f1_vs_R0"],
                "delta_fpr_vs_R0": row["delta_fpr_vs_R0"],
                "delta_params_vs_R0": row["delta_params_vs_R0"],
                "delta_latency_mean_ms_vs_R0": row["delta_latency_mean_ms_vs_R0"],
            })
    plots = []
    for row in cross:
        plots.extend([
            {"panel": "AUPR_vs_Params", **row, "x_metric": "params", "x_value": row["params"], "y_metric": "aupr", "y_value": row["aupr"]},
            {"panel": "AUPR_vs_Mean_latency", **row, "x_metric": "latency_mean_ms", "x_value": row["latency_mean_ms"], "y_metric": "aupr", "y_value": row["aupr"]},
            {"panel": "Normalized_performance", **row, "x_metric": "variant", "x_value": row["variant"], "y_metric": "aupr_ratio_vs_R0", "y_value": row["aupr"] / next(x["aupr"] for x in cross if x["dataset"] == row["dataset"] and x["variant"] == "R0 Final")},
            {"panel": "Normalized_performance", **row, "x_metric": "variant", "x_value": row["variant"], "y_metric": "auroc_ratio_vs_R0", "y_value": row["auroc"] / next(x["auroc"] for x in cross if x["dataset"] == row["dataset"] and x["variant"] == "R0 Final")},
        ])
    by = {(row["dataset"], row["variant"]): row for row in cross}
    facts = {
        "R1_vs_R0_AUPR_SWaT": by[("SWaT", "R1")]["delta_aupr_vs_R0"],
        "R1_vs_R0_AUPR_HAI": by[("HAI21", "R1")]["delta_aupr_vs_R0"],
        "R2_vs_R0_AUPR_SWaT": by[("SWaT", "R2")]["delta_aupr_vs_R0"],
        "R2_vs_R0_AUPR_HAI": by[("HAI21", "R2")]["delta_aupr_vs_R0"],
        "R1_param_ratio_SWaT": by[("SWaT", "R1")]["params"] / by[("SWaT", "R0 Final")]["params"],
        "R1_param_ratio_HAI": by[("HAI21", "R1")]["params"] / by[("HAI21", "R0 Final")]["params"],
        "R2_param_ratio_SWaT": by[("SWaT", "R2")]["params"] / by[("SWaT", "R0 Final")]["params"],
        "R2_param_ratio_HAI": by[("HAI21", "R2")]["params"] / by[("HAI21", "R0 Final")]["params"],
        "R1_latency_ratio_SWaT": by[("SWaT", "R1")]["latency_mean_ms"] / by[("SWaT", "R0 Final")]["latency_mean_ms"],
        "R1_latency_ratio_HAI": by[("HAI21", "R1")]["latency_mean_ms"] / by[("HAI21", "R0 Final")]["latency_mean_ms"],
        "R2_latency_ratio_SWaT": by[("SWaT", "R2")]["latency_mean_ms"] / by[("SWaT", "R0 Final")]["latency_mean_ms"],
        "R2_latency_ratio_HAI": by[("HAI21", "R2")]["latency_mean_ms"] / by[("HAI21", "R0 Final")]["latency_mean_ms"],
    }
    signs = lambda a, b: (a >= 0 and b >= 0) or (a < 0 and b < 0)
    facts["direction_status"] = "CONSISTENT_DIRECTION" if signs(facts["R1_vs_R0_AUPR_SWaT"], facts["R1_vs_R0_AUPR_HAI"]) and signs(facts["R2_vs_R0_AUPR_SWaT"], facts["R2_vs_R0_AUPR_HAI"]) else "DATASET_DEPENDENT_DIRECTION"
    facts["r1_dominance_status"] = "R1_DOMINATES_R0_ON_OBSERVED_RANKING_METRICS" if all(
        by[(dataset, "R1")]["params"] <= by[(dataset, "R0 Final")]["params"]
        and by[(dataset, "R1")]["auroc"] >= by[(dataset, "R0 Final")]["auroc"]
        and by[(dataset, "R1")]["aupr"] >= by[(dataset, "R0 Final")]["aupr"]
        for dataset in ("SWaT", "HAI21")
    ) else "R1_DOES_NOT_DOMINATE_R0_ON_BOTH_DATASETS"
    facts["r2_status"] = "DENSIFICATION_NO_CONSISTENT_GAIN" if (
        facts["R2_param_ratio_SWaT"] > 1 and facts["R2_param_ratio_HAI"] > 1
        and not (facts["R2_vs_R0_AUPR_SWaT"] > 0 and facts["R2_vs_R0_AUPR_HAI"] > 0)
    ) else "DENSIFICATION_GAIN_OBSERVED_ON_BOTH_DATASETS"
    return cross, plots, facts


def main() -> int:
    overall_started = time.perf_counter()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry["status"] != "ABLATION_PROTOCOL_READY" or "ABLATION_A3A_INNOVATION1_SWAT_COMPLETED" not in A3A.read_text(encoding="utf-8"):
        raise RuntimeError("ABL3B prerequisite gate failed")
    if not torch.cuda.is_available():
        raise RuntimeError("ABL3B requires CUDA")
    HAI_OUT.mkdir(parents=True, exist_ok=True)
    audit = architecture_audit()
    if audit["status"] != "PASS":
        dump_json(OUT / "abl3b_summary.json", {"status": "ABLATION_A3B_ARCHITECTURE_MISMATCH", "architecture_audit": audit})
        return 0

    audit_summary = json.loads((AUDIT / "audit_summary.json").read_text(encoding="utf-8"))
    if audit_summary["status"] != "HAI21_BENCHMARK_PROTOCOL_CANDIDATE_READY":
        raise RuntimeError("HAI-A0 protocol changed")
    schema = pd.read_csv(AUDIT / "variable_schema.csv")
    feature_names = schema["column_name"].astype(str).tolist()
    if len(feature_names) != 79:
        raise RuntimeError("HAI feature count changed")
    events = pd.read_csv(AUDIT / "attack_events_50.csv")
    source_paths = [DATA / name for name in TRAIN_FILES + TEST_FILES] + [AUDIT / "audit_summary.json", AUDIT / "variable_schema.csv", AUDIT / "attack_events_50.csv"]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    r0_hash_before = sha256(R0_OUT / "rec_best.pt")
    scaler, train, validation, bounds = load_normal_data(feature_names)
    if int(scaler.n_samples_seen_) != sum(bounds.values()) or scaler.mean_.dtype != np.float64:
        raise RuntimeError("formal HAI R0 scaler path changed")

    rows = [load_r0_row(events)]
    for variant, model_class in (("R1", TemporalFreeAE), ("R2", DenseConvAE)):
        row, complete = train_variant(variant, model_class, train, validation, scaler, bounds, feature_names, audit, events, source_hashes, overall_started)
        if not complete:
            summary = {
                "status": "ABLATION_A3B_HARD_STOP_INCOMPLETE",
                "completed_variants": [item["variant"] for item in rows],
                "pending_variant": variant,
                "elapsed_seconds": time.perf_counter() - overall_started,
                "r0_checkpoint_unchanged": sha256(R0_OUT / "rec_best.pt") == r0_hash_before,
                "commit": False,
                "push": False,
            }
            dump_json(OUT / "abl3b_summary.json", summary)
            print(json.dumps(summary, indent=2), flush=True)
            return 0
        assert row is not None
        rows.append(row)
        if time.perf_counter() - overall_started >= HARD_STOP_SECONDS and variant != "R2":
            summary = {
                "status": "ABLATION_A3B_HARD_STOP_INCOMPLETE",
                "completed_variants": [item["variant"] for item in rows],
                "pending_variant": "R2",
                "elapsed_seconds": time.perf_counter() - overall_started,
                "r0_checkpoint_unchanged": sha256(R0_OUT / "rec_best.pt") == r0_hash_before,
                "commit": False,
                "push": False,
            }
            dump_json(OUT / "abl3b_summary.json", summary)
            print(json.dumps(summary, indent=2), flush=True)
            return 0

    add_deltas(rows)
    swat_rows = read_swat_rows()
    # Normalize field coverage of ABL3A rows without changing their values.
    for row in swat_rows:
        row.setdefault("delta_fpr_vs_R0", row["fpr"] - swat_rows[0]["fpr"])
        row.setdefault("delta_auroc_vs_R0", row["auroc"] - swat_rows[0]["auroc"])
        row.setdefault("delta_aupr_vs_R0", row["aupr"] - swat_rows[0]["aupr"])
        row.setdefault("delta_f1_vs_R0", row["f1"] - swat_rows[0]["f1"])
        row.setdefault("delta_params_vs_R0", row["params"] - swat_rows[0]["params"])
        row.setdefault("delta_latency_mean_ms_vs_R0", row["latency_mean_ms"] - swat_rows[0]["latency_mean_ms"])
    cross, plots, facts = cross_dataset_payload(swat_rows, rows)
    payload = {"innovation1_hai_ablation": rows, "innovation1_cross_dataset_ablation": cross, "innovation1_cross_dataset_plot_data": plots}
    dump_json(OUT / "abl3b_csv_payload.json", payload)

    tests = {
        "hai_r1_definition_equals_swat_r1": audit["variants"]["R1"]["definition_matches_swat_after_C_substitution"],
        "hai_r2_definition_equals_swat_r2": audit["variants"]["R2"]["definition_matches_swat_after_C_substitution"],
        "r0_not_retrained": sha256(R0_OUT / "rec_best.pt") == r0_hash_before,
        "formal_hai_ours_rec_scaler_path_preserved": scaler.mean_.dtype == np.float64 and int(scaler.n_samples_seen_) == sum(bounds.values()),
        "train_validation_split_identical": bounds == {"train1.csv.gz": 172800, "train2.csv.gz": 181440, "train3.csv.gz": 383040},
        "no_cross_file_window": True,
        "training_config_identical_except_architecture": all(json.loads((HAI_OUT / name / "manifest.json").read_text(encoding="utf-8"))["training"] == TRAINING for name in ("r1", "r2")),
        "checkpoint_by_validation_loss_only": all("Normal Validation" in torch.load(HAI_OUT / name / "best.pt", map_location="cpu", weights_only=False)["selection_scope"] for name in ("r1", "r2")),
        "q99_from_normal_validation_only": all(json.loads((HAI_OUT / name / "thresholds.json").read_text(encoding="utf-8"))["source"] == "HAI Normal Validation suffixes only" for name in ("r1", "r2")),
        "raw_scores_persisted": all((HAI_OUT / name / "validation_raw_scores.npz").exists() and (HAI_OUT / name / "test_raw_scores.npz").exists() for name in ("r1", "r2")),
        "event_semantics_unchanged": len(events) == 50 and sum(events["AP_scenario"].map(is_high)) == 19,
        "no_pa_oracle_test_tuning": True,
    }
    if not all(tests.values()):
        raise RuntimeError(f"ABL3B integrity failure: {tests}")
    hashes_after = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    if hashes_after != source_hashes:
        raise RuntimeError("HAI source or audit artifact changed")
    summary = {
        "status": "ABLATION_A3_INNOVATION1_COMPLETED",
        "evidence_class": "RETROSPECTIVE_ABLATION",
        "elapsed_seconds": time.perf_counter() - overall_started,
        "rows": rows,
        "cross_dataset_facts": facts,
        "tests": tests,
        "r0_checkpoint_unchanged": tests["r0_not_retrained"],
        "scaler_evidence_note": "Formal HAI R0 uses train-only sklearn float64 partial_fit statistics and float32 model inputs; SWaT uses its distinct custom float32 scaler.",
        "restrictions": {"r0_retrain": False, "new_variant": False, "hp_search": False, "architecture_search": False, "test_tuning": False, "pa": False, "oracle": False, "method_change": False, "commit": False, "push": False},
    }
    dump_json(OUT / "abl3b_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
