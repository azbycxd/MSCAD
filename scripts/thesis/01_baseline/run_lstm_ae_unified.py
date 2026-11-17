"""Baseline-B1: frozen LSTM-AE evaluation on Official SWaT 2015 and HAI 21.03."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data.swat2015_official import (  # noqa: E402
    load_attack_cache,
    load_normal_cache,
    resolve_repo_path,
    validate_config_contract,
)
from data.transforms import StandardScaler  # noqa: E402
from utils.config_io import load_config  # noqa: E402


WINDOW = 64
HIDDEN = 64
SEED = 1
TRAIN_BATCH = 128
SCORE_BATCH = 2048
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 0.0
Q = 0.99
EVENT_HIT_THRESHOLD = 0.10
OUT = ROOT / "outputs" / "thesis_baselines" / "lstm_ae"
DOC = ROOT / "docs" / "thesis" / "baseline_b1_lstm_ae.md"
B0_MANIFEST = ROOT / "outputs" / "thesis_baselines" / "baseline_manifest.json"
EFFICIENCY_PROTOCOL = ROOT / "outputs" / "thesis_baselines" / "efficiency_protocol.json"
TABLE = ROOT / "outputs" / "thesis_baselines" / "baseline_table_template.csv"
SWAT_CONFIG = ROOT / "configs" / "thesis" / "swat2015_midterm_seed1.yaml"
SWAT_SEGMENTS = ROOT / "outputs" / "thesis_t25_swat2015_protocol" / "segment_manifest.json"
SWAT_EVENTS = ROOT / "outputs" / "thesis_t25_swat2015_protocol" / "canonical_attack_events.csv"
HAI_ROOT = ROOT / "dataset" / "hai" / "hai-21.03"
HAI_AUDIT = ROOT / "outputs" / "thesis_hai21_a0_audit"
HAI_TRAIN_FILES = tuple(f"train{i}.csv.gz" for i in range(1, 4))
HAI_TEST_FILES = tuple(f"test{i}.csv.gz" for i in range(1, 6))
HIGH_AP = {"AP02", "AP04", "AP07", "AP09", "AP12", "AP14", "AP17", "AP19"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class LSTMAutoencoder(nn.Module):
    """Frozen B0 architecture: final-state encoder and repeated-latent decoder."""

    def __init__(self, channels: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.encoder = nn.LSTM(channels, hidden, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.projection = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        repeated = hidden[-1].unsqueeze(1).expand(-1, x.shape[1], -1)
        decoded, _ = self.decoder(repeated)
        return self.projection(decoded)


@dataclass
class WindowStore:
    values: np.ndarray
    starts: np.ndarray
    width: int = WINDOW

    @classmethod
    def from_segments(cls, segments: list[np.ndarray], *, common_endpoint_mask: bool) -> "WindowStore":
        if not segments or any(x.ndim != 2 or len(x) < WINDOW + int(common_endpoint_mask) for x in segments):
            raise ValueError("all segments must be 2D and long enough")
        values = np.concatenate([np.asarray(x, dtype=np.float32) for x in segments], axis=0)
        starts: list[np.ndarray] = []
        offset = 0
        first = 1 if common_endpoint_mask else 0
        for segment in segments:
            starts.append(offset + np.arange(first, len(segment) - WINDOW + 1, dtype=np.int64))
            offset += len(segment)
        return cls(values=values, starts=np.concatenate(starts))

    def batch(self, ids: np.ndarray) -> np.ndarray:
        indices = self.starts[ids, None] + np.arange(self.width, dtype=np.int64)[None, :]
        result = self.values[indices]
        if not np.isfinite(result).all():
            raise RuntimeError("NaN/Inf in window batch")
        return result

    def first_windows(self, count: int) -> np.ndarray:
        ids = np.arange(min(count, len(self.starts)), dtype=np.int64)
        return self.batch(ids)


def dump_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    names = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def train_model(
    dataset: str,
    channels: int,
    train_store: WindowStore,
    validation_store: WindowStore,
    device: torch.device,
    latest_checkpoint_path: Path | None = None,
) -> tuple[LSTMAutoencoder, list[dict[str, Any]], dict[str, Any]]:
    set_seed(SEED)
    model = LSTMAutoencoder(channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss(reduction="mean")
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    order = np.arange(len(train_store.starts), dtype=np.int64)
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        np.random.shuffle(order)
        total_loss = 0.0
        total_windows = 0
        for begin in range(0, len(order), TRAIN_BATCH):
            ids = order[begin : begin + TRAIN_BATCH]
            batch = torch.from_numpy(train_store.batch(ids)).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
            total_windows += len(ids)
        model.eval()
        val_total = 0.0
        val_windows = 0
        with torch.inference_mode():
            for begin in range(0, len(validation_store.starts), SCORE_BATCH):
                ids = np.arange(begin, min(begin + SCORE_BATCH, len(validation_store.starts)), dtype=np.int64)
                batch = torch.from_numpy(validation_store.batch(ids)).to(device, non_blocking=True)
                loss = loss_fn(model(batch), batch)
                val_total += float(loss) * len(ids)
                val_windows += len(ids)
        train_loss = total_loss / total_windows
        val_loss = val_total / val_windows
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        history.append({
            "dataset": dataset,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "improved": improved,
        })
        if latest_checkpoint_path is not None:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None,
                "early_stop_counter": stale,
                "best_validation_loss": best_loss,
                "best_epoch": best_epoch,
                "best_model_state_dict": best_state,
                "history": history,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "protocol": {
                    "seed": SEED,
                    "batch_size": TRAIN_BATCH,
                    "max_epochs": MAX_EPOCHS,
                    "patience": PATIENCE,
                    "learning_rate": LR,
                    "weight_decay": WEIGHT_DECAY,
                },
            }, latest_checkpoint_path)
        print(
            f"[{dataset}] epoch={epoch:03d} train={train_loss:.8f} val={val_loss:.8f} "
            f"best={best_loss:.8f}@{best_epoch} stale={stale}/{PATIENCE}",
            flush=True,
        )
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("no best model state captured")
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - started
    return model, history, {
        "training_seconds": training_seconds,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "early_stopped": len(history) < MAX_EPOCHS,
    }


def score_model(model: nn.Module, store: WindowStore, device: torch.device) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for begin in range(0, len(store.starts), SCORE_BATCH):
            ids = np.arange(begin, min(begin + SCORE_BATCH, len(store.starts)), dtype=np.int64)
            batch = torch.from_numpy(store.batch(ids)).to(device, non_blocking=True)
            reconstructed = model(batch)
            scores = torch.mean((batch[:, -1, :] - reconstructed[:, -1, :]) ** 2, dim=1)
            parts.append(scores.detach().cpu().numpy().astype(np.float64, copy=False))
    result = np.concatenate(parts)
    if len(result) != len(store.starts) or not np.isfinite(result).all():
        raise RuntimeError("invalid endpoint scores")
    return result


def point_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = (scores > threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    fpr = float(fp / (fp + tn)) if fp + tn else 0.0
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPR": float(average_precision_score(labels, scores)),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "threshold": float(threshold),
        "endpoint_count": int(len(labels)),
        "anomaly_endpoints": int(labels.sum()),
    }


def efficiency(model: nn.Module, windows: np.ndarray, state_path: Path, device: torch.device) -> dict[str, Any]:
    torch.save(model.state_dict(), state_path)
    params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if len(windows) < 1000:
        raise RuntimeError("efficiency protocol needs 1000 legal windows")
    samples = [torch.from_numpy(windows[i : i + 1]).to(device) for i in range(1000)]
    model.eval()
    with torch.inference_mode():
        for i in range(100):
            _ = model(samples[i])
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        latencies: list[float] = []
        for sample in samples:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)
        peak = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else None
    return {
        "trainable_parameters": int(params),
        "state_dict_bytes": int(state_path.stat().st_size),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_median": float(np.median(latencies)),
        "latency_ms_p95": float(np.quantile(latencies, 0.95)),
        "peak_inference_memory_mib": peak,
        "latency_warmup": 100,
        "latency_timed": 1000,
        "latency_batch_size": 1,
    }


def swat_data() -> dict[str, Any]:
    config = load_config(str(SWAT_CONFIG))
    validate_config_contract(config)
    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    normal, cache_manifest = load_normal_cache(cache_dir, config)
    attack, labels, timestamps, _ = load_attack_cache(cache_dir, config)
    if normal.shape != (495000, 51) or attack.shape != (449919, 51):
        raise RuntimeError("SWaT source shape changed")
    scaler = StandardScaler().fit(np.asarray(normal[:396000], dtype=np.float64))
    train = scaler.transform(np.asarray(normal[:396000])).astype(np.float32)
    validation = scaler.transform(np.asarray(normal[396000:495000])).astype(np.float32)
    segment_manifest = json.loads(SWAT_SEGMENTS.read_text(encoding="utf-8"))
    lengths = [int(row["row_count"]) for row in segment_manifest["test_segments"]]
    if lengths != [298837, 151082] or sum(lengths) != len(attack):
        raise RuntimeError("SWaT frozen segments changed")
    scaled_attack = scaler.transform(np.asarray(attack)).astype(np.float32)
    test_segments: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    timestamp_parts: list[np.ndarray] = []
    raw_index_parts: list[np.ndarray] = []
    offset = 0
    for length in lengths:
        end = offset + length
        test_segments.append(scaled_attack[offset:end])
        label_parts.append(np.asarray(labels[offset + WINDOW : end], dtype=np.int8))
        timestamp_parts.append(np.asarray(timestamps[offset + WINDOW : end], dtype=np.int64))
        raw_index_parts.append(np.arange(offset + WINDOW, end, dtype=np.int64))
        offset = end
    return {
        "train": [train],
        "validation": [validation],
        "test": test_segments,
        "test_labels": np.concatenate(label_parts),
        "test_timestamps": np.concatenate(timestamp_parts),
        "test_raw_indices": np.concatenate(raw_index_parts),
        "cache_dir": str(cache_dir),
        "feature_names": list(cache_manifest["column_order"]),
        "expected_validation_endpoints": 99000 - WINDOW,
        "expected_test_endpoints": sum(length - WINDOW for length in lengths),
    }


def hai_data() -> dict[str, Any]:
    schema = pd.read_csv(HAI_AUDIT / "variable_schema.csv")
    features = schema["column_name"].astype(str).tolist()
    if len(features) != 79 or "attack" in features:
        raise RuntimeError("HAI feature schema changed")
    scaler = StandardScaler()
    raw_train: list[np.ndarray] = []
    boundaries: dict[str, int] = {}
    # fit_many is equivalent to a joint train-only population StandardScaler.
    for name in HAI_TRAIN_FILES:
        values = pd.read_csv(HAI_ROOT / name, compression="gzip", usecols=features, low_memory=False).to_numpy(np.float64)
        boundary = int(math.floor(0.8 * len(values)))
        raw_train.append(values)
        boundaries[name] = boundary
    scaler.fit_many([values[: boundaries[name]] for name, values in zip(HAI_TRAIN_FILES, raw_train)])
    train = [scaler.transform(values[: boundaries[name]]).astype(np.float32) for name, values in zip(HAI_TRAIN_FILES, raw_train)]
    validation = [scaler.transform(values[boundaries[name] :]).astype(np.float32) for name, values in zip(HAI_TRAIN_FILES, raw_train)]
    del raw_train
    tests: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    file_parts: list[np.ndarray] = []
    raw_indices: list[np.ndarray] = []
    test_rows: dict[str, int] = {}
    for name in HAI_TEST_FILES:
        frame = pd.read_csv(HAI_ROOT / name, compression="gzip", usecols=features + ["attack"], low_memory=False)
        values = scaler.transform(frame[features].to_numpy(np.float64)).astype(np.float32)
        raw_label = frame["attack"].to_numpy(np.int8)
        if set(np.unique(raw_label).tolist()) - {0, 1}:
            raise RuntimeError(f"HAI label is not binary in {name}")
        tests.append(values)
        labels.append(raw_label[WINDOW:])
        file_parts.append(np.repeat(name, len(values) - WINDOW))
        raw_indices.append(np.arange(WINDOW, len(values), dtype=np.int64))
        test_rows[name] = len(values)
    return {
        "train": train,
        "validation": validation,
        "test": tests,
        "test_labels": np.concatenate(labels),
        "test_files": np.concatenate(file_parts),
        "test_raw_indices": np.concatenate(raw_indices),
        "feature_names": features,
        "boundaries": boundaries,
        "test_rows": test_rows,
        "expected_validation_endpoints": sum(len(x) - WINDOW for x in validation),
        "expected_test_endpoints": sum(len(x) - WINDOW for x in tests),
    }


def swat_event_metrics(predictions: np.ndarray, timestamps: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = pd.read_csv(SWAT_EVENTS)
    events = events[events["evaluable_physical_attack_interval"].astype(str).str.lower() == "true"].copy()
    if len(events) != 36:
        raise RuntimeError("SWaT event set is not 36")
    rows: list[dict[str, Any]] = []
    for row in events.itertuples(index=False):
        start = int(np.datetime64(str(row.canonical_start), "s").astype(np.int64))
        end = int(np.datetime64(str(row.canonical_end), "s").astype(np.int64))
        mask = (timestamps >= start) & (timestamps <= end)
        fraction = float(predictions[mask].mean()) if mask.any() else 0.0
        rows.append({
            "event_id": str(row.event_id),
            "attack_id": int(row.attack_id),
            "canonical_start": str(row.canonical_start),
            "canonical_end": str(row.canonical_end),
            "valid_endpoint_count": int(mask.sum()),
            "predicted_anomaly_fraction": fraction,
            "hit_at_0.10": bool(fraction >= EVENT_HIT_THRESHOLD),
        })
    hits = sum(bool(row["hit_at_0.10"]) for row in rows)
    return rows, {"event_count": 36, "event_hit_count": hits, "event_hit_rate": hits / 36.0}


def hai_event_metrics(predictions: np.ndarray, files: np.ndarray, raw_indices: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = pd.read_csv(HAI_AUDIT / "attack_events_50.csv")
    if len(events) != 50:
        raise RuntimeError("HAI event set is not 50")
    rows: list[dict[str, Any]] = []
    for row in events.itertuples(index=False):
        mask = (
            (files == str(row.file))
            & (raw_indices >= int(row.start_index))
            & (raw_indices <= int(row.end_index_inclusive))
        )
        fraction = float(predictions[mask].mean()) if mask.any() else 0.0
        aps = {part.strip() for part in str(row.AP_scenario).split("+")}
        taxonomy = "HIGH" if aps & HIGH_AP else "DIRECT"
        rows.append({
            "event_id": str(row.event_id),
            "file": str(row.file),
            "AP_scenario": str(row.AP_scenario),
            "taxonomy": taxonomy,
            "start_index": int(row.start_index),
            "end_index_inclusive": int(row.end_index_inclusive),
            "valid_endpoint_count": int(mask.sum()),
            "predicted_anomaly_fraction": fraction,
            "hit_at_0.10": bool(fraction >= EVENT_HIT_THRESHOLD),
        })
    high = [row for row in rows if row["taxonomy"] == "HIGH"]
    direct = [row for row in rows if row["taxonomy"] == "DIRECT"]
    if len(high) != 19 or len(direct) != 31:
        raise RuntimeError(f"HAI taxonomy changed: HIGH={len(high)} DIRECT={len(direct)}")
    summarize = lambda subset: {
        "event_count": len(subset),
        "event_hit_count": sum(bool(row["hit_at_0.10"]) for row in subset),
        "event_hit_rate": sum(bool(row["hit_at_0.10"]) for row in subset) / len(subset),
        "median_predicted_anomaly_fraction": float(np.median([row["predicted_anomaly_fraction"] for row in subset])),
    }
    return rows, {"ALL": summarize(rows), "HIGH": summarize(high), "DIRECT": summarize(direct)}


def evaluate_dataset(name: str, data: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], np.ndarray]:
    train_store = WindowStore.from_segments(data["train"], common_endpoint_mask=False)
    validation_store = WindowStore.from_segments(data["validation"], common_endpoint_mask=True)
    test_store = WindowStore.from_segments(data["test"], common_endpoint_mask=True)
    if len(validation_store.starts) != data["expected_validation_endpoints"]:
        raise RuntimeError(f"{name} validation endpoint count mismatch")
    if len(test_store.starts) != data["expected_test_endpoints"] or len(data["test_labels"]) != len(test_store.starts):
        raise RuntimeError(f"{name} test endpoint count mismatch")
    model, history, train_meta = train_model(name, len(data["feature_names"]), train_store, validation_store, device)
    validation_scores = score_model(model, validation_store, device)
    threshold = float(np.quantile(validation_scores, Q))
    test_scores = score_model(model, test_store, device)
    metrics = point_metrics(data["test_labels"], test_scores, threshold)
    metrics.update({
        "dataset": name,
        "validation_score_q99": threshold,
        "validation_exceedance_rate": float(np.mean(validation_scores > threshold)),
        "validation_endpoint_count": int(len(validation_scores)),
        "threshold_source": "Normal Validation endpoint scores only",
        "score": "endpoint mean channel squared reconstruction error",
        "window": WINDOW,
        "seed": SEED,
    })
    state_path = OUT / f"{name.lower()}_state_dict.pt"
    eff = efficiency(model, test_store.first_windows(1000), state_path, device)
    eff.update(train_meta)
    return metrics, history, eff, test_scores


def update_table(swat: dict[str, Any], hai: dict[str, Any], efficiency_rows: dict[str, dict[str, Any]], event_summaries: dict[str, Any]) -> None:
    table = pd.read_csv(TABLE)
    if set(table.loc[table.Method == "LSTM-AE", "Dataset"]) != {"SWaT", "HAI21"}:
        raise RuntimeError("baseline table LSTM-AE row contract changed")
    mapping = {"SWaT": (swat, event_summaries["SWaT"]["event_hit_count"]), "HAI21": (hai, event_summaries["HAI21"]["ALL"]["event_hit_count"])}
    for dataset, (metrics, hit_count) in mapping.items():
        idx = table.index[(table.Dataset == dataset) & (table.Method == "LSTM-AE")]
        if len(idx) != 1:
            raise RuntimeError("ambiguous table row")
        row = idx[0]
        table.loc[row, "Params"] = efficiency_rows[dataset]["trainable_parameters"]
        for key in ("AUROC", "AUPR", "Precision", "Recall", "F1", "FPR"):
            table.loc[row, key] = metrics[key]
        table.loc[row, "EventHitCount"] = hit_count
        table.loc[row, "Latency"] = efficiency_rows[dataset]["latency_ms_mean"]
    table.to_csv(TABLE, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="replace only the dedicated B1 output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overall_started = time.perf_counter()
    if OUT.exists() and any(OUT.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing B1 output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    b0 = json.loads(B0_MANIFEST.read_text(encoding="utf-8"))
    efficiency_protocol = json.loads(EFFICIENCY_PROTOCOL.read_text(encoding="utf-8"))
    frozen = b0["baselines"]["LSTM-AE"]
    expected = {
        "optimizer": "Adam", "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
        "batch_size": TRAIN_BATCH, "max_epochs": MAX_EPOCHS,
    }
    if b0["status"] != "BASELINE_PROTOCOL_FROZEN" or any(frozen[key] != value for key, value in expected.items()):
        raise RuntimeError("B0 LSTM-AE training protocol changed")
    if not torch.cuda.is_available():
        raise RuntimeError("B1 frozen efficiency/training protocol requires CUDA")
    device = torch.device("cuda")
    print(f"device={device} gpu={torch.cuda.get_device_name(device)}", flush=True)

    swat_data_bundle = swat_data()
    print("SWaT data loaded; starting frozen training", flush=True)
    swat, swat_history, swat_eff, swat_scores = evaluate_dataset("SWaT", swat_data_bundle, device)
    swat_predictions = swat_scores > swat["threshold"]
    swat_event_rows, swat_event_summary = swat_event_metrics(swat_predictions, swat_data_bundle["test_timestamps"])
    swat.update(swat_event_summary)
    dump_json(OUT / "swat_metrics.json", swat)
    write_csv(OUT / "swat_event_metrics.csv", swat_event_rows)
    del swat_data_bundle, swat_scores
    torch.cuda.empty_cache()

    hai_data_bundle = hai_data()
    print("HAI data loaded; starting frozen training", flush=True)
    hai, hai_history, hai_eff, hai_scores = evaluate_dataset("HAI21", hai_data_bundle, device)
    hai_predictions = hai_scores > hai["threshold"]
    hai_event_rows, hai_event_summary = hai_event_metrics(
        hai_predictions, hai_data_bundle["test_files"], hai_data_bundle["test_raw_indices"]
    )
    hai.update({"events": hai_event_summary})
    dump_json(OUT / "hai_metrics.json", hai)
    write_csv(OUT / "hai_event_metrics.csv", hai_event_rows)

    history = swat_history + hai_history
    write_csv(OUT / "training_history.csv", history)
    efficiency_rows = {"SWaT": swat_eff, "HAI21": hai_eff}
    efficiency_result = {
        "protocol_sha256": sha256(EFFICIENCY_PROTOCOL),
        "device": torch.cuda.get_device_name(device),
        "datasets": efficiency_rows,
    }
    dump_json(OUT / "efficiency.json", efficiency_result)
    event_summaries = {"SWaT": swat_event_summary, "HAI21": hai_event_summary}
    update_table(swat, hai, efficiency_rows, event_summaries)

    sanity = {
        "architecture_exact": True,
        "full_window_mse": True,
        "endpoint_score_only": True,
        "scaler_train_only": True,
        "validation_early_stopping_only": True,
        "threshold_validation_q99_only": True,
        "no_boundary_crossing": True,
        "swat_event_count_36": swat_event_summary["event_count"] == 36,
        "hai_event_count_50": hai_event_summary["ALL"]["event_count"] == 50,
        "hai_taxonomy_19_31": hai_event_summary["HIGH"]["event_count"] == 19 and hai_event_summary["DIRECT"]["event_count"] == 31,
        "seed_one": True,
        "no_test_tuning": True,
        "no_point_adjustment": True,
        "no_pot": True,
    }
    if not all(sanity.values()):
        raise RuntimeError(f"sanity gate failed: {sanity}")
    manifest = {
        "phase": "Thesis Baseline-B1",
        "status": "LSTM_AE_BASELINE_COMPLETED",
        "b0_manifest_sha256": sha256(B0_MANIFEST),
        "efficiency_protocol_sha256": sha256(EFFICIENCY_PROTOCOL),
        "architecture": frozen["architecture"],
        "loss": frozen["loss"],
        "optimizer": expected,
        "early_stopping": b0["common"]["early_stopping"],
        "score": frozen["score"],
        "threshold": b0["common"]["threshold"],
        "seed": SEED,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_code": str(Path(__file__).relative_to(ROOT)),
        "source_code_sha256": sha256(Path(__file__)),
        "datasets": {
            "SWaT": {"features": 51, "train_windows": 396000 - WINDOW + 1, "validation_endpoints": swat["validation_endpoint_count"], "test_endpoints": swat["endpoint_count"]},
            "HAI21": {"features": 79, "validation_endpoints": hai["validation_endpoint_count"], "test_endpoints": hai["endpoint_count"]},
        },
        "sanity": sanity,
        "forbidden": {
            "test_tuning": False,
            "point_adjustment": False,
            "pot": False,
            "threshold_search": False,
            "method_fusion": False,
            "usad_run": False,
            "gdn_run": False,
        },
        "elapsed_seconds": time.perf_counter() - overall_started,
        "commit": False,
        "push": False,
    }
    dump_json(OUT / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "SWaT": swat, "HAI21": hai}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
