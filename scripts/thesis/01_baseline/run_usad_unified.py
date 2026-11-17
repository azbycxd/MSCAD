"""Baseline-B2: frozen official-core USAD evaluation on SWaT and HAI."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
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
from sklearn.preprocessing import MinMaxScaler
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data.swat2015_official import load_attack_cache, load_normal_cache, resolve_repo_path, validate_config_contract  # noqa: E402
from utils.config_io import load_config  # noqa: E402


WINDOW = 64
Z_SIZE = 6400
SEED = 1
BATCH = 256
SCORE_BATCH = 256
MAX_EPOCHS = 100
PATIENCE = 10
MIN_DELTA = 0.0
LR = 1e-3
WEIGHT_DECAY = 0.0
Q = 0.99
SOURCE_COMMIT = "e25af45c8e1c32783aed94fc7e5ab85effef2b1a"
OUT = ROOT / "outputs" / "thesis_baselines" / "usad"
LATEST = OUT / "latest_training_state.pt"
SWAT_CONFIG = ROOT / "configs" / "thesis" / "swat2015_midterm_seed1.yaml"
HAI_ROOT = ROOT / "dataset" / "hai" / "hai-21.03"
HAI_AUDIT = ROOT / "outputs" / "thesis_hai21_a0_audit"
BASELINE_MANIFEST = ROOT / "outputs" / "thesis_baselines" / "baseline_manifest.json"


def load_base_module():
    path = Path(__file__).with_name("run_lstm_ae_unified.py")
    spec = importlib.util.spec_from_file_location("baseline_b1_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Baseline-B1 helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class Encoder(nn.Module):
    def __init__(self, in_size: int, latent_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_size, in_size // 2)
        self.linear2 = nn.Linear(in_size // 2, in_size // 4)
        self.linear3 = nn.Linear(in_size // 4, latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear3(self.relu(self.linear2(self.relu(self.linear1(window))))))


class Decoder(nn.Module):
    def __init__(self, latent_size: int, out_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(latent_size, out_size // 4)
        self.linear2 = nn.Linear(out_size // 4, out_size // 2)
        self.linear3 = nn.Linear(out_size // 2, out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear3(self.relu(self.linear2(self.relu(self.linear1(latent))))))


class USAD(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.w_size = WINDOW * channels
        self.encoder = Encoder(self.w_size, Z_SIZE)
        self.decoder1 = Decoder(Z_SIZE, self.w_size)
        self.decoder2 = Decoder(Z_SIZE, self.w_size)

    def outputs(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(flat)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return w1, w2, w3


def losses(model: USAD, flat: torch.Tensor, epoch: int) -> tuple[torch.Tensor, torch.Tensor]:
    w1, w2, w3 = model.outputs(flat)
    fraction = 1.0 / epoch
    loss1 = fraction * torch.mean((flat - w1) ** 2) + (1.0 - fraction) * torch.mean((flat - w3) ** 2)
    loss2 = fraction * torch.mean((flat - w2) ** 2) - (1.0 - fraction) * torch.mean((flat - w3) ** 2)
    return loss1, loss2


def protocol() -> dict[str, Any]:
    manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    usad = manifest["baselines"]["USAD"]
    policy = manifest["normalization_policy"]["methods"]["USAD"]
    early = manifest["common"]["early_stopping"]
    checks = {
        "source_commit": usad["source_commit"] == SOURCE_COMMIT,
        "normalization": policy["scaler"] == "TRAIN_ONLY_MINMAX_0_1",
        "no_clip": policy["clip"] is False,
        "window": manifest["common"]["window"] == WINDOW,
        "seed": manifest["seed"] == SEED,
        "lr": usad["learning_rate"] == LR,
        "weight_decay": usad["weight_decay"] == WEIGHT_DECAY,
        "batch": usad["batch_size"] == BATCH,
        "epochs": usad["max_epochs"] == MAX_EPOCHS,
        "patience": early["patience"] == PATIENCE,
        "min_delta": early["min_delta"] == MIN_DELTA,
        "sigmoid": usad["decoder_output"] == "sigmoid",
        "score_weights": usad["score_weights"] == {"alpha": 0.5, "beta": 0.5},
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen protocol mismatch: {checks}")
    return checks


def swat_train_only() -> tuple[list[np.ndarray], MinMaxScaler]:
    config = load_config(str(SWAT_CONFIG))
    validate_config_contract(config)
    normal, _ = load_normal_cache(resolve_repo_path(ROOT, config["paths"]["cache_dir"]), config)
    raw = np.asarray(normal[:396000], dtype=np.float32)
    scaler = MinMaxScaler(feature_range=(0, 1), clip=False).fit(raw)
    return [scaler.transform(raw).astype(np.float32)], scaler


def hai_feature_names() -> list[str]:
    names = pd.read_csv(HAI_AUDIT / "variable_schema.csv")["column_name"].astype(str).tolist()
    if len(names) != 79 or "attack" in names:
        raise RuntimeError("HAI feature schema changed")
    return names


def hai_train_only() -> tuple[list[np.ndarray], MinMaxScaler, dict[str, int]]:
    names = hai_feature_names()
    row_counts = {"train1.csv.gz": 216001, "train2.csv.gz": 226801, "train3.csv.gz": 478801}
    boundaries = {name: math.floor(0.8 * rows) for name, rows in row_counts.items()}
    raw: list[np.ndarray] = []
    scaler = MinMaxScaler(feature_range=(0, 1), clip=False)
    for name, boundary in boundaries.items():
        values = pd.read_csv(HAI_ROOT / name, compression="gzip", usecols=names, nrows=boundary, low_memory=False).to_numpy(np.float32)
        if len(values) != boundary:
            raise RuntimeError(f"HAI train prefix changed: {name}")
        scaler.partial_fit(values)
        raw.append(values)
    return [scaler.transform(values).astype(np.float32) for values in raw], scaler, boundaries


def window_store(segments: list[np.ndarray], common_mask: bool):
    return BASE.WindowStore.from_segments(segments, common_endpoint_mask=common_mask)


def optimizer_pair(model: USAD):
    optimizer1 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder1.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer2 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder2.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    return optimizer1, optimizer2


def official_train_step(model: USAD, flat: torch.Tensor, epoch: int, optimizer1, optimizer2) -> tuple[float, float]:
    loss1, _ = losses(model, flat, epoch)
    loss1.backward()
    optimizer1.step()
    optimizer1.zero_grad()
    _, loss2 = losses(model, flat, epoch)
    loss2.backward()
    optimizer2.step()
    optimizer2.zero_grad()
    if not torch.isfinite(loss1) or not torch.isfinite(loss2):
        raise RuntimeError("non-finite USAD loss")
    return float(loss1.detach()), float(loss2.detach())


def preflight_dataset(dataset: str, channels: int, segments: list[np.ndarray], device: torch.device) -> dict[str, Any]:
    set_seed()
    model = USAD(channels).to(device)
    params = sum(p.numel() for p in model.parameters())
    store = window_store(segments, common_mask=False)
    batch = torch.from_numpy(store.batch(np.arange(BATCH, dtype=np.int64))).reshape(BATCH, -1).to(device)
    optimizer1, optimizer2 = optimizer_pair(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    loss1, loss2 = official_train_step(model, batch, 1, optimizer1, optimizer2)
    changed = any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    peak = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None
    result = {
        "dataset": dataset,
        "channels": channels,
        "w_size": WINDOW * channels,
        "z_size": Z_SIZE,
        "parameter_count": int(params),
        "estimated_parameter_bytes_float32": int(params * 4),
        "batch": BATCH,
        "loss1": loss1,
        "loss2": loss2,
        "parameters_updated": changed,
        "peak_allocated_mib": peak,
        "status": "PASS" if changed else "FAIL",
        "data_scope": "Normal Train only",
    }
    del before, optimizer1, optimizer2, batch, store, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not changed:
        raise RuntimeError(f"{dataset} preflight did not update parameters")
    return result


def validation_losses(model: USAD, store, device: torch.device, epoch: int) -> tuple[float, float]:
    model.eval()
    first, second = [], []
    with torch.inference_mode():
        for begin in range(0, len(store.starts), SCORE_BATCH):
            ids = np.arange(begin, min(begin + SCORE_BATCH, len(store.starts)), dtype=np.int64)
            flat = torch.from_numpy(store.batch(ids)).reshape(len(ids), -1).to(device)
            loss1, loss2 = losses(model, flat, epoch)
            first.append(float(loss1)); second.append(float(loss2))
    return float(np.mean(first)), float(np.mean(second))


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_latest(dataset: str, epoch: int, model: USAD, opt1, opt2, best_state, best_epoch: int, best_loss: float, stale: int, history, order, completed: bool) -> None:
    torch.save({
        "dataset": dataset,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer1_state_dict": opt1.state_dict(),
        "optimizer2_state_dict": opt2.state_dict(),
        "best_model_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "early_stop_counter": stale,
        "history": history,
        "order": order,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "completed": completed,
        "protocol": {"source_commit": SOURCE_COMMIT, "window": WINDOW, "z_size": Z_SIZE, "batch": BATCH, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA, "lr": LR, "weight_decay": WEIGHT_DECAY, "seed": SEED, "scaler": "TRAIN_ONLY_MINMAX_0_1", "clip": False, "validation_monitor": "official val_loss1"},
    }, LATEST)


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state["torch_cuda_rng_state_all"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])


def train_dataset(dataset: str, channels: int, train_segments: list[np.ndarray], validation_segments: list[np.ndarray], device: torch.device):
    train = window_store(train_segments, common_mask=False)
    validation = window_store(validation_segments, common_mask=True)
    set_seed()
    model = USAD(channels).to(device)
    opt1, opt2 = optimizer_pair(model)
    best_loss, best_epoch, best_state, stale = math.inf, 0, None, 0
    history: list[dict[str, Any]] = []
    order = np.arange(len(train.starts), dtype=np.int64)
    start_epoch = 1
    if LATEST.exists():
        state = torch.load(LATEST, map_location="cpu", weights_only=False)
        if state.get("dataset") == dataset and not state.get("completed", False):
            expected = {"source_commit": SOURCE_COMMIT, "window": WINDOW, "z_size": Z_SIZE, "batch": BATCH, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA, "lr": LR, "weight_decay": WEIGHT_DECAY, "seed": SEED, "scaler": "TRAIN_ONLY_MINMAX_0_1", "clip": False, "validation_monitor": "official val_loss1"}
            if state["protocol"] != expected:
                raise RuntimeError("resume protocol mismatch")
            model.load_state_dict(state["model_state_dict"]); opt1.load_state_dict(state["optimizer1_state_dict"]); opt2.load_state_dict(state["optimizer2_state_dict"])
            best_state = state["best_model_state_dict"]; best_epoch = int(state["best_epoch"]); best_loss = float(state["best_validation_loss"]); stale = int(state["early_stop_counter"])
            history = list(state["history"]); order = np.asarray(state["order"], dtype=np.int64); start_epoch = int(state["epoch"]) + 1
            restore_rng(state)
            print(f"[{dataset}] exact resume from epoch {start_epoch}", flush=True)
    started = time.perf_counter()
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train(); np.random.shuffle(order); train1, train2 = [], []
        for begin in range(0, len(order), BATCH):
            ids = order[begin:begin + BATCH]
            flat = torch.from_numpy(train.batch(ids)).reshape(len(ids), -1).to(device)
            loss1, loss2 = official_train_step(model, flat, epoch, opt1, opt2)
            train1.append(loss1); train2.append(loss2)
        val1, val2 = validation_losses(model, validation, device, epoch)
        improved = val1 < best_loss - MIN_DELTA
        if improved:
            best_loss, best_epoch, best_state, stale = val1, epoch, cpu_state(model), 0
        else:
            stale += 1
        row = {"dataset": dataset, "epoch": epoch, "train_loss1": float(np.mean(train1)), "train_loss2": float(np.mean(train2)), "validation_loss1": val1, "validation_loss2": val2, "improved": improved}
        history.append(row)
        completed = stale >= PATIENCE or epoch == MAX_EPOCHS
        save_latest(dataset, epoch, model, opt1, opt2, best_state, best_epoch, best_loss, stale, history, order, completed)
        print(f"[{dataset}] epoch={epoch:03d} val1={val1:.8f} val2={val2:.8f} best={best_loss:.8f}@{best_epoch} stale={stale}/{PATIENCE}", flush=True)
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("no best USAD state")
    model.load_state_dict(best_state)
    return model, history, {"training_seconds": time.perf_counter() - started, "epochs_ran": len(history), "best_epoch": best_epoch, "best_validation_loss": best_loss, "early_stopped": len(history) < MAX_EPOCHS, "validation_monitor": "official val_loss1"}, validation


def score_model(model: USAD, store, device: torch.device) -> np.ndarray:
    model.eval(); parts = []
    with torch.inference_mode():
        for begin in range(0, len(store.starts), SCORE_BATCH):
            ids = np.arange(begin, min(begin + SCORE_BATCH, len(store.starts)), dtype=np.int64)
            windows = torch.from_numpy(store.batch(ids)).to(device)
            flat = windows.reshape(len(ids), -1)
            w1, _, w3 = model.outputs(flat)
            x_endpoint = windows[:, -1, :]
            s1 = w1.reshape(len(ids), WINDOW, model.channels)[:, -1, :]
            s3 = w3.reshape(len(ids), WINDOW, model.channels)[:, -1, :]
            score = torch.mean(0.5 * (x_endpoint - s1) ** 2 + 0.5 * (x_endpoint - s3) ** 2, dim=1)
            parts.append(score.detach().cpu().numpy().astype(np.float64))
    result = np.concatenate(parts)
    if len(result) != len(store.starts) or not np.isfinite(result).all():
        raise RuntimeError("invalid USAD scores")
    return result


def formal_swat(device: torch.device):
    config = load_config(str(SWAT_CONFIG)); validate_config_contract(config)
    cache = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    normal, _ = load_normal_cache(cache, config)
    raw_train = np.asarray(normal[:396000], dtype=np.float32); raw_val = np.asarray(normal[396000:495000], dtype=np.float32)
    scaler = MinMaxScaler(feature_range=(0, 1), clip=False).fit(raw_train)
    train = [scaler.transform(raw_train).astype(np.float32)]; validation = [scaler.transform(raw_val).astype(np.float32)]
    model, history, training, val_store = train_dataset("SWaT", 51, train, validation, device)
    val_scores = score_model(model, val_store, device); threshold = float(np.quantile(val_scores, Q))
    attack, labels, timestamps, _ = load_attack_cache(cache, config)
    scaled = scaler.transform(np.asarray(attack, dtype=np.float32)).astype(np.float32)
    segment_manifest = json.loads(BASE.SWAT_SEGMENTS.read_text(encoding="utf-8")); lengths = [int(x["row_count"]) for x in segment_manifest["test_segments"]]
    segments, label_parts, time_parts = [], [], []; offset = 0
    for length in lengths:
        end = offset + length; segments.append(scaled[offset:end]); label_parts.append(np.asarray(labels[offset + WINDOW:end], dtype=np.int8)); time_parts.append(np.asarray(timestamps[offset + WINDOW:end], dtype=np.int64)); offset = end
    test_store = window_store(segments, common_mask=True); test_scores = score_model(model, test_store, device); test_labels = np.concatenate(label_parts); test_times = np.concatenate(time_parts)
    metrics = BASE.point_metrics(test_labels, test_scores, threshold); events, event_summary = BASE.swat_event_metrics((test_scores > threshold).astype(np.int8), test_times)
    metrics.update({"dataset": "SWaT", "validation_score_q99": threshold, "validation_exceedance_rate": float((val_scores > threshold).mean()), "score": "USAD endpoint 0.5*w1 MSE + 0.5*w3 MSE", "normalization": "TRAIN_ONLY_MINMAX_0_1", "event_count": 36, "event_hit_count": event_summary["event_hit_count"], "event_hit_rate": event_summary["event_hit_rate"]})
    return model, history, training, metrics, events, train[0][:WINDOW + 1000]


def formal_hai(device: torch.device):
    names = hai_feature_names(); files = ["train1.csv.gz", "train2.csv.gz", "train3.csv.gz"]; raw = [pd.read_csv(HAI_ROOT / name, compression="gzip", usecols=names, low_memory=False).to_numpy(np.float32) for name in files]
    boundaries = {name: math.floor(0.8 * len(values)) for name, values in zip(files, raw)}; scaler = MinMaxScaler(feature_range=(0, 1), clip=False)
    for name, values in zip(files, raw): scaler.partial_fit(values[:boundaries[name]])
    train = [scaler.transform(values[:boundaries[name]]).astype(np.float32) for name, values in zip(files, raw)]; validation = [scaler.transform(values[boundaries[name]:]).astype(np.float32) for name, values in zip(files, raw)]; del raw
    model, history, training, val_store = train_dataset("HAI21", 79, train, validation, device)
    val_scores = score_model(model, val_store, device); threshold = float(np.quantile(val_scores, Q))
    test_segments, labels, test_files, raw_indices = [], [], [], []
    for name in [f"test{i}.csv.gz" for i in range(1, 6)]:
        frame = pd.read_csv(HAI_ROOT / name, compression="gzip", usecols=names + ["attack"], low_memory=False); values = scaler.transform(frame[names].to_numpy(np.float32)).astype(np.float32); y = frame["attack"].to_numpy(np.int8)
        test_segments.append(values); labels.append(y[WINDOW:]); test_files.append(np.repeat(name, len(values) - WINDOW)); raw_indices.append(np.arange(WINDOW, len(values), dtype=np.int64))
    test_store = window_store(test_segments, common_mask=True); test_scores = score_model(model, test_store, device); test_labels = np.concatenate(labels); files_array = np.concatenate(test_files); indices = np.concatenate(raw_indices)
    metrics = BASE.point_metrics(test_labels, test_scores, threshold); events, event_summary = BASE.hai_event_metrics((test_scores > threshold).astype(np.int8), files_array, indices)
    metrics.update({"dataset": "HAI21", "validation_score_q99": threshold, "validation_exceedance_rate": float((val_scores > threshold).mean()), "score": "USAD endpoint 0.5*w1 MSE + 0.5*w3 MSE", "normalization": "TRAIN_ONLY_MINMAX_0_1", "events": event_summary})
    return model, history, training, metrics, events, train[0][:WINDOW + 1000]


def efficiency(model: USAD, samples: np.ndarray, state_path: Path, device: torch.device) -> dict[str, Any]:
    torch.save(model.state_dict(), state_path); store = window_store([samples], common_mask=False); windows = store.first_windows(1000); tensors = [torch.from_numpy(windows[i:i+1]).to(device) for i in range(1000)]
    model.eval()
    with torch.inference_mode():
        for i in range(100): model.outputs(tensors[i].reshape(1, -1))
        if device.type == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
        latency = []
        for tensor in tensors:
            if device.type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter(); model.outputs(tensor.reshape(1, -1))
            if device.type == "cuda": torch.cuda.synchronize()
            latency.append((time.perf_counter() - start) * 1000)
    return {"trainable_parameters": sum(p.numel() for p in model.parameters()), "state_dict_bytes": state_path.stat().st_size, "latency_ms_mean": float(np.mean(latency)), "latency_ms_median": float(np.median(latency)), "latency_ms_p95": float(np.quantile(latency, .95)), "peak_inference_memory_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None, "latency_warmup": 100, "latency_timed": 1000, "latency_batch_size": 1}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    checks = protocol()
    if not torch.cuda.is_available(): raise RuntimeError("RTX 4060 CUDA is unavailable")
    device = torch.device("cuda")
    try:
        swat_train, _ = swat_train_only(); swat_pf = preflight_dataset("SWaT", 51, swat_train, device); del swat_train
        hai_train, _, _ = hai_train_only(); hai_pf = preflight_dataset("HAI21", 79, hai_train, device); del hai_train
    except torch.cuda.OutOfMemoryError:
        print(json.dumps({"status": "USAD_FROZEN_BATCH_OOM"}), flush=True); return 2
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = {"status": "RESOURCE_PREFLIGHT_PASS", "device": torch.cuda.get_device_name(device), "datasets": {"SWaT": swat_pf, "HAI21": hai_pf}}
    BASE.dump_json(OUT / "resource_preflight.json", preflight); print(json.dumps(preflight), flush=True)
    histories, efficiencies, trainings = [], {}, {}
    if (OUT / "swat_metrics.json").exists() and (OUT / "swat_state_dict.pt").exists():
        raise FileExistsError("existing completed SWaT USAD outputs require an explicit resume-only runner")
    swat_model, swat_history, swat_training, swat_metrics, swat_events, swat_samples = formal_swat(device)
    BASE.dump_json(OUT / "swat_metrics.json", swat_metrics); write_csv(OUT / "swat_event_metrics.csv", swat_events); histories += swat_history; trainings["SWaT"] = swat_training
    efficiencies["SWaT"] = efficiency(swat_model, swat_samples, OUT / "swat_state_dict.pt", device); efficiencies["SWaT"].update(swat_training); del swat_model; torch.cuda.empty_cache()
    hai_model, hai_history, hai_training, hai_metrics, hai_events, hai_samples = formal_hai(device)
    BASE.dump_json(OUT / "hai_metrics.json", hai_metrics); write_csv(OUT / "hai_event_metrics.csv", hai_events); histories += hai_history; trainings["HAI21"] = hai_training
    efficiencies["HAI21"] = efficiency(hai_model, hai_samples, OUT / "hai_state_dict.pt", device); efficiencies["HAI21"].update(hai_training)
    write_csv(OUT / "training_history.csv", histories); BASE.dump_json(OUT / "efficiency.json", {"device": torch.cuda.get_device_name(device), "datasets": efficiencies})
    manifest = {"phase": "Thesis Baseline-B2", "status": "USAD_BASELINE_COMPLETED", "protocol_checks": checks, "resource_preflight": preflight, "source_commit": SOURCE_COMMIT, "architecture": {"shared_encoder": True, "decoder1": True, "decoder2": True, "z_size": Z_SIZE, "decoder_output": "Sigmoid"}, "normalization": {"scaler": "TRAIN_ONLY_MINMAX_0_1", "clip": False}, "training": trainings, "validation_monitor": "official val_loss1", "score": "endpoint mean channels: 0.5*(x-w1)^2 + 0.5*(x-w3)^2", "threshold": "Normal Validation q99", "no_test_tuning": True, "point_adjustment": False, "pot": False, "gdn_run": False, "ours_rerun": False, "commit": False, "push": False}
    BASE.dump_json(OUT / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "SWaT": swat_metrics, "HAI21": hai_metrics}), flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
