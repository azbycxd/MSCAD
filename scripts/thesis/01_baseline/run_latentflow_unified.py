from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "thesis_baselines" / "latentflow_2026"
OFFICIAL = ROOT / "external" / "baselines" / "LatentFlow"
PROTOCOL = OUT / "protocol.json"
PROTOCOL_DOC = ROOT / "docs" / "thesis" / "baseline_b4_latentflow_2026_protocol.md"
BASELINE_DOC = ROOT / "docs" / "thesis" / "baseline_protocol.md"
SOURCE_COMMIT = "3695f54317fcc1a3fdbfa0a7037d1cdf4369db27"

WINDOW = 128
PATCH = 64
PATCH_STRIDE = 8
D_MODEL = 128
TOP_K = 15
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 256
SCORE_BATCH = 256
MAX_EPOCHS = 30
PATIENCE = 5
MIN_DELTA = 0.0
LAMBDA_SPARSITY = 1e-3
SEEDS = (42, 43, 44)
Q = 0.99
ORDER = tuple((dataset, seed) for dataset in ("SWaT", "HAI21") for seed in SEEDS)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module("latentflow_b1_shared", ROOT / "scripts" / "thesis" / "01_baseline" / "run_lstm_ae_unified.py")
sys.dont_write_bytecode = True
sys.path.insert(0, str(OFFICIAL))
from models.LatentFlow import Model as OfficialLatentFlow  # noqa: E402


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
        + "\n",
        encoding="utf-8",
    )


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(OFFICIAL), *args], text=True).strip()


def frozen_protocol(seed: int, channels: int) -> dict[str, Any]:
    return {
        "source_commit": SOURCE_COMMIT,
        "window_size": WINDOW,
        "patch_size": PATCH,
        "patch_stride": PATCH_STRIDE,
        "d_model": D_MODEL,
        "k": TOP_K,
        "dropout": DROPOUT,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "lambda_sparsity": LAMBDA_SPARSITY,
        "lr_schedule": "type4",
        "seed": seed,
        "channels": channels,
        "normalization": "Train-only StandardScaler",
        "selection": "minimum Normal Validation reconstruction-plus-sparsity loss",
        "score": "endpoint mean-channel squared reconstruction residual",
        "threshold": "Normal Validation endpoint score q99",
    }


def verify_protocol() -> dict[str, Any]:
    if git_text("rev-parse", "HEAD") != SOURCE_COMMIT:
        raise RuntimeError("official LatentFlow commit mismatch")
    if git_text("status", "--short"):
        raise RuntimeError("official LatentFlow repository is dirty")
    manifest = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cfg = manifest["frozen_model_configuration"]
    expected = {
        "window_size": WINDOW,
        "patch_size": PATCH,
        "patch_stride": PATCH_STRIDE,
        "d_model": D_MODEL,
        "top_k_cross_channel_neighbors": TOP_K,
        "dropout": DROPOUT,
        "lambda_sparsity": LAMBDA_SPARSITY,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "seeds": list(SEEDS),
    }
    if manifest.get("status") != "LATENTFLOW_2026_PROTOCOL_FROZEN":
        raise RuntimeError("P0 protocol is not frozen")
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise RuntimeError(f"frozen protocol mismatch: {key}")
    if manifest["evaluation"]["primary_threshold"] != "Normal Validation endpoint score q99":
        raise RuntimeError("threshold protocol mismatch")
    combined_docs = PROTOCOL_DOC.read_text(encoding="utf-8") + BASELINE_DOC.read_text(encoding="utf-8")
    for token in (SOURCE_COMMIT, "window_size | 128", "patience | 5", "Normal Validation"):
        if token.lower() not in combined_docs.lower():
            raise RuntimeError(f"protocol document missing token: {token}")
    return {"status": "PASS", "source_commit": SOURCE_COMMIT, "official_repo_clean": True}


def model_config(channels: int) -> SimpleNamespace:
    return SimpleNamespace(
        enc_in=channels,
        window_size=WINDOW,
        patch_size=PATCH,
        patch_stride=PATCH_STRIDE,
        d_model=D_MODEL,
        k=TOP_K,
        dropout=DROPOUT,
    )


def make_model(channels: int) -> torch.nn.Module:
    return OfficialLatentFlow(model_config(channels)).float()


@dataclass
class WindowStore:
    values: np.ndarray
    starts: np.ndarray
    width: int = WINDOW

    @classmethod
    def from_segments(cls, segments: list[np.ndarray]) -> "WindowStore":
        if not segments or any(x.ndim != 2 or len(x) < WINDOW for x in segments):
            raise ValueError("all segments must be 2D and at least one window long")
        values = np.concatenate([np.asarray(x, dtype=np.float32) for x in segments], axis=0)
        starts: list[np.ndarray] = []
        offset = 0
        for segment in segments:
            starts.append(offset + np.arange(0, len(segment) - WINDOW + 1, dtype=np.int64))
            offset += len(segment)
        return cls(values=values, starts=np.concatenate(starts))

    def batch(self, ids: np.ndarray) -> np.ndarray:
        indices = self.starts[ids, None] + np.arange(self.width, dtype=np.int64)[None, :]
        result = self.values[indices]
        if not np.isfinite(result).all():
            raise RuntimeError("NaN/Inf in LatentFlow window batch")
        return result

    def first_windows(self, count: int) -> np.ndarray:
        return self.batch(np.arange(min(count, len(self.starts)), dtype=np.int64))


def normal_data(dataset: str) -> dict[str, Any]:
    """Load only frozen Normal Train/Validation. Never reads Attack/Test."""
    if dataset == "SWaT":
        config = BASE.load_config(str(BASE.SWAT_CONFIG))
        BASE.validate_config_contract(config)
        normal, cache_manifest = BASE.load_normal_cache(BASE.resolve_repo_path(ROOT, config["paths"]["cache_dir"]), config)
        scaler = BASE.StandardScaler().fit(np.asarray(normal[:396000], dtype=np.float64))
        return {
            "train": [scaler.transform(np.asarray(normal[:396000])).astype(np.float32)],
            "validation": [scaler.transform(np.asarray(normal[396000:495000])).astype(np.float32)],
            "feature_names": list(cache_manifest["column_order"]),
        }
    if dataset == "HAI21":
        import pandas as pd

        features = pd.read_csv(BASE.HAI_AUDIT / "variable_schema.csv")["column_name"].astype(str).tolist()
        if len(features) != 79 or "attack" in features:
            raise RuntimeError("HAI feature schema changed")
        raw, boundaries = [], {}
        for name in BASE.HAI_TRAIN_FILES:
            values = pd.read_csv(BASE.HAI_ROOT / name, compression="gzip", usecols=features, low_memory=False).to_numpy(np.float64)
            raw.append(values)
            boundaries[name] = math.floor(0.8 * len(values))
        scaler = BASE.StandardScaler()
        scaler.fit_many([values[: boundaries[name]] for name, values in zip(BASE.HAI_TRAIN_FILES, raw)])
        return {
            "train": [scaler.transform(values[: boundaries[name]]).astype(np.float32) for name, values in zip(BASE.HAI_TRAIN_FILES, raw)],
            "validation": [scaler.transform(values[boundaries[name] :]).astype(np.float32) for name, values in zip(BASE.HAI_TRAIN_FILES, raw)],
            "feature_names": features,
        }
    raise ValueError(dataset)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_parts(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon, truth, adjs = model(x, inference_inertia=False)
    reconstruction = F.mse_loss(recon, truth)
    sparsity = torch.mean(torch.abs(adjs))
    return reconstruction + LAMBDA_SPARSITY * sparsity, reconstruction, sparsity


def aggregate_reconstruction(recon_seq: torch.Tensor) -> torch.Tensor:
    batch, steps, channels, patch = recon_seq.shape
    canvas = torch.zeros((batch, WINDOW, channels), device=recon_seq.device, dtype=recon_seq.dtype)
    counts = torch.zeros_like(canvas)
    for step in range(steps):
        start = step * PATCH_STRIDE
        end = start + patch
        patch_prediction = recon_seq[:, step].permute(0, 2, 1)
        canvas[:, start:end] += patch_prediction
        counts[:, start:end] += 1.0
    return canvas / (counts + 1e-8)


def resource_preflight(dataset: str, train_segments: list[np.ndarray], channels: int, device: torch.device) -> dict[str, Any]:
    set_seed(42)
    store = WindowStore.from_segments(train_segments)
    if len(store.starts) < BATCH:
        raise RuntimeError("insufficient Normal Train windows for frozen preflight batch")
    model = make_model(channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    x = torch.from_numpy(store.batch(np.arange(BATCH, dtype=np.int64))).to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, reconstruction, sparsity = loss_parts(model, x)
    total.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak = float(torch.cuda.max_memory_allocated(device) / 1024**2)
    else:
        peak = None
    steps = (WINDOW - PATCH) // PATCH_STRIDE + 1
    estimated_adj = BATCH * steps * channels * channels * 4 / 1024**2
    result = {
        "dataset": dataset,
        "status": "RESOURCE_PREFLIGHT_PASS",
        "channels": channels,
        "batch_size": BATCH,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "peak_allocation_mib": peak,
        "estimated_adjacency_sequence_mib": float(estimated_adj),
        "loss": float(total.detach()),
        "reconstruction_loss": float(reconstruction.detach()),
        "sparsity_loss": float(sparsity.detach()),
        "validation_data_read": False,
        "test_data_read": False,
    }
    del model, optimizer, x, total, reconstruction, sparsity, store
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def rng_state() -> dict[str, Any]:
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("torch_cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])


def lr_after_epoch(epoch: int) -> float:
    return LR if epoch < 10 else LR * (0.5 ** (epoch // 10))


def save_latest(
    path: Path,
    dataset: str,
    seed: int,
    channels: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_state: dict[str, torch.Tensor],
    best_epoch: int,
    best_loss: float,
    stale: int,
    history: list[dict[str, Any]],
    order: np.ndarray,
    training_seconds: float,
    completed: bool,
) -> None:
    payload = {
        "dataset": dataset,
        "seed": seed,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_model_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "early_stop_counter": stale,
        "history": history,
        "sample_order": order,
        "training_seconds": training_seconds,
        "protocol": frozen_protocol(seed, channels),
        "completed": completed,
        **rng_state(),
    }
    atomic_torch_save(payload, path)


def validation_loss(model: torch.nn.Module, store: WindowStore, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    totals: list[float] = []
    reconstructions: list[float] = []
    sparsities: list[float] = []
    with torch.inference_mode():
        for begin in range(0, len(store.starts), SCORE_BATCH):
            ids = np.arange(begin, min(begin + SCORE_BATCH, len(store.starts)), dtype=np.int64)
            x = torch.from_numpy(store.batch(ids)).to(device)
            total, reconstruction, sparsity = loss_parts(model, x)
            totals.append(float(total))
            reconstructions.append(float(reconstruction))
            sparsities.append(float(sparsity))
    return float(np.mean(totals)), float(np.mean(reconstructions)), float(np.mean(sparsities))


def train_seed(
    dataset: str,
    seed: int,
    data: dict[str, Any],
    device: torch.device,
    overall_started: float,
    wall_seconds: float,
) -> tuple[torch.nn.Module | None, dict[str, Any]]:
    channels = len(data["feature_names"])
    seed_dir = OUT / ("swat" if dataset == "SWaT" else "hai") / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    latest = seed_dir / "latest_training_state.pt"
    final_manifest = seed_dir / "manifest.json"
    if final_manifest.exists():
        existing = json.loads(final_manifest.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            print(f"[{dataset} seed={seed}] already completed; skipping", flush=True)
            return None, existing

    train_store = WindowStore.from_segments(data["train"])
    validation_store = WindowStore.from_segments(data["validation"])
    set_seed(seed)
    model = make_model(channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch, best_loss, stale = 0, math.inf, 0
    history: list[dict[str, Any]] = []
    prior_seconds, start_epoch = 0.0, 1
    order = np.arange(len(train_store.starts), dtype=np.int64)
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state.get("dataset") != dataset or int(state.get("seed", -1)) != seed:
            raise RuntimeError(f"checkpoint identity mismatch: {latest}")
        if state["protocol"] != frozen_protocol(seed, channels):
            raise RuntimeError(f"checkpoint protocol mismatch: {latest}")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        best_state = state["best_model_state_dict"]
        best_epoch = int(state["best_epoch"])
        best_loss = float(state["best_validation_loss"])
        stale = int(state["early_stop_counter"])
        history = list(state["history"])
        order = np.asarray(state["sample_order"], dtype=np.int64)
        prior_seconds = float(state.get("training_seconds", 0.0))
        start_epoch = int(state["epoch"]) + 1
        restore_rng(state)
        if state.get("completed"):
            model.load_state_dict(best_state)
            best_path = seed_dir / "best_state_dict.pt"
            if not best_path.exists():
                atomic_torch_save(best_state, best_path)
            pending = {
                "status": "TRAINING_COMPLETED_PENDING_EVALUATION",
                "dataset": dataset,
                "seed": seed,
                "total_epochs": int(state["epoch"]),
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "stop_reason": "early_stopping" if stale >= PATIENCE else "max_epochs",
                "training_seconds": prior_seconds,
                "checkpoint": str(best_path),
                "latest_training_state": str(latest),
                "protocol": frozen_protocol(seed, channels),
            }
            dump_json(final_manifest, pending)
            print(f"[{dataset} seed={seed}] recovered completed training state; evaluation pending", flush=True)
            return model, pending
        print(f"[{dataset} seed={seed}] exact resume from epoch {start_epoch}", flush=True)

    session_started = time.perf_counter()
    stop_reason = None
    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()
        np.random.shuffle(order)
        train_losses: list[float] = []
        train_reconstruction: list[float] = []
        train_sparsity: list[float] = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for begin in range(0, len(order), BATCH):
            ids = order[begin : begin + BATCH]
            x = torch.from_numpy(train_store.batch(ids)).to(device)
            optimizer.zero_grad(set_to_none=True)
            total, reconstruction, sparsity = loss_parts(model, x)
            total.backward()
            optimizer.step()
            train_losses.append(float(total.detach()))
            train_reconstruction.append(float(reconstruction.detach()))
            train_sparsity.append(float(sparsity.detach()))
        val_loss, val_reconstruction, val_sparsity = validation_loss(model, validation_store, device)
        improved = val_loss < (best_loss - MIN_DELTA)
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            best_state = cpu_state(model)
            stale = 0
        else:
            stale += 1
        next_lr = lr_after_epoch(epoch)
        for group in optimizer.param_groups:
            group["lr"] = next_lr
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak = float(torch.cuda.max_memory_allocated(device) / 1024**2)
        else:
            peak = None
        final_epoch = epoch
        elapsed_training = prior_seconds + (time.perf_counter() - session_started)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_reconstruction_loss": float(np.mean(train_reconstruction)),
            "train_sparsity_loss": float(np.mean(train_sparsity)),
            "validation_loss": val_loss,
            "validation_reconstruction_loss": val_reconstruction,
            "validation_sparsity_loss": val_sparsity,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "early_stop_counter": stale,
            "learning_rate_for_next_epoch": next_lr,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "peak_training_allocation_mib": peak,
        }
        history.append(row)
        if best_state is None:
            raise RuntimeError("best model state was not initialized")
        natural_stop = stale >= PATIENCE or epoch >= MAX_EPOCHS
        save_latest(
            latest,
            dataset,
            seed,
            channels,
            epoch,
            model,
            optimizer,
            best_state,
            best_epoch,
            best_loss,
            stale,
            history,
            order,
            elapsed_training,
            natural_stop,
        )
        dump_json(seed_dir / "training_history.json", history)
        print(
            f"[{dataset} seed={seed}] epoch={epoch} train={row['train_loss']:.8f} "
            f"val={val_loss:.8f} best={best_loss:.8f}@{best_epoch} stale={stale}/{PATIENCE} "
            f"seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )
        if natural_stop:
            stop_reason = "early_stopping" if stale >= PATIENCE else "max_epochs"
            break
        if time.perf_counter() - overall_started >= wall_seconds:
            return None, {
                "status": "HARD_STOP_INCOMPLETE",
                "dataset": dataset,
                "seed": seed,
                "last_completed_epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "early_stop_counter": stale,
                "checkpoint": str(latest),
                "training_seconds": elapsed_training,
            }

    if stop_reason is None or best_state is None:
        raise RuntimeError("training ended without a natural stop")
    model.load_state_dict(best_state)
    best_path = seed_dir / "best_state_dict.pt"
    atomic_torch_save(best_state, best_path)
    training_seconds = prior_seconds + (time.perf_counter() - session_started)
    meta = {
        "status": "TRAINING_COMPLETED_PENDING_EVALUATION",
        "dataset": dataset,
        "seed": seed,
        "total_epochs": final_epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "stop_reason": stop_reason,
        "training_seconds": training_seconds,
        "checkpoint": str(best_path),
        "latest_training_state": str(latest),
        "protocol": frozen_protocol(seed, channels),
    }
    dump_json(final_manifest, meta)
    return model, meta


def score_model(model: torch.nn.Module, store: WindowStore, device: torch.device) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for begin in range(0, len(store.starts), SCORE_BATCH):
            ids = np.arange(begin, min(begin + SCORE_BATCH, len(store.starts)), dtype=np.int64)
            x = torch.from_numpy(store.batch(ids)).to(device)
            recon_seq, _, _ = model(x, inference_inertia=True)
            reconstructed = aggregate_reconstruction(recon_seq)
            score = torch.mean((x[:, -1] - reconstructed[:, -1]) ** 2, dim=-1)
            scores.append(score.detach().cpu().numpy().astype(np.float64, copy=False))
    result = np.concatenate(scores)
    if not np.isfinite(result).all():
        raise RuntimeError("NaN/Inf in LatentFlow scores")
    return result


def point_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = scores > threshold
    tp = int(np.sum(predictions & (labels == 1)))
    fp = int(np.sum(predictions & (labels == 0)))
    fn = int(np.sum((~predictions) & (labels == 1)))
    tn = int(np.sum((~predictions) & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPR": float(average_precision_score(labels, scores)),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "FPR": float(fp / (fp + tn)) if fp + tn else 0.0,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "threshold": threshold,
    }


def load_evaluation_data(dataset: str) -> dict[str, Any]:
    full = BASE.swat_data() if dataset == "SWaT" else BASE.hai_data()
    shift = WINDOW - 1 - BASE.WINDOW
    if shift != 63:
        raise RuntimeError("common endpoint alignment contract changed")
    old_lengths = [len(segment) - BASE.WINDOW for segment in full["test"]]

    def align_parts(values: np.ndarray) -> np.ndarray:
        parts = []
        offset = 0
        for length in old_lengths:
            parts.append(values[offset + shift : offset + length])
            offset += length
        if offset != len(values):
            raise RuntimeError("shared endpoint metadata length changed")
        return np.concatenate(parts)

    full["test_labels"] = align_parts(full["test_labels"])
    if dataset == "SWaT":
        full["test_timestamps"] = align_parts(full["test_timestamps"])
        full["test_raw_indices"] = align_parts(full["test_raw_indices"])
    else:
        full["test_files"] = align_parts(full["test_files"])
        full["test_raw_indices"] = align_parts(full["test_raw_indices"])
    return full


def efficiency(model: torch.nn.Module, windows: np.ndarray, best_path: Path, device: torch.device) -> dict[str, Any]:
    if len(windows) < 1000:
        raise RuntimeError("efficiency protocol needs 1000 legal windows")
    samples = [torch.from_numpy(windows[i : i + 1]).to(device) for i in range(1000)]
    model.eval()
    with torch.inference_mode():
        for index in range(100):
            recon, _, _ = model(samples[index], inference_inertia=True)
            _ = aggregate_reconstruction(recon)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        latencies = []
        for sample in samples:
            torch.cuda.synchronize()
            started = time.perf_counter()
            recon, _, _ = model(sample, inference_inertia=True)
            reconstructed = aggregate_reconstruction(recon)
            _ = torch.mean((sample[:, -1] - reconstructed[:, -1]) ** 2, dim=-1)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)
        peak = float(torch.cuda.max_memory_allocated(device) / 1024**2)
    return {
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "state_dict_bytes": int(best_path.stat().st_size),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_median": float(np.median(latencies)),
        "latency_ms_p95": float(np.quantile(latencies, 0.95)),
        "peak_inference_memory_mib": peak,
        "latency_warmup": 100,
        "latency_timed": 1000,
        "latency_batch_size": 1,
    }


def evaluate_seed(dataset: str, seed: int, model: torch.nn.Module, train_meta: dict[str, Any], device: torch.device) -> dict[str, Any]:
    seed_dir = OUT / ("swat" if dataset == "SWaT" else "hai") / f"seed{seed}"
    data = load_evaluation_data(dataset)
    validation_store = WindowStore.from_segments(data["validation"])
    test_store = WindowStore.from_segments(data["test"])
    validation_scores = score_model(model, validation_store, device)
    threshold = float(np.quantile(validation_scores, Q))
    test_scores = score_model(model, test_store, device)
    if len(test_scores) != len(data["test_labels"]):
        raise RuntimeError(f"{dataset} LatentFlow score/label alignment mismatch")
    metrics = point_metrics(data["test_labels"], test_scores, threshold)
    metrics.update(
        {
            "dataset": dataset,
            "seed": seed,
            "validation_score_q99": threshold,
            "validation_endpoint_count": int(len(validation_scores)),
            "test_endpoint_count": int(len(test_scores)),
            "threshold_source": "Normal Validation endpoint scores only",
            "endpoint_local_index_minimum": WINDOW - 1,
            "point_adjustment": False,
            "spot_pot": False,
            "test_best_threshold": False,
        }
    )
    predictions = test_scores > threshold
    if dataset == "SWaT":
        event_rows, event_summary = BASE.swat_event_metrics(predictions, data["test_timestamps"])
    else:
        event_rows, event_summary = BASE.hai_event_metrics(predictions, data["test_files"], data["test_raw_indices"])
    metrics["events"] = event_summary
    np.save(seed_dir / "test_scores.npy", test_scores.astype(np.float32))
    np.save(seed_dir / "validation_scores.npy", validation_scores.astype(np.float32))
    dump_json(seed_dir / "metrics.json", metrics)
    dump_json(seed_dir / "event_metrics.json", event_rows)
    eff = None
    if seed == 42:
        eff = efficiency(model, test_store.first_windows(1000), seed_dir / "best_state_dict.pt", device)
        eff["dataset"] = dataset
        eff["seed"] = seed
        dump_json(seed_dir / "efficiency.json", eff)
    manifest = {
        **train_meta,
        "status": "COMPLETED",
        "metrics": str(seed_dir / "metrics.json"),
        "event_metrics": str(seed_dir / "event_metrics.json"),
        "test_scores": str(seed_dir / "test_scores.npy"),
        "validation_scores": str(seed_dir / "validation_scores.npy"),
        "test_labels_read_after_training_completed": True,
        "efficiency": eff,
    }
    dump_json(seed_dir / "manifest.json", manifest)
    del data, validation_store, test_store, validation_scores, test_scores
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return manifest


def root_manifest(status: str, verification: dict[str, Any], preflight: dict[str, Any], current: dict[str, Any] | None, started: float) -> None:
    completed = []
    for dataset, seed in ORDER:
        path = OUT / ("swat" if dataset == "SWaT" else "hai") / f"seed{seed}" / "manifest.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("status") == "COMPLETED":
                completed.append({"dataset": dataset, "seed": seed, "manifest": str(path)})
    dump_json(
        OUT / "manifest.json",
        {
            "phase": "Thesis Baseline-B4",
            "status": status,
            "protocol_verification": verification,
            "resource_preflight": preflight,
            "seed_order": [{"dataset": d, "seed": s} for d, s in ORDER],
            "completed_seeds": completed,
            "completed_seed_count": len(completed),
            "current": current,
            "elapsed_seconds": time.perf_counter() - started,
            "no_hp_search": True,
            "no_test_tuning": True,
            "no_spot_pot": True,
            "no_point_adjustment": True,
            "no_other_method_rerun": True,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-wall-minutes", type=float, default=45.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    wall_seconds = args.max_wall_minutes * 60.0
    OUT.mkdir(parents=True, exist_ok=True)
    verification = verify_protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("LatentFlow frozen training protocol requires CUDA")
    device = torch.device("cuda")
    print(f"device={device} gpu={torch.cuda.get_device_name(device)}", flush=True)

    normal_bundles: dict[str, dict[str, Any]] = {}
    preflight_rows: dict[str, Any] = {}
    try:
        for dataset in ("SWaT", "HAI21"):
            data = normal_data(dataset)
            normal_bundles[dataset] = data
            preflight_rows[dataset] = resource_preflight(dataset, data["train"], len(data["feature_names"]), device)
            print(f"[{dataset}] RESOURCE_PREFLIGHT_PASS peak={preflight_rows[dataset]['peak_allocation_mib']:.1f} MiB", flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        preflight_rows["error"] = str(exc)
        dump_json(OUT / "resource_preflight.json", preflight_rows)
        root_manifest("LATENTFLOW_FROZEN_BATCH_OOM", verification, preflight_rows, None, started)
        return 2
    dump_json(OUT / "resource_preflight.json", preflight_rows)

    current: dict[str, Any] | None = None
    for dataset, seed in ORDER:
        if time.perf_counter() - started >= wall_seconds:
            current = {"next_dataset": dataset, "next_seed": seed, "reason": "hard stop reached before starting next seed"}
            root_manifest("HARD_STOP_INCOMPLETE", verification, preflight_rows, current, started)
            return 3
        data = normal_bundles.get(dataset)
        if data is None:
            data = normal_data(dataset)
            normal_bundles[dataset] = data
        model, train_meta = train_seed(dataset, seed, data, device, started, wall_seconds)
        if train_meta.get("status") == "HARD_STOP_INCOMPLETE":
            root_manifest("HARD_STOP_INCOMPLETE", verification, preflight_rows, train_meta, started)
            return 3
        if model is None:
            continue
        manifest = evaluate_seed(dataset, seed, model, train_meta, device)
        print(
            f"[{dataset} seed={seed}] COMPLETED AUROC={json.loads(Path(manifest['metrics']).read_text())['AUROC']:.6f}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    root_manifest("LATENTFLOW_2026_BASELINE_COMPLETED", verification, preflight_rows, None, started)
    print("LATENTFLOW_2026_BASELINE_COMPLETED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
