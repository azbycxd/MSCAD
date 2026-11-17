"""Baseline-B3: frozen GDN evaluation on Official SWaT 2015 and HAI 21.03.

The graph layer below is a dependency-free transcription of d-ailin/GDN
commit 9853899da860682669a134e4af315d036aab4eca.  It evaluates the same
single-head attention algebra without requiring torch_geometric's scatter API.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[3]
BASE_PATH = ROOT / "scripts" / "thesis" / "01_baseline" / "run_lstm_ae_unified.py"
OUT = ROOT / "outputs" / "thesis_baselines" / "gdn"
DOC = ROOT / "docs" / "thesis" / "baseline_b3_gdn.md"
TABLE = ROOT / "outputs" / "thesis_baselines" / "baseline_table_template.csv"
OURS = ROOT / "outputs" / "thesis_innovation3b_tail_fusion_final" / "cross_dataset_metrics.csv"
PROTOCOL_DOC = ROOT / "docs" / "thesis" / "baseline_protocol.md"
PROTOCOL_MANIFEST = ROOT / "outputs" / "thesis_baselines" / "baseline_manifest.json"

SOURCE_COMMIT = "9853899da860682669a134e4af315d036aab4eca"
HISTORY = 64
DIM = 64
TOP_K = 5
OUT_LAYER_NUM = 1
OUT_INTER_DIM = 128
BATCH = 32
SCORE_BATCH = 2048
MAX_EPOCHS = 30
PATIENCE = 10
MIN_DELTA = 0.0
LR = 1e-3
WEIGHT_DECAY = 0.0
SEED = 1
EPSILON = 1e-2
Q = 0.99


def load_base_module():
    spec = importlib.util.spec_from_file_location("baseline_b1_shared", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen B1 data helpers")
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


@dataclass
class ForecastStore:
    values: np.ndarray
    starts: np.ndarray
    history: int = HISTORY

    @classmethod
    def from_segments(cls, segments: list[np.ndarray]) -> "ForecastStore":
        if not segments or any(x.ndim != 2 or len(x) <= HISTORY for x in segments):
            raise ValueError("all forecast segments must be 2D and longer than history")
        values = np.concatenate([np.asarray(x, dtype=np.float32) for x in segments], axis=0)
        starts, offset = [], 0
        for segment in segments:
            starts.append(offset + np.arange(0, len(segment) - HISTORY, dtype=np.int64))
            offset += len(segment)
        return cls(values=values, starts=np.concatenate(starts))

    def batch(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        starts = self.starts[np.asarray(ids, dtype=np.int64)]
        indices = starts[:, None] + np.arange(self.history, dtype=np.int64)[None, :]
        x = self.values[indices].transpose(0, 2, 1)
        y = self.values[starts + self.history]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise RuntimeError("NaN/Inf in forecast batch")
        return x, y


class OfficialGraphAttention(nn.Module):
    """Official GraphLayer single-head equations, batched as [B, target, source]."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.att_i = nn.Parameter(torch.empty(out_channels))
        self.att_j = nn.Parameter(torch.empty(out_channels))
        self.att_em_i = nn.Parameter(torch.empty(out_channels))
        self.att_em_j = nn.Parameter(torch.empty(out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_i.view(1, -1))
        nn.init.xavier_uniform_(self.att_j.view(1, -1))
        nn.init.zeros_(self.att_em_i)
        nn.init.zeros_(self.att_em_j)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        projected = self.lin(x)
        batch, nodes, dim = projected.shape
        k = neighbors.shape[1]
        source = projected[:, neighbors.reshape(-1), :].reshape(batch, nodes, k, dim)
        target = projected[:, :, None, :].expand(-1, -1, k, -1)
        source_embedding = embedding[neighbors]
        target_embedding = embedding[:, None, :].expand(-1, k, -1)
        logits = (
            (target * self.att_i).sum(-1)
            + (source * self.att_j).sum(-1)
            + (target_embedding * self.att_em_i).sum(-1)[None, :, :]
            + (source_embedding * self.att_em_j).sum(-1)[None, :, :]
        )
        alpha = torch.softmax(F.leaky_relu(logits, negative_slope=0.2), dim=-1)
        return (source * alpha[..., None]).sum(dim=2) + self.bias


class FrozenGDN(nn.Module):
    """Official GDN core with the B0-frozen history/dim/top-k/output settings."""

    def __init__(self, node_num: int) -> None:
        super().__init__()
        self.node_num = node_num
        self.embedding = nn.Embedding(node_num, DIM)
        self.graph_attention = OfficialGraphAttention(HISTORY, DIM)
        self.graph_bn = nn.BatchNorm1d(DIM)
        self.output_bn = nn.BatchNorm1d(DIM)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(DIM, 1)
        self.learned_graph: torch.Tensor | None = None
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def neighbor_graph(self) -> torch.Tensor:
        weights = self.embedding.weight.detach()
        cosine = weights @ weights.T
        cosine = cosine / (weights.norm(dim=-1)[:, None] * weights.norm(dim=-1)[None, :]).clamp_min(1e-12)
        return torch.topk(cosine, TOP_K, dim=-1).indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.node_num, HISTORY):
            raise ValueError(f"expected [B,{self.node_num},{HISTORY}], got {tuple(x.shape)}")
        embedding = self.embedding(torch.arange(self.node_num, device=x.device))
        neighbors = self.neighbor_graph()
        self.learned_graph = neighbors
        hidden = self.graph_attention(x.detach(), embedding, neighbors)
        hidden = F.relu(self.graph_bn(hidden.reshape(-1, DIM))).reshape(x.shape[0], self.node_num, DIM)
        hidden = hidden * embedding[None, :, :]
        hidden = hidden.permute(0, 2, 1)
        hidden = F.relu(self.output_bn(hidden)).permute(0, 2, 1)
        return self.output(self.dropout(hidden)).squeeze(-1)


def frozen_protocol() -> dict[str, Any]:
    doc = PROTOCOL_DOC.read_text(encoding="utf-8")
    manifest = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))
    gdn = manifest["baselines"]["GDN"]
    expected = {
        "source_commit": SOURCE_COMMIT,
        "history": HISTORY,
        "dim": DIM,
        "top_k_graph_neighbors": TOP_K,
        "out_layer_num": OUT_LAYER_NUM,
        "out_layer_inter_dim": OUT_INTER_DIM,
        "batch_size": BATCH,
        "max_epochs": MAX_EPOCHS,
        "optimizer": "Adam",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
    }
    for key, value in expected.items():
        if gdn.get(key) != value:
            raise RuntimeError(f"frozen GDN manifest mismatch for {key}: {gdn.get(key)!r} != {value!r}")
    common = manifest["common"]
    early = common["early_stopping"]
    if manifest["seed"] != SEED or early["patience"] != PATIENCE or early["min_delta"] != MIN_DELTA:
        raise RuntimeError("frozen seed/early-stopping protocol mismatch")
    for token in (SOURCE_COMMIT, "history length: 64", "graph neighbors: top-k 5", "four-point trailing mean"):
        if token.lower() not in doc.lower():
            raise RuntimeError(f"protocol document is missing {token}")
    return {"manifest": str(PROTOCOL_MANIFEST), "document": str(PROTOCOL_DOC), **expected, "patience": PATIENCE, "min_delta": MIN_DELTA, "seed": SEED}


def normal_data(dataset: str) -> dict[str, Any]:
    """Load only Normal Train/Validation; attack/Test files are untouched here."""
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
        features = pd.read_csv(BASE.HAI_AUDIT / "variable_schema.csv")["column_name"].astype(str).tolist()
        if len(features) != 79 or "attack" in features:
            raise RuntimeError("HAI feature schema changed")
        raw, boundaries = [], {}
        for name in BASE.HAI_TRAIN_FILES:
            values = pd.read_csv(BASE.HAI_ROOT / name, compression="gzip", usecols=features, low_memory=False).to_numpy(np.float64)
            raw.append(values); boundaries[name] = math.floor(0.8 * len(values))
        scaler = BASE.StandardScaler()
        scaler.fit_many([values[:boundaries[name]] for name, values in zip(BASE.HAI_TRAIN_FILES, raw)])
        return {
            "train": [scaler.transform(values[:boundaries[name]]).astype(np.float32) for name, values in zip(BASE.HAI_TRAIN_FILES, raw)],
            "validation": [scaler.transform(values[boundaries[name]:]).astype(np.float32) for name, values in zip(BASE.HAI_TRAIN_FILES, raw)],
            "feature_names": features,
        }
    raise ValueError(dataset)


def resource_preflight(dataset: str, train_segments: list[np.ndarray], channels: int, device: torch.device) -> dict[str, Any]:
    set_seed()
    store = ForecastStore.from_segments(train_segments)
    model = FrozenGDN(channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ids = np.arange(BATCH, dtype=np.int64)
    x_np, y_np = store.batch(ids)
    x, y = torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak = float(torch.cuda.max_memory_allocated(device) / 1024**2)
    else:
        peak = None
    result = {"dataset": dataset, "status": "PASS", "channels": channels, "batch": BATCH, "history": HISTORY, "loss": float(loss.detach()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "peak_allocation_mib": peak, "validation_data_read": False, "test_data_read": False}
    del model, optimizer, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_latest(dataset: str, epoch: int, model: FrozenGDN, optimizer, best_state, best_epoch: int, best_loss: float, stale: int, history, order, training_seconds: float, completed: bool) -> None:
    torch.save({
        "dataset": dataset, "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "best_model_state_dict": best_state, "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "early_stop_counter": stale, "history": history, "sample_order": order, "training_seconds": training_seconds,
        "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(), "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "protocol": frozen_protocol(), "completed": completed,
    }, OUT / "latest_training_state.pt")


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state["torch_cuda_rng_state_all"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])


def validation_loss(model: FrozenGDN, store: ForecastStore, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for begin in range(0, len(store.starts), SCORE_BATCH):
            ids = np.arange(begin, min(begin + SCORE_BATCH, len(store.starts)), dtype=np.int64)
            x_np, y_np = store.batch(ids)
            x, y = torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)
            loss = F.mse_loss(model(x), y)
            total += float(loss) * len(ids)
            count += len(ids)
    return total / count


def train_dataset(dataset: str, train_segments: list[np.ndarray], validation_segments: list[np.ndarray], channels: int, device: torch.device, overall_start: float, wall_seconds: float):
    train_store, val_store = ForecastStore.from_segments(train_segments), ForecastStore.from_segments(validation_segments)
    checkpoint = OUT / "latest_training_state.pt"
    set_seed()
    model = FrozenGDN(channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state, best_epoch, best_loss, stale, history, prior_seconds, start_epoch = None, 0, math.inf, 0, [], 0.0, 1
    order = np.arange(len(train_store.starts), dtype=np.int64)
    if checkpoint.exists():
        # RNG state must remain a CPU ByteTensor; model/optimizer loaders move
        # their own tensors to the already-created CUDA parameters.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("dataset") == dataset and not state.get("completed", False):
            if state["protocol"] != frozen_protocol():
                raise RuntimeError("resume checkpoint protocol mismatch")
            model.load_state_dict(state["model_state_dict"]); optimizer.load_state_dict(state["optimizer_state_dict"])
            best_state, best_epoch, best_loss = state["best_model_state_dict"], int(state["best_epoch"]), float(state["best_validation_loss"])
            stale, history, order = int(state["early_stop_counter"]), list(state["history"]), np.asarray(state["sample_order"], dtype=np.int64)
            prior_seconds, start_epoch = float(state.get("training_seconds", 0.0)), int(state["epoch"]) + 1
            restore_rng(state)
            print(f"[{dataset}] exact resume from epoch {start_epoch}", flush=True)
    session_start = time.perf_counter()
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train(); np.random.shuffle(order); total, count = 0.0, 0
        for begin in range(0, len(order), BATCH):
            ids = order[begin:begin + BATCH]
            x_np, y_np = train_store.batch(ids)
            x, y = torch.from_numpy(x_np).to(device, non_blocking=True), torch.from_numpy(y_np).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True); prediction = model(x); loss = F.mse_loss(prediction, y); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(ids); count += len(ids)
        val = validation_loss(model, val_store, device)
        improved = val < best_loss - MIN_DELTA
        if improved:
            best_loss, best_epoch, best_state, stale = val, epoch, cpu_state(model), 0
        else:
            stale += 1
        train_loss = total / count
        history.append({"dataset": dataset, "epoch": epoch, "train_loss": train_loss, "validation_loss": val, "improved": improved})
        training_seconds = prior_seconds + time.perf_counter() - session_start
        natural_done = stale >= PATIENCE or epoch == MAX_EPOCHS
        save_latest(dataset, epoch, model, optimizer, best_state, best_epoch, best_loss, stale, history, order, training_seconds, natural_done)
        print(f"[{dataset}] epoch={epoch:02d} train={train_loss:.8f} val={val:.8f} best={best_loss:.8f}@{best_epoch} stale={stale}/{PATIENCE} total_wall_min={(time.perf_counter()-overall_start)/60:.2f}", flush=True)
        if natural_done:
            if best_state is None:
                raise RuntimeError("no best GDN state")
            model.load_state_dict(best_state)
            return model, history, {"training_seconds": training_seconds, "epochs_ran": epoch, "best_epoch": best_epoch, "best_validation_loss": best_loss, "stop_reason": "early_stopping" if stale >= PATIENCE else "max_epochs"}, False
        if time.perf_counter() - overall_start >= wall_seconds:
            return None, history, {"training_seconds": training_seconds, "epochs_ran": epoch, "best_epoch": best_epoch, "best_validation_loss": best_loss, "stop_reason": "hard_stop"}, True
    raise AssertionError("unreachable")


def raw_errors(model: FrozenGDN, store: ForecastStore, device: torch.device) -> np.ndarray:
    parts = []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(store.starts), SCORE_BATCH):
            ids = np.arange(begin, min(begin + SCORE_BATCH, len(store.starts)), dtype=np.int64)
            x_np, y_np = store.batch(ids)
            prediction = model(torch.from_numpy(x_np).to(device)).cpu().numpy()
            parts.append(np.abs(prediction - y_np).astype(np.float64, copy=False))
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite GDN forecast error")
    return result


def native_scores(errors: np.ndarray, median: np.ndarray, iqr: np.ndarray, segments: list[np.ndarray]) -> np.ndarray:
    standardized = (errors - median[None, :]) / (np.abs(iqr)[None, :] + EPSILON)
    smoothed = np.zeros_like(standardized)
    offset = 0
    for segment in segments:
        count = len(segment) - HISTORY
        local = standardized[offset:offset + count]
        for lag in range(4):
            if lag == 0:
                smoothed += 0  # shape assertion without changing values
            if len(local) > 3:
                # Official code leaves the first three scores at exactly zero.
                pass
        if len(local) > 3:
            csum = np.cumsum(local, axis=0)
            sums = csum[3:].copy()
            sums[1:] -= csum[:-4]
            smoothed[offset + 3:offset + count] = sums / 4.0
        offset += count
    scores = np.max(smoothed, axis=1)
    if not np.isfinite(scores).all():
        raise RuntimeError("non-finite native GDN score")
    return scores


def graph_diagnostics(model: FrozenGDN, feature_names: list[str], dataset: str) -> dict[str, Any]:
    graph = model.neighbor_graph().detach().cpu().numpy()
    selected = Counter(int(x) for x in graph.reshape(-1))
    process = lambda name: (re.search(r"(?:^|_)P([1-6])(?:_|$)", str(name), re.I).group(1) if re.search(r"(?:^|_)P([1-6])(?:_|$)", str(name), re.I) else "UNKNOWN")
    within = cross = unknown = 0
    edges = []
    for target in range(len(feature_names)):
        for source in graph[target]:
            tp, sp = process(feature_names[target]), process(feature_names[int(source)])
            if "UNKNOWN" in (tp, sp): unknown += 1
            elif tp == sp: within += 1
            else: cross += 1
            edges.append({"dataset": dataset, "target": feature_names[target], "source": feature_names[int(source)], "target_process": tp, "source_process": sp})
    return {"dataset": dataset, "edge_count": int(graph.size), "within_process_edges": within, "cross_process_edges": cross, "unknown_process_edges": unknown, "top_frequently_selected_neighbors": [{"channel": feature_names[i], "count": count} for i, count in selected.most_common(15)], "edges": edges}


def efficiency(model: FrozenGDN, store: ForecastStore, state_path: Path, device: torch.device) -> dict[str, Any]:
    torch.save(model.state_dict(), state_path)
    x_np, _ = store.batch(np.arange(1000, dtype=np.int64))
    samples = [torch.from_numpy(x_np[i:i + 1]).to(device) for i in range(1000)]
    model.eval()
    with torch.inference_mode():
        for i in range(100): model(samples[i])
        if device.type == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for sample in samples:
            if device.type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter(); model(sample)
            if device.type == "cuda": torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
        peak = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None
    return {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "state_dict_bytes": state_path.stat().st_size, "latency_ms_mean": float(np.mean(timings)), "latency_ms_median": float(np.median(timings)), "latency_ms_p95": float(np.quantile(timings, 0.95)), "peak_inference_memory_mib": peak, "latency_batch_size": 1, "latency_warmup": 100, "latency_timed": 1000}


def slug(dataset: str) -> str:
    return "swat" if dataset == "SWaT" else "hai"


def recover_completed_efficiency(dataset: str, device: torch.device) -> dict[str, Any]:
    """Re-run inference timing only; never retrain an already completed dataset."""
    key = slug(dataset)
    path = OUT / f"{key}_efficiency.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = BASE.swat_data() if dataset == "SWaT" else BASE.hai_data()
    model = FrozenGDN(len(data["feature_names"])).to(device)
    state_path = OUT / f"{key}_state_dict.pt"
    model.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    result = efficiency(model, ForecastStore.from_segments(data["test"]), state_path, device)
    result.update({"dataset": dataset, "training_seconds": None, "training_time_note": "N/A: prior completed-run wall time was not persisted; no retraining performed"})
    BASE.dump_json(path, result)
    return result


def evaluate_dataset(dataset: str, data: dict[str, Any], device: torch.device, overall_start: float, wall_seconds: float):
    model, history, training, stopped = train_dataset(dataset, data["train"], data["validation"], len(data["feature_names"]), device, overall_start, wall_seconds)
    if stopped:
        return {"stopped": True, "history": history, "training": training}
    # Test is loaded only after Normal-Validation model selection has finished.
    full_data = BASE.swat_data() if dataset == "SWaT" else BASE.hai_data()
    if full_data["feature_names"] != data["feature_names"]:
        raise RuntimeError("post-freeze Test feature contract mismatch")
    full_data["validation"] = data["validation"]
    data = full_data
    val_store, test_store = ForecastStore.from_segments(data["validation"]), ForecastStore.from_segments(data["test"])
    val_error = raw_errors(model, val_store, device)
    median = np.median(val_error, axis=0); iqr = np.quantile(val_error, 0.75, axis=0) - np.quantile(val_error, 0.25, axis=0)
    val_scores = native_scores(val_error, median, iqr, data["validation"])
    threshold = float(np.quantile(val_scores, Q))
    test_scores = native_scores(raw_errors(model, test_store, device), median, iqr, data["test"])
    metrics = BASE.point_metrics(data["test_labels"], test_scores, threshold)
    predictions = (test_scores > threshold).astype(np.int8)
    if dataset == "SWaT":
        event_rows, event_summary = BASE.swat_event_metrics(predictions, data["test_timestamps"])
    else:
        event_rows, event_summary = BASE.hai_event_metrics(predictions, data["test_files"], data["test_raw_indices"])
    metrics.update({"dataset": dataset, "validation_score_q99": threshold, "validation_exceedance_rate": float((val_scores > threshold).mean()), "score": "official median/IQR+1e-2 standardized per-sensor absolute forecast error; four-point trailing mean; max sensor", "history": HISTORY, "seed": SEED, "events": event_summary})
    graph = graph_diagnostics(model, data["feature_names"], dataset)
    eff = efficiency(model, test_store, OUT / f"{slug(dataset)}_state_dict.pt", device); eff.update(training); eff["dataset"] = dataset
    return {"stopped": False, "metrics": metrics, "events": event_rows, "history": history, "training": training, "efficiency": eff, "graph": graph}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def write_partial_manifest(status: str, protocol: dict[str, Any], preflight: list[dict[str, Any]], results: dict[str, Any], elapsed: float) -> None:
    BASE.dump_json(OUT / "manifest.json", {"phase": "Thesis Baseline-B3", "status": status, "protocol": protocol, "resource_preflight": preflight, "results_completed": list(results), "elapsed_seconds": elapsed, "no_hp_search": True, "no_test_tuning": True, "point_adjustment": False, "pot": False, "commit": False, "push": False})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-wall-minutes", type=float, default=45.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args(); overall_start = time.perf_counter(); wall_seconds = args.max_wall_minutes * 60
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = frozen_protocol()
    if not torch.cuda.is_available(): raise RuntimeError("RTX 4060 CUDA device is required")
    device = torch.device("cuda")
    preflight = []
    try:
        for dataset in ("SWaT", "HAI21"):
            normal = normal_data(dataset)
            segments, channels = normal["train"], len(normal["feature_names"])
            preflight.append(resource_preflight(dataset, segments, channels, device))
            print(f"[{dataset}] RESOURCE_PREFLIGHT_PASS params={preflight[-1]['trainable_parameters']} peak_mib={preflight[-1]['peak_allocation_mib']:.2f}", flush=True)
            del segments
        BASE.dump_json(OUT / "resource_preflight.json", {"status": "RESOURCE_PREFLIGHT_PASS", "datasets": preflight})
    except torch.cuda.OutOfMemoryError as exc:
        write_partial_manifest("GDN_FROZEN_BATCH_OOM", protocol, preflight, {}, time.perf_counter() - overall_start)
        print(f"GDN_FROZEN_BATCH_OOM: {exc}", flush=True); return 2

    results: dict[str, Any] = {}
    completed_efficiencies: dict[str, Any] = {}
    for dataset in ("SWaT", "HAI21"):
        key = slug(dataset)
        metrics_path = OUT / f"{key}_metrics.json"
        if metrics_path.exists() and (OUT / f"{key}_state_dict.pt").exists():
            print(f"[{dataset}] frozen result exists; skipping retraining", flush=True)
            completed_efficiencies[dataset] = recover_completed_efficiency(dataset, device)
            continue
        data = normal_data(dataset)
        result = evaluate_dataset(dataset, data, device, overall_start, wall_seconds)
        if result["stopped"]:
            write_csv(OUT / "training_history.csv", result["history"])
            write_partial_manifest("HARD_STOP_INCOMPLETE", protocol, preflight, results, time.perf_counter() - overall_start)
            print("HARD_STOP_INCOMPLETE", flush=True); return 3
        results[dataset] = result
        BASE.dump_json(metrics_path, result["metrics"]); write_csv(OUT / f"{key}_event_metrics.csv", result["events"])
        BASE.dump_json(OUT / f"{key}_graph_diagnostics.json", {name: value for name, value in result["graph"].items() if name != "edges"})
        write_csv(OUT / f"{key}_learned_graph_edges.csv", result["graph"]["edges"])
        BASE.dump_json(OUT / f"{key}_efficiency.json", result["efficiency"])
        completed_efficiencies[dataset] = result["efficiency"]
        if time.perf_counter() - overall_start >= wall_seconds and dataset != "HAI21":
            histories = [row for value in results.values() for row in value["history"]]; write_csv(OUT / "training_history.csv", histories)
            write_partial_manifest("HARD_STOP_INCOMPLETE", protocol, preflight, results, time.perf_counter() - overall_start)
            print("HARD_STOP_INCOMPLETE after completed dataset", flush=True); return 3

    # Completion finalization is deliberately separated so partial runs never update the comparison table.
    if not all((OUT / f"{name}_metrics.json").exists() for name in ("swat", "hai")):
        write_partial_manifest("HARD_STOP_INCOMPLETE", protocol, preflight, results, time.perf_counter() - overall_start); return 3
    BASE.dump_json(OUT / "efficiency.json", completed_efficiencies)
    histories = [row for value in results.values() for row in value["history"]]; write_csv(OUT / "training_history.csv", histories)
    write_partial_manifest("GDN_BASELINE_COMPLETED", protocol, preflight, results, time.perf_counter() - overall_start)
    print("GDN_BASELINE_COMPLETED", flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
