from __future__ import annotations

import os
import time
from typing import Any

import torch

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ram_usage_mb() -> float | None:
    if psutil is None:
        return None
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024.0 * 1024.0)


@torch.no_grad()
def benchmark_inference_ms(model: torch.nn.Module, batch: torch.Tensor, warmup: int = 20, runs: int = 50) -> float:
    model.eval()
    device = next(model.parameters()).device
    x = batch.to(device, non_blocking=True)
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / runs


def gpu_peak_mem_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device=device) / (1024.0 * 1024.0)
