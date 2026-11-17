from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

from data.factory import build_dataloaders
from models.mtad import MTADCore
from utils.config_io import load_config, save_json
from utils.metrics import binary_metrics, z_norm
from utils.pot import POTThreshold
from utils.profiler import benchmark_inference_ms, count_parameters, gpu_peak_mem_mb, ram_usage_mb
from utils.seed import set_seed


def build_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(xp, kernel, mode="valid")


def stats(x: np.ndarray) -> tuple[float, float]:
    return float(np.mean(x)), float(np.std(x) + 1e-8)


def get_effective_fuse_weights(cfg: dict) -> tuple[float, float]:
    model_variant = str(cfg["model"].get("variant", "core")).lower()
    a = float(cfg["eval"].get("alpha", 0.5))
    b = float(cfg["eval"].get("beta", 0.5))
    if model_variant == "r":
        b = 0.0
    elif model_variant == "p":
        a = 0.0
    s = a + b
    if s <= 0:
        raise ValueError("Invalid fusion weights: alpha + beta must be > 0")
    return a, b


def is_ot_svd_enabled(cfg: dict) -> bool:
    variant = str(cfg["model"].get("variant", "core")).lower()
    if variant == "full":
        return True
    if variant in {"w", "core", "r", "p"}:
        return False
    return bool(cfg["model"].get("ot_svd_enabled", False))


def get_decoder_type(cfg: dict) -> str:
    variant = str(cfg["model"].get("variant", "core")).lower()
    if variant in {"w", "full"}:
        return "conv"
    return str(cfg["model"].get("decoder_type", "mlp")).lower()


@torch.no_grad()
def collect_components(model: MTADCore, loader, device: torch.device):
    model.eval()
    rec, pred, y = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        if xb.size(1) < 2:
            continue
        x_ctx = xb[:, :-1, :]
        x_next = xb[:, -1, :]
        x_rec, x_pred = model(x_ctx)
        rec_s = ((x_rec - x_ctx) ** 2).mean(dim=(1, 2))
        pred_s = ((x_pred - x_next) ** 2).mean(dim=1)
        rec.append(rec_s.detach().cpu().numpy())
        pred.append(pred_s.detach().cpu().numpy())
        y.append(yb.numpy())
    return np.concatenate(rec, axis=0), np.concatenate(pred, axis=0), np.concatenate(y, axis=0)


def fuse_score(
    rec_score: np.ndarray,
    pred_score: np.ndarray,
    rec_stat: tuple[float, float],
    pred_stat: tuple[float, float],
    alpha: float,
    beta: float,
) -> np.ndarray:
    zr = z_norm(rec_score, rec_stat[0], rec_stat[1])
    zp = z_norm(pred_score, pred_stat[0], pred_stat[1])
    return alpha * zr + beta * zp


def run_eval(cfg: dict, ckpt_path: str | None = None) -> dict:
    set_seed(int(cfg["seed"]))
    device = build_device(cfg.get("device", "cuda"))
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_dataloaders(cfg, pin_memory=(device.type == "cuda"))

    model_cfg = cfg["model"]
    model = MTADCore(
        in_channels=len(bundle.feature_names),
        d_model=int(model_cfg.get("d_model", 64)),
        nhead=int(model_cfg.get("nhead", 4)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        ff_dim=int(model_cfg.get("ff_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_len=int(model_cfg.get("max_len", 512)),
        decoder_type=get_decoder_type(cfg),
    ).to(device)

    if ckpt_path is None:
        ckpt_path = str(out_dir / "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    tr_rec, tr_pred, _ = collect_components(model, bundle.train_loader, device)
    te_rec, te_pred, te_y = collect_components(model, bundle.test_loader, device)
    rec_stat = stats(tr_rec)
    pred_stat = stats(tr_pred)

    alpha, beta = get_effective_fuse_weights(cfg)
    tr_score = fuse_score(tr_rec, tr_pred, rec_stat, pred_stat, alpha=alpha, beta=beta)
    te_score = fuse_score(te_rec, te_pred, rec_stat, pred_stat, alpha=alpha, beta=beta)

    smooth_w = int(cfg["eval"].get("smooth_window", 1))
    tr_score = moving_average(tr_score, smooth_w)
    te_score = moving_average(te_score, smooth_w)

    mode = str(cfg["eval"].get("threshold_mode", "pot")).lower()
    if mode == "pot":
        pot = POTThreshold(
            q=float(cfg["eval"].get("pot_q", 0.98)),
            target_prob=float(cfg["eval"].get("pot_target_prob", 1e-3)),
        )
        pot.fit(tr_score)
        tau = float(pot.tau)
    elif mode == "quantile":
        tau = float(np.quantile(tr_score, float(cfg["eval"].get("threshold_quantile", 0.99))))
    elif mode == "mean_k_std":
        k = float(cfg["eval"].get("mean_k", 3.0))
        tau = float(np.mean(tr_score) + k * np.std(tr_score))
    elif mode == "mean3std":
        tau = float(np.mean(tr_score) + 3.0 * np.std(tr_score))
    else:
        raise ValueError(f"Unsupported threshold mode: {mode}")

    te_pred_label = (te_score > tau).astype(int)
    m = binary_metrics(te_y, te_pred_label, te_score)
    m["y_true_pos"] = int((te_y == 1).sum())
    m["y_true_neg"] = int((te_y == 0).sum())
    m["y_pred_pos"] = int((te_pred_label == 1).sum())
    m["y_pred_neg"] = int((te_pred_label == 0).sum())

    first_batch = next(iter(bundle.test_loader))[0]
    first_ctx = first_batch[:, :-1, :]
    infer_ms = benchmark_inference_ms(model, first_ctx, warmup=10, runs=30)
    profile = {
        "params": int(count_parameters(model)),
        "inference_ms_per_batch": float(infer_ms),
        "gpu_peak_mem_mb": gpu_peak_mem_mb(device),
        "cpu_ram_mb": ram_usage_mb(),
        "threshold_tau": tau,
        "threshold_mode": mode,
    }

    result = {
        "metrics": m,
        "profile": profile,
        "train_score_stats": {
            "rec": {"mean": rec_stat[0], "std": rec_stat[1]},
            "pred": {"mean": pred_stat[0], "std": pred_stat[1]},
        },
        "fuse_weights": {"alpha": float(alpha), "beta": float(beta)},
        "variant": str(cfg["model"].get("variant", "core")),
    }
    np.save(str(out_dir / "train_rec_score.npy"), tr_rec.astype(np.float32))
    np.save(str(out_dir / "train_pred_score.npy"), tr_pred.astype(np.float32))
    np.save(str(out_dir / "test_rec_score.npy"), te_rec.astype(np.float32))
    np.save(str(out_dir / "test_pred_score.npy"), te_pred.astype(np.float32))
    np.save(str(out_dir / "train_score.npy"), tr_score.astype(np.float32))
    np.save(str(out_dir / "test_score.npy"), te_score.astype(np.float32))
    np.save(str(out_dir / "test_label.npy"), te_y.astype(np.int64))
    np.save(str(out_dir / "test_pred.npy"), te_pred_label.astype(np.int64))
    save_json(str(out_dir / "eval_metrics.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mtad_swat.yaml")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_eval(cfg, ckpt_path=args.ckpt)
    m = result["metrics"]
    p = result["profile"]
    print(
        f"Eval done | F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
        f"AUPR={m['aupr']:.4f} AUROC={m['auroc']:.4f} | "
        f"y_true(pos/neg)={m['y_true_pos']}/{m['y_true_neg']} "
        f"y_pred(pos/neg)={m['y_pred_pos']}/{m['y_pred_neg']} | "
        f"Params={p['params']} Infer(ms/batch)={p['inference_ms_per_batch']:.3f}"
    )


if __name__ == "__main__":
    main()
