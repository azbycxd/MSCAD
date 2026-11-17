from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must be set before importing numpy/torch on some Windows Conda stacks.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

from data.factory import build_dataloaders
from models.mscad import MSCAD
from utils.config_io import load_config, save_json
from utils.metrics import fuse_scores, binary_metrics
from utils.pot import POTThreshold
from utils.profiler import count_parameters, benchmark_inference_ms, gpu_peak_mem_mb, ram_usage_mb
from utils.seed import set_seed


def build_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def collect_components(model: MSCAD, loader, device: torch.device):
    model.eval()
    rec, short, long, y = [], [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        xh = model(xb)
        comp = model.score_components(xb, xh)
        rec.append(comp["rec"].detach().cpu().numpy())
        short.append(comp["short"].detach().cpu().numpy())
        long.append(comp["long"].detach().cpu().numpy())
        y.append(yb.numpy())
    return (
        np.concatenate(rec, axis=0),
        np.concatenate(short, axis=0),
        np.concatenate(long, axis=0),
        np.concatenate(y, axis=0),
    )


def stats(x: np.ndarray) -> tuple[float, float]:
    return float(np.mean(x)), float(np.std(x) + 1e-8)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(xp, kernel, mode="valid")


def run_eval(cfg: dict, ckpt_path: str | None = None) -> dict:
    set_seed(int(cfg["seed"]))
    device = build_device(cfg.get("device", "cuda"))
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    bundle = build_dataloaders(cfg, pin_memory=(device.type == "cuda"))

    model_cfg = cfg["model"]
    model = MSCAD(
        in_channels=len(bundle.feature_names),
        hidden_channels=int(model_cfg["hidden_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
        dropout=float(model_cfg["dropout"]),
        cov_type=cfg["train"]["cov_type"],
        short_window=int(data_cfg["short_window"]),
        backbone_variant=str(model_cfg.get("backbone_variant", "base")),
    ).to(device)

    if ckpt_path is None:
        ckpt_path = str(out_dir / "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    if "structure_pairs" in ckpt:
        model.set_structure_pairs(ckpt["structure_pairs"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    tr_rec, tr_short, tr_long, _ = collect_components(model, bundle.train_loader, device)
    te_rec, te_short, te_long, te_y = collect_components(model, bundle.test_loader, device)

    rec_stat = stats(tr_rec)
    short_stat = stats(tr_short)
    long_stat = stats(tr_long)

    eval_cfg = cfg["eval"]
    tr_score = fuse_scores(
        tr_rec,
        tr_short,
        tr_long,
        rec_stat=rec_stat,
        short_stat=short_stat,
        long_stat=long_stat,
        alpha=float(eval_cfg["alpha"]),
        beta=float(eval_cfg["beta"]),
        gamma=float(eval_cfg["gamma"]),
    )
    te_score = fuse_scores(
        te_rec,
        te_short,
        te_long,
        rec_stat=rec_stat,
        short_stat=short_stat,
        long_stat=long_stat,
        alpha=float(eval_cfg["alpha"]),
        beta=float(eval_cfg["beta"]),
        gamma=float(eval_cfg["gamma"]),
    )
    smooth_w = int(eval_cfg.get("smooth_window", 1))
    tr_score = moving_average(tr_score, smooth_w)
    te_score = moving_average(te_score, smooth_w)

    mode = str(eval_cfg.get("threshold_mode", "pot")).lower()
    if mode == "pot":
        pot = POTThreshold(q=float(eval_cfg.get("pot_q", 0.98)), target_prob=float(eval_cfg.get("pot_target_prob", 1e-3)))
        pot.fit(tr_score)
        tau = float(pot.tau)
    elif mode == "quantile":
        tau = float(np.quantile(tr_score, float(eval_cfg.get("threshold_quantile", 0.99))))
    elif mode == "mean_k_std":
        k = float(eval_cfg.get("mean_k", 3.0))
        tau = float(np.mean(tr_score) + k * np.std(tr_score))
    elif mode == "mean3std":
        tau = float(np.mean(tr_score) + 3.0 * np.std(tr_score))
    else:
        raise ValueError(f"Unsupported threshold mode: {mode}")

    te_pred = (te_score > tau).astype(int)
    m = binary_metrics(te_y, te_pred, te_score)
    m["y_true_pos"] = int((te_y == 1).sum())
    m["y_true_neg"] = int((te_y == 0).sum())
    m["y_pred_pos"] = int((te_pred == 1).sum())
    m["y_pred_neg"] = int((te_pred == 0).sum())

    first_batch = next(iter(bundle.test_loader))[0]
    infer_ms = benchmark_inference_ms(model, first_batch, warmup=10, runs=30)
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
            "short": {"mean": short_stat[0], "std": short_stat[1]},
            "long": {"mean": long_stat[0], "std": long_stat[1]},
        },
    }
    np.save(str(out_dir / "train_score.npy"), tr_score.astype(np.float32))
    np.save(str(out_dir / "test_score.npy"), te_score.astype(np.float32))
    np.save(str(out_dir / "test_label.npy"), te_y.astype(np.int64))
    np.save(str(out_dir / "test_pred.npy"), te_pred.astype(np.int64))
    save_json(str(out_dir / "eval_metrics.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/swat.yaml")
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
