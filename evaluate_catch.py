from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

from data.factory import build_dataloaders
from models.catch import CATCHModel
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


def get_variant(cfg: dict) -> str:
    return str(cfg["model"].get("variant", "core")).lower()


def get_effective_fuse_weights(cfg: dict) -> tuple[float, float]:
    v = get_variant(cfg)
    a = float(cfg["eval"].get("alpha", 0.5))
    b = float(cfg["eval"].get("beta", 0.5))
    if v == "timeonly":
        b = 0.0
    s = a + b
    if s <= 0:
        raise ValueError("Invalid fusion weights: alpha + beta must be > 0")
    return a, b


@torch.no_grad()
def collect_components(model: CATCHModel, loader, device: torch.device):
    model.eval()
    ts, fs, y = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        out = model(xb)
        t_score = (out["x_time_hat"] - xb).abs().mean(dim=(1, 2))
        f_score = ((out["real_hat"] - out["real"]) ** 2).mean(dim=(1, 2)) + (
            (out["imag_hat"] - out["imag"]) ** 2
        ).mean(dim=(1, 2))
        ts.append(t_score.detach().cpu().numpy())
        fs.append(f_score.detach().cpu().numpy())
        y.append(yb.numpy())
    return np.concatenate(ts, axis=0), np.concatenate(fs, axis=0), np.concatenate(y, axis=0)


def run_eval(cfg: dict, ckpt_path: str | None = None) -> dict:
    set_seed(int(cfg["seed"]))
    device = build_device(cfg.get("device", "cuda"))
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_dataloaders(cfg, pin_memory=(device.type == "cuda"))

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    model = CATCHModel(
        in_channels=len(bundle.feature_names),
        window_size=int(data_cfg["window_size"]),
        patch_size=int(model_cfg.get("patch_size", 8)),
        patch_stride=int(model_cfg.get("patch_stride", 4)),
        d_model=int(model_cfg.get("d_model", 64)),
        nhead=int(model_cfg.get("nhead", 4)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        ff_dim=int(model_cfg.get("ff_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        variant=get_variant(cfg),
        mask_hidden=int(model_cfg.get("mask_hidden", 32)),
    ).to(device)

    if ckpt_path is None:
        ckpt_path = str(out_dir / "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    tr_t, tr_f, _ = collect_components(model, bundle.train_loader, device)
    te_t, te_f, te_y = collect_components(model, bundle.test_loader, device)
    t_stat = stats(tr_t)
    f_stat = stats(tr_f)

    alpha, beta = get_effective_fuse_weights(cfg)
    tr_score = alpha * z_norm(tr_t, t_stat[0], t_stat[1]) + beta * z_norm(tr_f, f_stat[0], f_stat[1])
    te_score = alpha * z_norm(te_t, t_stat[0], t_stat[1]) + beta * z_norm(te_f, f_stat[0], f_stat[1])

    sw = int(cfg["eval"].get("smooth_window", 1))
    tr_score = moving_average(tr_score, sw)
    te_score = moving_average(te_score, sw)

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
            "time": {"mean": t_stat[0], "std": t_stat[1]},
            "freq": {"mean": f_stat[0], "std": f_stat[1]},
        },
        "fuse_weights": {"alpha": float(alpha), "beta": float(beta)},
        "variant": get_variant(cfg),
    }
    np.save(str(out_dir / "train_time_score.npy"), tr_t.astype(np.float32))
    np.save(str(out_dir / "train_freq_score.npy"), tr_f.astype(np.float32))
    np.save(str(out_dir / "test_time_score.npy"), te_t.astype(np.float32))
    np.save(str(out_dir / "test_freq_score.npy"), te_f.astype(np.float32))
    np.save(str(out_dir / "train_score.npy"), tr_score.astype(np.float32))
    np.save(str(out_dir / "test_score.npy"), te_score.astype(np.float32))
    np.save(str(out_dir / "test_label.npy"), te_y.astype(np.int64))
    np.save(str(out_dir / "test_pred.npy"), te_pred.astype(np.int64))
    save_json(str(out_dir / "eval_metrics.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/catch_swat.yaml")
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
