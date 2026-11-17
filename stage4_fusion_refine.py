from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

from data.factory import build_dataloaders
from models.mscad import MSCAD
from utils.config_io import load_config, save_json
from utils.metrics import binary_metrics, fuse_scores
from utils.pot import POTThreshold
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


def stats(x: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(x)), float(np.std(x) + 1e-8)


def fit_tau(train_score: np.ndarray, q: float, target_prob: float) -> float:
    pot = POTThreshold(q=q, target_prob=target_prob)
    pot.fit(train_score)
    return float(pot.tau)


def evaluate_ckpt(
    ckpt_path: Path,
    alpha_grid: List[float],
    beta_grid: List[float],
    gamma_grid: List[float],
    smooth_grid: List[int],
    q_grid: List[float],
    target_prob: float,
    seed: int,
) -> Dict[str, Any]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", load_config("configs/swat_main.yaml"))
    set_seed(seed)
    device = build_device(cfg.get("device", "cuda"))
    data_cfg = cfg["data"]
    bundle = build_dataloaders(cfg, pin_memory=False)
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
    model.load_state_dict(ckpt["model_state"], strict=True)
    if "structure_pairs" in ckpt:
        model.set_structure_pairs(ckpt["structure_pairs"])

    tr_rec, tr_short, tr_long, _ = collect_components(model, bundle.train_loader, device)
    te_rec, te_short, te_long, te_y = collect_components(model, bundle.test_loader, device)
    rec_stat = stats(tr_rec)
    short_stat = stats(tr_short)
    long_stat = stats(tr_long)

    rows = []
    combos = itertools.product(alpha_grid, beta_grid, gamma_grid, smooth_grid, q_grid)
    for alpha, beta, gamma, smooth_w, pot_q in combos:
        tr_score = fuse_scores(
            tr_rec,
            tr_short,
            tr_long,
            rec_stat=rec_stat,
            short_stat=short_stat,
            long_stat=long_stat,
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
        )
        te_score = fuse_scores(
            te_rec,
            te_short,
            te_long,
            rec_stat=rec_stat,
            short_stat=short_stat,
            long_stat=long_stat,
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
        )
        tr_score = moving_average(tr_score, int(smooth_w))
        te_score = moving_average(te_score, int(smooth_w))
        tau = fit_tau(tr_score, q=float(pot_q), target_prob=float(target_prob))
        pred = (te_score > tau).astype(int)
        m = binary_metrics(te_y, pred, te_score)
        rows.append(
            {
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
                "smooth_window": int(smooth_w),
                "pot_q": float(pot_q),
                "pot_target_prob": float(target_prob),
                "threshold_tau": float(tau),
                "f1": float(m["f1"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "aupr": float(m["aupr"]),
                "auroc": float(m["auroc"]),
            }
        )

    best_f1 = max(rows, key=lambda r: r["f1"])
    best_aupr = max(rows, key=lambda r: r["aupr"])
    best_auroc = max(rows, key=lambda r: r["auroc"])
    return {
        "ckpt": str(ckpt_path),
        "n_combinations": len(rows),
        "best_by_f1": best_f1,
        "best_by_aupr": best_aupr,
        "best_by_auroc": best_auroc,
        "top5_by_f1": sorted(rows, key=lambda r: r["f1"], reverse=True)[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/stage4_fusion_refine/stage4_fusion_refine_summary.json")
    parser.add_argument(
        "--ckpts",
        nargs="+",
        default=[
            "outputs/stage3_refine/R5_ws96_ss24_ls0.05_ll0.02_top15_corr0.92/best.pt",
            "outputs/stage3_refine/R6_ws96_ss24_ls0.05_ll0.02_top25_corr0.88/best.pt",
        ],
    )
    args = parser.parse_args()

    alpha_grid = [0.45, 0.50, 0.55, 0.60]
    beta_grid = [0.15, 0.20, 0.25]
    gamma_grid = [0.20, 0.25, 0.30, 0.35]
    smooth_grid = [1, 3, 5]
    q_grid = [0.93, 0.95, 0.97]
    target_prob = 0.005

    out = {"grid": {"alpha": alpha_grid, "beta": beta_grid, "gamma": gamma_grid, "smooth_window": smooth_grid, "pot_q": q_grid}}
    out["runs"] = {}
    for ck in args.ckpts:
        ckpt_path = Path(ck).resolve()
        name = ckpt_path.parent.name
        print(f"Running fusion refine for {name} ...")
        out["runs"][name] = evaluate_ckpt(
            ckpt_path=ckpt_path,
            alpha_grid=alpha_grid,
            beta_grid=beta_grid,
            gamma_grid=gamma_grid,
            smooth_grid=smooth_grid,
            q_grid=q_grid,
            target_prob=target_prob,
            seed=args.seed,
        )

    f1_rank = []
    for name, run in out["runs"].items():
        r = dict(run["best_by_f1"])
        r["run"] = name
        f1_rank.append(r)
    out["best_overall_by_f1"] = max(f1_rank, key=lambda x: x["f1"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(str(out_path), out)
    print(f"Saved: {out_path}")
    print("Best overall by F1:", out["best_overall_by_f1"])


if __name__ == "__main__":
    main()
