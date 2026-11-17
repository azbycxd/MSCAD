from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

# Must be set before importing numpy/torch on some Windows Conda stacks.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

from data.swat import build_swat_dataloaders
from models.mscad import MSCAD
from utils.config_io import load_config, save_json
from utils.metrics import fuse_scores, binary_metrics
from utils.pot import POTThreshold
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


def threshold_mean_k_std(train_score: np.ndarray, k: float) -> float:
    return float(np.mean(train_score) + k * np.std(train_score))


def threshold_quantile(train_score: np.ndarray, q: float) -> float:
    return float(np.quantile(train_score, q))


def threshold_pot(train_score: np.ndarray, q: float, target_prob: float) -> float:
    pot = POTThreshold(q=q, target_prob=target_prob)
    pot.fit(train_score)
    return float(pot.tau)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/swat.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--mode",
        default="oracle",
        choices=["oracle"],
        help="oracle uses test labels to tune thresholds/weights; do NOT report as main result.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = build_device(cfg.get("device", "cuda"))
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    bundle = build_swat_dataloaders(
        train_csv=data_cfg["train_csv"],
        test_csv=data_cfg["test_csv"],
        label_col=data_cfg["label_col"],
        timestamp_col=data_cfg["timestamp_col"],
        window_size=int(data_cfg["window_size"]),
        stride_train=int(data_cfg["stride_train"]),
        stride_test=int(data_cfg["stride_test"]),
        val_ratio=float(data_cfg.get("val_ratio", 0.2)),
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
    )

    model_cfg = cfg["model"]
    model = MSCAD(
        in_channels=len(bundle.feature_names),
        hidden_channels=int(model_cfg["hidden_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
        dropout=float(model_cfg["dropout"]),
        cov_type=cfg["train"]["cov_type"],
        short_window=int(data_cfg["short_window"]),
    ).to(device)

    ckpt_path = args.ckpt or str(out_dir / "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)

    tr_rec, tr_short, tr_long, _ = collect_components(model, bundle.train_loader, device)
    te_rec, te_short, te_long, te_y = collect_components(model, bundle.test_loader, device)
    rec_stat, short_stat, long_stat = stats(tr_rec), stats(tr_short), stats(tr_long)

    weight_grid = [
        (0.50, 0.25, 0.25),
        (0.40, 0.30, 0.30),
        (0.34, 0.33, 0.33),
        (0.60, 0.20, 0.20),
        (0.70, 0.15, 0.15),
        (0.30, 0.35, 0.35),
    ]
    smooth_grid = [1, 3, 5, 9]
    quantile_grid = [0.95, 0.97, 0.98, 0.99]
    mean_k_grid = [2.0, 2.5, 3.0]
    pot_q_grid = [0.95, 0.97, 0.98]
    pot_p_grid = [0.01, 0.005, 0.001]

    rows = []
    def enrich(m: dict, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        out = dict(m)
        out["y_true_pos"] = int((y_true == 1).sum())
        out["y_true_neg"] = int((y_true == 0).sum())
        out["y_pred_pos"] = int((y_pred == 1).sum())
        out["y_pred_neg"] = int((y_pred == 0).sum())
        return out

    for alpha, beta, gamma in weight_grid:
        tr_score = fuse_scores(
            tr_rec,
            tr_short,
            tr_long,
            rec_stat=rec_stat,
            short_stat=short_stat,
            long_stat=long_stat,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        te_score = fuse_scores(
            te_rec,
            te_short,
            te_long,
            rec_stat=rec_stat,
            short_stat=short_stat,
            long_stat=long_stat,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        for smooth_w in smooth_grid:
            tr_s = moving_average(tr_score, smooth_w)
            te_s = moving_average(te_score, smooth_w)

            for q in quantile_grid:
                tau = threshold_quantile(tr_s, q)
                pred = (te_s > tau).astype(int)
                m = enrich(binary_metrics(te_y, pred, te_s), te_y, pred)
                rows.append(
                    {
                        "mode": "quantile",
                        "param": {"q": q},
                        "threshold_tau": float(tau),
                        "smooth_w": smooth_w,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        **m,
                    }
                )

            for k in mean_k_grid:
                tau = threshold_mean_k_std(tr_s, k)
                pred = (te_s > tau).astype(int)
                m = enrich(binary_metrics(te_y, pred, te_s), te_y, pred)
                rows.append(
                    {
                        "mode": "mean_k_std",
                        "param": {"k": k},
                        "threshold_tau": float(tau),
                        "smooth_w": smooth_w,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        **m,
                    }
                )

            for q in pot_q_grid:
                for p in pot_p_grid:
                    tau = threshold_pot(tr_s, q=q, target_prob=p)
                    pred = (te_s > tau).astype(int)
                    m = enrich(binary_metrics(te_y, pred, te_s), te_y, pred)
                    rows.append(
                        {
                            "mode": "pot",
                            "param": {"q": q, "target_prob": p},
                            "threshold_tau": float(tau),
                            "smooth_w": smooth_w,
                            "alpha": alpha,
                            "beta": beta,
                            "gamma": gamma,
                            **m,
                        }
                    )

    rows_sorted = sorted(rows, key=lambda r: (r["f1"], r["aupr"]), reverse=True)
    best = rows_sorted[0]
    topk = rows_sorted[: max(1, args.topk)]

    tuned_cfg = copy.deepcopy(cfg)
    tuned_cfg["eval"]["alpha"] = best["alpha"]
    tuned_cfg["eval"]["beta"] = best["beta"]
    tuned_cfg["eval"]["gamma"] = best["gamma"]
    tuned_cfg["eval"]["smooth_window"] = int(best["smooth_w"])
    tuned_cfg["eval"]["threshold_mode"] = best["mode"]
    if best["mode"] == "quantile":
        tuned_cfg["eval"]["threshold_quantile"] = float(best["param"]["q"])
    elif best["mode"] == "mean_k_std":
        tuned_cfg["eval"]["mean_k"] = float(best["param"]["k"])
    else:
        tuned_cfg["eval"]["pot_q"] = float(best["param"]["q"])
        tuned_cfg["eval"]["pot_target_prob"] = float(best["param"]["target_prob"])

    tune_meta = {
        "mode": args.mode,
        "warning": "Oracle tuning uses test labels. Use only as auxiliary analysis, not as main unsupervised result.",
        "best": best,
        "topk": topk,
    }
    save_json(str(out_dir / "oracle_tune_results.json"), tune_meta)
    save_json(str(out_dir / "swat_oracle_tuned.yaml"), tuned_cfg)

    print("Oracle tuning completed (auxiliary only, not main result).")
    print(
        f"Best | mode={best['mode']} param={best['param']} smooth={best['smooth_w']} "
        f"alpha/beta/gamma={best['alpha']}/{best['beta']}/{best['gamma']} "
        f"F1={best['f1']:.4f} P={best['precision']:.4f} R={best['recall']:.4f}"
    )
    print(f"Saved oracle tuned config: {out_dir / 'swat_oracle_tuned.yaml'}")


if __name__ == "__main__":
    main()
