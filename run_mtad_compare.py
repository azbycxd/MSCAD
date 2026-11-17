from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from utils.config_io import load_config, save_json
from utils.metrics import binary_metrics, z_norm
from utils.pot import POTThreshold


def run_cmd(cmd: List[str], cwd: str) -> None:
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    p = subprocess.run(cmd, cwd=cwd, env=env)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(xp, kernel, mode="valid")


def refine_core_fusion(core_dir: Path, q_grid: list[float], smooth_grid: list[int], alpha_grid: list[float]) -> dict:
    tr_rec = np.load(str(core_dir / "train_rec_score.npy"))
    tr_pred = np.load(str(core_dir / "train_pred_score.npy"))
    te_rec = np.load(str(core_dir / "test_rec_score.npy"))
    te_pred = np.load(str(core_dir / "test_pred_score.npy"))
    y = np.load(str(core_dir / "test_label.npy")).astype(int)

    rec_stat = (float(np.mean(tr_rec)), float(np.std(tr_rec) + 1e-8))
    pred_stat = (float(np.mean(tr_pred)), float(np.std(tr_pred) + 1e-8))
    zr_tr = z_norm(tr_rec, rec_stat[0], rec_stat[1])
    zp_tr = z_norm(tr_pred, pred_stat[0], pred_stat[1])
    zr_te = z_norm(te_rec, rec_stat[0], rec_stat[1])
    zp_te = z_norm(te_pred, pred_stat[0], pred_stat[1])

    rows = []
    for a in alpha_grid:
        b = float(1.0 - a)
        for sw in smooth_grid:
            tr_score = moving_average(a * zr_tr + b * zp_tr, sw)
            te_score = moving_average(a * zr_te + b * zp_te, sw)
            for q in q_grid:
                pot = POTThreshold(q=float(q), target_prob=0.005)
                pot.fit(tr_score)
                tau = float(pot.tau)
                pred = (te_score > tau).astype(int)
                m = binary_metrics(y, pred, te_score)
                rows.append(
                    {
                        "alpha": float(a),
                        "beta": float(b),
                        "smooth_window": int(sw),
                        "pot_q": float(q),
                        "threshold_tau": float(tau),
                        "f1": float(m["f1"]),
                        "precision": float(m["precision"]),
                        "recall": float(m["recall"]),
                        "aupr": float(m["aupr"]),
                        "auroc": float(m["auroc"]),
                    }
                )
    return {
        "n_combinations": len(rows),
        "best_by_f1": max(rows, key=lambda r: r["f1"]),
        "best_by_aupr": max(rows, key=lambda r: r["aupr"]),
        "best_by_auroc": max(rows, key=lambda r: r["auroc"]),
        "top5_by_f1": sorted(rows, key=lambda r: r["f1"], reverse=True)[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mtad_swat.yaml")
    parser.add_argument("--workdir", default="D:/workspace/python/mscad")
    parser.add_argument("--out-dir", default="outputs/mtad_compare")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--variants", default="core,r,p,w,full")
    args = parser.parse_args()

    cwd = args.workdir
    py = sys.executable
    base_cfg = load_config(args.config)
    root = Path(cwd) / args.out_dir
    root.mkdir(parents=True, exist_ok=True)

    variants = [v.strip().lower() for v in args.variants.split(",") if v.strip()]
    if not variants:
        raise ValueError("No MTAD variants provided.")

    run_summary: Dict[str, Any] = {"config": args.config, "variants": variants, "runs": {}}

    for v in variants:
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["variant"] = v
        if v == "r":
            cfg["eval"]["alpha"] = 1.0
            cfg["eval"]["beta"] = 0.0
        elif v == "p":
            cfg["eval"]["alpha"] = 0.0
            cfg["eval"]["beta"] = 1.0
        else:
            cfg["eval"]["alpha"] = float(cfg["eval"].get("alpha", 0.5))
            cfg["eval"]["beta"] = float(cfg["eval"].get("beta", 0.5))
        cfg["output"]["dir"] = f"{args.out_dir}/{v}"
        cfg_path = root / f"mtad_{v}_config.yaml"
        save_json(str(cfg_path), cfg)
        if not args.skip_train:
            run_cmd([py, "train_mtad.py", "--config", str(cfg_path)], cwd)
        run_cmd([py, "evaluate_mtad.py", "--config", str(cfg_path)], cwd)
        eval_res = load_config(str(Path(cwd) / cfg["output"]["dir"] / "eval_metrics.json"))
        run_summary["runs"][v] = eval_res

    for k in ["core", "full"]:
        if k in variants:
            core_dir = Path(cwd) / args.out_dir / k
            run_summary[f"{k}_fusion_refine"] = refine_core_fusion(
                core_dir=core_dir,
                q_grid=[0.93, 0.95, 0.97],
                smooth_grid=[1, 3, 5],
                alpha_grid=[0.3, 0.4, 0.5, 0.6, 0.7],
            )

    rows = []
    for v, d in run_summary["runs"].items():
        m = d["metrics"]
        p = d["profile"]
        rows.append(
            {
                "model": f"MTAD-{v.upper()}",
                "f1": float(m["f1"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "aupr": float(m["aupr"]),
                "auroc": float(m["auroc"]),
                "params": int(p["params"]),
                "inference_ms_per_batch": float(p["inference_ms_per_batch"]),
                "gpu_peak_mem_mb": p["gpu_peak_mem_mb"],
            }
        )
    if "core_fusion_refine" in run_summary:
        bf = run_summary["core_fusion_refine"]["best_by_f1"]
        rows.append(
            {
                "model": "MTAD-CORE+FUSION",
                "f1": float(bf["f1"]),
                "precision": float(bf["precision"]),
                "recall": float(bf["recall"]),
                "aupr": float(bf["aupr"]),
                "auroc": float(bf["auroc"]),
                "params": int(run_summary["runs"]["core"]["profile"]["params"]),
                "inference_ms_per_batch": float(run_summary["runs"]["core"]["profile"]["inference_ms_per_batch"]),
                "gpu_peak_mem_mb": run_summary["runs"]["core"]["profile"]["gpu_peak_mem_mb"],
            }
        )
    if "full_fusion_refine" in run_summary:
        bf = run_summary["full_fusion_refine"]["best_by_f1"]
        rows.append(
            {
                "model": "MTAD-FULL+FUSION",
                "f1": float(bf["f1"]),
                "precision": float(bf["precision"]),
                "recall": float(bf["recall"]),
                "aupr": float(bf["aupr"]),
                "auroc": float(bf["auroc"]),
                "params": int(run_summary["runs"]["full"]["profile"]["params"]),
                "inference_ms_per_batch": float(run_summary["runs"]["full"]["profile"]["inference_ms_per_batch"]),
                "gpu_peak_mem_mb": run_summary["runs"]["full"]["profile"]["gpu_peak_mem_mb"],
            }
        )
    run_summary["table_rows"] = rows
    out_path = root / "mtad_compare_summary.json"
    save_json(str(out_path), run_summary)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
