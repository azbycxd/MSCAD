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


def refine_fusion(run_dir: Path, q_grid: list[float], smooth_grid: list[int], alpha_grid: list[float]) -> dict:
    tr_t = np.load(str(run_dir / "train_time_score.npy"))
    tr_f = np.load(str(run_dir / "train_freq_score.npy"))
    te_t = np.load(str(run_dir / "test_time_score.npy"))
    te_f = np.load(str(run_dir / "test_freq_score.npy"))
    y = np.load(str(run_dir / "test_label.npy")).astype(int)

    t_stat = (float(np.mean(tr_t)), float(np.std(tr_t) + 1e-8))
    f_stat = (float(np.mean(tr_f)), float(np.std(tr_f) + 1e-8))
    zt_tr = z_norm(tr_t, t_stat[0], t_stat[1])
    zf_tr = z_norm(tr_f, f_stat[0], f_stat[1])
    zt_te = z_norm(te_t, t_stat[0], t_stat[1])
    zf_te = z_norm(te_f, f_stat[0], f_stat[1])

    rows = []
    for a in alpha_grid:
        b = float(1.0 - a)
        for sw in smooth_grid:
            tr_score = moving_average(a * zt_tr + b * zf_tr, sw)
            te_score = moving_average(a * zt_te + b * zf_te, sw)
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
    parser.add_argument("--config", default="configs/catch_swat.yaml")
    parser.add_argument("--workdir", default="D:/workspace/python/mscad")
    parser.add_argument("--out-dir", default="outputs/catch_compare")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--variants", default="core,w,full")
    args = parser.parse_args()

    cwd = args.workdir
    py = sys.executable
    base_cfg = load_config(args.config)
    root = Path(cwd) / args.out_dir
    root.mkdir(parents=True, exist_ok=True)

    variants = [v.strip().lower() for v in args.variants.split(",") if v.strip()]
    if not variants:
        raise ValueError("No CATCH variants provided.")

    summary: Dict[str, Any] = {"config": args.config, "variants": variants, "runs": {}}
    for v in variants:
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["variant"] = v
        if v == "timeonly":
            cfg["eval"]["alpha"] = 1.0
            cfg["eval"]["beta"] = 0.0
        cfg["output"]["dir"] = f"{args.out_dir}/{v}"
        cfg_path = root / f"catch_{v}_config.yaml"
        save_json(str(cfg_path), cfg)
        if not args.skip_train:
            run_cmd([py, "train_catch.py", "--config", str(cfg_path)], cwd)
        run_cmd([py, "evaluate_catch.py", "--config", str(cfg_path)], cwd)
        summary["runs"][v] = load_config(str(Path(cwd) / cfg["output"]["dir"] / "eval_metrics.json"))

    for v in ["core", "full"]:
        if v in variants:
            summary[f"{v}_fusion_refine"] = refine_fusion(
                Path(cwd) / args.out_dir / v,
                q_grid=[0.93, 0.95, 0.97],
                smooth_grid=[1, 3, 5],
                alpha_grid=[0.3, 0.4, 0.5, 0.6, 0.7],
            )

    rows = []
    for v, d in summary["runs"].items():
        m = d["metrics"]
        p = d["profile"]
        rows.append(
            {
                "model": f"CATCH-{v.upper()}",
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
    if "core_fusion_refine" in summary:
        b = summary["core_fusion_refine"]["best_by_f1"]
        p = summary["runs"]["core"]["profile"]
        rows.append(
            {
                "model": "CATCH-CORE+FUSION",
                "f1": float(b["f1"]),
                "precision": float(b["precision"]),
                "recall": float(b["recall"]),
                "aupr": float(b["aupr"]),
                "auroc": float(b["auroc"]),
                "params": int(p["params"]),
                "inference_ms_per_batch": float(p["inference_ms_per_batch"]),
                "gpu_peak_mem_mb": p["gpu_peak_mem_mb"],
            }
        )
    if "full_fusion_refine" in summary:
        b = summary["full_fusion_refine"]["best_by_f1"]
        p = summary["runs"]["full"]["profile"]
        rows.append(
            {
                "model": "CATCH-FULL+FUSION",
                "f1": float(b["f1"]),
                "precision": float(b["precision"]),
                "recall": float(b["recall"]),
                "aupr": float(b["aupr"]),
                "auroc": float(b["auroc"]),
                "params": int(p["params"]),
                "inference_ms_per_batch": float(p["inference_ms_per_batch"]),
                "gpu_peak_mem_mb": p["gpu_peak_mem_mb"],
            }
        )
    summary["table_rows"] = rows
    out_path = root / "catch_compare_summary.json"
    save_json(str(out_path), summary)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
