from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from utils.config_io import load_config, save_json


def run_cmd(cmd: List[str], cwd: str) -> None:
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    p = subprocess.run(cmd, cwd=cwd, env=env)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def run_train_eval(cwd: str, py: str, cfg_path: Path, init_ckpt: str) -> Dict[str, Any]:
    run_cmd([py, "train.py", "--config", str(cfg_path), "--init-ckpt", init_ckpt], cwd=cwd)
    run_cmd([py, "evaluate.py", "--config", str(cfg_path)], cwd=cwd)
    cfg = load_config(str(cfg_path))
    out_dir = Path(cwd) / cfg["output"]["dir"]
    return {
        "eval": load_config(str(out_dir / "eval_metrics.json")),
        "train": load_config(str(out_dir / "train_metrics.json")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/swat_main.yaml")
    parser.add_argument("--workdir", default="D:/workspace/python/mscad")
    parser.add_argument(
        "--init-ckpt",
        default="outputs/stage2_grid/A0_backbone_wo_struct/best.pt",
        help="Backbone checkpoint path used for stage-2 initialization.",
    )
    args = parser.parse_args()

    cwd = args.workdir
    py = sys.executable
    base_cfg = load_config(args.base_config)
    root = Path(cwd) / "outputs" / "stage3_refine"
    root.mkdir(parents=True, exist_ok=True)
    init_ckpt = str((Path(cwd) / args.init_ckpt).resolve())

    grid = [
        ("R1_ws96_ss24_ls0.05_ll0.02_top20_corr0.9", 96, 24, 0.05, 0.02, 20, 0.90),
        ("R2_ws96_ss24_ls0.05_ll0.03_top20_corr0.9", 96, 24, 0.05, 0.03, 20, 0.90),
        ("R3_ws80_ss20_ls0.05_ll0.02_top20_corr0.9", 80, 20, 0.05, 0.02, 20, 0.90),
        ("R4_ws64_ss16_ls0.04_ll0.02_top20_corr0.9", 64, 16, 0.04, 0.02, 20, 0.90),
        ("R5_ws96_ss24_ls0.05_ll0.02_top15_corr0.92", 96, 24, 0.05, 0.02, 15, 0.92),
        ("R6_ws96_ss24_ls0.05_ll0.02_top25_corr0.88", 96, 24, 0.05, 0.02, 25, 0.88),
    ]

    summary: Dict[str, Any] = {"runs": {}, "init_ckpt": init_ckpt}
    rows = []
    for name, ws, ss, ls, ll, topk, min_corr in grid:
        cfg = copy.deepcopy(base_cfg)
        cfg["data"]["window_size"] = ws
        cfg["data"]["short_window"] = ss
        cfg["train"]["lambda_s"] = ls
        cfg["train"]["lambda_l"] = ll
        cfg["train"]["structure_warmup_epochs"] = 10
        cfg["train"]["structure_pairs"] = {"enabled": True, "topk": topk, "min_abs_corr": min_corr}
        cfg["train"]["epochs"] = min(int(cfg["train"]["epochs"]), 20)
        cfg["train"]["patience"] = min(int(cfg["train"]["patience"]), 5)
        cfg["eval"]["alpha"] = 0.5
        cfg["eval"]["beta"] = 0.25
        cfg["eval"]["gamma"] = 0.25
        cfg["output"]["dir"] = f"outputs/stage3_refine/{name}"

        cfg_path = root / f"{name}_config.yaml"
        save_json(str(cfg_path), cfg)
        out = run_train_eval(cwd, py, cfg_path, init_ckpt)
        summary["runs"][name] = out
        m = out["eval"]["metrics"]
        rows.append(
            {
                "name": name,
                "f1": float(m["f1"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "aupr": float(m["aupr"]),
                "auroc": float(m["auroc"]),
            }
        )

    rows = sorted(rows, key=lambda x: x["f1"], reverse=True)
    summary["ranked_by_f1"] = rows
    summary["best_by_f1"] = rows[0]
    summary["best_by_aupr"] = sorted(rows, key=lambda x: x["aupr"], reverse=True)[0]
    summary["best_by_auroc"] = sorted(rows, key=lambda x: x["auroc"], reverse=True)[0]
    save_json(str(root / "stage3_refine_summary.json"), summary)
    print(f"Saved: {root / 'stage3_refine_summary.json'}")
    for r in rows:
        print(f"{r['name']}: F1={r['f1']:.4f} AUPR={r['aupr']:.4f} AUROC={r['auroc']:.4f}")


if __name__ == "__main__":
    main()
