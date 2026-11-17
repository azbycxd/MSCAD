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
    parser.add_argument("--init-ckpt", default="outputs/stage2_grid/A0_backbone_wo_struct/best.pt")
    args = parser.parse_args()

    cwd = args.workdir
    py = sys.executable
    base_cfg = load_config(args.base_config)
    init_ckpt = str((Path(cwd) / args.init_ckpt).resolve())
    root = Path(cwd) / "outputs" / "stage4_train_light"
    root.mkdir(parents=True, exist_ok=True)

    # Keep protocol fixed; only small training-side changes.
    runs = [
        ("L1_ls0.03_ll0.01_unfrozen", 0.03, 0.01, False),
        ("L2_ls0.04_ll0.01_unfrozen", 0.04, 0.01, False),
        ("L3_ls0.04_ll0.015_unfrozen", 0.04, 0.015, False),
        ("L4_ls0.03_ll0.01_freeze_backbone", 0.03, 0.01, True),
        ("L5_ls0.04_ll0.01_freeze_backbone", 0.04, 0.01, True),
        ("L6_ls0.04_ll0.015_freeze_backbone", 0.04, 0.015, True),
    ]
    summary: Dict[str, Any] = {"runs": {}, "init_ckpt": init_ckpt}
    rows = []
    for name, ls, ll, freeze in runs:
        cfg = copy.deepcopy(base_cfg)
        cfg["data"]["window_size"] = 96
        cfg["data"]["short_window"] = 24
        cfg["train"]["lambda_s"] = ls
        cfg["train"]["lambda_l"] = ll
        cfg["train"]["structure_warmup_epochs"] = 2
        cfg["train"]["structure_pairs"] = {"enabled": True, "topk": 15, "min_abs_corr": 0.92}
        cfg["train"]["epochs"] = 2
        cfg["train"]["patience"] = 1
        cfg["train"]["freeze_backbone"] = bool(freeze)
        cfg["train"]["unfreeze_keywords"] = ["backbone.decoder", "backbone.out", "backbone.skip"]
        cfg["eval"]["alpha"] = 0.5
        cfg["eval"]["beta"] = 0.25
        cfg["eval"]["gamma"] = 0.25
        cfg["output"]["dir"] = f"outputs/stage4_train_light/{name}"
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
    out_path = root / "stage4_train_light_summary.json"
    save_json(str(out_path), summary)
    print(f"Saved: {out_path}")
    for r in rows:
        print(f"{r['name']}: F1={r['f1']:.4f} AUPR={r['aupr']:.4f} AUROC={r['auroc']:.4f}")


if __name__ == "__main__":
    main()
