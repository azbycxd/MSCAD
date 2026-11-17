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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cmapss_ncmapss.yaml")
    parser.add_argument("--workdir", default="D:/workspace/python/mscad")
    parser.add_argument("--out-dir", default="outputs/cmapss_compare")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    cwd = args.workdir
    py = sys.executable
    base_cfg = load_config(args.config)
    root = Path(cwd) / args.out_dir
    root.mkdir(parents=True, exist_ok=True)

    # A0: backbone-only
    a0 = copy.deepcopy(base_cfg)
    a0["output"]["dir"] = f"{args.out_dir}/A0_default"
    a0["train"]["lambda_s"] = 0.0
    a0["train"]["lambda_l"] = 0.0
    a0["train"]["structure_warmup_epochs"] = 0
    a0["train"]["structure_pairs"] = {"enabled": False}
    a0["eval"]["alpha"] = 1.0
    a0["eval"]["beta"] = 0.0
    a0["eval"]["gamma"] = 0.0
    a0_cfg = root / "a0_config.yaml"
    save_json(str(a0_cfg), a0)

    # R5: fixed structure protocol on CMAPSS
    r5 = copy.deepcopy(base_cfg)
    r5["output"]["dir"] = f"{args.out_dir}/R5_default"
    r5_cfg = root / "r5_config.yaml"
    save_json(str(r5_cfg), r5)

    if not args.skip_train:
        run_cmd([py, "train.py", "--config", str(a0_cfg)], cwd)
    run_cmd([py, "evaluate.py", "--config", str(a0_cfg)], cwd)
    a0_ckpt = str((Path(cwd) / a0["output"]["dir"] / "best.pt").resolve())

    if not args.skip_train:
        run_cmd([py, "train.py", "--config", str(r5_cfg), "--init-ckpt", a0_ckpt], cwd)
    run_cmd([py, "evaluate.py", "--config", str(r5_cfg)], cwd)
    r5_ckpt = str((Path(cwd) / r5["output"]["dir"] / "best.pt").resolve())

    fusion_out = f"{args.out_dir}/r5_fusion_refine_summary.json"
    run_cmd(
        [
            py,
            "stage4_fusion_refine.py",
            "--out",
            fusion_out,
            "--ckpts",
            r5_ckpt,
        ],
        cwd,
    )

    a0_eval = load_config(str(Path(cwd) / a0["output"]["dir"] / "eval_metrics.json"))
    r5_eval = load_config(str(Path(cwd) / r5["output"]["dir"] / "eval_metrics.json"))
    r5_fusion = load_config(str(Path(cwd) / fusion_out))
    run_name = next(iter(r5_fusion["runs"].keys()))

    summary: Dict[str, Any] = {
        "dataset_config": args.config,
        "A0_default": {
            "metrics": a0_eval["metrics"],
            "profile": a0_eval["profile"],
        },
        "R5_default": {
            "metrics": r5_eval["metrics"],
            "profile": r5_eval["profile"],
        },
        "R5_fusion_best_by_f1": r5_fusion["runs"][run_name]["best_by_f1"],
        "R5_fusion_best_by_aupr": r5_fusion["runs"][run_name]["best_by_aupr"],
        "R5_fusion_best_by_auroc": r5_fusion["runs"][run_name]["best_by_auroc"],
    }
    out_summary = root / "cmapss_compare_summary.json"
    save_json(str(out_summary), summary)
    print(f"Saved: {out_summary}")


if __name__ == "__main__":
    main()
