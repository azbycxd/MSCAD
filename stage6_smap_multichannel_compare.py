from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

from utils.config_io import load_config, save_json


def run_cmd(cmd: List[str], cwd: str) -> None:
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    p = subprocess.run(cmd, cwd=cwd, env=env)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def _stats(vals: List[float]) -> Dict[str, float]:
    if len(vals) == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(mean(vals)), "std": float(pstdev(vals)), "n": len(vals)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="D:/workspace/python/mscad")
    parser.add_argument("--base-config", default="configs/smap.yaml")
    parser.add_argument("--channels", default="A-1,E-1,P-1,T-1,M-6")
    parser.add_argument("--out-root", default="outputs/stage6_smap_multichannel")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--short-window", type=int, default=16)
    args = parser.parse_args()

    cwd = args.workdir
    py = sys.executable
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    base_cfg = load_config(args.base_config)
    out_root = Path(cwd) / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "channels": channels,
        "protocol": {
            "window_size": int(args.window_size),
            "short_window": int(args.short_window),
            "stride_train": 1,
            "stride_test": 1,
            "fixed_same_window_for_A0_R5": True,
        },
        "per_channel": {},
    }

    for ch in channels:
        ch_dir = out_root / ch
        ch_dir.mkdir(parents=True, exist_ok=True)

        # Shared protocol on this channel
        common = copy.deepcopy(base_cfg)
        common["data"]["channel"] = ch
        common["data"]["window_size"] = int(args.window_size)
        common["data"]["short_window"] = int(args.short_window)
        common["data"]["stride_train"] = 1
        common["data"]["stride_test"] = 1
        common["train"]["epochs"] = int(args.epochs)
        common["train"]["patience"] = int(args.patience)

        # A0
        a0 = copy.deepcopy(common)
        a0["output"]["dir"] = f"{args.out_root}/{ch}/A0"
        a0["train"]["lambda_s"] = 0.0
        a0["train"]["lambda_l"] = 0.0
        a0["train"]["structure_warmup_epochs"] = 0
        a0["train"]["structure_pairs"] = {"enabled": False}
        a0["eval"]["alpha"] = 1.0
        a0["eval"]["beta"] = 0.0
        a0["eval"]["gamma"] = 0.0
        a0_cfg = ch_dir / "a0_config.yaml"
        save_json(str(a0_cfg), a0)
        run_cmd([py, "train.py", "--config", str(a0_cfg)], cwd)
        run_cmd([py, "evaluate.py", "--config", str(a0_cfg)], cwd)
        a0_ckpt = str((Path(cwd) / a0["output"]["dir"] / "best.pt").resolve())
        a0_eval = load_config(str(Path(cwd) / a0["output"]["dir"] / "eval_metrics.json"))

        # R5-like
        r5 = copy.deepcopy(common)
        r5["output"]["dir"] = f"{args.out_root}/{ch}/R5"
        r5["train"]["lambda_s"] = 0.05
        r5["train"]["lambda_l"] = 0.02
        r5["train"]["structure_warmup_epochs"] = 10
        r5["train"]["structure_pairs"] = {"enabled": True, "topk": 15, "min_abs_corr": 0.92}
        r5["eval"]["alpha"] = 0.5
        r5["eval"]["beta"] = 0.25
        r5["eval"]["gamma"] = 0.25
        r5_cfg = ch_dir / "r5_config.yaml"
        save_json(str(r5_cfg), r5)
        run_cmd([py, "train.py", "--config", str(r5_cfg), "--init-ckpt", a0_ckpt], cwd)
        run_cmd([py, "evaluate.py", "--config", str(r5_cfg)], cwd)
        r5_ckpt = str((Path(cwd) / r5["output"]["dir"] / "best.pt").resolve())
        r5_eval = load_config(str(Path(cwd) / r5["output"]["dir"] / "eval_metrics.json"))

        # Fusion refine on the same channel/protocol for BOTH A0 and R5
        fusion_out = f"{args.out_root}/{ch}/fusion_refine_summary.json"
        run_cmd(
            [py, "stage4_fusion_refine.py", "--out", fusion_out, "--ckpts", a0_ckpt, r5_ckpt],
            cwd,
        )
        fusion = load_config(str(Path(cwd) / fusion_out))
        fusion_runs = fusion["runs"]
        a0_name = Path(a0_ckpt).parent.name
        r5_name = Path(r5_ckpt).parent.name

        same_labels = (
            int(a0_eval["metrics"]["y_true_pos"]) == int(r5_eval["metrics"]["y_true_pos"])
            and int(a0_eval["metrics"]["y_true_neg"]) == int(r5_eval["metrics"]["y_true_neg"])
        )

        summary["per_channel"][ch] = {
            "same_label_protocol": same_labels,
            "A0_default": a0_eval["metrics"],
            "R5_default": r5_eval["metrics"],
            "A0_fusion_best_by_f1": fusion_runs[a0_name]["best_by_f1"],
            "R5_fusion_best_by_f1": fusion_runs[r5_name]["best_by_f1"],
            "A0_fusion_best_by_aupr": fusion_runs[a0_name]["best_by_aupr"],
            "R5_fusion_best_by_aupr": fusion_runs[r5_name]["best_by_aupr"],
        }

    # Aggregate mean ± std across channels
    rows = summary["per_channel"]
    agg = {}
    for key, path in {
        "A0_default_aupr": ("A0_default", "aupr"),
        "A0_default_auroc": ("A0_default", "auroc"),
        "R5_default_aupr": ("R5_default", "aupr"),
        "R5_default_auroc": ("R5_default", "auroc"),
        "A0_fusion_aupr": ("A0_fusion_best_by_aupr", "aupr"),
        "A0_fusion_auroc": ("A0_fusion_best_by_aupr", "auroc"),
        "R5_fusion_aupr": ("R5_fusion_best_by_aupr", "aupr"),
        "R5_fusion_auroc": ("R5_fusion_best_by_aupr", "auroc"),
    }.items():
        vals = [float(rows[ch][path[0]][path[1]]) for ch in channels]
        agg[key] = _stats(vals)
    # Win counts by channel (AUPR/AUROC)
    wins = {"r5_default_gt_a0_default_aupr": 0, "r5_default_gt_a0_default_auroc": 0, "r5_fusion_gt_a0_fusion_aupr": 0, "r5_fusion_gt_a0_fusion_auroc": 0}
    for ch in channels:
        r = rows[ch]
        if float(r["R5_default"]["aupr"]) > float(r["A0_default"]["aupr"]):
            wins["r5_default_gt_a0_default_aupr"] += 1
        if float(r["R5_default"]["auroc"]) > float(r["A0_default"]["auroc"]):
            wins["r5_default_gt_a0_default_auroc"] += 1
        if float(r["R5_fusion_best_by_aupr"]["aupr"]) > float(r["A0_fusion_best_by_aupr"]["aupr"]):
            wins["r5_fusion_gt_a0_fusion_aupr"] += 1
        if float(r["R5_fusion_best_by_aupr"]["auroc"]) > float(r["A0_fusion_best_by_aupr"]["auroc"]):
            wins["r5_fusion_gt_a0_fusion_auroc"] += 1
    summary["aggregate"] = agg
    summary["wins"] = wins

    out_file = out_root / "smap_multichannel_summary.json"
    save_json(str(out_file), summary)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
