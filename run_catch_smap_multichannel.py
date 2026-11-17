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
from run_catch_compare import refine_fusion


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


def _build_cfg(base_cfg: dict, channel: str, variant: str, out_dir: str, epochs: int, patience: int) -> dict:
    c = copy.deepcopy(base_cfg)
    c["data"]["channel"] = channel
    c["train"]["epochs"] = int(epochs)
    c["train"]["patience"] = int(patience)
    c["model"]["variant"] = variant
    c["output"]["dir"] = out_dir
    return c


def _collect_metrics(path: Path) -> dict:
    return load_config(str(path / "eval_metrics.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="D:/workspace/python/mscad")
    parser.add_argument("--base-config", default="configs/catch_smap.yaml")
    parser.add_argument("--channels", default="A-1,E-2,D-1,P-1,S-1")
    parser.add_argument("--out-root", default="outputs/catch_smap_multichannel_5ch")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    cwd = args.workdir
    py = sys.executable
    base_cfg = load_config(args.base_config)
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    out_root = Path(cwd) / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "channels": channels,
        "config": args.base_config,
        "per_channel": {},
    }

    for ch in channels:
        ch_dir = out_root / ch
        ch_dir.mkdir(parents=True, exist_ok=True)

        run_data: Dict[str, Any] = {}
        for v in ["core", "full"]:
            out_dir = f"{args.out_root}/{ch}/{v}"
            cfg = _build_cfg(base_cfg, channel=ch, variant=v, out_dir=out_dir, epochs=args.epochs, patience=args.patience)
            cfg_path = ch_dir / f"catch_{v}_config.yaml"
            save_json(str(cfg_path), cfg)
            if not args.skip_train:
                run_cmd([py, "train_catch.py", "--config", str(cfg_path)], cwd)
            run_cmd([py, "evaluate_catch.py", "--config", str(cfg_path)], cwd)
            eval_res = _collect_metrics(Path(cwd) / out_dir)
            run_data[v] = eval_res

        # fusion refine for core/full
        for v in ["core", "full"]:
            rf = refine_fusion(
                run_dir=Path(cwd) / args.out_root / ch / v,
                q_grid=[0.93, 0.95, 0.97],
                smooth_grid=[1, 3, 5],
                alpha_grid=[0.3, 0.4, 0.5, 0.6, 0.7],
            )
            run_data[f"{v}_fusion_best_by_f1"] = rf.get("best_by_f1")
            run_data[f"{v}_fusion_best_by_aupr"] = rf.get("best_by_aupr")

        # protocol consistency check
        m_core = run_data["core"]["metrics"]
        m_full = run_data["full"]["metrics"]
        run_data["same_label_protocol"] = (
            int(m_core["y_true_pos"]) == int(m_full["y_true_pos"])
            and int(m_core["y_true_neg"]) == int(m_full["y_true_neg"])
        )
        summary["per_channel"][ch] = run_data

    # aggregate
    agg = {}
    for key, path in {
        "core_aupr": ("core", "metrics", "aupr"),
        "core_auroc": ("core", "metrics", "auroc"),
        "full_aupr": ("full", "metrics", "aupr"),
        "full_auroc": ("full", "metrics", "auroc"),
        "core_fusion_aupr": ("core_fusion_best_by_aupr", "aupr"),
        "core_fusion_auroc": ("core_fusion_best_by_aupr", "auroc"),
        "full_fusion_aupr": ("full_fusion_best_by_aupr", "aupr"),
        "full_fusion_auroc": ("full_fusion_best_by_aupr", "auroc"),
    }.items():
        vals = []
        for ch in channels:
            row = summary["per_channel"][ch]
            if len(path) == 3:
                vals.append(float(row[path[0]][path[1]][path[2]]))
            else:
                vals.append(float(row[path[0]][path[1]]))
        agg[key] = _stats(vals)

    wins = {
        "full_gt_core_aupr": 0,
        "full_gt_core_auroc": 0,
        "full_fusion_gt_core_fusion_aupr": 0,
        "full_fusion_gt_core_fusion_auroc": 0,
    }
    for ch in channels:
        row = summary["per_channel"][ch]
        if float(row["full"]["metrics"]["aupr"]) > float(row["core"]["metrics"]["aupr"]):
            wins["full_gt_core_aupr"] += 1
        if float(row["full"]["metrics"]["auroc"]) > float(row["core"]["metrics"]["auroc"]):
            wins["full_gt_core_auroc"] += 1
        if float(row["full_fusion_best_by_aupr"]["aupr"]) > float(row["core_fusion_best_by_aupr"]["aupr"]):
            wins["full_fusion_gt_core_fusion_aupr"] += 1
        if float(row["full_fusion_best_by_aupr"]["auroc"]) > float(row["core_fusion_best_by_aupr"]["auroc"]):
            wins["full_fusion_gt_core_fusion_auroc"] += 1

    summary["aggregate"] = agg
    summary["wins"] = wins
    out_file = out_root / "catch_smap_multichannel_summary.json"
    save_json(str(out_file), summary)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
