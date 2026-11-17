from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

from utils.config_io import load_config, save_json


def _stats(vals: List[float]) -> Dict[str, float]:
    if len(vals) == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(mean(vals)), "std": float(pstdev(vals)), "n": len(vals)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/stage6_smap_multichannel")
    parser.add_argument("--channels", default="A-1,E-1,P-1,T-1,D-1")
    parser.add_argument("--out", default="outputs/stage6_smap_multichannel/smap_multichannel_summary_5ch.json")
    args = parser.parse_args()

    root = Path(args.root)
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]

    per = {}
    for ch in channels:
        ch_dir = root / ch
        a0_eval = load_config(str(ch_dir / "A0" / "eval_metrics.json"))["metrics"]
        r5_eval = load_config(str(ch_dir / "R5" / "eval_metrics.json"))["metrics"]
        fusion = load_config(str(ch_dir / "fusion_refine_summary.json"))["runs"]
        a0_f = fusion["A0"]["best_by_aupr"]
        r5_f = fusion["R5"]["best_by_aupr"]
        per[ch] = {
            "same_label_protocol": int(a0_eval["y_true_pos"]) == int(r5_eval["y_true_pos"]) and int(a0_eval["y_true_neg"]) == int(r5_eval["y_true_neg"]),
            "A0_default": a0_eval,
            "R5_default": r5_eval,
            "A0_fusion_best_by_aupr": a0_f,
            "R5_fusion_best_by_aupr": r5_f,
        }

    agg = {}
    for name, key_path in {
        "A0_default_aupr": ("A0_default", "aupr"),
        "A0_default_auroc": ("A0_default", "auroc"),
        "R5_default_aupr": ("R5_default", "aupr"),
        "R5_default_auroc": ("R5_default", "auroc"),
        "A0_fusion_aupr": ("A0_fusion_best_by_aupr", "aupr"),
        "A0_fusion_auroc": ("A0_fusion_best_by_aupr", "auroc"),
        "R5_fusion_aupr": ("R5_fusion_best_by_aupr", "aupr"),
        "R5_fusion_auroc": ("R5_fusion_best_by_aupr", "auroc"),
    }.items():
        vals = [float(per[ch][key_path[0]][key_path[1]]) for ch in channels]
        agg[name] = _stats(vals)

    wins = {
        "r5_default_gt_a0_default_aupr": 0,
        "r5_default_gt_a0_default_auroc": 0,
        "r5_fusion_gt_a0_fusion_aupr": 0,
        "r5_fusion_gt_a0_fusion_auroc": 0,
    }
    for ch in channels:
        p = per[ch]
        if float(p["R5_default"]["aupr"]) > float(p["A0_default"]["aupr"]):
            wins["r5_default_gt_a0_default_aupr"] += 1
        if float(p["R5_default"]["auroc"]) > float(p["A0_default"]["auroc"]):
            wins["r5_default_gt_a0_default_auroc"] += 1
        if float(p["R5_fusion_best_by_aupr"]["aupr"]) > float(p["A0_fusion_best_by_aupr"]["aupr"]):
            wins["r5_fusion_gt_a0_fusion_aupr"] += 1
        if float(p["R5_fusion_best_by_aupr"]["auroc"]) > float(p["A0_fusion_best_by_aupr"]["auroc"]):
            wins["r5_fusion_gt_a0_fusion_auroc"] += 1

    out = {"channels": channels, "per_channel": per, "aggregate": agg, "wins": wins}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(str(out_path), out)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
