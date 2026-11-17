from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

from utils.config_io import load_config, save_json


def _fmt_stats(vals: List[float]) -> Dict[str, float]:
    if len(vals) == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(mean(vals)), "std": float(pstdev(vals)), "n": len(vals)}


def _collect_metric(items: List[dict], path: List[str]) -> List[float]:
    out = []
    for it in items:
        cur = it
        ok = True
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok:
            out.append(float(cur))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/correlation_suite")
    parser.add_argument("--suffix", default="v3")
    parser.add_argument("--seeds", nargs="+", default=["42", "123", "2026"])
    parser.add_argument("--presets", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--out", default="outputs/stage4_protocol_repeat/stage4_protocol_repeat_summary.json")
    args = parser.parse_args()

    root = Path(args.out_dir)
    records = {"a0": {}, "r5": {}}
    for model in ["a0", "r5"]:
        for preset in args.presets:
            rows = []
            for s in args.seeds:
                p = root / f"{model}_{preset}_s{s}_{args.suffix}" / "correlation_suite_results.json"
                rows.append(load_config(str(p)))
            records[model][preset] = rows

    summary = {"seeds": args.seeds, "presets": args.presets, "models": ["a0", "r5"], "sections": {}}
    for preset in args.presets:
        sec = {}
        for model in ["a0", "r5"]:
            rows = records[model][preset]
            sec[model] = {
                "syn1_mscad_multi": {
                    "f1": _fmt_stats(_collect_metric(rows, ["dataset_syn_1", "MSCAD-multi", "f1"])),
                    "aupr": _fmt_stats(_collect_metric(rows, ["dataset_syn_1", "MSCAD-multi", "aupr"])),
                    "auroc": _fmt_stats(_collect_metric(rows, ["dataset_syn_1", "MSCAD-multi", "auroc"])),
                    "cohen_d": _fmt_stats(_collect_metric(rows, ["dataset_syn_1", "MSCAD-multi", "score_gap", "cohen_d"])),
                },
                "syn2_mscad_multi": {
                    "f1": _fmt_stats(_collect_metric(rows, ["dataset_syn_2", "MSCAD-multi", "f1"])),
                    "aupr": _fmt_stats(_collect_metric(rows, ["dataset_syn_2", "MSCAD-multi", "aupr"])),
                    "auroc": _fmt_stats(_collect_metric(rows, ["dataset_syn_2", "MSCAD-multi", "auroc"])),
                    "cohen_d": _fmt_stats(_collect_metric(rows, ["dataset_syn_2", "MSCAD-multi", "score_gap", "cohen_d"])),
                },
            }

        # Win counts on Syn1 for the model under study (MSCAD-multi).
        a0_rows = records["a0"][preset]
        r5_rows = records["r5"][preset]
        wins = {"r5_gt_a0_aupr_syn1": 0, "r5_gt_a0_auroc_syn1": 0, "r5_gt_a0_cohen_d_syn1": 0}
        for a, r in zip(a0_rows, r5_rows):
            a1 = a["dataset_syn_1"]["MSCAD-multi"]
            r1 = r["dataset_syn_1"]["MSCAD-multi"]
            if float(r1["aupr"]) > float(a1["aupr"]):
                wins["r5_gt_a0_aupr_syn1"] += 1
            if float(r1["auroc"]) > float(a1["auroc"]):
                wins["r5_gt_a0_auroc_syn1"] += 1
            if float(r1["score_gap"]["cohen_d"]) > float(a1["score_gap"]["cohen_d"]):
                wins["r5_gt_a0_cohen_d_syn1"] += 1
        sec["wins_syn1_mscad_multi"] = wins
        summary["sections"][preset] = sec

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(str(out_path), summary)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
