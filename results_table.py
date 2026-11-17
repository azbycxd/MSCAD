from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_eval_metrics(eval_json: Dict[str, Any]) -> Dict[str, Any]:
    m = eval_json.get("metrics", {})
    p = eval_json.get("profile", {})
    return {
        "F1": m.get("f1"),
        "Precision": m.get("precision"),
        "Recall": m.get("recall"),
        "AUPR": m.get("aupr"),
        "AUROC": m.get("auroc"),
        "Params": p.get("params"),
        "InferMsBatch": p.get("inference_ms_per_batch"),
        "GpuMemMB": p.get("gpu_peak_mem_mb"),
        "CpuRamMB": p.get("cpu_ram_mb"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mscad-eval", default="outputs/swat_main/eval_metrics.json")
    parser.add_argument("--baseline-summary", default="outputs/baselines/baselines_summary.json")
    parser.add_argument("--ablation-summary", default="outputs/ablation_main_summary.json")
    parser.add_argument("--out-csv", default="outputs/paper_tables/main_comparison.csv")
    parser.add_argument("--out-ablation-csv", default="outputs/paper_tables/ablation_table.csv")
    parser.add_argument("--out-md", default="outputs/paper_tables/main_comparison.md")
    args = parser.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    mscad_path = Path(args.mscad_eval)
    if mscad_path.exists():
        m = extract_eval_metrics(load_json(mscad_path))
        rows.append({"Model": "MSCAD (Main)", **m})

    baseline_path = Path(args.baseline_summary)
    if baseline_path.exists():
        bsum = load_json(baseline_path)
        for name, res in bsum.items():
            if "error" in res:
                rows.append({"Model": name, "Note": res["error"]})
                continue
            rows.append(
                {
                    "Model": name,
                    "F1": res.get("f1"),
                    "Precision": res.get("precision"),
                    "Recall": res.get("recall"),
                    "AUPR": res.get("aupr"),
                    "AUROC": res.get("auroc"),
                    "Params": res.get("params"),
                    "InferMsBatch": res.get("inference_ms_per_batch"),
                    "GpuMemMB": res.get("gpu_peak_mem_mb"),
                    "CpuRamMB": res.get("cpu_ram_mb"),
                }
            )

    headers = ["Model", "F1", "Precision", "Recall", "AUPR", "AUROC", "Params", "InferMsBatch", "GpuMemMB", "CpuRamMB", "Note"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # markdown
    md_path = Path(args.out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        vals = [r.get(h, "") for h in headers]
        lines.append("| " + " | ".join("" if v is None else str(v) for v in vals) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    abl_rows: List[Dict[str, Any]] = []
    abl_path = Path(args.ablation_summary)
    if abl_path.exists():
        ab = load_json(abl_path)
        for name, res in ab.items():
            m = res.get("metrics", {})
            abl_rows.append(
                {
                    "Variant": name,
                    "F1": m.get("f1"),
                    "Precision": m.get("precision"),
                    "Recall": m.get("recall"),
                    "AUPR": m.get("aupr"),
                    "AUROC": m.get("auroc"),
                }
            )

    out_ab_csv = Path(args.out_ablation_csv)
    out_ab_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_ab_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Variant", "F1", "Precision", "Recall", "AUPR", "AUROC"])
        w.writeheader()
        for r in abl_rows:
            w.writerow(r)

    print(f"Saved main table: {out_csv}")
    print(f"Saved markdown table: {md_path}")
    print(f"Saved ablation table: {out_ab_csv}")


if __name__ == "__main__":
    main()
