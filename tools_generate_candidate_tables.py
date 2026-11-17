import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAPPING_OUT = ROOT / "results" / "method_alias_mapping.csv"
TABLE_DIR = ROOT / "results" / "candidate_tables"
CSV_OUT = TABLE_DIR / "mscad_candidate_methods.csv"
MD_OUT = TABLE_DIR / "mscad_candidate_methods.md"


MAPPING_ROWS = [
    {
        "original_name": "MSCAD main",
        "paper_alias": "MSCAD-Joint",
        "role": "joint training with reconstruction plus short/long structure losses",
        "evidence_status": "confirmed_ablation",
        "source_result_file": "outputs/swat_main/eval_metrics.json",
        "notes": "Keep original experiment directory. Paper alias is for candidate writing only; ablation evidence also summarized in outputs/ablation_main_summary.json.",
    },
    {
        "original_name": "rec-only",
        "paper_alias": "MSCAD-Backbone",
        "role": "reconstruction-only backbone baseline",
        "evidence_status": "confirmed_backbone",
        "source_result_file": "outputs/swat_improve_v3_rec_only/eval_metrics.json",
        "notes": "Backbone/reference method; useful to separate architecture value from structure-score value.",
    },
    {
        "original_name": "struct-finetune",
        "paper_alias": "MSCAD-FT-candidate",
        "role": "two-stage weak-structure fine-tuning candidate",
        "evidence_status": "candidate_need_multiseed",
        "source_result_file": "outputs/swat_improve_v4_struct_finetune/eval_metrics.json",
        "notes": "Promising candidate because AUPR/AUROC improve, but needs multi-seed confirmation before final paper naming.",
    },
    {
        "original_name": "R5 fusion best",
        "paper_alias": "MSCAD-Score-candidate",
        "role": "score-level fusion candidate optimized by F1",
        "evidence_status": "candidate_need_sensitivity",
        "source_result_file": "outputs/stage4_fusion_refine/stage4_fusion_refine_summary.json",
        "notes": "Uses R5 checkpoint and stage4 score fusion search; candidate table only, not final paper result.",
    },
    {
        "original_name": "R6 best by AUPR",
        "paper_alias": "MSCAD-Score-candidate",
        "role": "score-level fusion candidate optimized by AUPR/AUROC",
        "evidence_status": "candidate_need_sensitivity",
        "source_result_file": "outputs/stage4_fusion_refine/stage4_fusion_refine_summary.json",
        "notes": "Uses R6 checkpoint and stage4 score fusion search; candidate table only, not final paper result.",
    },
]


def refuse_overwrite(paths):
    existing = [str(path.relative_to(ROOT)) for path in paths if path.exists()]
    if existing:
        raise SystemExit(
            "Refusing to overwrite existing files:\n"
            + "\n".join(f"- {path}" for path in existing)
        )


def load_eval_metrics(relative_path):
    data = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
    metrics = data.get("metrics", data)
    return {
        "F1": metrics["f1"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "AUPR": metrics["aupr"],
        "AUROC": metrics["auroc"],
    }


def load_stage4_metrics(original_name):
    data = json.loads(
        (ROOT / "outputs/stage4_fusion_refine/stage4_fusion_refine_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if original_name == "R5 fusion best":
        metrics = data["runs"]["R5_ws96_ss24_ls0.05_ll0.02_top15_corr0.92"]["best_by_f1"]
    elif original_name == "R6 best by AUPR":
        metrics = data["runs"]["R6_ws96_ss24_ls0.05_ll0.02_top25_corr0.88"]["best_by_aupr"]
    else:
        raise KeyError(original_name)
    return {
        "F1": metrics["f1"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "AUPR": metrics["aupr"],
        "AUROC": metrics["auroc"],
    }


def collect_candidate_rows():
    rows = []
    for mapping in MAPPING_ROWS:
        if mapping["original_name"].startswith("R5") or mapping["original_name"].startswith("R6"):
            metrics = load_stage4_metrics(mapping["original_name"])
        else:
            metrics = load_eval_metrics(mapping["source_result_file"])
        rows.append({**mapping, **metrics})
    return rows


def write_mapping():
    MAPPING_OUT.parent.mkdir(parents=True, exist_ok=True)
    with MAPPING_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "original_name",
                "paper_alias",
                "role",
                "evidence_status",
                "source_result_file",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(MAPPING_ROWS)


def write_candidate_csv(rows):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "original_name",
                "paper_alias",
                "role",
                "evidence_status",
                "F1",
                "Precision",
                "Recall",
                "AUPR",
                "AUROC",
                "source_result_file",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_candidate_md(rows):
    lines = [
        "# MSCAD Candidate Paper Table",
        "",
        "说明：本表只用于论文候选命名和候选结果整理，不作为最终论文表。原始实验目录和原始实验名保持不变。",
        "",
        "| Original name | Paper alias | Role | Evidence status | F1 | Precision | Recall | AUPR | AUROC | Source |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {original_name} | {paper_alias} | {role} | {evidence_status} | "
            "{F1:.4f} | {Precision:.4f} | {Recall:.4f} | {AUPR:.4f} | {AUROC:.4f} | "
            "`{source_result_file}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `MSCAD-Joint` 是 `MSCAD main` 的论文候选别名，不重命名 `outputs/swat_main/`。",
            "- `MSCAD-Backbone` 是 `rec-only` 的论文候选别名，用于表示仅重构主干。",
            "- `MSCAD-FT-candidate` 暂时只是候选名，需要多随机种子结果支撑。",
            "- `MSCAD-Score-candidate` 暂时用于 R5/R6 score-level fusion 候选，需要权重敏感性分析支撑。",
            "- 本目录下文件均为候选表，不覆盖 `outputs/` 中已有结果。",
        ]
    )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    refuse_overwrite([MAPPING_OUT, CSV_OUT, MD_OUT])
    rows = collect_candidate_rows()
    write_mapping()
    write_candidate_csv(rows)
    write_candidate_md(rows)
    print("Generated candidate alias files under results/.")


if __name__ == "__main__":
    main()
