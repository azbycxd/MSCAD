"""Build the final thesis ablation figures and audit metadata from frozen CSVs only.

This script is deliberately presentation-only: it does not import model code, read
datasets, train, infer, tune, or recompute experiment metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "thesis_ablation"
FIG_DIR = ROOT / "outputs" / "thesis_figures" / "ablation"
DOC_DIR = ROOT / "docs" / "thesis"

SOURCES = {
    "innovation1": OUT / "innovation1" / "innovation1_cross_dataset_ablation.csv",
    "innovation2_real": OUT / "innovation2" / "summary" / "innovation2_real_ablation.csv",
    "innovation2_mechanism": OUT / "innovation2" / "summary" / "innovation2_mechanism_ablation.csv",
    "normal_tail": OUT / "innovation3" / "normal_tail_balance.csv",
    "fusion": OUT / "innovation3" / "fusion_detection_ablation.csv",
}

REQUIRED_STATUSES = {
    DOC_DIR / "ablation_a1_innovation2_results.md": "ABLATION_A1_INNOVATION2_COMPLETED",
    DOC_DIR / "ablation_a2_innovation3_results.md": "ABLATION_A2_INNOVATION3_COMPLETED",
    DOC_DIR / "ablation_a3b_innovation1_hai_results.md": "ABLATION_A3_INNOVATION1_COMPLETED",
}

STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.3,
    "lines.markersize": 6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f(value: str) -> float:
    return float(value)


def add_audit(
    audit: list[dict[str, object]], figure: str, panel: str, dataset: str,
    variant: str, metric: str, value: float, source: Path,
) -> None:
    audit.append({
        "figure": figure,
        "panel": panel,
        "dataset": dataset,
        "variant": variant,
        "metric": metric,
        "value": value,
        "source_csv": source.relative_to(ROOT).as_posix(),
    })


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_lightweight(rows: list[dict[str, str]], audit: list[dict[str, object]]) -> None:
    fig_name = "fig_ablation_lightweight_rec"
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25))
    dataset_style = {
        "SWaT": {"marker": "o", "linestyle": "-", "label": "SWaT"},
        "HAI21": {"marker": "s", "linestyle": "--", "label": "HAI 21.03"},
    }
    variant_color = {"R0 Final": "#666666", "R1": "#111111", "R2": "#aaaaaa"}
    label_offsets = {
        ("a", "R0 Final"): (5, -12),
        ("a", "R1"): (5, 5),
        ("a", "R2"): (5, 5),
        ("b", "R0 Final"): (5, 5),
        ("b", "R1"): (-22, 5),
        ("b", "R2"): (5, -12),
    }
    for panel, x_field, x_label in [
        ("a", "params", "Trainable parameters"),
        ("b", "latency_mean_ms", "Mean inference latency (ms)"),
    ]:
        ax = axes[0 if panel == "a" else 1]
        for dataset, ds_style in dataset_style.items():
            ds_rows = [row for row in rows if row["dataset"] == dataset]
            for row in ds_rows:
                x = f(row[x_field])
                y = f(row["aupr"])
                ax.scatter(
                    x, y, color=variant_color[row["variant"]], edgecolor="black",
                    marker=ds_style["marker"], linewidth=0.7, zorder=3,
                )
                ax.annotate(row["variant"].replace(" Final", ""), (x, y),
                            xytext=label_offsets[(panel, row["variant"])],
                            textcoords="offset points", fontsize=7)
                add_audit(audit, fig_name, panel, dataset, row["variant"], x_field, x, SOURCES["innovation1"])
                add_audit(audit, fig_name, panel, dataset, row["variant"], "aupr", y, SOURCES["innovation1"])
            xs = [f(row[x_field]) for row in ds_rows]
            ys = [f(row["aupr"]) for row in ds_rows]
            ax.plot(xs, ys, color="#444444", marker=ds_style["marker"],
                    linestyle=ds_style["linestyle"], alpha=0.55, label=ds_style["label"])
        ax.set_xlabel(x_label)
        ax.set_ylabel("AUPR")
        ax.set_ylim(0, 1)
        max_x = max(f(row[x_field]) for row in rows)
        ax.set_xlim(0, max_x * 1.10)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
        ax.set_title(f"({panel}) " + ("Performance vs parameters" if panel == "a" else "Performance vs latency"))
    axes[0].legend(frameon=False, loc="lower right")
    save_figure(fig, fig_name)


def grouped_bars(ax: plt.Axes, rows: list[dict[str, str]], metrics: list[str], title: str,
                 fig_name: str, panel: str, source: Path, audit: list[dict[str, object]]) -> None:
    variants = ["D0 Final", "D1 Short-lag", "D2 No-Gate"]
    x = np.arange(len(variants))
    width = 0.34
    colors = ["#333333", "#b0b0b0"]
    hatches = ["///", "..."]
    for idx, metric in enumerate(metrics):
        values = []
        for variant in variants:
            row = next(row for row in rows if row["variant"] == variant)
            value = f(row[metric])
            values.append(value)
            add_audit(audit, fig_name, panel, row["dataset"], variant, metric, value, source)
        ax.bar(x + (idx - 0.5) * width, values, width, color=colors[idx], edgecolor="black",
               hatch=hatches[idx], linewidth=0.7, label=metric.upper())
    ax.set_xticks(x, ["D0 Final", "D1 Short-lag", "D2 No-Gate"], rotation=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")


def figure_dual(real_rows: list[dict[str, str]], mechanism_rows: list[dict[str, str]],
                audit: list[dict[str, object]]) -> None:
    fig_name = "fig_ablation_dual_view"
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.35))
    hai = [row for row in real_rows if row["dataset"] == "HAI" and row["variant"] in {"D0 Final", "D1 Short-lag", "D2 No-Gate"}]
    controlled = [row for row in mechanism_rows if row["dataset"] == "SWaT controlled"]
    grouped_bars(axes[0], hai, ["aupr", "f1"], "(a) HAI real attacks",
                 fig_name, "a", SOURCES["innovation2_real"], audit)
    grouped_bars(axes[1], controlled, ["auroc", "aupr"], "(b) Controlled SWaT relation breaking",
                 fig_name, "b", SOURCES["innovation2_mechanism"], audit)
    save_figure(fig, fig_name)


def figure_tail(tail_rows: list[dict[str, str]], fusion_rows: list[dict[str, str]],
                audit: list[dict[str, object]]) -> None:
    fig_name = "fig_ablation_tail_calibration"
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.45))
    datasets = ["SWaT", "HAI21", "WADI.A2"]
    variants = ["F3 Robust-Z", "F4 Tail-Calibrated"]
    labels, rec, dual = [], [], []
    for dataset in datasets:
        for variant in variants:
            row = next(row for row in tail_rows if row["dataset"] == dataset and row["variant"] == variant)
            labels.append(f"{dataset}\n{variant.split()[0]}")
            rec_value = f(row["top1_rec_share"])
            dual_value = f(row["top1_dual_share"])
            rec.append(rec_value)
            dual.append(dual_value)
            add_audit(audit, fig_name, "a", dataset, variant, "top1_rec_share", rec_value, SOURCES["normal_tail"])
            add_audit(audit, fig_name, "a", dataset, variant, "top1_dual_share", dual_value, SOURCES["normal_tail"])
    x = np.arange(len(labels))
    axes[0].bar(x, rec, color="white", edgecolor="black", hatch="///", linewidth=0.7, label="REC")
    axes[0].bar(x, dual, bottom=rec, color="#777777", edgecolor="black", hatch="...", linewidth=0.7, label="DUAL")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("Normal top-1% evidence share")
    axes[0].set_title("(a) Evidence contribution")
    axes[0].grid(axis="y", color="#d9d9d9", linewidth=0.6)
    axes[0].legend(frameon=False, loc="upper center", ncol=2)

    fusion_variants = ["F0 REC", "F1 DUAL", "F4 Tail-Calibrated"]
    x = np.arange(2)
    width = 0.24
    colors = ["white", "#777777", "#222222"]
    hatches = ["///", "...", "xxx"]
    for idx, variant in enumerate(fusion_variants):
        values = []
        for dataset in ["SWaT", "HAI21"]:
            row = next(row for row in fusion_rows if row["dataset"] == dataset and row["variant"] == variant)
            value = f(row["aupr"])
            values.append(value)
            add_audit(audit, fig_name, "b", dataset, variant, "aupr", value, SOURCES["fusion"])
        axes[1].bar(x + (idx - 1) * width, values, width, color=colors[idx], edgecolor="black",
                    hatch=hatches[idx], linewidth=0.7, label=variant.replace("F4 ", ""))
    axes[1].set_xticks(x, ["SWaT", "HAI 21.03"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("AUPR")
    axes[1].set_title("(b) Detection consequence")
    axes[1].grid(axis="y", color="#d9d9d9", linewidth=0.6)
    axes[1].legend(frameon=False, loc="upper center")
    save_figure(fig, fig_name)


def bold(value: float, maximum: float) -> str:
    rendered = f"{value:.6f}"
    return rf"\textbf{{{rendered}}}" if abs(value - maximum) < 1e-12 else rendered


def latex_tables(data: dict[str, list[dict[str, str]]]) -> str:
    lines = [
        "% Auto-generated from frozen final CSVs; no model metric was recomputed.",
        "% Requires: \\usepackage{booktabs}",
    ]
    rows = data["innovation1"]
    lines += [
        "\\begin{table}[htbp]", "\\centering", "\\caption{Table Abl-1: Innovation-1 structural and efficiency ablation.}",
        "\\label{tab:abl1}", "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "Dataset & Variant & Params & AUROC & AUPR & F1 & FPR & Latency (ms) \\\\", "\\midrule",
    ]
    for dataset in ["SWaT", "HAI21"]:
        subset = [row for row in rows if row["dataset"] == dataset]
        max_auroc = max(f(row["auroc"]) for row in subset)
        max_aupr = max(f(row["aupr"]) for row in subset)
        for row in subset:
            lines.append(
                f"{dataset} & {row['variant']} & {int(f(row['params'])):,} & {bold(f(row['auroc']), max_auroc)} & "
                f"{bold(f(row['aupr']), max_aupr)} & {f(row['f1']):.6f} & {f(row['fpr']):.6f} & {f(row['latency_mean_ms']):.6f} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    rows = [row for row in data["innovation2_real"] if row["variant"] in {"D0 Final", "D1 Short-lag", "D2 No-Gate"}]
    lines += [
        "\\begin{table}[htbp]", "\\centering", "\\caption{Table Abl-2: Innovation-2 real-attack ablation.}",
        "\\label{tab:abl2}", "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "Dataset & Variant & Targets & AUROC & AUPR & F1 & FPR & Event hits \\\\", "\\midrule",
    ]
    for dataset in ["SWaT", "HAI"]:
        subset = [row for row in rows if row["dataset"] == dataset]
        max_auroc = max(f(row["auroc"]) for row in subset)
        max_aupr = max(f(row["aupr"]) for row in subset)
        for row in subset:
            lines.append(
                f"{dataset} & {row['variant']} & {int(f(row['selected_target_count']))} & {bold(f(row['auroc']), max_auroc)} & "
                f"{bold(f(row['aupr']), max_aupr)} & {f(row['f1']):.6f} & {f(row['fpr']):.6f} & {int(f(row['event_hit_count']))} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    rows = data["innovation2_mechanism"]
    lines += [
        "\\begin{table}[htbp]", "\\centering", "\\caption{Table Abl-3: Innovation-2 mechanism ablation.}",
        "\\label{tab:abl3}", "\\begin{tabular}{llrrrr}", "\\toprule",
        "Dataset & Variant & AUROC & AUPR & Recall & FPR \\\\", "\\midrule",
    ]
    for row in rows:
        lines.append(f"{row['dataset']} & {row['variant']} & {f(row['auroc']):.6f} & {f(row['aupr']):.6f} & {f(row['recall']):.6f} & {f(row['fpr']):.6f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    rows = [row for row in data["fusion"] if row["dataset"] in {"SWaT", "HAI21"}]
    lines += [
        "\\begin{table}[htbp]", "\\centering", "\\caption{Table Abl-4: Innovation-3 fusion ablation.}",
        "\\label{tab:abl4}", "\\begin{tabular}{llrrrrr}", "\\toprule",
        "Dataset & Variant & AUROC & AUPR & F1 & FPR & Role \\\\", "\\midrule",
    ]
    for dataset in ["SWaT", "HAI21"]:
        subset = [row for row in rows if row["dataset"] == dataset]
        max_aupr = max(f(row["aupr"]) for row in subset)
        for row in subset:
            lines.append(
                f"{dataset} & {row['variant']} & {f(row['auroc']):.6f} & {bold(f(row['aupr']), max_aupr)} & "
                f"{f(row['f1']):.6f} & {f(row['fpr']):.6f} & {row['role'].replace('_', ' ')} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def inventory_rows() -> list[dict[str, object]]:
    paths: set[Path] = set()
    paths.update(DOC_DIR.glob("ablation*"))
    paths.update((ROOT / "scripts" / "thesis" / "03_ablation").rglob("*"))
    paths.update((ROOT / "tests").glob("test_ablation*"))
    paths.update(OUT.rglob("*"))
    paths.update(FIG_DIR.glob("fig_ablation_*"))
    rows: list[dict[str, object]] = []
    final_csv_names = {
        "innovation1_swat_ablation.csv", "innovation1_hai_ablation.csv",
        "innovation1_cross_dataset_ablation.csv", "innovation2_real_ablation.csv",
        "innovation2_mechanism_ablation.csv", "fusion_detection_ablation.csv",
        "normal_tail_balance.csv", "gate_stage_details.csv",
    }
    for path in sorted((p for p in paths if p.is_file()), key=lambda p: p.as_posix().lower()):
        rel = path.relative_to(ROOT).as_posix()
        suffix = path.suffix.lower()
        size = path.stat().st_size
        if rel.startswith("docs/thesis/"):
            category, action, reason = "THESIS_DOC", "COMMIT", "Human-readable frozen protocol/result wording"
        elif rel.startswith("scripts/"):
            category, action, reason = "RUNNER_SOURCE", "COMMIT", "Reviewable reproducibility source"
        elif rel.startswith("tests/"):
            category, action, reason = "TEST_SOURCE", "COMMIT", "Deterministic protocol test"
        elif rel.startswith("outputs/thesis_figures/"):
            category, action, reason = "THESIS_FIGURE", "REVIEW", "Generated figure; repository policy requires selection review"
        elif suffix in {".pt", ".pth", ".ckpt"}:
            category, action, reason = "MODEL_OR_RESUME_STATE", "ARCHIVE_EXTERNALLY", "Binary training state is unsuitable for normal Git"
        elif suffix in {".npz", ".npy"}:
            category, action, reason = "RAW_SCORE_OR_ARRAY", "ARCHIVE_EXTERNALLY", "Binary evidence array should remain outside Git"
        elif path.name in final_csv_names:
            category, action, reason = "FINAL_SUMMARY_CSV", "REVIEW", "Small final evidence table under blanket-ignored outputs"
        elif suffix in {".csv", ".json"} and size < 2_000_000:
            category, action, reason = "SMALL_OUTPUT_METADATA", "REVIEW", "Potential evidence metadata under ignored outputs"
        elif size >= 2_000_000:
            category, action, reason = "LARGE_OUTPUT", "ARCHIVE_EXTERNALLY", "Large derived artifact"
        else:
            category, action, reason = "GENERATED_OR_TEMPORARY", "GITIGNORE", "Regenerable output or temporary artifact"
        rows.append({"path": rel, "size_bytes": size, "category": category,
                     "recommended_action": action, "reason": reason})
    return rows


def main() -> None:
    for path, status in REQUIRED_STATUSES.items():
        if status not in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"Required status missing: {status} in {path}")
    data = {name: read_rows(path) for name, path in SOURCES.items()}
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    audit: list[dict[str, object]] = []
    figure_lightweight(data["innovation1"], audit)
    figure_dual(data["innovation2_real"], data["innovation2_mechanism"], audit)
    figure_tail(data["normal_tail"], data["fusion"], audit)

    (FIG_DIR / "ablation_figure_data_audit_payload.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "ablation_git_inventory_payload.json").write_text(
        json.dumps(inventory_rows(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DOC_DIR / "ablation_tables_latex.tex").write_text(latex_tables(data), encoding="utf-8")
    summary = {
        "status": "ABLATION_FINAL_PACKAGE_READY",
        "source_sha256": {name: sha256(path) for name, path in SOURCES.items()},
        "figure_audit_rows": len(audit),
        "prohibited_actions": {
            "training": False, "inference": False, "new_experiment": False,
            "result_recomputation": False, "test_tuning": False,
            "file_delete": False, "git_add": False, "commit": False, "push": False,
        },
    }
    (OUT / "ablation_final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
