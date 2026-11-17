from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
if str(DEFAULT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ROOT))

from data.swat import SWaTWindowDataset  # noqa: E402
from scripts.stage4_2a_analyze_relation_stability import (  # noqa: E402
    StreamingPairStatistics,
    _window_pair_correlations_tensor,
)
from scripts.stage4_2b_diagnose_relation_validity import (  # noqa: E402
    VARIANCE_EPSILON,
    channel_activity_rows,
    load_normal_training_data,
)
from utils.config_io import load_config  # noqa: E402


WINDOW_SIZES = (64, 128)
VALID_THRESHOLDS = (0.80, 0.90, 0.95, 0.99)
STRENGTH_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)
INSTABILITY_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50)

PAIR_FIELDS = [
    "pair_id",
    "channel_i",
    "channel_j",
    "channel_i_index",
    "channel_j_index",
    "window_size",
    "channel_i_activity_class",
    "channel_j_activity_class",
    "pair_activity_group",
    "total_window_count",
    "valid_window_count",
    "valid_ratio",
    "invalid_i_constant_count",
    "invalid_j_constant_count",
    "invalid_both_constant_count",
    "invalid_numeric_count",
    "invalid_i_constant_ratio",
    "invalid_j_constant_ratio",
    "invalid_both_constant_ratio",
    "invalid_numeric_ratio",
    "mean_corr",
    "median_corr",
    "abs_median_corr",
    "std_corr",
    "q05_corr",
    "q25_corr",
    "q75_corr",
    "q95_corr",
    "q05_q95_width",
    "median_fisher",
    "mad_fisher",
    "relation_instability",
]

ACTIVITY_FIELDS = [
    "channel",
    "channel_index",
    "window_size",
    "total_window_count",
    "active_window_count",
    "window_active_ratio",
    "activity_class",
    "global_unique_count",
    "global_change_ratio",
    "global_min",
    "global_max",
    "global_std",
]

THRESHOLD_FIELDS = [
    "window_size",
    "min_valid_ratio",
    "min_strength",
    "max_instability",
    "candidate_pair_count",
    "involved_channel_count",
    "dense_dense_count",
    "other_activity_group_count",
    "median_valid_ratio",
    "median_strength",
    "median_instability",
    "median_q05_q95_width",
]


def classify_activity(window_active_ratio: float) -> str:
    if window_active_ratio >= 0.95:
        return "dense_active"
    if window_active_ratio >= 0.20:
        return "moderately_active"
    return "sparse_switching"


def pair_activity_group(class_i: str, class_j: str) -> str:
    short = {
        "dense_active": "dense",
        "moderately_active": "moderate",
        "sparse_switching": "sparse",
    }
    order = {"dense": 0, "moderate": 1, "sparse": 2}
    left, right = short[class_i], short[class_j]
    if order[left] > order[right]:
        left, right = right, left
    return f"{left}_{right}"


def generate_unordered_pairs(channel_count: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(channel_count) for j in range(i + 1, channel_count)]


def finite_candidate(row: dict[str, Any]) -> bool:
    return all(
        math.isfinite(float(row[field]))
        for field in ("valid_ratio", "abs_median_corr", "relation_instability", "q05_q95_width")
    )


def candidate_matches(row: dict[str, Any], valid: float, strength: float, instability: float) -> bool:
    return (
        finite_candidate(row)
        and float(row["valid_ratio"]) >= valid
        and float(row["abs_median_corr"]) >= strength
        and float(row["relation_instability"]) <= instability
    )


def safe_median(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
    return float(np.median(values)) if values else float("nan")


def threshold_sensitivity_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for window_size in WINDOW_SIZES:
        window_rows = [row for row in pair_rows if int(row["window_size"]) == window_size]
        for valid, strength, instability in itertools.product(
            VALID_THRESHOLDS, STRENGTH_THRESHOLDS, INSTABILITY_THRESHOLDS
        ):
            candidates = [
                row for row in window_rows if candidate_matches(row, valid, strength, instability)
            ]
            channels = {
                int(row[index])
                for row in candidates
                for index in ("channel_i_index", "channel_j_index")
            }
            dense_count = sum(row["pair_activity_group"] == "dense_dense" for row in candidates)
            output.append(
                {
                    "window_size": window_size,
                    "min_valid_ratio": valid,
                    "min_strength": strength,
                    "max_instability": instability,
                    "candidate_pair_count": len(candidates),
                    "involved_channel_count": len(channels),
                    "dense_dense_count": dense_count,
                    "other_activity_group_count": len(candidates) - dense_count,
                    "median_valid_ratio": safe_median(candidates, "valid_ratio"),
                    "median_strength": safe_median(candidates, "abs_median_corr"),
                    "median_instability": safe_median(candidates, "relation_instability"),
                    "median_q05_q95_width": safe_median(candidates, "q05_q95_width"),
                }
            )
    return output


def pareto_candidate_flags(rows: list[dict[str, Any]]) -> list[bool]:
    flags = [False] * len(rows)
    valid_indices = [index for index, row in enumerate(rows) if finite_candidate(row)]
    for index in valid_indices:
        current = rows[index]
        v = float(current["valid_ratio"])
        s = float(current["abs_median_corr"])
        instability = float(current["relation_instability"])
        dominated = False
        for other_index in valid_indices:
            if other_index == index:
                continue
            other = rows[other_index]
            ov = float(other["valid_ratio"])
            os_ = float(other["abs_median_corr"])
            oi = float(other["relation_instability"])
            if ov >= v and os_ >= s and oi <= instability and (ov > v or os_ > s or oi < instability):
                dominated = True
                break
        flags[index] = not dominated
    return flags


def pareto_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for window_size in WINDOW_SIZES:
        group = [row for row in pair_rows if int(row["window_size"]) == window_size]
        flags = pareto_candidate_flags(group)
        for row, flag in zip(group, flags):
            if flag:
                output.append({**row, "pareto_candidate": True, "pareto_front": 1})
    return output


def candidate_channel_concentration(
    candidates: list[dict[str, Any]], candidate_definition: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for window_size in WINDOW_SIZES:
        group = [row for row in candidates if int(row["window_size"]) == window_size]
        counts: Counter[tuple[int, str]] = Counter()
        for row in group:
            counts[(int(row["channel_i_index"]), str(row["channel_i"]))] += 1
            counts[(int(row["channel_j_index"]), str(row["channel_j"]))] += 1
        for (channel_index, channel), count in sorted(counts.items()):
            ratio = count / max(1, len(group))
            output.append(
                {
                    "channel": channel,
                    "channel_index": channel_index,
                    "window_size": window_size,
                    "candidate_definition": candidate_definition,
                    "candidate_pair_count": len(group),
                    "candidate_occurrence_count": count,
                    "candidate_occurrence_ratio": ratio,
                    "concentration_warning": ratio > 0.30,
                }
            )
    return output


def _activity_output_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in raw_rows:
        ratio = float(row["window_active_ratio"])
        output.append(
            {
                "channel": row["channel"],
                "channel_index": row["channel_index"],
                "window_size": row["window_size"],
                "total_window_count": row["total_window_count"],
                "active_window_count": row["valid_variance_window_count"],
                "window_active_ratio": ratio,
                "activity_class": classify_activity(ratio),
                "global_unique_count": row["global_unique_count"],
                "global_change_ratio": row["global_change_ratio"],
                "global_min": row["global_min"],
                "global_max": row["global_max"],
                "global_std": row["global_std"],
            }
        )
    return output


def _analyze_window(
    values: np.ndarray,
    feature_names: list[str],
    window_size: int,
    activity_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    channel_count = len(feature_names)
    pairs = generate_unordered_pairs(channel_count)
    pair_i = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
    pair_j = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
    pair_count = len(pairs)
    statistics = StreamingPairStatistics(pair_count=pair_count)
    reason_counts = {
        "invalid_i_constant_count": np.zeros(pair_count, dtype=np.int64),
        "invalid_j_constant_count": np.zeros(pair_count, dtype=np.int64),
        "invalid_both_constant_count": np.zeros(pair_count, dtype=np.int64),
        "invalid_numeric_count": np.zeros(pair_count, dtype=np.int64),
    }

    _, active_masks = channel_activity_rows(
        values, list(range(channel_count)), feature_names, (window_size,), VARIANCE_EPSILON
    )
    active_all = np.column_stack([active_masks[(window_size, index)] for index in range(channel_count)])
    dataset = SWaTWindowDataset(
        values.astype(np.float32, copy=False),
        np.zeros(len(values), dtype=np.int64),
        window_size=window_size,
        stride=1,
    )
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair_i_device = pair_i.to(device)
    pair_j_device = pair_j.to(device)
    offset = 0
    for batch_index, (windows, _) in enumerate(loader, start=1):
        windows = windows.to(device, dtype=torch.float32, non_blocking=True)
        corr, _ = _window_pair_correlations_tensor(
            windows, pair_i_device, pair_j_device, VARIANCE_EPSILON
        )
        corr_np = corr.detach().cpu().numpy()
        end = offset + len(corr_np)
        batch_active = active_all[offset:end]
        active_i = batch_active[:, pair_i.numpy()]
        active_j = batch_active[:, pair_j.numpy()]
        numeric_invalid = active_i & active_j & ~np.isfinite(corr_np)
        valid = active_i & active_j & np.isfinite(corr_np)
        reason_counts["invalid_i_constant_count"] += (~active_i & active_j).sum(axis=0)
        reason_counts["invalid_j_constant_count"] += (active_i & ~active_j).sum(axis=0)
        reason_counts["invalid_both_constant_count"] += (~active_i & ~active_j).sum(axis=0)
        reason_counts["invalid_numeric_count"] += numeric_invalid.sum(axis=0)
        statistics.update(corr_np, valid)
        offset = end
        if batch_index % 100 == 0:
            print(f"window={window_size}: processed {offset}/{len(dataset)} windows")

    class_map = {
        int(row["channel_index"]): str(row["activity_class"])
        for row in activity_rows
        if int(row["window_size"]) == window_size
    }
    rows: list[dict[str, Any]] = []
    for index, (i, j) in enumerate(pairs):
        stats = statistics.finalize_pair(index)
        total = int(stats["total_window_count"])
        row: dict[str, Any] = {
            "pair_id": f"{i:02d}-{j:02d}:{feature_names[i].strip()}__{feature_names[j].strip()}",
            "channel_i": feature_names[i],
            "channel_j": feature_names[j],
            "channel_i_index": i,
            "channel_j_index": j,
            "window_size": window_size,
            "channel_i_activity_class": class_map[i],
            "channel_j_activity_class": class_map[j],
            "pair_activity_group": pair_activity_group(class_map[i], class_map[j]),
            **stats,
        }
        for field, counts in reason_counts.items():
            count = int(counts[index])
            row[field] = count
            row[field.replace("_count", "_ratio")] = count / max(1, total)
        conserved = int(row["valid_window_count"]) + sum(
            int(row[field]) for field in reason_counts
        )
        if conserved != total:
            raise RuntimeError(f"Validity counts do not sum to total for {row['pair_id']}")
        row["q05_q95_width"] = (
            float(row["q95_corr"]) - float(row["q05_corr"])
            if math.isfinite(float(row["q95_corr"])) and math.isfinite(float(row["q05_corr"]))
            else float("nan")
        )
        rows.append(row)
    return rows


def _candidate_tables(pair_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], ...]:
    dense_valid = [
        row for row in pair_rows
        if row["pair_activity_group"] == "dense_dense" and float(row["valid_ratio"]) >= 0.95
    ]
    strong_dense = [
        row for row in dense_valid
        if math.isfinite(float(row["abs_median_corr"])) and float(row["abs_median_corr"]) >= 0.30
    ]
    balanced_all = [row for row in pair_rows if candidate_matches(row, 0.95, 0.30, 0.40)]
    balanced_examples: list[dict[str, Any]] = []
    for window_size in WINDOW_SIZES:
        group = [row for row in balanced_all if int(row["window_size"]) == window_size]
        group.sort(
            key=lambda row: (
                -float(row["valid_ratio"]),
                -float(row["abs_median_corr"]),
                float(row["relation_instability"]),
            )
        )
        balanced_examples.extend(group[:50])
    return dense_valid, strong_dense, balanced_all, balanced_examples


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if isinstance(value, float) and not math.isfinite(value) else value
                    for key, value in row.items()
                }
            )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def _threshold_lookup(
    sensitivity: list[dict[str, Any]], window_size: int, valid: float, strength: float, instability: float
) -> dict[str, Any]:
    return next(
        row for row in sensitivity
        if int(row["window_size"]) == window_size
        and float(row["min_valid_ratio"]) == valid
        and float(row["min_strength"]) == strength
        and float(row["max_instability"]) == instability
    )


def create_figures(
    pair_rows: list[dict[str, Any]], sensitivity: list[dict[str, Any]], output_dir: Path
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=False)
    for window_size in WINDOW_SIZES:
        matrix = np.asarray(
            [
                [
                    _threshold_lookup(sensitivity, window_size, 0.95, strength, instability)["candidate_pair_count"]
                    for instability in INSTABILITY_THRESHOLDS
                ]
                for strength in STRENGTH_THRESHOLDS
            ],
            dtype=float,
        )
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        image = ax.imshow(matrix, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(INSTABILITY_THRESHOLDS)), INSTABILITY_THRESHOLDS)
        ax.set_yticks(range(len(STRENGTH_THRESHOLDS)), STRENGTH_THRESHOLDS)
        ax.set_xlabel("max instability")
        ax.set_ylabel("min strength")
        ax.set_title(f"Candidate count, window={window_size}, valid ratio >= 0.95")
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                ax.text(x, y, str(int(matrix[y, x])), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="candidate pair count")
        fig.tight_layout()
        fig.savefig(figure_dir / f"candidate_count_heatmap_window{window_size}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for window_size, color in ((64, "#2563eb"), (128, "#dc2626")):
        rows = [
            row for row in pair_rows
            if row["window_size"] == window_size and row["pair_activity_group"] == "dense_dense"
            and finite_candidate(row)
        ]
        ax.scatter(
            [row["abs_median_corr"] for row in rows],
            [row["relation_instability"] for row in rows],
            s=18,
            alpha=0.55,
            color=color,
            label=f"window {window_size}",
        )
    ax.set_xlabel("abs median correlation")
    ax.set_ylabel("relation instability")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "strength_vs_instability_dense_pairs.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for window_size, color in ((64, "#2563eb"), (128, "#dc2626")):
        rows = [row for row in pair_rows if row["window_size"] == window_size and finite_candidate(row)]
        ax.scatter(
            [row["valid_ratio"] for row in rows],
            [row["abs_median_corr"] for row in rows],
            s=12,
            alpha=0.35,
            color=color,
            label=f"window {window_size}",
        )
    ax.set_xlabel("valid ratio")
    ax.set_ylabel("abs median correlation")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "valid_ratio_vs_strength_all_pairs.png", dpi=180)
    plt.close(fig)


def _activity_counts(activity_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for window_size in WINDOW_SIZES:
        counts = Counter(
            row["activity_class"] for row in activity_rows if int(row["window_size"]) == window_size
        )
        output[str(window_size)] = dict(counts)
    return output


def _pair_group_counts(pair_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for window_size in WINDOW_SIZES:
        counts = Counter(
            row["pair_activity_group"] for row in pair_rows if int(row["window_size"]) == window_size
        )
        output[str(window_size)] = dict(counts)
    return output


def _instability_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray(
        [float(row["relation_instability"]) for row in rows if math.isfinite(float(row["relation_instability"]))]
    )
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "le_0_15_count": int((values <= 0.15).sum()),
        "le_0_25_count": int((values <= 0.25).sum()),
        "le_0_30_count": int((values <= 0.30).sum()),
        "le_0_40_count": int((values <= 0.40).sum()),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    output_dir = project_root / (
        "outputs/stage4_2c_local_relation_candidates_smoke"
        if args.smoke
        else args.output_dir
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    stage4_2a = project_root / "scripts/stage4_2a_analyze_relation_stability.py"
    stage4_2b = project_root / "scripts/stage4_2b_diagnose_relation_validity.py"
    config_path = project_root / "configs/swat_improve_v4_struct_finetune.yaml"
    for required in (stage4_2a, stage4_2b, config_path):
        if not required.exists():
            raise FileNotFoundError(required)
    print("Reuse:")
    print("- data.swat._load_swat_csv and data.transforms.StandardScaler via stage4_2b")
    print("- stage4_2b.load_normal_training_data and channel_activity_rows")
    print("- stage4_2a StreamingPairStatistics and batch Pearson implementation")
    print(f"Analyze: 51 channels, {len(generate_unordered_pairs(51))} unordered pairs, windows={WINDOW_SIZES}")
    print("Not modified: model, training, checkpoint, configs, current structure_pairs, stage4_2a/2b outputs")
    print(f"Output: {output_dir}")
    if args.dry_run:
        return {"dry_run": True, "output_dir": str(output_dir)}

    cfg = load_config(str(config_path))
    values, feature_names, preprocessing = load_normal_training_data(project_root, cfg, args.smoke)
    raw_activity, _ = channel_activity_rows(
        values, list(range(len(feature_names))), feature_names, WINDOW_SIZES, VARIANCE_EPSILON
    )
    activity_rows = _activity_output_rows(raw_activity)
    pair_rows: list[dict[str, Any]] = []
    for window_size in WINDOW_SIZES:
        print(f"Analyzing all local pairs for window={window_size}")
        pair_rows.extend(_analyze_window(values, feature_names, window_size, activity_rows))

    sensitivity = threshold_sensitivity_rows(pair_rows)
    pareto = pareto_rows(pair_rows)
    dense_valid, strong_dense, balanced_all, balanced_examples = _candidate_tables(pair_rows)
    concentration = []
    for rows, definition in (
        (dense_valid, "dense_valid"),
        (strong_dense, "strong_dense"),
        (balanced_all, "balanced_valid095_strength030_instability040"),
    ):
        concentration.extend(candidate_channel_concentration(rows, definition))
    concentration_warnings = [row for row in concentration if row["concentration_warning"]]

    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "channel_activity_classes.csv", activity_rows, ACTIVITY_FIELDS)
    write_csv(output_dir / "all_local_pair_stats.csv", pair_rows, PAIR_FIELDS)
    write_csv(output_dir / "threshold_sensitivity.csv", sensitivity, THRESHOLD_FIELDS)
    write_csv(output_dir / "pareto_local_relation_candidates.csv", pareto, PAIR_FIELDS + ["pareto_candidate", "pareto_front"])
    write_csv(output_dir / "dense_valid_candidates.csv", dense_valid, PAIR_FIELDS)
    write_csv(output_dir / "strong_dense_candidates.csv", strong_dense, PAIR_FIELDS)
    write_csv(output_dir / "balanced_candidate_examples.csv", balanced_examples, PAIR_FIELDS)
    concentration_fields = [
        "channel", "channel_index", "window_size", "candidate_definition", "candidate_pair_count",
        "candidate_occurrence_count", "candidate_occurrence_ratio", "concentration_warning",
    ]
    write_csv(output_dir / "candidate_channel_concentration.csv", concentration, concentration_fields)
    create_figures(pair_rows, sensitivity, output_dir)

    activity_counts = _activity_counts(activity_rows)
    pair_groups = _pair_group_counts(pair_rows)
    core_threshold_counts = {
        str(window_size): {
            str(instability): _threshold_lookup(sensitivity, window_size, 0.95, 0.30, instability)[
                "candidate_pair_count"
            ]
            for instability in (0.15, 0.25, 0.30, 0.40)
        }
        for window_size in WINDOW_SIZES
    }
    valid_strength_counts = {
        str(window_size): {
            "valid095_strength030_no_instability_limit": sum(
                finite_candidate(row)
                and float(row["valid_ratio"]) >= 0.95
                and float(row["abs_median_corr"]) >= 0.30
                for row in pair_rows
                if int(row["window_size"]) == window_size
            ),
            "valid095_strength050_no_instability_limit": sum(
                finite_candidate(row)
                and float(row["valid_ratio"]) >= 0.95
                and float(row["abs_median_corr"]) >= 0.50
                for row in pair_rows
                if int(row["window_size"]) == window_size
            ),
        }
        for window_size in WINDOW_SIZES
    }
    strong_dense_instability = {
        str(window_size): _instability_summary(
            [row for row in strong_dense if int(row["window_size"]) == window_size]
        )
        for window_size in WINDOW_SIZES
    }
    balanced_counts = {
        str(window_size): sum(int(row["window_size"]) == window_size for row in balanced_all)
        for window_size in WINDOW_SIZES
    }
    pareto_counts = {
        str(window_size): sum(int(row["window_size"]) == window_size for row in pareto)
        for window_size in WINDOW_SIZES
    }
    max_concentration = {
        str(window_size): max(
            (
                float(row["candidate_occurrence_ratio"])
                for row in concentration
                if int(row["window_size"]) == window_size
                and row["candidate_definition"] == "balanced_valid095_strength030_instability040"
            ),
            default=0.0,
        )
        for window_size in WINDOW_SIZES
    }
    enough_balanced = {
        str(window_size): 10 <= balanced_counts[str(window_size)]
        and max_concentration[str(window_size)] <= 0.30
        for window_size in WINDOW_SIZES
    }
    summary = {
        "preprocessing": {
            **preprocessing,
            "window_sizes": list(WINDOW_SIZES),
            "variance_epsilon": VARIANCE_EPSILON,
            "channel_count": len(feature_names),
            "all_unordered_pair_count": len(generate_unordered_pairs(len(feature_names))),
            "quantile_method": "stage4_2a 4001-bin streaming histogram",
        },
        "channel_activity_class_counts": activity_counts,
        "pair_activity_group_counts": pair_groups,
        "threshold_grid": {
            "valid_ratio": list(VALID_THRESHOLDS),
            "strength": list(STRENGTH_THRESHOLDS),
            "instability": list(INSTABILITY_THRESHOLDS),
            "combination_count_per_window": len(VALID_THRESHOLDS) * len(STRENGTH_THRESHOLDS) * len(INSTABILITY_THRESHOLDS),
        },
        "core_threshold_candidate_counts": core_threshold_counts,
        "valid095_strength_counts_without_instability_limit": valid_strength_counts,
        "strong_dense_instability_distribution": strong_dense_instability,
        "pareto_candidate_counts": pareto_counts,
        "dense_valid_candidate_counts": {
            str(window_size): sum(int(row["window_size"]) == window_size for row in dense_valid)
            for window_size in WINDOW_SIZES
        },
        "strong_dense_candidate_counts": {
            str(window_size): sum(int(row["window_size"]) == window_size for row in strong_dense)
            for window_size in WINDOW_SIZES
        },
        "balanced_candidate_counts": balanced_counts,
        "candidate_channel_concentration": {
            "max_balanced_occurrence_ratio": max_concentration,
            "warning_threshold": 0.30,
            "warning_count": len(concentration_warnings),
            "warnings": concentration_warnings,
        },
        "window_comparison": {
            "window64_balanced_count": balanced_counts["64"],
            "window128_balanced_count": balanced_counts["128"],
            "window128_has_more_balanced_candidates": balanced_counts["128"] > balanced_counts["64"],
            "window128_strong_dense_median_instability_lower": (
                strong_dense_instability["128"].get("median", float("inf"))
                < strong_dense_instability["64"].get("median", float("inf"))
            ),
        },
        "instability_015_diagnosis": {
            "counts_vs_relaxed_thresholds": core_threshold_counts,
            "is_clearly_restrictive": any(
                core_threshold_counts[str(window_size)]["0.15"]
                < core_threshold_counts[str(window_size)]["0.4"]
                for window_size in WINDOW_SIZES
            ),
        },
        "enough_balanced_candidates": enough_balanced,
        "supports_next_candidate_relation_set_step": any(enough_balanced.values()),
        "recommendation_scope": (
            "Diagnostic evidence only: consider dense-active Pearson candidates in a later step; "
            "do not replace current structure_pairs from this output."
        ),
        "test_set_or_labels_read": False,
        "current_structure_pairs_modified": False,
        "model_or_training_modified": False,
        "warnings": [
            "No threshold combination was selected as final.",
            "Balanced examples are capped at 50 per window and are not model structure_pairs.",
            "Sparse-switching handling is outside Step 4.2C.",
        ],
    }
    write_json(output_dir / "local_relation_candidate_summary.json", summary)
    print(json.dumps(json_safe({
        "activity_counts": activity_counts,
        "dense_dense_pair_counts": {w: pair_groups[w].get("dense_dense", 0) for w in pair_groups},
        "core_threshold_candidate_counts": core_threshold_counts,
        "pareto_candidate_counts": pareto_counts,
        "balanced_candidate_counts": balanced_counts,
    }), indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 4.2C: diagnose all local relation candidates and thresholds.")
    parser.add_argument("--project-root", default="D:/workspace/python/mscad")
    parser.add_argument("--output-dir", default="outputs/stage4_2c_local_relation_candidates")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
