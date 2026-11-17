from __future__ import annotations

import argparse
import csv
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


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
if str(DEFAULT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ROOT))

from data.swat import _load_swat_csv  # noqa: E402
from data.transforms import StandardScaler  # noqa: E402
from scripts.stage4_2a_analyze_relation_stability import (  # noqa: E402
    fisher_transform,
    summarize_correlation_samples,
)
from utils.config_io import load_config  # noqa: E402


WINDOW_SIZES = (64, 96, 128, 256)
VARIANCE_EPSILON = 1e-8

CHANNEL_FIELDS = [
    "channel",
    "channel_index",
    "window_size",
    "total_window_count",
    "valid_variance_window_count",
    "invalid_variance_window_count",
    "window_active_ratio",
    "global_std",
    "global_min",
    "global_max",
    "global_unique_count",
    "global_change_ratio",
]

PAIR_FIELDS = [
    "pair_id",
    "current_pair_rank",
    "channel_i",
    "channel_j",
    "channel_i_index",
    "channel_j_index",
    "window_size",
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
    "min_corr",
    "max_corr",
    "median_fisher",
    "mad_fisher",
    "relation_instability",
    "diagnostic_valid",
    "diagnostic_strong",
    "diagnostic_stable",
    "diagnostic_strong_and_stable",
    "valid_ratio_delta_vs_64",
    "strength_delta_vs_64",
    "instability_delta_vs_64",
    "q05_q95_width",
    "q05_q95_width_delta_vs_64",
]

INVALID_REASON_FIELDS = [
    "pair_id",
    "current_pair_rank",
    "channel_i",
    "channel_j",
    "window_size",
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
]


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def rolling_sum(values: np.ndarray, window_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("rolling_sum expects a one-dimensional array")
    if window_size <= 0 or window_size > values.size:
        raise ValueError("window_size must be in [1, len(values)]")
    cumulative = np.empty(values.size + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(values, dtype=np.float64, out=cumulative[1:])
    return cumulative[window_size:] - cumulative[:-window_size]


def rolling_channel_moments(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    sums = rolling_sum(values, window_size)
    sum_squares = rolling_sum(np.square(np.asarray(values, dtype=np.float64)), window_size)
    centered_ss = sum_squares - np.square(sums) / float(window_size)
    centered_ss = np.maximum(centered_ss, 0.0)
    sample_variance = centered_ss / float(max(1, window_size - 1))
    return sums, sample_variance


def rolling_pair_diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    variance_epsilon: float = VARIANCE_EPSILON,
) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays with matching shape")
    sum_x, var_x = rolling_channel_moments(x, window_size)
    sum_y, var_y = rolling_channel_moments(y, window_size)
    sum_xy = rolling_sum(x * y, window_size)
    centered_cross = sum_xy - sum_x * sum_y / float(window_size)
    ss_x = var_x * float(max(1, window_size - 1))
    ss_y = var_y * float(max(1, window_size - 1))

    active_i = var_x > variance_epsilon
    active_j = var_y > variance_epsilon
    invalid_i = ~active_i & active_j
    invalid_j = active_i & ~active_j
    invalid_both = ~active_i & ~active_j
    variance_valid = active_i & active_j

    denominator = np.sqrt(np.maximum(ss_x * ss_y, 0.0))
    correlations = np.full(x.size - window_size + 1, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations[variance_valid] = centered_cross[variance_valid] / denominator[variance_valid]
    invalid_numeric = variance_valid & ~np.isfinite(correlations)
    valid = variance_valid & np.isfinite(correlations)
    correlations[valid] = np.clip(correlations[valid], -0.999, 0.999)
    correlations[~valid] = np.nan

    counts = {
        "valid_window_count": int(valid.sum()),
        "invalid_i_constant_count": int(invalid_i.sum()),
        "invalid_j_constant_count": int(invalid_j.sum()),
        "invalid_both_constant_count": int(invalid_both.sum()),
        "invalid_numeric_count": int(invalid_numeric.sum()),
    }
    total = int(correlations.size)
    if sum(counts.values()) != total:
        raise RuntimeError("Mutually exclusive validity counts do not sum to total windows")
    return {
        "correlations": correlations,
        "active_i": active_i,
        "active_j": active_j,
        "valid": valid,
        "total_window_count": total,
        **counts,
    }


def pair_row_from_diagnostics(
    diagnostics: dict[str, Any],
    pair: tuple[int, int],
    rank: int,
    feature_names: list[str],
    window_size: int,
    min_valid_ratio: float,
    min_strength: float,
    max_instability: float,
) -> dict[str, Any]:
    i, j = pair
    total = int(diagnostics["total_window_count"])
    statistics = summarize_correlation_samples(diagnostics["correlations"], total_window_count=total)
    valid_ratio = float(statistics["valid_ratio"])
    strength = float(statistics["relation_strength"])
    instability = float(statistics["relation_instability"])
    finite_strength = math.isfinite(strength)
    finite_instability = math.isfinite(instability)
    row: dict[str, Any] = {
        "pair_id": f"{i:02d}-{j:02d}:{feature_names[i].strip()}__{feature_names[j].strip()}",
        "current_pair_rank": rank,
        "channel_i": feature_names[i],
        "channel_j": feature_names[j],
        "channel_i_index": i,
        "channel_j_index": j,
        "window_size": window_size,
        "total_window_count": total,
        "valid_window_count": int(statistics["valid_window_count"]),
        "valid_ratio": valid_ratio,
        "mean_corr": statistics["mean_corr"],
        "median_corr": statistics["median_corr"],
        "abs_median_corr": statistics["abs_median_corr"],
        "std_corr": statistics["std_corr"],
        "q05_corr": statistics["q05_corr"],
        "q25_corr": statistics["q25_corr"],
        "q75_corr": statistics["q75_corr"],
        "q95_corr": statistics["q95_corr"],
        "min_corr": statistics["min_corr"],
        "max_corr": statistics["max_corr"],
        "median_fisher": statistics["median_fisher"],
        "mad_fisher": statistics["mad_fisher"],
        "relation_instability": instability,
    }
    for reason in (
        "invalid_i_constant_count",
        "invalid_j_constant_count",
        "invalid_both_constant_count",
        "invalid_numeric_count",
    ):
        count = int(diagnostics[reason])
        row[reason] = count
        row[reason.replace("_count", "_ratio")] = count / max(1, total)
    row["diagnostic_valid"] = valid_ratio >= min_valid_ratio
    row["diagnostic_strong"] = finite_strength and strength >= min_strength
    row["diagnostic_stable"] = finite_instability and instability <= max_instability
    row["diagnostic_strong_and_stable"] = bool(
        row["diagnostic_valid"] and row["diagnostic_strong"] and row["diagnostic_stable"]
    )
    width = float(row["q95_corr"]) - float(row["q05_corr"]) if finite_strength else float("nan")
    row["q05_q95_width"] = width
    return row


def load_normal_training_data(project_root: Path, cfg: dict[str, Any], smoke: bool) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    data_cfg = cfg["data"]
    train_path = resolve_path(project_root, data_cfg["train_csv"])
    x_all, y_all, feature_names = _load_swat_csv(
        str(train_path), str(data_cfg["label_col"]), str(data_cfg["timestamp_col"])
    )
    base_window = int(data_cfg["window_size"])
    n = len(x_all)
    n_val = int(max(base_window * 2, round(n * float(data_cfg.get("val_ratio", 0.2)))))
    n_val = min(max(base_window * 2, n_val), n - base_window - 1)
    split = n - n_val
    x_train_raw = x_all[:split]
    y_train_raw = y_all[:split]
    if np.any(y_train_raw != 0):
        raise ValueError("SWaT normal training split contains non-normal labels")
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train_raw).astype(np.float64, copy=False)
    full_rows = len(x_train_norm)
    if smoke:
        x_train_norm = x_train_norm[: min(full_rows, 8192)]
    return x_train_norm, feature_names, {
        "train_csv": str(Path(data_cfg["train_csv"]).as_posix()),
        "test_data_read": False,
        "normal_csv_row_count": int(n),
        "training_split_row_count": int(split),
        "validation_tail_row_count": int(n_val),
        "full_training_rows": int(full_rows),
        "analyzed_training_rows": int(len(x_train_norm)),
        "scaler": "data.transforms.StandardScaler fitted on x_train_raw",
        "loader": "data.swat._load_swat_csv",
        "missing_value_handling": "pd.to_numeric(errors=coerce).fillna(0.0)",
        "stride": 1,
        "smoke": bool(smoke),
    }


def load_current_pairs(project_root: Path) -> tuple[list[tuple[int, int]], list[str], dict[str, Any]]:
    pairs_by_seed: dict[int, list[tuple[int, int]]] = {}
    names_by_seed: dict[int, list[str]] = {}
    for seed in (1, 2, 3):
        checkpoint = project_root / "results" / "multiseed" / f"seed_{seed}" / "ft" / "best.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        pairs_by_seed[seed] = [tuple(sorted((int(i), int(j)))) for i, j in state["structure_pairs"]]
        names_by_seed[seed] = [str(name) for name in state["feature_names"]]
    if not (pairs_by_seed[1] == pairs_by_seed[2] == pairs_by_seed[3]):
        raise ValueError("Step 4.2B expects identical structure_pairs across seeds 1, 2, and 3")
    if not (names_by_seed[1] == names_by_seed[2] == names_by_seed[3]):
        raise ValueError("Feature-name order differs across seed checkpoints")
    return pairs_by_seed[1], names_by_seed[1], {
        "source": "results/multiseed/seed_1/ft/best.pt::structure_pairs",
        "selection_code": "train.py::select_corr_pairs(bundle.x_train_norm)",
        "pair_count": len(pairs_by_seed[1]),
        "same_pairs_across_seeds": True,
    }


def channel_activity_rows(
    values: np.ndarray,
    channel_indices: list[int],
    feature_names: list[str],
    window_sizes: tuple[int, ...],
    variance_epsilon: float,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    active_masks: dict[tuple[int, int], np.ndarray] = {}
    for channel_index in channel_indices:
        channel_values = values[:, channel_index]
        global_std = float(np.std(channel_values))
        global_min = float(np.min(channel_values))
        global_max = float(np.max(channel_values))
        global_unique_count = int(np.unique(channel_values).size)
        global_change_ratio = float(np.mean(np.abs(np.diff(channel_values)) > variance_epsilon))
        for window_size in window_sizes:
            _, variance = rolling_channel_moments(channel_values, window_size)
            active = variance > variance_epsilon
            active_masks[(window_size, channel_index)] = active
            valid_count = int(active.sum())
            total = int(active.size)
            rows.append(
                {
                    "channel": feature_names[channel_index],
                    "channel_index": channel_index,
                    "window_size": window_size,
                    "total_window_count": total,
                    "valid_variance_window_count": valid_count,
                    "invalid_variance_window_count": total - valid_count,
                    "window_active_ratio": valid_count / max(1, total),
                    "global_std": global_std,
                    "global_min": global_min,
                    "global_max": global_max,
                    "global_unique_count": global_unique_count,
                    "global_change_ratio": global_change_ratio,
                }
            )
    return rows, active_masks


def apply_deltas(pair_rows: list[dict[str, Any]]) -> None:
    baseline = {row["pair_id"]: row for row in pair_rows if int(row["window_size"]) == 64}
    for row in pair_rows:
        base = baseline[row["pair_id"]]
        row["valid_ratio_delta_vs_64"] = float(row["valid_ratio"]) - float(base["valid_ratio"])
        row["strength_delta_vs_64"] = float(row["abs_median_corr"]) - float(base["abs_median_corr"])
        row["instability_delta_vs_64"] = float(row["relation_instability"]) - float(base["relation_instability"])
        row["q05_q95_width_delta_vs_64"] = float(row["q05_q95_width"]) - float(base["q05_q95_width"])


def summarize_windows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window_size in WINDOW_SIZES:
        group = [row for row in pair_rows if int(row["window_size"]) == window_size]
        finite_instability = [float(row["relation_instability"]) for row in group if math.isfinite(float(row["relation_instability"]))]
        rows.append(
            {
                "window_size": window_size,
                "current_pair_count": len(group),
                "valid_ratio_ge_095_count": sum(bool(row["diagnostic_valid"]) for row in group),
                "strong_count": sum(bool(row["diagnostic_strong"]) for row in group),
                "stable_count": sum(bool(row["diagnostic_stable"]) for row in group),
                "strong_and_stable_count": sum(bool(row["diagnostic_strong_and_stable"]) for row in group),
                "median_valid_ratio": float(np.median([row["valid_ratio"] for row in group])),
                "median_strength": float(np.median([row["abs_median_corr"] for row in group])),
                "median_instability": float(np.median(finite_instability)) if finite_instability else float("nan"),
                "median_q05_q95_width": float(np.median([row["q05_q95_width"] for row in group])),
                "valid_ratio_improved_pair_count_vs_64": sum(float(row["valid_ratio_delta_vs_64"]) > 0.01 for row in group),
                "strength_improved_pair_count_vs_64": sum(float(row["strength_delta_vs_64"]) > 0.01 for row in group),
                "instability_reduced_pair_count_vs_64": sum(float(row["instability_delta_vs_64"]) < -0.01 for row in group),
            }
        )
    return rows


def low_valid_root_causes(
    pair_rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    activity = {
        (int(row["window_size"]), int(row["channel_index"])): float(row["window_active_ratio"])
        for row in channel_rows
    }
    result: list[dict[str, Any]] = []
    reason_columns = {
        "invalid_i_constant": "invalid_i_constant_ratio",
        "invalid_j_constant": "invalid_j_constant_ratio",
        "invalid_both_constant": "invalid_both_constant_ratio",
        "invalid_numeric": "invalid_numeric_ratio",
    }
    for row in pair_rows:
        if int(row["window_size"]) != 64 or float(row["valid_ratio"]) >= 0.95:
            continue
        reason = max(reason_columns, key=lambda key: float(row[reason_columns[key]]))
        if reason == "invalid_i_constant":
            channel = row["channel_i"]
        elif reason == "invalid_j_constant":
            channel = row["channel_j"]
        elif reason == "invalid_both_constant":
            channel = f"{str(row['channel_i']).strip()}|{str(row['channel_j']).strip()}"
        else:
            channel = ""
        result.append(
            {
                "pair_id": row["pair_id"],
                "current_pair_rank": row["current_pair_rank"],
                "channel_i": row["channel_i"],
                "channel_j": row["channel_j"],
                "valid_ratio": row["valid_ratio"],
                "dominant_invalid_reason": reason,
                "dominant_invalid_channel": channel,
                "dominant_invalid_ratio": row[reason_columns[reason]],
                "channel_i_window_active_ratio": activity[(64, int(row["channel_i_index"]))],
                "channel_j_window_active_ratio": activity[(64, int(row["channel_j_index"]))],
            }
        )
    return result


def frequent_invalid_channels(low_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pair_frequency: Counter[str] = Counter()
    dominant_frequency: Counter[str] = Counter()
    for row in low_rows:
        for channel in (str(row["channel_i"]).strip(), str(row["channel_j"]).strip()):
            pair_frequency[channel] += 1
        dominant = str(row["dominant_invalid_channel"])
        for channel in filter(None, dominant.split("|")):
            dominant_frequency[channel] += 1
    channels = set(pair_frequency) | set(dominant_frequency)
    return sorted(
        (
            {
                "channel": channel,
                "low_valid_pair_occurrence_count": pair_frequency[channel],
                "dominant_cause_count": dominant_frequency[channel],
            }
            for channel in channels
        ),
        key=lambda row: (-int(row["dominant_cause_count"]), -int(row["low_valid_pair_occurrence_count"]), row["channel"]),
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def create_figures(pair_rows: list[dict[str, Any]], output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=False)
    colors = {64: "#2563eb", 96: "#0891b2", 128: "#16a34a", 256: "#dc2626"}

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for pair_id, group in _group_by_pair(pair_rows).items():
        ordered = sorted(group, key=lambda row: int(row["window_size"]))
        ax.plot([r["window_size"] for r in ordered], [r["valid_ratio"] for r in ordered],
                color="#94a3b8", alpha=0.35, linewidth=0.8)
    medians = [np.median([r["valid_ratio"] for r in pair_rows if r["window_size"] == w]) for w in WINDOW_SIZES]
    ax.plot(WINDOW_SIZES, medians, color="#111827", marker="o", linewidth=2.2, label="median")
    ax.axhline(0.95, color="#dc2626", linestyle="--", linewidth=1, label="diagnostic threshold")
    ax.set_xticks(WINDOW_SIZES)
    ax.set_xlabel("window size")
    ax.set_ylabel("valid ratio")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "valid_ratio_by_window_size.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for _, group in _group_by_pair(pair_rows).items():
        ordered = sorted(group, key=lambda row: int(row["window_size"]))
        ax.plot([r["window_size"] for r in ordered], [r["relation_instability"] for r in ordered],
                color="#94a3b8", alpha=0.35, linewidth=0.8)
    medians = [np.nanmedian([r["relation_instability"] for r in pair_rows if r["window_size"] == w]) for w in WINDOW_SIZES]
    ax.plot(WINDOW_SIZES, medians, color="#111827", marker="o", linewidth=2.2, label="median")
    ax.axhline(0.15, color="#dc2626", linestyle="--", linewidth=1, label="diagnostic threshold")
    ax.set_xticks(WINDOW_SIZES)
    ax.set_xlabel("window size")
    ax.set_ylabel("relation instability")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "instability_by_window_size.png", dpi=180)
    plt.close(fig)

    rows64 = sorted((row for row in pair_rows if row["window_size"] == 64), key=lambda row: int(row["current_pair_rank"]))
    y = np.arange(len(rows64))
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    left = np.zeros(len(rows64), dtype=float)
    for field, label, color in (
        ("invalid_i_constant_ratio", "i constant", "#2563eb"),
        ("invalid_j_constant_ratio", "j constant", "#0891b2"),
        ("invalid_both_constant_ratio", "both constant", "#f59e0b"),
        ("invalid_numeric_ratio", "numeric", "#dc2626"),
    ):
        values = np.asarray([row[field] for row in rows64], dtype=float)
        ax.barh(y, values, left=left, label=label, color=color)
        left += values
    ax.set_yticks(y, [row["pair_id"] for row in rows64], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("invalid window ratio at window size 64")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "invalid_reason_by_pair.png", dpi=180)
    plt.close(fig)


def _group_by_pair(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    return grouped


def inspect_stage4_2a(project_root: Path) -> dict[str, Any]:
    base = project_root / "outputs" / "stage4_2a_relation_stability"
    summary_path = base / "relation_stability_summary.json"
    if not summary_path.exists():
        summary_path = base / "relation_stability_summary_new.json"
    diagnostics = base / "current_selected_pair_diagnostics.csv"
    if not summary_path.exists() or not diagnostics.exists():
        raise FileNotFoundError("Required Step 4.2A outputs are missing")
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with diagnostics.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "summary_path": str(summary_path.relative_to(project_root).as_posix()),
        "diagnostics_path": str(diagnostics.relative_to(project_root).as_posix()),
        "selected_pair_count": len(rows),
        "reported_low_valid_pair_count": sum(float(row["valid_ratio"]) < 0.95 for row in rows),
        "summary": summary,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    cfg = load_config(str(project_root / "configs" / "swat_improve_v4_struct_finetune.yaml"))
    stage4_2a = inspect_stage4_2a(project_root)
    pairs, checkpoint_names, pair_source = load_current_pairs(project_root)
    output_dir = resolve_path(project_root, args.output_dir)
    if args.smoke:
        output_dir = project_root / "outputs" / "stage4_2b_relation_validity_smoke"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    print("Reuse files:")
    print(f"- {stage4_2a['summary_path']}")
    print(f"- {stage4_2a['diagnostics_path']}")
    print("- data/swat.py::_load_swat_csv")
    print("- data/transforms.py::StandardScaler")
    print("- scripts/stage4_2a_analyze_relation_stability.py statistics helpers")
    print("- results/multiseed/seed_1/ft/best.pt::structure_pairs")
    print("New files:")
    print("- scripts/stage4_2b_diagnose_relation_validity.py")
    print("- tests/test_relation_validity_diagnosis.py")
    print(f"Current pairs: {len(pairs)} from {pair_source['source']}")
    print(f"Training rows: {stage4_2a['summary']['preprocessing_summary']['training_split_row_count']}")
    print("Not modified: models/, losses/, train.py, evaluate.py, data/, configs/, structure_pairs, stage4_2a outputs")
    print(f"Output: {output_dir}")
    if args.dry_run:
        return {"dry_run": True, "output_dir": str(output_dir)}

    values, feature_names, preprocessing = load_normal_training_data(project_root, cfg, args.smoke)
    if feature_names != checkpoint_names:
        raise ValueError("Normal training channel order differs from checkpoint feature_names")
    involved_channels = sorted({index for pair in pairs for index in pair})
    channel_rows, _ = channel_activity_rows(
        values, involved_channels, feature_names, WINDOW_SIZES, VARIANCE_EPSILON
    )

    pair_rows: list[dict[str, Any]] = []
    for window_size in WINDOW_SIZES:
        print(f"Analyzing window_size={window_size}")
        for rank, pair in enumerate(pairs, start=1):
            i, j = pair
            diagnostics = rolling_pair_diagnostics(values[:, i], values[:, j], window_size, VARIANCE_EPSILON)
            pair_rows.append(
                pair_row_from_diagnostics(
                    diagnostics,
                    pair,
                    rank,
                    feature_names,
                    window_size,
                    args.min_valid_ratio,
                    args.min_strength,
                    args.max_instability,
                )
            )
    apply_deltas(pair_rows)
    window_summary = summarize_windows(pair_rows)
    low_rows = low_valid_root_causes(pair_rows, channel_rows)
    frequent_channels = frequent_invalid_channels(low_rows)

    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "channel_window_activity.csv", channel_rows, CHANNEL_FIELDS)
    write_csv(output_dir / "current_pair_window_scale_diagnostics.csv", pair_rows, PAIR_FIELDS)
    write_csv(output_dir / "pair_invalid_reason_summary.csv", pair_rows, INVALID_REASON_FIELDS)
    window_fields = list(window_summary[0].keys())
    write_csv(output_dir / "window_scale_summary.csv", window_summary, window_fields)
    low_fields = [
        "pair_id", "current_pair_rank", "channel_i", "channel_j", "valid_ratio",
        "dominant_invalid_reason", "dominant_invalid_channel", "dominant_invalid_ratio",
        "channel_i_window_active_ratio", "channel_j_window_active_ratio",
    ]
    write_csv(output_dir / "low_valid_ratio_root_causes.csv", low_rows, low_fields)
    create_figures(pair_rows, output_dir)

    summary_by_window = {str(row["window_size"]): row for row in window_summary}
    baseline = window_summary[0]
    longest = window_summary[-1]
    pair_groups = _group_by_pair(pair_rows)
    valid_improvement_counts = {
        str(window): sum(
            float(next(row for row in group if row["window_size"] == window)["valid_ratio_delta_vs_64"]) > 0.01
            for group in pair_groups.values()
        )
        for window in WINDOW_SIZES[1:]
    }
    instability_improvement_counts = {
        str(window): sum(
            float(next(row for row in group if row["window_size"] == window)["instability_delta_vs_64"]) < -0.01
            for group in pair_groups.values()
        )
        for window in WINDOW_SIZES[1:]
    }
    root_reason_counts = Counter(row["dominant_invalid_reason"] for row in low_rows)
    summary = {
        "current_pairs": {
            **pair_source,
            "involved_channel_count": len(involved_channels),
            "stage4_2a_reported_low_valid_pair_count": stage4_2a["reported_low_valid_pair_count"],
        },
        "window_sizes": list(WINDOW_SIZES),
        "window_scale_summary": summary_by_window,
        "low_valid_ratio_at_64": {
            "pair_count": len(low_rows),
            "dominant_reason_counts": dict(root_reason_counts),
            "frequent_invalid_channels": frequent_channels[:15],
        },
        "window_length_effect": {
            "valid_ratio_median_delta_256_vs_64": float(longest["median_valid_ratio"] - baseline["median_valid_ratio"]),
            "strength_median_delta_256_vs_64": float(longest["median_strength"] - baseline["median_strength"]),
            "instability_median_delta_256_vs_64": float(longest["median_instability"] - baseline["median_instability"]),
            "valid_ratio_improved_pair_count_by_window": valid_improvement_counts,
            "instability_reduced_pair_count_by_window": instability_improvement_counts,
            "max_strong_and_stable_count": max(row["strong_and_stable_count"] for row in window_summary),
            "window_with_max_strong_and_stable_count": [
                row["window_size"] for row in window_summary
                if row["strong_and_stable_count"] == max(item["strong_and_stable_count"] for item in window_summary)
            ],
            "any_window_has_all_40_strong_and_stable": any(row["strong_and_stable_count"] == 40 for row in window_summary),
        },
        "diagnostic_thresholds": {
            "variance_epsilon": VARIANCE_EPSILON,
            "min_valid_ratio": args.min_valid_ratio,
            "min_strength": args.min_strength,
            "max_instability": args.max_instability,
            "status": "diagnostic_only_not_written_to_model",
        },
        "preprocessing": preprocessing,
        "stage4_2a_sources": {
            "summary": stage4_2a["summary_path"],
            "selected_pair_diagnostics": stage4_2a["diagnostics_path"],
        },
        "test_set_or_labels_read": False,
        "model_modified": False,
        "structure_pairs_modified": False,
        "warnings": [
            "Window sizes 96, 128, and 256 are diagnostic only and do not change the trained model.",
            "No new relation pairs were selected.",
            "No test data, test labels, anomaly metrics, or fusion experiments were used.",
        ],
    }
    write_json(output_dir / "relation_validity_diagnosis_summary.json", summary)
    print(json.dumps(json_safe({
        "low_valid_pair_count_at_64": len(low_rows),
        "dominant_reason_counts": dict(root_reason_counts),
        "window_scale_summary": window_summary,
    }), indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 4.2B: diagnose current relation validity and window scale.")
    parser.add_argument("--project-root", default="D:/workspace/python/mscad")
    parser.add_argument("--output-dir", default="outputs/stage4_2b_relation_validity")
    parser.add_argument("--min-valid-ratio", type=float, default=0.95)
    parser.add_argument("--min-strength", type=float, default=0.50)
    parser.add_argument("--max-instability", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
