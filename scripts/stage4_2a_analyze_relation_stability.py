from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
if str(DEFAULT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ROOT))

from data.swat import SWaTWindowDataset, _load_swat_csv  # noqa: E402
from data.transforms import StandardScaler  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.cov import covariance_matrix  # noqa: E402
from utils.seed import set_seed  # noqa: E402


PAIR_FIELDS = [
    "pair_id",
    "channel_i",
    "channel_j",
    "channel_i_index",
    "channel_j_index",
    "total_window_count",
    "valid_window_count",
    "valid_ratio",
    "mean_corr",
    "median_corr",
    "std_corr",
    "mad_corr",
    "min_corr",
    "max_corr",
    "q05_corr",
    "q25_corr",
    "q75_corr",
    "q95_corr",
    "abs_median_corr",
    "mean_fisher",
    "median_fisher",
    "std_fisher",
    "mad_fisher",
    "robust_scale_fisher",
    "q05_fisher",
    "q95_fisher",
    "relation_strength",
    "relation_instability",
    "used_by_current_method",
    "current_pair_rank",
    "current_selection_source",
    "current_min_abs_corr",
    "current_top_k",
    "selected_seed_1",
    "selected_seed_2",
    "selected_seed_3",
    "selected_seed_count",
    "strong_relation",
    "stable_relation",
    "strong_and_stable",
    "diagnostic_exclusion_reason",
]


def fisher_transform(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.arctanh(np.clip(values, -0.999, 0.999))


def _median_absolute_deviation(values: np.ndarray, median: float) -> float:
    return float(np.median(np.abs(values - median)))


def summarize_correlation_samples(values: np.ndarray, total_window_count: int | None = None) -> dict[str, float | int]:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    total = int(raw.size if total_window_count is None else total_window_count)
    valid = raw[np.isfinite(raw)]
    if valid.size == 0:
        return empty_statistics(total)
    valid = np.clip(valid, -0.999, 0.999)
    fisher = fisher_transform(valid)
    median_corr = float(np.median(valid))
    median_fisher = float(np.median(fisher))
    mad_fisher = _median_absolute_deviation(fisher, median_fisher)
    return {
        "total_window_count": total,
        "valid_window_count": int(valid.size),
        "valid_ratio": float(valid.size / max(1, total)),
        "mean_corr": float(np.mean(valid)),
        "median_corr": median_corr,
        "std_corr": float(np.std(valid)),
        "mad_corr": _median_absolute_deviation(valid, median_corr),
        "min_corr": float(np.min(valid)),
        "max_corr": float(np.max(valid)),
        "q05_corr": float(np.quantile(valid, 0.05)),
        "q25_corr": float(np.quantile(valid, 0.25)),
        "q75_corr": float(np.quantile(valid, 0.75)),
        "q95_corr": float(np.quantile(valid, 0.95)),
        "abs_median_corr": abs(median_corr),
        "mean_fisher": float(np.mean(fisher)),
        "median_fisher": median_fisher,
        "std_fisher": float(np.std(fisher)),
        "mad_fisher": mad_fisher,
        "robust_scale_fisher": float(1.4826 * mad_fisher),
        "q05_fisher": float(np.quantile(fisher, 0.05)),
        "q95_fisher": float(np.quantile(fisher, 0.95)),
        "relation_strength": abs(median_corr),
        "relation_instability": float(1.4826 * mad_fisher),
    }


def empty_statistics(total_window_count: int) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "total_window_count": int(total_window_count),
        "valid_window_count": 0,
        "valid_ratio": 0.0,
    }
    for name in (
        "mean_corr", "median_corr", "std_corr", "mad_corr", "min_corr", "max_corr",
        "q05_corr", "q25_corr", "q75_corr", "q95_corr", "abs_median_corr",
        "mean_fisher", "median_fisher", "std_fisher", "mad_fisher",
        "robust_scale_fisher", "q05_fisher", "q95_fisher", "relation_strength",
        "relation_instability",
    ):
        row[name] = float("nan")
    return row


def window_pair_correlations(
    windows: np.ndarray | torch.Tensor,
    variance_epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = torch.as_tensor(windows, dtype=torch.float64)
    channel_count = int(tensor.shape[-1])
    pair_i, pair_j = torch.triu_indices(channel_count, channel_count, offset=1)
    corr, valid = _window_pair_correlations_tensor(tensor, pair_i, pair_j, variance_epsilon)
    out = corr.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()
    out[~valid_np] = np.nan
    return out, pair_i.numpy(), pair_j.numpy()


def _window_pair_correlations_tensor(
    windows: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    cov = covariance_matrix(windows)
    variance = torch.diagonal(cov, dim1=-2, dim2=-1)
    vi = variance[:, pair_i]
    vj = variance[:, pair_j]
    valid = (vi > variance_epsilon) & (vj > variance_epsilon)
    denominator = torch.sqrt(torch.clamp(vi * vj, min=variance_epsilon**2))
    corr = cov[:, pair_i, pair_j] / denominator
    corr = torch.clamp(corr, -0.999, 0.999)
    return corr, valid


def classify_row(row: dict[str, Any], min_strength: float, max_instability: float, min_valid_ratio: float) -> None:
    valid_ratio = float(row["valid_ratio"])
    strength = float(row["relation_strength"])
    instability = float(row["relation_instability"])
    has_valid = int(row["valid_window_count"]) > 0
    strong = has_valid and math.isfinite(strength) and strength >= min_strength
    stable = has_valid and math.isfinite(instability) and instability <= max_instability
    selected = valid_ratio >= min_valid_ratio and strong and stable
    row["strong_relation"] = bool(strong)
    row["stable_relation"] = bool(stable)
    row["strong_and_stable"] = bool(selected)
    if not has_valid:
        reason = "near_constant_channel"
    elif valid_ratio < min_valid_ratio:
        reason = "low_valid_ratio"
    elif not strong and not stable:
        reason = "weak_and_unstable"
    elif not strong:
        reason = "weak_relation"
    elif not stable:
        reason = "unstable_relation"
    else:
        reason = "selected"
    row["diagnostic_exclusion_reason"] = reason


@dataclass
class StreamingPairStatistics:
    pair_count: int
    histogram_bins: int = 4001

    def __post_init__(self) -> None:
        p = self.pair_count
        self.total_windows = 0
        self.count = np.zeros(p, dtype=np.int64)
        self.sum_corr = np.zeros(p, dtype=np.float64)
        self.sum_sq_corr = np.zeros(p, dtype=np.float64)
        self.sum_fisher = np.zeros(p, dtype=np.float64)
        self.sum_sq_fisher = np.zeros(p, dtype=np.float64)
        self.min_corr = np.full(p, np.inf, dtype=np.float64)
        self.max_corr = np.full(p, -np.inf, dtype=np.float64)
        self.histogram = np.zeros((p, self.histogram_bins), dtype=np.uint32)
        self._offsets = np.arange(p, dtype=np.int64) * self.histogram_bins
        self._centers = np.linspace(-0.999, 0.999, self.histogram_bins, dtype=np.float64)

    def update(self, correlations: np.ndarray, valid: np.ndarray) -> None:
        corr = np.asarray(correlations, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)
        self.total_windows += corr.shape[0]
        safe = np.where(valid, np.clip(corr, -0.999, 0.999), 0.0)
        fisher = np.where(valid, fisher_transform(safe), 0.0)
        batch_count = valid.sum(axis=0, dtype=np.int64)
        self.count += batch_count
        self.sum_corr += safe.sum(axis=0, dtype=np.float64)
        self.sum_sq_corr += np.square(safe).sum(axis=0, dtype=np.float64)
        self.sum_fisher += fisher.sum(axis=0, dtype=np.float64)
        self.sum_sq_fisher += np.square(fisher).sum(axis=0, dtype=np.float64)
        batch_min = np.min(np.where(valid, safe, np.inf), axis=0)
        batch_max = np.max(np.where(valid, safe, -np.inf), axis=0)
        self.min_corr = np.minimum(self.min_corr, batch_min)
        self.max_corr = np.maximum(self.max_corr, batch_max)

        bin_index = np.rint((safe + 0.999) * (self.histogram_bins - 1) / 1.998).astype(np.int64)
        composite = bin_index + self._offsets[None, :]
        counts = np.bincount(composite[valid], minlength=self.pair_count * self.histogram_bins)
        self.histogram += counts.reshape(self.pair_count, self.histogram_bins).astype(np.uint32)

    def _hist_quantile(self, pair: int, q: float) -> float:
        hist = self.histogram[pair].astype(np.int64)
        count = int(hist.sum())
        if count == 0:
            return float("nan")
        target = max(1, int(math.ceil(q * count)))
        index = int(np.searchsorted(np.cumsum(hist), target, side="left"))
        return float(self._centers[min(index, self.histogram_bins - 1)])

    def _hist_mad(self, pair: int, median: float, fisher: bool = False) -> float:
        hist = self.histogram[pair].astype(np.int64)
        if int(hist.sum()) == 0:
            return float("nan")
        centers = fisher_transform(self._centers) if fisher else self._centers
        distances = np.abs(centers - median)
        order = np.argsort(distances)
        ordered_counts = hist[order]
        target = max(1, int(math.ceil(0.5 * int(hist.sum()))))
        selected = int(np.searchsorted(np.cumsum(ordered_counts), target, side="left"))
        return float(distances[order[min(selected, len(order) - 1)]])

    def finalize_pair(self, pair: int) -> dict[str, float | int]:
        count = int(self.count[pair])
        if count == 0:
            return empty_statistics(self.total_windows)
        mean_corr = self.sum_corr[pair] / count
        mean_fisher = self.sum_fisher[pair] / count
        std_corr = math.sqrt(max(0.0, self.sum_sq_corr[pair] / count - mean_corr**2))
        std_fisher = math.sqrt(max(0.0, self.sum_sq_fisher[pair] / count - mean_fisher**2))
        median_corr = self._hist_quantile(pair, 0.50)
        median_fisher = float(np.arctanh(np.clip(median_corr, -0.999, 0.999)))
        mad_corr = self._hist_mad(pair, median_corr, fisher=False)
        mad_fisher = self._hist_mad(pair, median_fisher, fisher=True)
        robust = 1.4826 * mad_fisher
        return {
            "total_window_count": int(self.total_windows),
            "valid_window_count": count,
            "valid_ratio": float(count / max(1, self.total_windows)),
            "mean_corr": float(mean_corr),
            "median_corr": float(median_corr),
            "std_corr": float(std_corr),
            "mad_corr": float(mad_corr),
            "min_corr": float(self.min_corr[pair]),
            "max_corr": float(self.max_corr[pair]),
            "q05_corr": self._hist_quantile(pair, 0.05),
            "q25_corr": self._hist_quantile(pair, 0.25),
            "q75_corr": self._hist_quantile(pair, 0.75),
            "q95_corr": self._hist_quantile(pair, 0.95),
            "abs_median_corr": abs(float(median_corr)),
            "mean_fisher": float(mean_fisher),
            "median_fisher": median_fisher,
            "std_fisher": float(std_fisher),
            "mad_fisher": float(mad_fisher),
            "robust_scale_fisher": float(robust),
            "q05_fisher": float(np.arctanh(np.clip(self._hist_quantile(pair, 0.05), -0.999, 0.999))),
            "q95_fisher": float(np.arctanh(np.clip(self._hist_quantile(pair, 0.95), -0.999, 0.999))),
            "relation_strength": abs(float(median_corr)),
            "relation_instability": float(robust),
        }


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_current_pair_metadata(project_root: Path, cfg: dict[str, Any]) -> tuple[dict[int, list[list[int]]], dict[str, Any]]:
    pairs_by_seed: dict[int, list[list[int]]] = {}
    feature_names: list[str] | None = None
    for seed in (1, 2, 3):
        checkpoint = project_root / "results" / "multiseed" / f"seed_{seed}" / "ft" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing FT checkpoint for seed {seed}: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        pairs_by_seed[seed] = [[int(a), int(b)] for a, b in state.get("structure_pairs", [])]
        names = [str(name) for name in state.get("feature_names", [])]
        if names:
            if feature_names is not None and names != feature_names:
                raise ValueError("Feature-name order differs across seed checkpoints")
            feature_names = names
    pair_cfg = cfg["train"]["structure_pairs"]
    return pairs_by_seed, {
        "feature_names": feature_names,
        "top_k": int(pair_cfg.get("topk", 0)),
        "min_abs_corr": float(pair_cfg.get("min_abs_corr", 0.0)),
        "source": "checkpoint.structure_pairs generated by train.py::select_corr_pairs(bundle.x_train_norm)",
        "same_pairs_across_seeds": pairs_by_seed[1] == pairs_by_seed[2] == pairs_by_seed[3],
    }


def build_normal_training_loader(
    project_root: Path,
    cfg: dict[str, Any],
    smoke: bool,
) -> tuple[DataLoader, list[str], dict[str, Any]]:
    data_cfg = cfg["data"]
    train_path = resolve_path(project_root, data_cfg["train_csv"])
    x_all, y_all, feature_names = _load_swat_csv(
        str(train_path), str(data_cfg["label_col"]), str(data_cfg["timestamp_col"])
    )
    window_size = int(data_cfg["window_size"])
    n = len(x_all)
    n_val = int(max(window_size * 2, round(n * float(data_cfg.get("val_ratio", 0.2)))))
    n_val = min(max(window_size * 2, n_val), n - window_size - 1)
    split = n - n_val
    x_train_raw = x_all[:split]
    y_train_raw = y_all[:split]
    if np.any(y_train_raw != 0):
        raise ValueError("Configured SWaT normal training split contains non-normal labels")
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train_raw)
    dataset = SWaTWindowDataset(
        x_train_norm,
        y_train_raw,
        window_size=window_size,
        stride=int(data_cfg["stride_train"]),
    )
    full_window_count = len(dataset)
    if smoke:
        dataset = Subset(dataset, range(min(4096, full_window_count)))
    loader = DataLoader(
        dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    preprocessing = {
        "dataset": "SWaT normal training only",
        "train_csv": str(Path(data_cfg["train_csv"]).as_posix()),
        "test_data_read": False,
        "loader_reused": "data.swat._load_swat_csv",
        "scaler_reused": "data.transforms.StandardScaler",
        "window_dataset_reused": "data.swat.SWaTWindowDataset",
        "missing_value_handling": "pd.to_numeric(errors=coerce).fillna(0.0)",
        "scaler_fit_scope": "x_train_raw before validation tail, identical to build_swat_dataloaders",
        "normal_csv_row_count": int(n),
        "training_split_row_count": int(split),
        "validation_tail_row_count": int(n_val),
        "window_size": window_size,
        "stride_train": int(data_cfg["stride_train"]),
        "short_window_not_analyzed": int(data_cfg["short_window"]),
        "full_training_window_count": int(full_window_count),
        "analyzed_window_count": int(len(dataset)),
        "smoke": bool(smoke),
    }
    return loader, feature_names, preprocessing


def enrich_selection_fields(
    row: dict[str, Any],
    pair: tuple[int, int],
    pairs_by_seed: dict[int, list[list[int]]],
    metadata: dict[str, Any],
) -> None:
    ranks: list[int] = []
    selected_count = 0
    for seed in (1, 2, 3):
        rank_map = {tuple(sorted(item)): rank for rank, item in enumerate(pairs_by_seed[seed], start=1)}
        selected = pair in rank_map
        row[f"selected_seed_{seed}"] = bool(selected)
        if selected:
            selected_count += 1
            ranks.append(rank_map[pair])
    row["selected_seed_count"] = selected_count
    row["used_by_current_method"] = selected_count > 0
    row["current_pair_rank"] = min(ranks) if ranks else ""
    row["current_selection_source"] = metadata["source"] if ranks else ""
    row["current_min_abs_corr"] = metadata["min_abs_corr"] if ranks else ""
    row["current_top_k"] = metadata["top_k"] if ranks else ""


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def create_figures(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=False, exist_ok=False)
    valid = [row for row in rows if math.isfinite(float(row["relation_strength"]))]
    selected = [row for row in valid if row["used_by_current_method"]]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter([r["relation_strength"] for r in valid], [r["relation_instability"] for r in valid],
               s=12, alpha=0.35, color="#4b5563", label="all valid pairs")
    ax.scatter([r["relation_strength"] for r in selected], [r["relation_instability"] for r in selected],
               s=30, alpha=0.9, color="#dc2626", label="current pairs")
    ax.set_xlabel("relation strength |median corr|")
    ax.set_ylabel("relation instability (1.4826 x Fisher MAD)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "strength_vs_instability.png", dpi=180)
    plt.close(fig)

    selected_sorted = sorted(selected, key=lambda row: int(row["current_pair_rank"]))
    fig_height = max(6.0, len(selected_sorted) * 0.23)
    fig, ax = plt.subplots(figsize=(9.0, fig_height))
    y = np.arange(len(selected_sorted))
    medians = np.asarray([r["median_corr"] for r in selected_sorted], dtype=float)
    lower = medians - np.asarray([r["q05_corr"] for r in selected_sorted], dtype=float)
    upper = np.asarray([r["q95_corr"] for r in selected_sorted], dtype=float) - medians
    ax.errorbar(medians, y, xerr=np.vstack([lower, upper]), fmt="o", color="#2563eb", ecolor="#94a3b8",
                capsize=2, markersize=4)
    ax.axvline(0.0, color="#111827", linewidth=0.8)
    ax.set_yticks(y, [r["pair_id"] for r in selected_sorted], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("window correlation (q05 / median / q95)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "current_pair_corr_interval.png", dpi=180)
    plt.close(fig)

    for field, filename, label in (
        ("relation_strength", "relation_strength_histogram.png", "relation strength |median corr|"),
        ("relation_instability", "relation_instability_histogram.png", "relation instability"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        values = [float(row[field]) for row in valid if math.isfinite(float(row[field]))]
        ax.hist(values, bins=50, color="#2563eb", alpha=0.75)
        ax.set_xlabel(label)
        ax.set_ylabel("pair count")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=180)
        plt.close(fig)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    config_path = project_root / "configs" / "swat_improve_v4_struct_finetune.yaml"
    cfg = load_config(str(config_path))
    pairs_by_seed, pair_metadata = load_current_pair_metadata(project_root, cfg)
    output_dir = resolve_path(project_root, args.output_dir)
    if args.smoke:
        output_dir = project_root / "outputs" / "stage4_2a_relation_stability_smoke"

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    print("Reuse files:")
    for value in (
        "data/swat.py (_load_swat_csv, SWaTWindowDataset)",
        "data/transforms.py (StandardScaler)",
        "utils/cov.py (covariance_matrix)",
        "configs/swat_improve_v4_struct_finetune.yaml",
        "results/multiseed/seed_{1,2,3}/ft/best.pt (structure_pairs only)",
    ):
        print(f"- {value}")
    print("New files:")
    print("- scripts/stage4_2a_analyze_relation_stability.py")
    print("- tests/test_relation_stability_analysis.py")
    print(f"Config: window_size={cfg['data']['window_size']} stride={cfg['data']['stride_train']} "
          f"short_window={cfg['data']['short_window']} channel_count={cfg['model']['in_channels']}")
    print("Pair selection: train.py::select_corr_pairs at train.py:27")
    print("Not modified: models/, losses/, train.py, evaluate.py, utils/cov.py, data/, configs/")
    print(f"Output: {output_dir}")
    if args.dry_run:
        return {"dry_run": True, "output_dir": str(output_dir)}

    set_seed(int(args.seed))
    loader, feature_names, preprocessing = build_normal_training_loader(project_root, cfg, args.smoke)
    if pair_metadata["feature_names"] and feature_names != pair_metadata["feature_names"]:
        raise ValueError("Current normal-data channel order differs from checkpoint feature_names")

    channel_count = len(feature_names)
    pair_i, pair_j = torch.triu_indices(channel_count, channel_count, offset=1)
    pair_count = int(pair_i.numel())
    statistics = StreamingPairStatistics(pair_count=pair_count)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair_i_device = pair_i.to(device)
    pair_j_device = pair_j.to(device)
    for batch_index, (windows, _) in enumerate(loader, start=1):
        windows = windows.to(device, dtype=torch.float32, non_blocking=True)
        corr, valid = _window_pair_correlations_tensor(windows, pair_i_device, pair_j_device, 1e-8)
        statistics.update(corr.detach().cpu().numpy(), valid.detach().cpu().numpy())
        if batch_index % 50 == 0:
            print(f"Processed {statistics.total_windows}/{len(loader.dataset)} normal windows")

    rows: list[dict[str, Any]] = []
    for index, (i, j) in enumerate(zip(pair_i.tolist(), pair_j.tolist())):
        row: dict[str, Any] = {
            "pair_id": f"{i:02d}-{j:02d}:{feature_names[i].strip()}__{feature_names[j].strip()}",
            "channel_i": feature_names[i],
            "channel_j": feature_names[j],
            "channel_i_index": i,
            "channel_j_index": j,
            **statistics.finalize_pair(index),
        }
        enrich_selection_fields(row, (i, j), pairs_by_seed, pair_metadata)
        classify_row(row, args.min_strength, args.max_instability, args.min_valid_ratio)
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "all_pair_relation_stats.csv", rows, PAIR_FIELDS)
    current = sorted(
        (row for row in rows if row["used_by_current_method"]),
        key=lambda row: (-int(row["used_by_current_method"]), -float(row["relation_strength"])),
    )
    write_csv(output_dir / "current_selected_pair_diagnostics.csv", current, PAIR_FIELDS)
    strong_unstable = [
        row for row in rows
        if float(row["relation_strength"]) >= args.min_strength
        and float(row["relation_instability"]) > args.max_instability
    ]
    write_csv(output_dir / "strong_but_unstable_pairs.csv", strong_unstable, PAIR_FIELDS)
    candidates = [row for row in rows if row["strong_and_stable"]]
    write_csv(output_dir / "strong_and_stable_candidates.csv", candidates, PAIR_FIELDS)
    window_fields = [
        "pair_id", "channel_i", "channel_j", "current_pair_rank", "relation_strength",
        "relation_instability", "q05_corr", "median_corr", "q95_corr", "valid_ratio",
        "strong_relation", "stable_relation", "strong_and_stable",
    ]
    write_csv(output_dir / "current_selected_pair_window_stats.csv", current, window_fields)

    valid_pair_count = sum(float(row["valid_ratio"]) >= args.min_valid_ratio for row in rows)
    all_strong = sum(bool(row["strong_relation"]) for row in rows)
    all_stable = sum(bool(row["stable_relation"]) for row in rows)
    selected_strong = sum(bool(row["strong_relation"]) for row in current)
    selected_stable = sum(bool(row["stable_relation"]) for row in current)
    selected_both = sum(bool(row["strong_and_stable"]) for row in current)
    warnings = [
        "Only the SWaT normal training split was read; test data and test labels were not accessed.",
        "Quantiles and MAD use a 4001-bin streaming histogram over clipped correlations; resolution is approximately 0.0005.",
        "Diagnostic thresholds are provisional and were not written back to model configuration.",
        "Short-window relations were intentionally not analyzed in stage 4.2a.",
    ]
    if pair_metadata["same_pairs_across_seeds"]:
        warnings.append("Seeds 1, 2, and 3 use the same ordered set of current structure pairs.")
    summary = {
        "channel_count": channel_count,
        "all_pair_count": pair_count,
        "valid_pair_count": int(valid_pair_count),
        "current_selected_pair_count": len(current),
        "current_selected_strong_count": int(selected_strong),
        "current_selected_stable_count": int(selected_stable),
        "current_selected_strong_and_stable_count": int(selected_both),
        "all_strong_pair_count": int(all_strong),
        "all_stable_pair_count": int(all_stable),
        "all_strong_and_stable_count": len(candidates),
        "strong_but_unstable_pair_count": len(strong_unstable),
        "thresholds_used_for_diagnosis": {
            "min_valid_ratio": float(args.min_valid_ratio),
            "min_strength": float(args.min_strength),
            "max_instability": float(args.max_instability),
            "status": "diagnostic_only_not_final_method_parameters",
        },
        "preprocessing_summary": preprocessing,
        "current_pair_source": {
            **pair_metadata,
            "feature_names": feature_names,
            "pair_counts_by_seed": {str(seed): len(pairs) for seed, pairs in pairs_by_seed.items()},
        },
        "warnings": warnings,
    }
    write_json(output_dir / "relation_stability_summary.json", json_safe(summary))
    summary_rows = [
        {"metric": key, "value": value, "ratio": value / pair_count if "current_selected" not in key else value / max(1, len(current))}
        for key, value in (
            ("channel_count", channel_count),
            ("all_pair_count", pair_count),
            ("valid_pair_count", valid_pair_count),
            ("current_selected_pair_count", len(current)),
            ("current_selected_strong_count", selected_strong),
            ("current_selected_stable_count", selected_stable),
            ("current_selected_strong_and_stable_count", selected_both),
            ("all_strong_pair_count", all_strong),
            ("all_stable_pair_count", all_stable),
            ("all_strong_and_stable_count", len(candidates)),
            ("strong_but_unstable_pair_count", len(strong_unstable)),
        )
    ]
    write_csv(output_dir / "relation_stability_summary.csv", summary_rows, ["metric", "value", "ratio"])
    create_figures(rows, output_dir / "figures")
    print(json.dumps({key: summary[key] for key in (
        "channel_count", "all_pair_count", "current_selected_pair_count",
        "current_selected_strong_count", "current_selected_stable_count",
        "current_selected_strong_and_stable_count", "strong_but_unstable_pair_count",
    )}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4.2a: diagnose normal long-window relation stability only.")
    parser.add_argument("--project-root", default="D:/workspace/python/mscad")
    parser.add_argument("--output-dir", default="outputs/stage4_2a_relation_stability")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-strength", type=float, default=0.50)
    parser.add_argument("--max-instability", type=float, default=0.15)
    parser.add_argument("--min-valid-ratio", type=float, default=0.95)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
