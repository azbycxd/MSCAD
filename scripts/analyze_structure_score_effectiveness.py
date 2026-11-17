from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

# Keep the Windows Conda OpenMP behavior consistent with evaluate.py.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.factory import build_dataloaders  # noqa: E402
from models.mscad import MSCAD  # noqa: E402
from scripts.run_multiseed_swat import load_registry  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.metrics import binary_metrics  # noqa: E402
from utils.pot import POTThreshold  # noqa: E402
from utils.seed import set_seed  # noqa: E402


SCORE_NAMES = ("rec", "short", "long")
PAIR_NAMES = (("rec", "short"), ("rec", "long"), ("short", "long"))


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    return seeds


def build_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def registry_ft_sources(registry: dict[str, Any], seed: int) -> tuple[Path, Path]:
    try:
        ft = registry["methods"]["ft"]
        checkpoint_pattern = str(ft["checkpoint_pattern"]["multiseed"])
        train_config = str(ft["train_config"])
    except KeyError as exc:
        raise KeyError(f"Missing FT metadata in experiment registry: {exc}") from exc

    if checkpoint_pattern.startswith("TODO") or train_config.startswith("TODO"):
        raise ValueError("FT checkpoint_pattern.multiseed and train_config must not be TODO")

    checkpoint = Path(checkpoint_pattern.format(seed=seed))
    config = Path(train_config)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not config.is_absolute():
        config = ROOT / config
    return checkpoint, config


def planned_outputs(output_root: Path, seeds: list[int]) -> list[Path]:
    paths = [
        output_root / "single_score_metrics.csv",
        output_root / "score_correlations.csv",
        output_root / "complementary_detection.csv",
    ]
    for seed in seeds:
        seed_dir = output_root / f"seed_{seed}"
        for split in ("train", "test"):
            paths.extend(seed_dir / f"{split}_{name}_score.npy" for name in SCORE_NAMES)
            paths.append(seed_dir / f"{split}_label.npy")
        paths.append(seed_dir / "score_distributions.png")
    return paths


def refuse_overwrite(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    shown = "\n".join(f"- {path.relative_to(ROOT)}" for path in existing[:20])
    suffix = "\n- ..." if len(existing) > 20 else ""
    raise SystemExit(f"Refusing to overwrite existing structure-analysis outputs:\n{shown}{suffix}")


@torch.no_grad()
def collect_scores(
    model: MSCAD, loader: torch.utils.data.DataLoader, device: torch.device
) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = {name: [] for name in SCORE_NAMES}
    labels: list[np.ndarray] = []
    model.eval()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        x_hat = model(xb)
        components = model.score_components(xb, x_hat)
        for name in SCORE_NAMES:
            values[name].append(components[name].detach().cpu().numpy())
        labels.append(yb.detach().cpu().numpy())
    return {
        **{name: np.concatenate(values[name]).astype(np.float32) for name in SCORE_NAMES},
        "label": np.concatenate(labels).astype(np.int64),
    }


def load_model(cfg: dict[str, Any], checkpoint: Path, feature_count: int, device: torch.device) -> MSCAD:
    model_cfg = cfg["model"]
    model = MSCAD(
        in_channels=feature_count,
        hidden_channels=int(model_cfg["hidden_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
        dropout=float(model_cfg["dropout"]),
        cov_type=str(cfg["train"]["cov_type"]),
        short_window=int(cfg["data"]["short_window"]),
        backbone_variant=str(model_cfg.get("backbone_variant", "base")),
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"], strict=True)
    if "structure_pairs" in state:
        model.set_structure_pairs(state["structure_pairs"])
    return model


def fit_train_threshold(train_score: np.ndarray, eval_cfg: dict[str, Any]) -> tuple[float, str]:
    score = np.asarray(train_score, dtype=float)
    score = score[np.isfinite(score)]
    if score.size == 0:
        raise ValueError("Training score contains no finite values")

    mode = str(eval_cfg.get("threshold_mode", "pot")).lower()
    if mode == "pot":
        q = float(eval_cfg.get("pot_q", 0.98))
        target_prob = float(eval_cfg.get("pot_target_prob", 1e-3))
        try:
            threshold = POTThreshold(q=q, target_prob=target_prob).fit(score).tau
            if threshold is not None and np.isfinite(threshold):
                return float(threshold), "pot_train_only"
        except (ValueError, FloatingPointError):
            pass
        return float(np.quantile(score, min(0.999, max(q, 0.995)))), "pot_train_only_quantile_fallback"
    if mode == "quantile":
        q = float(eval_cfg.get("threshold_quantile", 0.99))
        return float(np.quantile(score, q)), "quantile_train_only"
    if mode in {"mean_k_std", "mean3std"}:
        k = 3.0 if mode == "mean3std" else float(eval_cfg.get("mean_k", 3.0))
        return float(np.mean(score) + k * np.std(score)), f"{mode}_train_only"
    raise ValueError(f"Unsupported threshold mode: {mode}")


def safe_correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan"), float("nan"), int(x.size)
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic), int(x.size)


def save_arrays(seed_dir: Path, split: str, values: dict[str, np.ndarray]) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    for name in SCORE_NAMES:
        np.save(seed_dir / f"{split}_{name}_score.npy", values[name])
    np.save(seed_dir / f"{split}_label.npy", values["label"])


def distribution_limit(train: np.ndarray, normal: np.ndarray, anomaly: np.ndarray) -> float:
    chunks = [part[np.isfinite(part)] for part in (train, normal, anomaly) if part.size]
    merged = np.concatenate(chunks) if chunks else np.asarray([1.0])
    upper = float(np.quantile(merged, 0.995))
    return upper if upper > 0 else float(np.max(merged) + 1.0)


def save_distribution_plot(
    seed: int,
    train_values: dict[str, np.ndarray],
    test_values: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    labels = test_values["label"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"train": "#6b7280", "normal": "#2563eb", "anomaly": "#dc2626"}
    for axis, name in zip(axes, SCORE_NAMES):
        train_score = train_values[name]
        normal_score = test_values[name][labels == 0]
        anomaly_score = test_values[name][labels == 1]
        upper = distribution_limit(train_score, normal_score, anomaly_score)
        bins = np.linspace(0.0, upper, 80)
        axis.hist(np.clip(train_score, 0.0, upper), bins=bins, density=True, histtype="step", linewidth=1.4,
                  color=colors["train"], label="train")
        axis.hist(np.clip(normal_score, 0.0, upper), bins=bins, density=True, alpha=0.35,
                  color=colors["normal"], label="test normal")
        axis.hist(np.clip(anomaly_score, 0.0, upper), bins=bins, density=True, alpha=0.35,
                  color=colors["anomaly"], label="test anomaly")
        axis.set_title(f"{name} score")
        axis.set_xlabel("score (clipped at pooled 99.5th percentile)")
        axis.set_ylabel("density")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Seed {seed}: single-score distributions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_seed(
    seed: int, registry: dict[str, Any], output_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    checkpoint, config_path = registry_ft_sources(registry, seed)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Registry checkpoint does not exist for seed {seed}: {checkpoint}")
    if not config_path.exists():
        raise FileNotFoundError(f"Registry train_config does not exist: {config_path}")

    cfg = load_config(str(config_path))
    cfg["seed"] = seed
    set_seed(seed)
    device = build_device(str(cfg.get("device", "cuda")))
    bundle = build_dataloaders(cfg, pin_memory=(device.type == "cuda"))
    model = load_model(cfg, checkpoint, len(bundle.feature_names), device)

    train_values = collect_scores(model, bundle.train_loader, device)
    test_values = collect_scores(model, bundle.test_loader, device)
    seed_dir = output_root / f"seed_{seed}"
    save_arrays(seed_dir, "train", train_values)
    save_arrays(seed_dir, "test", test_values)

    metrics_rows: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    predictions: dict[str, np.ndarray] = {}
    for name in SCORE_NAMES:
        threshold, threshold_method = fit_train_threshold(train_values[name], cfg["eval"])
        prediction = (test_values[name] > threshold).astype(np.int64)
        metrics = binary_metrics(test_values["label"], prediction, test_values[name])
        thresholds[name] = threshold
        predictions[name] = prediction
        metrics_rows.append(
            {
                "seed": seed,
                "score_type": name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "aupr": metrics["aupr"],
                "auroc": metrics["auroc"],
                "threshold": threshold,
                "threshold_method": threshold_method,
                "train_count": len(train_values[name]),
                "test_count": len(test_values[name]),
                "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            }
        )

    correlation_rows: list[dict[str, Any]] = []
    for split, values in (("train", train_values), ("test", test_values)):
        for left, right in PAIR_NAMES:
            pearson, spearman, count = safe_correlations(values[left], values[right])
            correlation_rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "score_pair": f"{left}_vs_{right}",
                    "pearson": pearson,
                    "spearman": spearman,
                    "n_samples": count,
                }
            )

    anomaly = test_values["label"] == 1
    rec_missed = anomaly & (predictions["rec"] == 0)
    rec_missed_count = int(rec_missed.sum())
    short_complement = int((rec_missed & (predictions["short"] == 1)).sum())
    long_complement = int((rec_missed & (predictions["long"] == 1)).sum())
    complementary_row = {
        "seed": seed,
        "true_anomaly_count": int(anomaly.sum()),
        "rec_detected_count": int((anomaly & (predictions["rec"] == 1)).sum()),
        "rec_missed_count": rec_missed_count,
        "rec_missed_short_detected_count": short_complement,
        "rec_missed_long_detected_count": long_complement,
        "short_complement_rate_among_rec_missed": short_complement / max(1, rec_missed_count),
        "long_complement_rate_among_rec_missed": long_complement / max(1, rec_missed_count),
        "rec_threshold": thresholds["rec"],
        "short_threshold": thresholds["short"],
        "long_threshold": thresholds["long"],
    }

    save_distribution_plot(seed, train_values, test_values, seed_dir / "score_distributions.png")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics_rows, correlation_rows, complementary_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze rec/short/long score effectiveness from fixed multiseed FT checkpoints."
    )
    parser.add_argument("--registry", default="configs/experiment_registry.yaml")
    parser.add_argument("--seeds", type=parse_seeds, default=[1, 2, 3])
    parser.add_argument("--output-root", default="results/structure_analysis")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = ROOT / registry_path
    registry = load_registry(str(registry_path.relative_to(ROOT)))
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    targets = planned_outputs(output_root, args.seeds)
    refuse_overwrite(targets)
    for seed in args.seeds:
        checkpoint, config = registry_ft_sources(registry, seed)
        status = "exists" if checkpoint.exists() else "MISSING"
        print(f"seed={seed} checkpoint={checkpoint.relative_to(ROOT)} [{status}]")
        print(f"seed={seed} config={config.relative_to(ROOT)} [{'exists' if config.exists() else 'MISSING'}]")
        print(f"seed={seed} output={output_root.relative_to(ROOT) / f'seed_{seed}'}")
    if args.dry_run:
        return

    metric_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    complementary_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        print(f"Analyzing seed {seed}...")
        metrics, correlations, complementary = analyze_seed(seed, registry, output_root)
        metric_rows.extend(metrics)
        correlation_rows.extend(correlations)
        complementary_rows.append(complementary)

    write_csv(
        output_root / "single_score_metrics.csv",
        metric_rows,
        [
            "seed", "score_type", "precision", "recall", "f1", "aupr", "auroc", "threshold",
            "threshold_method", "train_count", "test_count", "checkpoint",
        ],
    )
    write_csv(
        output_root / "score_correlations.csv",
        correlation_rows,
        ["seed", "split", "score_pair", "pearson", "spearman", "n_samples"],
    )
    write_csv(
        output_root / "complementary_detection.csv",
        complementary_rows,
        [
            "seed", "true_anomaly_count", "rec_detected_count", "rec_missed_count",
            "rec_missed_short_detected_count", "rec_missed_long_detected_count",
            "short_complement_rate_among_rec_missed", "long_complement_rate_among_rec_missed",
            "rec_threshold", "short_threshold", "long_threshold",
        ],
    )
    print(f"Saved analysis to {output_root.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
