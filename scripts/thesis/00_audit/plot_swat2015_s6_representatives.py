"""Render at most four preregistered S6 representative figures without PyTorch."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs" / "thesis_s6_swat_phenomenon_audit"
LAGS = np.asarray([1, 5, 10, 20, 30], dtype=np.int64)


def robust(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values)); scale = max(1.4826 * float(np.median(np.abs(values - center))), 1e-6)
    return center, scale


def self_residual(values: np.ndarray, channel: int, coefficient: np.ndarray, intercept: float, start: int, end: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    times = np.arange(start + int(LAGS.max()), end, dtype=np.int64)
    history = np.column_stack([values[times - lag, channel] for lag in LAGS]).astype(np.float64)
    result[times] = np.abs(values[times, channel] - (history @ coefficient + intercept))
    return result


def main() -> None:
    summary = json.loads((OUTPUT / "audit_summary.json").read_text(encoding="utf-8"))
    with (OUTPUT / "event_audit.csv").open(encoding="utf-8") as handle:
        events = {int(row["event_id"]): row for row in csv.DictReader(handle)}
    cache = ROOT / "outputs" / "thesis_midterm_s1" / "cache"
    manifest = json.loads((cache / "cache_manifest.json").read_text(encoding="utf-8"))
    names = manifest["column_order"]; index = {name: i for i, name in enumerate(names)}
    normal = np.load(cache / "normal_v1_features.npy", mmap_mode="r")
    attack = np.load(cache / "attack_v0_features.npy", mmap_mode="r")
    timestamps = np.load(cache / "attack_v0_timestamps.npy", mmap_mode="r")
    score_file = np.load(ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline" / "test_scores.npz")
    score_times = score_file["endpoint_timestamp"]; scores = score_file["endpoint_mae"]
    threshold = float(summary["a0_validation_q99"])
    scaler_file = np.load(ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline" / "scaler.npz")
    mean = scaler_file["mean"]; scale = scaler_file["effective_scale"]
    scaled_train = (np.asarray(normal[:396000], dtype=np.float32) - mean) / scale
    scaled_val = (np.asarray(normal[396000:], dtype=np.float32) - mean) / scale
    scaled_attack = (np.asarray(attack, dtype=np.float32) - mean) / scale
    v3i = json.loads((ROOT / "outputs" / "thesis_v3i_innovation" / "v3i1_predictor_manifest.json").read_text(encoding="utf-8"))
    segment_bounds = [(0, 298837), (298837, 449919)]

    for event_id in summary["representative_figure_event_ids"][:4]:
        row = events[int(event_id)]; start = int(np.datetime64(row["start"], "s").astype(np.int64)); end = int(np.datetime64(row["end"], "s").astype(np.int64))
        context = np.flatnonzero((timestamps >= start - 120) & (timestamps <= end + 120))
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True); plotted = 0
        for point in row["canonical_attack_points"].split(";")[:3]:
            if point not in index:
                continue
            channel = index[point]; axes[0].plot(timestamps[context], attack[context, channel], label=point, linewidth=.9)
            low, high = np.quantile(np.asarray(normal[:396000, channel]), [.005, .995]); axes[0].fill_between(timestamps[context], low, high, alpha=.08)
            if point in v3i["self_models"]:
                model = v3i["self_models"][point]; coefficient = np.asarray(model["coefficient"]); intercept = float(model["intercept"])
                train_r = self_residual(scaled_train, channel, coefficient, intercept, 0, len(scaled_train)); val_r = self_residual(scaled_val, channel, coefficient, intercept, 0, len(scaled_val))
                attack_r = np.full(len(attack), np.nan)
                for left, right in segment_bounds:
                    partial = self_residual(scaled_attack, channel, coefficient, intercept, left, right); valid = np.isfinite(partial); attack_r[valid] = partial[valid]
                center, rscale = robust(train_r[np.isfinite(train_r)]); attack_z = np.abs(attack_r - center) / rscale; val_z = np.abs(val_r[np.isfinite(val_r)] - center) / rscale
                axes[1].plot(timestamps[context], attack_z[context], label=point, linewidth=.9); axes[1].axhline(np.quantile(val_z, .995), linestyle="--", linewidth=.7)
            else:
                axes[1].plot(timestamps[context], np.r_[np.nan, np.abs(np.diff(attack[context, channel]))], label=f"{point} transition", linewidth=.9)
            plotted += 1
        if not plotted:
            axes[0].text(.5, .5, "No attack point maps to the 51-channel schema", transform=axes[0].transAxes, ha="center")
        score_mask = (score_times >= start - 120) & (score_times <= end + 120)
        axes[2].plot(score_times[score_mask], scores[score_mask], color="black", linewidth=.9, label="A0 endpoint MAE"); axes[2].axhline(threshold, color="red", linestyle="--", label="Validation q99")
        for axis in axes:
            axis.axvline(start, color="orange", linestyle="--"); axis.axvline(end, color="orange", linestyle="--"); axis.legend(loc="upper right", fontsize=7)
        axes[0].set_ylabel("raw value"); axes[1].set_ylabel("self z / transition"); axes[2].set_ylabel("A0 score"); axes[2].set_xlabel("Unix timestamp (s)")
        fig.suptitle(f"Event {event_id} — {row['auto_contextual_class']}"); fig.tight_layout(); fig.savefig(OUTPUT / "figures" / f"event_{int(event_id):02d}.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
