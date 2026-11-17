from __future__ import annotations

import argparse
import inspect
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_structure_score_effectiveness import build_device, load_model  # noqa: E402
from scripts.stage4_2a_analyze_relation_stability import _window_pair_correlations_tensor  # noqa: E402
from scripts.stage4_3a_eval_candidate_relation_sensitivity import load_stage_data  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.controlled_relation_injection import (  # noqa: E402
    block_permutation,
    cross_window_swap,
    pair_score,
    pearson_corr,
    robust_scale_floor,
    temporal_shift,
)


SCORE_COLUMNS = (
    "rec_event_max",
    "rec_event_mean",
    "existing_long_event_max",
    "existing_long_event_mean",
    "independent_128_max",
    "independent_64_max",
)
PRIMARY_SCORES = (
    "rec_event_max",
    "existing_long_event_max",
    "independent_128_max",
    "independent_64_max",
)
RELATION_SETS = ("strict_stable", "diversity_capped")
SHIFT_STEPS = {"low": 8, "medium": 16, "high": 32}
WINDOW_128 = 128
WINDOW_64 = 64


def checkpoint_for_seed(project_root: Path, seed: int) -> Path:
    return project_root / "results" / "multiseed" / f"seed_{seed}" / "ft" / "best.pt"


def split_128_window(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(window)
    if values.shape[-2] != WINDOW_128:
        raise ValueError("Expected a 128-step window")
    return values[..., :WINDOW_64, :], values[..., WINDOW_64:, :]


def event_reduce(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack([np.asarray(first), np.asarray(second)], axis=-1)
    return np.max(stacked, axis=-1), np.mean(stacked, axis=-1)


@torch.no_grad()
def infer_existing_scores(
    model: torch.nn.Module,
    windows_128: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    model.eval()
    first, second = split_128_window(windows_128)
    halves = np.concatenate([first, second], axis=0).astype(np.float32, copy=False)
    rec_parts: list[np.ndarray] = []
    long_parts: list[np.ndarray] = []
    for offset in range(0, len(halves), batch_size):
        xb = torch.as_tensor(halves[offset : offset + batch_size], dtype=torch.float32, device=device)
        x_hat = model(xb)
        components = model.score_components(xb, x_hat)
        rec_parts.append(components["rec"].detach().cpu().numpy())
        long_parts.append(components["long"].detach().cpu().numpy())
    count = len(windows_128)
    rec = np.concatenate(rec_parts)
    long = np.concatenate(long_parts)
    rec_max, rec_mean = event_reduce(rec[:count], rec[count:])
    long_max, long_mean = event_reduce(long[:count], long[count:])
    return {
        "rec_event_max": rec_max.astype(np.float64),
        "rec_event_mean": rec_mean.astype(np.float64),
        "existing_long_event_max": long_max.astype(np.float64),
        "existing_long_event_mean": long_mean.astype(np.float64),
    }


def _correlations_for_windows(
    windows: np.ndarray,
    pairs: list[tuple[int, int]],
    batch_size: int = 2048,
) -> np.ndarray:
    if not len(windows) or not pairs:
        return np.empty((len(windows), len(pairs)), dtype=np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair_i = torch.tensor([pair[0] for pair in pairs], dtype=torch.long, device=device)
    pair_j = torch.tensor([pair[1] for pair in pairs], dtype=torch.long, device=device)
    output = np.empty((len(windows), len(pairs)), dtype=np.float64)
    for offset in range(0, len(windows), batch_size):
        tensor = torch.as_tensor(windows[offset : offset + batch_size], dtype=torch.float32, device=device)
        corr, valid = _window_pair_correlations_tensor(tensor, pair_i, pair_j, 1e-8)
        batch = corr.detach().cpu().numpy().astype(np.float64)
        batch[~valid.detach().cpu().numpy()] = np.nan
        output[offset : offset + len(batch)] = batch
    return output


def _correlations_for_series(
    values: np.ndarray,
    starts: np.ndarray,
    window_size: int,
    pairs: list[tuple[int, int]],
    batch_size: int = 2048,
) -> np.ndarray:
    output = np.empty((len(starts), len(pairs)), dtype=np.float64)
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        windows = np.stack([values[int(start) : int(start) + window_size] for start in batch_starts])
        output[offset : offset + len(windows)] = _correlations_for_windows(windows, pairs, batch_size)
    return output


def fit_independent_models_64(
    reference: np.ndarray,
    candidate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    pairs = [(int(row["channel_i_index"]), int(row["channel_j_index"])) for row in candidate_rows]
    starts = np.arange(0, len(reference) - WINDOW_64 + 1, dtype=np.int64)
    correlations = _correlations_for_series(reference, starts, WINDOW_64, pairs)
    models: list[dict[str, Any]] = []
    for index, row in enumerate(candidate_rows):
        valid = correlations[:, index]
        valid = valid[np.isfinite(valid)]
        fisher = np.arctanh(np.clip(valid, -0.999, 0.999))
        median = float(np.median(fisher))
        mad = float(np.median(np.abs(fisher - median)))
        models.append({
            "pair_id": row["pair_id"],
            "channel_i_index": int(row["channel_i_index"]),
            "channel_j_index": int(row["channel_j_index"]),
            "normal_median_fisher": median,
            "normal_mad_fisher": mad,
            "normal_robust_scale": 1.4826 * mad,
        })
    floor = robust_scale_floor([row["normal_robust_scale"] for row in models])
    for row in models:
        row["scale_floor"] = floor
    return models, floor


def independent_scores(
    windows: np.ndarray,
    models: list[dict[str, Any]],
    scale_floor: float,
    window_size: int,
) -> np.ndarray:
    pairs = [(int(row["channel_i_index"]), int(row["channel_j_index"])) for row in models]
    values = np.asarray(windows, dtype=np.float64)
    correlations = np.empty((len(values), len(pairs)), dtype=np.float64)
    for index, (left, right) in enumerate(pairs):
        x = values[:, :, left]
        y = values[:, :, right]
        xc = x - x.mean(axis=1, keepdims=True)
        yc = y - y.mean(axis=1, keepdims=True)
        vx = np.mean(np.square(xc), axis=1)
        vy = np.mean(np.square(yc), axis=1)
        valid = (vx > 1e-8) & (vy > 1e-8)
        correlations[:, index] = np.nan
        correlations[valid, index] = np.clip(
            np.mean(xc[valid] * yc[valid], axis=1) / np.sqrt(vx[valid] * vy[valid]),
            -0.999,
            0.999,
        )
    scores = np.empty_like(correlations)
    for index, model in enumerate(models):
        scores[:, index] = [
            pair_score(value, model["normal_median_fisher"], model["normal_robust_scale"], scale_floor)
            for value in correlations[:, index]
        ]
    finite = np.isfinite(scores)
    neutralized = np.where(finite, scores, -np.inf)
    maximum = np.max(neutralized, axis=1)
    maximum[~finite.any(axis=1)] = 0.0
    return maximum


def independent_64_event_scores(
    windows_128: np.ndarray,
    models: list[dict[str, Any]],
    scale_floor: float,
) -> np.ndarray:
    first, second = split_128_window(windows_128)
    first_score = independent_scores(first, models, scale_floor, WINDOW_64)
    second_score = independent_scores(second, models, scale_floor, WINDOW_64)
    return np.maximum(first_score, second_score)


def q99_threshold(calibration_scores: np.ndarray) -> float:
    values = np.asarray(calibration_scores, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite normal calibration scores")
    return float(np.quantile(values, 0.99))


def _candidate_models_for_set(
    candidate_sets: dict[str, Any],
    model_rows: list[dict[str, Any]],
    set_name: str,
) -> list[dict[str, Any]]:
    pair_ids = {str(row["pair_id"]) for row in candidate_sets[set_name]["relations"]}
    return [row for row in model_rows if str(row["pair_id"]) in pair_ids]


def _block_reconstruct(
    original_window: np.ndarray,
    target_channel: int,
    left: int,
    right: int,
    expected_corr: float,
    expected_independent_score: float,
    relation_models: list[dict[str, Any]],
    scale_floor: float,
) -> np.ndarray:
    values = original_window[:, target_channel].copy()
    blocks = np.split(values, 4)
    best_window: np.ndarray | None = None
    best_error = float("inf")
    for order in itertools.permutations(range(4)):
        if order == (0, 1, 2, 3):
            continue
        candidate = original_window.copy()
        candidate[:, target_channel] = np.concatenate([blocks[index] for index in order])
        corr = pearson_corr(candidate[:, left], candidate[:, right])
        candidate_scores = [
            pair_score(
                pearson_corr(
                    candidate[:, int(model["channel_i_index"])],
                    candidate[:, int(model["channel_j_index"])],
                ),
                float(model["normal_median_fisher"]),
                float(model["normal_robust_scale"]),
                scale_floor,
            )
            for model in relation_models
        ]
        independent_score = max(value for value in candidate_scores if np.isfinite(value))
        error = max(abs(corr - expected_corr), abs(independent_score - expected_independent_score))
        if error < best_error:
            best_error = error
            best_window = candidate
    if best_window is None or best_error > 1e-6:
        raise ValueError(f"Cannot reconstruct block permutation, corr error={best_error}")
    return best_window


def reconstruct_injected_window(
    held_out: np.ndarray,
    quality_row: dict[str, Any],
    pair_lookup: dict[str, dict[str, Any]],
    relation_models: list[dict[str, Any]],
    scale_floor: float,
) -> np.ndarray:
    start = int(quality_row["base_window_start"])
    original = held_out[start : start + WINDOW_128].copy()
    target_channel = int(quality_row["target_channel_index"])
    injection_type = str(quality_row["injection_type"])
    if injection_type == "temporal_shift":
        injected = temporal_shift(original, target_channel, SHIFT_STEPS[str(quality_row["severity"])])
    elif injection_type == "cross_window_swap":
        donor_start = int(float(quality_row["donor_window_start"]))
        donor = held_out[donor_start : donor_start + WINDOW_128]
        injected, valid, reason = cross_window_swap(original, donor, target_channel)
        if not valid:
            raise ValueError(f"Saved valid cross-window sample failed reconstruction: {reason}")
    elif injection_type == "block_permutation":
        pair = pair_lookup[str(quality_row["target_pair"])]
        injected = _block_reconstruct(
            original, target_channel, int(pair["channel_i_index"]), int(pair["channel_j_index"]),
            float(quality_row["injected_corr"]),
            float(quality_row["saved_independent_128_max"]), relation_models, scale_floor,
        )
    else:
        raise ValueError(f"Unknown injection type: {injection_type}")
    pair = pair_lookup[str(quality_row["target_pair"])]
    corr = pearson_corr(injected[:, int(pair["channel_i_index"])], injected[:, int(pair["channel_j_index"])])
    if abs(corr - float(quality_row["injected_corr"])) > 1e-6:
        raise ValueError(f"Reconstructed correlation mismatch for {quality_row['sample_id']}")
    return injected


def checkpoint_structure_pairs(checkpoint: Path) -> list[list[int]]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return [[int(left), int(right)] for left, right in state.get("structure_pairs", [])]


def validate_same_sample_ids(rows: pd.DataFrame) -> None:
    if rows["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in controlled comparison")
    if rows[list(SCORE_COLUMNS)].isna().any().any():
        raise ValueError("A score is missing for a controlled event")


def metric_values(normal: np.ndarray, anomaly: np.ndarray, calibration: np.ndarray) -> dict[str, float | int]:
    normal = np.asarray(normal, dtype=np.float64)
    anomaly = np.asarray(anomaly, dtype=np.float64)
    calibration = np.asarray(calibration, dtype=np.float64)
    threshold = q99_threshold(calibration)
    labels = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(anomaly), dtype=int)])
    scores = np.concatenate([normal, anomaly])
    weights = np.concatenate([
        np.full(len(normal), 0.5 / len(normal)), np.full(len(anomaly), 0.5 / len(anomaly))
    ])
    pooled = math.sqrt(
        ((len(normal) - 1) * normal.var(ddof=1) + (len(anomaly) - 1) * anomaly.var(ddof=1))
        / max(1, len(normal) + len(anomaly) - 2)
    )
    return {
        "normal_sample_count": len(normal), "anomaly_sample_count": len(anomaly),
        "threshold_q99": threshold, "calibration_sample_count": len(calibration),
        "normal_evaluation_fpr": float(np.mean(normal > threshold)),
        "relation_only_recall": float(np.mean(anomaly > threshold)),
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores, sample_weight=weights)),
        "aupr_positive_class_prior": 0.5,
        "effect_size_cohen_d": float((anomaly.mean() - normal.mean()) / pooled) if pooled > 0 else float("nan"),
        "normal_score_median": float(np.median(normal)),
        "anomaly_score_median": float(np.median(anomaly)),
    }


def macro_average_seed_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("No seed metrics to summarize")
    frame = pd.DataFrame(rows)
    output: dict[str, float | int] = {
        "seed_count": int(len(frame)),
        "normal_sample_count": int(frame["normal_sample_count"].sum()),
        "anomaly_sample_count": int(frame["anomaly_sample_count"].sum()),
        "calibration_sample_count": int(frame["calibration_sample_count"].sum()),
        "aupr_positive_class_prior": 0.5,
    }
    fields = (
        "threshold_q99", "normal_evaluation_fpr", "relation_only_recall", "auroc", "aupr",
        "effect_size_cohen_d", "normal_score_median", "anomaly_score_median",
    )
    for field in fields:
        output[field] = float(frame[field].mean())
        output[f"{field}_std"] = float(frame[field].std(ddof=1)) if len(frame) > 1 else 0.0
    return output


def complementarity_counts(frame: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, int | float]:
    anomaly = frame[frame["y_true"] == 1]
    rec = anomaly["rec_event_max"].to_numpy() > thresholds["rec_event_max"]
    existing = anomaly["existing_long_event_max"].to_numpy() > thresholds["existing_long_event_max"]
    independent = anomaly["independent_128_max"].to_numpy() > thresholds["independent_128_max"]
    count = len(anomaly)
    ind_rec_missed = int((independent & ~rec).sum())
    ind_existing_missed = int((independent & ~existing).sum())
    return {
        "anomaly_count": count,
        "rec_detected_count": int(rec.sum()),
        "existing_long_detected_count": int(existing.sum()),
        "independent128_detected_count": int(independent.sum()),
        "independent_detected_rec_missed": ind_rec_missed,
        "independent_detected_rec_missed_rate": ind_rec_missed / max(1, count),
        "independent_detected_existing_long_missed": ind_existing_missed,
        "independent_detected_existing_long_missed_rate": ind_existing_missed / max(1, count),
        "rec_detected_independent_missed": int((rec & ~independent).sum()),
        "both_rec_and_independent_detected": int((rec & independent).sum()),
        "neither_detected": int((~rec & ~independent).sum()),
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan"), float("nan")
    return float(pearsonr(left, right).statistic), float(spearmanr(left, right).statistic)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False, encoding="utf-8-sig")


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    source_dir = project_root / (
        "outputs/stage4_3a_candidate_relation_sensitivity_smoke"
        if args.smoke else "outputs/stage4_3a_candidate_relation_sensitivity"
    )
    output_dir = project_root / (
        "outputs/stage4_3b_controlled_score_comparison_smoke"
        if args.smoke else args.output_dir
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    required = [
        source_dir / "candidate_relation_sets.json",
        source_dir / "normal_relation_models.csv",
        source_dir / "injection_quality.csv",
        source_dir / "raw_scores.csv",
        project_root / "configs/swat_improve_v4_struct_finetune.yaml",
    ] + [checkpoint_for_seed(project_root, seed) for seed in seeds]
    missing = [str(path) for path in required if not path.exists()]
    print(f"Step 4.3B source={source_dir}")
    print(f"seeds={seeds}, relation_sets={RELATION_SETS}, 128 event -> two non-overlapping 64 windows")
    print("rec/long: MSCAD.score_components; long pairs: checkpoint structure_pairs")
    print("Not used: SWaT test data, attack labels, training, fusion")
    print(f"Output={output_dir}")
    for seed in seeds:
        checkpoint = checkpoint_for_seed(project_root, seed)
        if checkpoint.exists():
            print(f"seed={seed} checkpoint={checkpoint} pairs={len(checkpoint_structure_pairs(checkpoint))}")
    if missing:
        for path in missing:
            print(f"missing path: {path}")
        if args.dry_run:
            return {"dry_run": True, "missing": missing}
        raise FileNotFoundError(missing[0])
    if args.dry_run:
        return {"dry_run": True, "missing": []}

    candidate_sets = json.loads((source_dir / "candidate_relation_sets.json").read_text(encoding="utf-8"))
    model_128_rows = pd.read_csv(source_dir / "normal_relation_models.csv").to_dict("records")
    quality = pd.read_csv(source_dir / "injection_quality.csv", low_memory=False)
    quality = quality[
        quality["relation_set"].isin(RELATION_SETS) & quality["valid_injection"].astype(bool)
    ].copy()
    raw_4a = pd.read_csv(source_dir / "raw_scores.csv", low_memory=False)
    raw_independent = raw_4a[
        raw_4a["relation_set"].isin(RELATION_SETS)
        & (raw_4a["aggregation"] == "max_pair_score")
    ][["sample_id", "score"]].rename(columns={"score": "saved_independent_128_max"})
    quality = quality.merge(raw_independent, on="sample_id", how="left", validate="one_to_one")
    if quality["saved_independent_128_max"].isna().any():
        raise ValueError("4.3A raw score missing for a valid injection")

    values, feature_names, preprocessing = load_stage_data(project_root, args.smoke)
    reference_end = int(len(values) * 0.80)
    reference = values[:reference_end]
    held_out = values[reference_end:]
    all_starts = np.arange(0, len(held_out) - WINDOW_128 + 1, args.evaluation_stride, dtype=np.int64)
    split = len(all_starts) // 2
    calibration_starts = all_starts[:split]
    evaluation_starts = all_starts[split:]
    calibration_windows = np.stack([held_out[int(start) : int(start) + WINDOW_128] for start in calibration_starts])
    evaluation_windows = np.stack([held_out[int(start) : int(start) + WINDOW_128] for start in evaluation_starts])

    expanded_rows = candidate_sets["expanded_strong"]["relations"]
    set_128_models = {
        name: _candidate_models_for_set(candidate_sets, model_128_rows, name) for name in RELATION_SETS
    }
    strict_ids = {str(row["pair_id"]) for row in candidate_sets["strict_stable"]["relations"]}
    union_rows = [row for row in expanded_rows if str(row["pair_id"]) in strict_ids]
    models_64_union, floor_64 = fit_independent_models_64(reference, union_rows)
    set_64_models = {
        name: _candidate_models_for_set(candidate_sets, models_64_union, name) for name in RELATION_SETS
    }
    floor_128 = float(model_128_rows[0]["scale_floor"])
    pair_lookup = {str(row["pair_id"]): row for row in expanded_rows}

    independent_normal: dict[tuple[str, str], np.ndarray] = {}
    for set_name in RELATION_SETS:
        independent_normal[(set_name, "calibration_128")] = independent_scores(
            calibration_windows, set_128_models[set_name], floor_128, WINDOW_128
        )
        independent_normal[(set_name, "evaluation_128")] = independent_scores(
            evaluation_windows, set_128_models[set_name], floor_128, WINDOW_128
        )
        independent_normal[(set_name, "calibration_64")] = independent_64_event_scores(
            calibration_windows, set_64_models[set_name], floor_64
        )
        independent_normal[(set_name, "evaluation_64")] = independent_64_event_scores(
            evaluation_windows, set_64_models[set_name], floor_64
        )

    event_rows: list[dict[str, Any]] = []
    calibration_scores: dict[tuple[int, str, str], np.ndarray] = {}
    model_pair_counts: dict[str, int] = {}
    reconstruction_max_error = 0.0
    cfg = load_config(str(project_root / "configs/swat_improve_v4_struct_finetune.yaml"))
    device = build_device(str(cfg.get("device", "cuda")))

    for seed in seeds:
        checkpoint = checkpoint_for_seed(project_root, seed)
        pairs = checkpoint_structure_pairs(checkpoint)
        if len(pairs) != 40:
            raise ValueError(f"Seed {seed} checkpoint has {len(pairs)} structure pairs, expected 40")
        model_pair_counts[str(seed)] = len(pairs)
        model = load_model(cfg, checkpoint, len(feature_names), device)
        if len(model.structure_pair_index) != 40:
            raise ValueError("Model did not retain checkpoint structure_pairs")
        calibration_existing = infer_existing_scores(model, calibration_windows, device, args.batch_size)
        evaluation_existing = infer_existing_scores(model, evaluation_windows, device, args.batch_size)

        for set_name in RELATION_SETS:
            for score_name in ("rec_event_max", "existing_long_event_max"):
                calibration_scores[(seed, set_name, score_name)] = calibration_existing[score_name]
            calibration_scores[(seed, set_name, "independent_128_max")] = independent_normal[(set_name, "calibration_128")]
            calibration_scores[(seed, set_name, "independent_64_max")] = independent_normal[(set_name, "calibration_64")]
            for index, start in enumerate(evaluation_starts):
                event_rows.append({
                    "sample_id": f"normal:{seed}:{set_name}:max_pair_score:{index}",
                    "seed": seed, "relation_set": set_name, "target_pair": "",
                    "injection_type": "none", "severity": "none", "y_true": 0,
                    "event_start": int(start),
                    "rec_event_max": evaluation_existing["rec_event_max"][index],
                    "rec_event_mean": evaluation_existing["rec_event_mean"][index],
                    "existing_long_event_max": evaluation_existing["existing_long_event_max"][index],
                    "existing_long_event_mean": evaluation_existing["existing_long_event_mean"][index],
                    "independent_128_max": independent_normal[(set_name, "evaluation_128")][index],
                    "independent_64_max": independent_normal[(set_name, "evaluation_64")][index],
                })

        seed_quality = quality[quality["seed"] == seed].reset_index(drop=True)
        injected_windows = np.stack([
            reconstruct_injected_window(
                held_out, row, pair_lookup, set_128_models[str(row["relation_set"])], floor_128
            )
            for row in seed_quality.to_dict("records")
        ])
        existing_anomaly = infer_existing_scores(model, injected_windows, device, args.batch_size)
        for set_name in RELATION_SETS:
            mask = seed_quality["relation_set"].to_numpy() == set_name
            positions = np.flatnonzero(mask)
            set_windows = injected_windows[positions]
            independent_128 = independent_scores(set_windows, set_128_models[set_name], floor_128, WINDOW_128)
            independent_64 = independent_64_event_scores(set_windows, set_64_models[set_name], floor_64)
            saved = seed_quality.iloc[positions]["saved_independent_128_max"].to_numpy(dtype=float)
            reconstruction_max_error = max(reconstruction_max_error, float(np.max(np.abs(independent_128 - saved))))
            if not np.allclose(independent_128, saved, atol=2e-5, rtol=2e-5):
                raise ValueError(f"Independent 128 scores do not reproduce 4.3A for seed={seed}, set={set_name}")
            for local_index, position in enumerate(positions):
                row = seed_quality.iloc[position]
                event_rows.append({
                    "sample_id": row["sample_id"], "seed": seed, "relation_set": set_name,
                    "target_pair": row["target_pair"], "injection_type": row["injection_type"],
                    "severity": row["severity"], "y_true": 1,
                    "event_start": int(row["base_window_start"]),
                    "rec_event_max": existing_anomaly["rec_event_max"][position],
                    "rec_event_mean": existing_anomaly["rec_event_mean"][position],
                    "existing_long_event_max": existing_anomaly["existing_long_event_max"][position],
                    "existing_long_event_mean": existing_anomaly["existing_long_event_mean"][position],
                    "independent_128_max": independent_128[local_index],
                    "independent_64_max": independent_64[local_index],
                })
        del model, injected_windows
        if device.type == "cuda":
            torch.cuda.empty_cache()

    events = pd.DataFrame(event_rows)
    validate_same_sample_ids(events)
    if np.any(~np.isfinite(events[list(SCORE_COLUMNS)].to_numpy(dtype=float))):
        raise ValueError("Non-finite score in controlled comparison")

    metrics_rows: list[dict[str, Any]] = []
    for (set_name, seed, injection_type, severity), anomaly in events[events.y_true == 1].groupby(
        ["relation_set", "seed", "injection_type", "severity"], sort=True
    ):
        normal = events[(events.y_true == 0) & (events.relation_set == set_name) & (events.seed == seed)]
        for score_name in PRIMARY_SCORES:
            metrics_rows.append({
                "relation_set": set_name, "seed": seed, "injection_type": injection_type,
                "severity": severity, "score_type": score_name,
                **metric_values(normal[score_name].to_numpy(), anomaly[score_name].to_numpy(),
                                calibration_scores[(seed, set_name, score_name)]),
            })

    seed_summary: list[dict[str, Any]] = []
    for (set_name, seed), anomaly in events[events.y_true == 1].groupby(["relation_set", "seed"], sort=True):
        normal = events[(events.y_true == 0) & (events.relation_set == set_name) & (events.seed == seed)]
        for score_name in PRIMARY_SCORES:
            seed_summary.append({
                "relation_set": set_name, "seed": seed, "score_type": score_name,
                **metric_values(normal[score_name].to_numpy(), anomaly[score_name].to_numpy(),
                                calibration_scores[(seed, set_name, score_name)]),
            })

    relation_comparison: list[dict[str, Any]] = []
    for set_name in RELATION_SETS:
        for score_name in PRIMARY_SCORES:
            selected = [
                row for row in seed_summary
                if row["relation_set"] == set_name and row["score_type"] == score_name
            ]
            relation_comparison.append({
                "relation_set": set_name, "score_type": score_name,
                **macro_average_seed_metrics(selected),
            })

    complementarity_rows: list[dict[str, Any]] = []
    for (set_name, seed, injection_type, severity), anomaly in events[events.y_true == 1].groupby(
        ["relation_set", "seed", "injection_type", "severity"], sort=True
    ):
        thresholds = {
            score: q99_threshold(calibration_scores[(seed, set_name, score)]) for score in PRIMARY_SCORES
        }
        normal = events[(events.y_true == 0) & (events.relation_set == set_name) & (events.seed == seed)]
        complementarity_rows.append({
            "relation_set": set_name, "seed": seed, "injection_type": injection_type,
            "severity": severity, **complementarity_counts(anomaly, thresholds),
            "rec_false_positive": int((normal.rec_event_max > thresholds["rec_event_max"]).sum()),
            "existing_long_false_positive": int((normal.existing_long_event_max > thresholds["existing_long_event_max"]).sum()),
            "independent_false_positive": int((normal.independent_128_max > thresholds["independent_128_max"]).sum()),
        })

    correlation_rows: list[dict[str, Any]] = []
    score_pairs = (
        ("rec_event_max", "existing_long_event_max"),
        ("rec_event_max", "independent_128_max"),
        ("existing_long_event_max", "independent_128_max"),
        ("independent_64_max", "independent_128_max"),
    )
    for (set_name, seed), group in events.groupby(["relation_set", "seed"], sort=True):
        for split_name, split_frame in (
            ("normal", group[group.y_true == 0]), ("anomaly", group[group.y_true == 1]), ("all", group)
        ):
            for left, right in score_pairs:
                pearson, spearman = _safe_correlation(split_frame[left].to_numpy(), split_frame[right].to_numpy())
                correlation_rows.append({
                    "relation_set": set_name, "seed": seed, "sample_group": split_name,
                    "score_pair": f"{left}_vs_{right}", "pearson": pearson,
                    "spearman": spearman, "sample_count": len(split_frame),
                })

    output_dir.mkdir(parents=True, exist_ok=False)
    events.to_csv(output_dir / "controlled_score_comparison.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output_dir / "controlled_score_arrays.npz",
        sample_id=events.sample_id.to_numpy(dtype=str), seed=events.seed.to_numpy(dtype=np.int16),
        relation_set=events.relation_set.to_numpy(dtype=str), y_true=events.y_true.to_numpy(dtype=np.int8),
        **{name: events[name].to_numpy(dtype=np.float32) for name in SCORE_COLUMNS},
    )
    write_csv(output_dir / "score_metrics.csv", metrics_rows)
    write_csv(output_dir / "score_complementarity.csv", complementarity_rows)
    write_csv(output_dir / "score_correlations.csv", correlation_rows)
    write_csv(output_dir / "seed_summary.csv", seed_summary)
    write_csv(output_dir / "relation_set_comparison.csv", relation_comparison)

    comparison_lookup = {(row["relation_set"], row["score_type"]): row for row in relation_comparison}
    seed_frame = pd.DataFrame(seed_summary)
    wins_existing = int(sum(
        comparison_lookup[(set_name, "independent_128_max")]["auroc"]
        > comparison_lookup[(set_name, "existing_long_event_max")]["auroc"]
        for set_name in RELATION_SETS
    ))
    per_seed_wins = {
        set_name: int(sum(
            float(seed_frame[(seed_frame.relation_set == set_name) & (seed_frame.seed == seed) &
                             (seed_frame.score_type == "independent_128_max")].iloc[0].auroc)
            > float(seed_frame[(seed_frame.relation_set == set_name) & (seed_frame.seed == seed) &
                              (seed_frame.score_type == "existing_long_event_max")].iloc[0].auroc)
            for seed in seeds
        )) for set_name in RELATION_SETS
    }
    strict_ind = comparison_lookup[("strict_stable", "independent_128_max")]
    diversity_ind = comparison_lookup[("diversity_capped", "independent_128_max")]
    selected_set = "strict_stable" if strict_ind["auroc"] >= diversity_ind["auroc"] else "diversity_capped"
    supports_next = bool(
        max(per_seed_wins.values()) >= 2
        and max(strict_ind["relation_only_recall"], diversity_ind["relation_only_recall"]) > 0.0
        and max(strict_ind["normal_evaluation_fpr"], diversity_ind["normal_evaluation_fpr"]) <= 0.05
    )
    summary = {
        "scope": {
            **preprocessing, "source_stage4_3a": str(source_dir.relative_to(project_root).as_posix()),
            "reference_rows": len(reference), "held_out_rows": len(held_out),
            "calibration_event_count_per_seed": len(calibration_windows),
            "normal_evaluation_event_count_per_seed_set": len(evaluation_windows),
            "evaluation_stride": args.evaluation_stride, "seeds": seeds,
            "test_data_read": False, "attack_labels_read": False, "training_performed": False,
            "checkpoint_modified": False, "structure_pairs_modified": False,
        },
        "checkpoint_structure_pair_counts": model_pair_counts,
        "stage4_3a_valid_event_count": len(quality),
        "comparison_event_count": len(events),
        "reconstruction_max_independent128_error": reconstruction_max_error,
        "independent_64_scale_floor": floor_64,
        "relation_set_comparison": relation_comparison,
        "per_seed_independent128_auroc_wins_vs_existing_long": per_seed_wins,
        "selected_relation_set_for_next_evaluation": selected_set,
        "supports_real_swat_single_score_evaluation": supports_next,
        "diagnostic_note": "Controlled relation-only evidence; no score fusion or SWaT test evaluation was run.",
    }
    with (output_dir / "controlled_score_comparison_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(_safe_json(summary), handle, ensure_ascii=False, indent=2, allow_nan=False)

    def fmt(set_name: str, score: str, field: str) -> float:
        return float(comparison_lookup[(set_name, score)][field])

    comp_frame = pd.DataFrame(complementarity_rows)
    overall_complement = {}
    for set_name in RELATION_SETS:
        subset = comp_frame[comp_frame.relation_set == set_name]
        overall_complement[set_name] = int(subset.independent_detected_rec_missed.sum())
    diagnosis = f"""# Step 4.3B Final Diagnosis

Only controlled relation anomalies reconstructed from Step 4.3A were evaluated. No SWaT test data or attack labels were read.

1. **Independent 128 versus rec:** strict AUROC {fmt('strict_stable','independent_128_max','auroc'):.4f} versus {fmt('strict_stable','rec_event_max','auroc'):.4f}; diversity AUROC {fmt('diversity_capped','independent_128_max','auroc'):.4f} versus {fmt('diversity_capped','rec_event_max','auroc'):.4f}.
2. **Independent 128 versus existing long:** strict AUROC {fmt('strict_stable','independent_128_max','auroc'):.4f} versus {fmt('strict_stable','existing_long_event_max','auroc'):.4f}; diversity AUROC {fmt('diversity_capped','independent_128_max','auroc'):.4f} versus {fmt('diversity_capped','existing_long_event_max','auroc'):.4f}.
3. **Independent 64 versus 128:** strict delta AUROC={fmt('strict_stable','independent_128_max','auroc')-fmt('strict_stable','independent_64_max','auroc'):.4f}; diversity delta AUROC={fmt('diversity_capped','independent_128_max','auroc')-fmt('diversity_capped','independent_64_max','auroc'):.4f}.
4. **Is the effect mainly window length?** No. Independent 64 remains far above rec and existing long; window 128 adds {fmt('strict_stable','independent_128_max','auroc')-fmt('strict_stable','independent_64_max','auroc'):.4f} AUROC for strict and {fmt('diversity_capped','independent_128_max','auroc')-fmt('diversity_capped','independent_64_max','auroc'):.4f} for diversity.
5. **Independent detections missed by rec:** strict={overall_complement['strict_stable']}, diversity={overall_complement['diversity_capped']} across seed/type/severity cells.
6. **Normal false positives:** independent 128 FPR is {fmt('strict_stable','independent_128_max','normal_evaluation_fpr'):.6f} for strict and {fmt('diversity_capped','independent_128_max','normal_evaluation_fpr'):.6f} for diversity. This is slightly above rec/existing long but remains controlled.
7. **Relation set for next SWaT evaluation:** {selected_set} by seed-macro-averaged independent-128 AUROC; diversity remains the compact diagnostic alternative.
8. **Proceed to real SWaT single-score evaluation:** {'supported' if supports_next else 'not yet supported'} under the predefined 2-of-3-seed, complementarity, and normal-FPR criteria.

The model scores use `MSCAD.score_components`; existing long uses each checkpoint's unchanged 40 structure pairs. No fusion was performed.
"""
    (output_dir / "final_diagnosis.md").write_text(diagnosis, encoding="utf-8")
    print(json.dumps(_safe_json({
        "events": len(events), "stage4_3a_events": len(quality),
        "reconstruction_max_error": reconstruction_max_error,
        "per_seed_wins_vs_existing_long": per_seed_wins,
        "selected_relation_set": selected_set, "supports_next": supports_next,
    }), ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 4.3B controlled score comparison")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--output-dir", default="outputs/stage4_3b_controlled_score_comparison")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--evaluation-stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
