from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
if str(DEFAULT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ROOT))

from scripts.stage4_2a_analyze_relation_stability import _window_pair_correlations_tensor  # noqa: E402
from scripts.stage4_2b_diagnose_relation_validity import (  # noqa: E402
    VARIANCE_EPSILON,
    channel_activity_rows,
    load_normal_training_data,
)
from scripts.stage4_2c_analyze_local_relation_candidates import (  # noqa: E402
    _activity_output_rows,
    _analyze_window,
)
from utils.config_io import load_config  # noqa: E402
from utils.controlled_relation_injection import (  # noqa: E402
    aggregate_pair_scores,
    block_permutation,
    cross_window_swap,
    localization_result,
    pair_score,
    pearson_corr,
    relation_set_summary,
    robust_scale_floor,
    select_diversity_capped,
    temporal_shift,
)


WINDOW_SIZE = 128
SEVERITIES = ("low", "medium", "high")
SHIFT_STEPS = {"low": 8, "medium": 16, "high": 32}
AGGREGATIONS = ("mean_pair_score", "top3_mean_pair_score", "max_pair_score")

QUALITY_FIELDS = [
    "sample_id", "seed", "relation_set", "target_pair", "injection_type", "severity",
    "base_window_start", "donor_window_start", "target_channel", "target_channel_index",
    "original_corr", "injected_corr", "corr_change", "original_mean", "injected_mean",
    "mean_change", "original_std", "injected_std", "std_change_ratio",
    "marginal_q05_change", "marginal_q50_change", "marginal_q95_change",
    "valid_injection", "invalid_reason",
]


def load_stage_data(project_root: Path, smoke: bool) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    config_path = project_root / "configs/swat_improve_v4_struct_finetune.yaml"
    cfg = load_config(str(config_path))
    return load_normal_training_data(project_root, cfg, smoke)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def _pair_tuple(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["channel_i_index"]), int(row["channel_j_index"])


def build_candidate_sets(pair_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    expanded = [
        row for row in pair_rows
        if row["pair_activity_group"] == "dense_dense"
        and float(row["valid_ratio"]) >= 0.95
        and math.isfinite(float(row["abs_median_corr"]))
        and float(row["abs_median_corr"]) >= 0.30
        and math.isfinite(float(row["relation_instability"]))
    ]
    strict = [row for row in expanded if float(row["relation_instability"]) <= 0.25]
    return {
        "strict_stable": sorted(strict, key=lambda row: str(row["pair_id"])),
        "diversity_capped": select_diversity_capped(strict, max_channel_degree=2),
        "expanded_strong": sorted(expanded, key=lambda row: str(row["pair_id"])),
    }


def correlations_for_starts(
    values: np.ndarray,
    starts: np.ndarray,
    pairs: list[tuple[int, int]],
    batch_size: int = 2048,
) -> np.ndarray:
    if len(starts) == 0 or len(pairs) == 0:
        return np.empty((len(starts), len(pairs)), dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair_i = torch.tensor([pair[0] for pair in pairs], dtype=torch.long, device=device)
    pair_j = torch.tensor([pair[1] for pair in pairs], dtype=torch.long, device=device)
    output = np.empty((len(starts), len(pairs)), dtype=np.float32)
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        windows = np.stack([values[int(start) : int(start) + WINDOW_SIZE] for start in batch_starts])
        tensor = torch.as_tensor(windows, dtype=torch.float32, device=device)
        corr, valid = _window_pair_correlations_tensor(tensor, pair_i, pair_j, VARIANCE_EPSILON)
        batch = corr.detach().cpu().numpy()
        batch[~valid.detach().cpu().numpy()] = np.nan
        output[offset : offset + len(batch)] = batch
    return output


def fit_normal_relation_models(
    reference: np.ndarray,
    starts: np.ndarray,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    pairs = [_pair_tuple(row) for row in rows]
    correlations = correlations_for_starts(reference, starts, pairs)
    models: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        valid = correlations[:, index]
        valid = valid[np.isfinite(valid)]
        fisher = np.arctanh(np.clip(valid.astype(np.float64), -0.999, 0.999))
        median_fisher = float(np.median(fisher))
        mad_fisher = float(np.median(np.abs(fisher - median_fisher)))
        models.append({
            "pair_id": row["pair_id"],
            "channel_i": row["channel_i"], "channel_j": row["channel_j"],
            "channel_i_index": row["channel_i_index"], "channel_j_index": row["channel_j_index"],
            "normal_median_fisher": median_fisher,
            "normal_mad_fisher": mad_fisher,
            "normal_robust_scale": 1.4826 * mad_fisher,
            "valid_ratio": len(valid) / max(1, len(starts)),
            "median_corr": float(np.tanh(median_fisher)),
            "relation_instability": 1.4826 * mad_fisher,
        })
    scale_floor = robust_scale_floor([model["normal_robust_scale"] for model in models])
    for index, model in enumerate(models):
        scores = np.asarray([
            pair_score(corr, model["normal_median_fisher"], model["normal_robust_scale"], scale_floor)
            for corr in correlations[:, index]
        ])
        finite = scores[np.isfinite(scores)]
        model["scale_floor"] = scale_floor
        model["normal_pair_score_q99"] = float(np.quantile(finite, 0.99)) if finite.size else float("nan")
    return models, correlations, scale_floor


def score_matrix(correlations: np.ndarray, models: list[dict[str, Any]], scale_floor: float) -> np.ndarray:
    output = np.empty_like(correlations, dtype=np.float64)
    for index, model in enumerate(models):
        values = correlations[:, index].astype(np.float64)
        fisher = np.arctanh(np.clip(values, -0.999, 0.999))
        denominator = max(float(model["normal_robust_scale"]), scale_floor)
        output[:, index] = np.abs(fisher - float(model["normal_median_fisher"])) / denominator
        output[~np.isfinite(values), index] = np.nan
    return output


def _set_indices(candidate_sets: dict[str, list[dict[str, Any]]], model_rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    lookup = {str(row["pair_id"]): index for index, row in enumerate(model_rows)}
    return {name: [lookup[str(row["pair_id"])] for row in rows] for name, rows in candidate_sets.items()}


def aggregate_normal_scores(
    pair_scores: np.ndarray,
    indices_by_set: dict[str, list[int]],
) -> dict[tuple[str, str], np.ndarray]:
    output: dict[tuple[str, str], np.ndarray] = {}
    for set_name, indices in indices_by_set.items():
        values = pair_scores[:, indices]
        aggregates = [aggregate_pair_scores(row) for row in values]
        for aggregation in AGGREGATIONS:
            output[(set_name, aggregation)] = np.asarray(
                [row[aggregation] for row in aggregates], dtype=np.float64
            )
    return output


def _stable_rng(seed: int, set_index: int, pair_index: int, type_index: int, severity_index: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([seed, set_index, pair_index, type_index, severity_index])
    return np.random.default_rng(sequence)


def _quality_row(
    sample_id: str,
    seed: int,
    set_name: str,
    pair_row: dict[str, Any],
    injection_type: str,
    severity: str,
    base_start: int,
    donor_start: int | None,
    target_channel: int,
    feature_names: list[str],
    original_window: np.ndarray,
    injected_window: np.ndarray,
    valid_precondition: bool,
    invalid_reason: str,
) -> dict[str, Any]:
    left, right = _pair_tuple(pair_row)
    original_corr = pearson_corr(original_window[:, left], original_window[:, right], VARIANCE_EPSILON)
    injected_corr = pearson_corr(injected_window[:, left], injected_window[:, right], VARIANCE_EPSILON)
    original = original_window[:, target_channel].astype(np.float64)
    injected = injected_window[:, target_channel].astype(np.float64)
    original_std = float(original.std())
    injected_std = float(injected.std())
    corr_change = abs(injected_corr - original_corr) if np.isfinite(injected_corr) and np.isfinite(original_corr) else float("nan")
    mean_change = float(injected.mean() - original.mean())
    std_change_ratio = abs(injected_std - original_std) / max(original_std, VARIANCE_EPSILON)
    quantile_change = np.quantile(injected, [0.05, 0.50, 0.95]) - np.quantile(original, [0.05, 0.50, 0.95])
    valid = bool(valid_precondition and np.isfinite(injected_corr) and corr_change >= 0.20)
    if not valid and not invalid_reason:
        if not np.isfinite(injected_corr):
            invalid_reason = "injected_correlation_invalid"
        elif corr_change < 0.20:
            invalid_reason = "correlation_change_below_0.20"
    return {
        "sample_id": sample_id, "seed": seed, "relation_set": set_name,
        "target_pair": pair_row["pair_id"], "injection_type": injection_type, "severity": severity,
        "base_window_start": base_start, "donor_window_start": "" if donor_start is None else donor_start,
        "target_channel": feature_names[target_channel], "target_channel_index": target_channel,
        "original_corr": original_corr, "injected_corr": injected_corr, "corr_change": corr_change,
        "original_mean": float(original.mean()), "injected_mean": float(injected.mean()),
        "mean_change": mean_change, "original_std": original_std, "injected_std": injected_std,
        "std_change_ratio": std_change_ratio, "marginal_q05_change": float(quantile_change[0]),
        "marginal_q50_change": float(quantile_change[1]), "marginal_q95_change": float(quantile_change[2]),
        "valid_injection": valid, "invalid_reason": invalid_reason,
    }


def generate_injections(
    held_out: np.ndarray,
    evaluation_starts: np.ndarray,
    evaluation_correlations: np.ndarray,
    evaluation_pair_scores: np.ndarray,
    candidate_sets: dict[str, list[dict[str, Any]]],
    model_rows: list[dict[str, Any]],
    scale_floor: float,
    feature_names: list[str],
    seeds: list[int],
    max_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model_lookup = {str(row["pair_id"]): index for index, row in enumerate(model_rows)}
    quality_rows: list[dict[str, Any]] = []
    raw_anomaly_rows: list[dict[str, Any]] = []
    localization_rows: list[dict[str, Any]] = []
    donor_cache: dict[int, np.ndarray] = {}
    injection_types = ("temporal_shift", "block_permutation", "cross_window_swap")

    for set_index, (set_name, set_rows) in enumerate(candidate_sets.items()):
        set_model_indices = [model_lookup[str(row["pair_id"])] for row in set_rows]
        for pair_index, pair_row in enumerate(set_rows):
            model_index = model_lookup[str(pair_row["pair_id"])]
            model = model_rows[model_index]
            original_corrs = evaluation_correlations[:, model_index]
            original_scores = evaluation_pair_scores[:, model_index]
            direction_ok = original_corrs * float(model["median_corr"]) > 0
            eligible = np.flatnonzero(
                np.isfinite(original_corrs)
                & direction_ok
                & (original_scores <= float(model["normal_pair_score_q99"]))
            )
            target_channel = int(pair_row["channel_j_index"])
            if target_channel not in donor_cache:
                donor_cache[target_channel] = np.stack([
                    held_out[int(start) : int(start) + WINDOW_SIZE, target_channel]
                    for start in evaluation_starts
                ]).astype(np.float32)
            donor_values = donor_cache[target_channel]
            donor_means = donor_values.mean(axis=1)
            donor_stds = donor_values.std(axis=1)

            for seed in seeds:
                for type_index, injection_type in enumerate(injection_types):
                    for severity_index, severity in enumerate(SEVERITIES):
                        rng = _stable_rng(seed, set_index, pair_index, type_index, severity_index)
                        base_order = rng.permutation(eligible)
                        valid_count = 0
                        attempt_limit = min(len(base_order), max(50, max_samples * 20))
                        for attempt, base_index in enumerate(base_order[:attempt_limit]):
                            base_start = int(evaluation_starts[int(base_index)])
                            original_window = held_out[base_start : base_start + WINDOW_SIZE].copy()
                            injected_window = original_window.copy()
                            donor_start: int | None = None
                            precondition = True
                            invalid_reason = ""
                            if injection_type == "temporal_shift":
                                injected_window = temporal_shift(original_window, target_channel, SHIFT_STEPS[severity])
                            elif injection_type == "block_permutation":
                                injected_window = block_permutation(original_window, target_channel, rng)
                            else:
                                base_mean = float(donor_means[int(base_index)])
                                base_std = float(donor_stds[int(base_index)])
                                donor_candidates = np.flatnonzero(
                                    (np.abs(donor_means - base_mean) <= 0.10 * base_std)
                                    & (np.abs(donor_stds - base_std) / max(base_std, VARIANCE_EPSILON) <= 0.10)
                                )
                                donor_candidates = donor_candidates[donor_candidates != int(base_index)]
                                if donor_candidates.size:
                                    donor_index = int(rng.choice(donor_candidates))
                                    donor_start = int(evaluation_starts[donor_index])
                                    donor_window = held_out[donor_start : donor_start + WINDOW_SIZE]
                                    injected_window, precondition, invalid_reason = cross_window_swap(
                                        original_window, donor_window, target_channel
                                    )
                                else:
                                    precondition = False
                                    invalid_reason = "no_compatible_donor"

                            sample_id = f"{set_name}:{pair_index}:{seed}:{injection_type}:{severity}:{attempt}"
                            quality = _quality_row(
                                sample_id, seed, set_name, pair_row, injection_type, severity,
                                base_start, donor_start, target_channel, feature_names,
                                original_window, injected_window, precondition, invalid_reason,
                            )
                            quality_rows.append(quality)
                            if not quality["valid_injection"]:
                                continue

                            all_corr = np.asarray([
                                pearson_corr(injected_window[:, int(row["channel_i_index"])],
                                             injected_window[:, int(row["channel_j_index"])])
                                for row in model_rows
                            ])
                            all_scores = np.asarray([
                                pair_score(all_corr[index], row["normal_median_fisher"],
                                           row["normal_robust_scale"], scale_floor)
                                for index, row in enumerate(model_rows)
                            ])
                            set_scores = all_scores[set_model_indices]
                            aggregates = aggregate_pair_scores(set_scores)
                            for aggregation, score in aggregates.items():
                                raw_anomaly_rows.append({
                                    "sample_id": sample_id, "sample_kind": "injected_anomaly", "seed": seed,
                                    "relation_set": set_name, "aggregation": aggregation, "score": score,
                                    "target_pair": pair_row["pair_id"], "injection_type": injection_type,
                                    "severity": severity,
                                })
                            target_local_index = next(
                                index for index, row in enumerate(set_rows)
                                if str(row["pair_id"]) == str(pair_row["pair_id"])
                            )
                            localization_rows.append({
                                "sample_id": sample_id, "seed": seed, "relation_set": set_name,
                                "target_pair": pair_row["pair_id"], "injection_type": injection_type,
                                "severity": severity, **localization_result(set_scores, target_local_index),
                            })
                            valid_count += 1
                            if valid_count >= max_samples:
                                break
    return quality_rows, raw_anomaly_rows, localization_rows


def _cohen_d(normal: np.ndarray, anomaly: np.ndarray) -> float:
    if len(normal) < 2 or len(anomaly) < 2:
        return float("nan")
    pooled = math.sqrt(((len(normal) - 1) * normal.var(ddof=1) + (len(anomaly) - 1) * anomaly.var(ddof=1)) / (len(normal) + len(anomaly) - 2))
    return float((anomaly.mean() - normal.mean()) / pooled) if pooled > 0 else float("nan")


def metric_values(normal: np.ndarray, anomaly: np.ndarray, calibration: np.ndarray) -> dict[str, float | int]:
    normal = normal[np.isfinite(normal)]
    anomaly = anomaly[np.isfinite(anomaly)]
    calibration = calibration[np.isfinite(calibration)]
    if not len(normal) or not len(anomaly) or not len(calibration):
        return {key: float("nan") for key in (
            "auroc", "aupr", "normal_score_median", "normal_score_q95", "normal_score_q99",
            "anomaly_score_median", "effect_size_cohen_d", "relation_only_recall_at_normal_q99",
            "normal_false_positive_rate_at_q99",
        )} | {"normal_sample_count": len(normal), "anomaly_sample_count": len(anomaly)}
    labels = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(anomaly), dtype=int)])
    scores = np.concatenate([normal, anomaly])
    aupr_weights = np.concatenate([
        np.full(len(normal), 0.5 / len(normal), dtype=np.float64),
        np.full(len(anomaly), 0.5 / len(anomaly), dtype=np.float64),
    ])
    threshold = float(np.quantile(calibration, 0.99))
    return {
        "normal_sample_count": len(normal), "anomaly_sample_count": len(anomaly),
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores, sample_weight=aupr_weights)),
        "aupr_positive_class_prior": 0.5,
        "normal_score_median": float(np.median(normal)), "normal_score_q95": float(np.quantile(normal, 0.95)),
        "normal_score_q99": threshold, "anomaly_score_median": float(np.median(anomaly)),
        "effect_size_cohen_d": _cohen_d(normal, anomaly),
        "relation_only_recall_at_normal_q99": float(np.mean(anomaly > threshold)),
        "normal_false_positive_rate_at_q99": float(np.mean(normal > threshold)),
    }


def build_metric_tables(
    anomaly_df: pd.DataFrame,
    normal_eval: dict[tuple[str, str], np.ndarray],
    normal_calibration: dict[tuple[str, str], np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sensitivity: list[dict[str, Any]] = []
    seed_summary: list[dict[str, Any]] = []
    relation_summary: list[dict[str, Any]] = []
    if anomaly_df.empty:
        return sensitivity, seed_summary, relation_summary
    target_keys = ["relation_set", "aggregation", "injection_type", "severity", "seed", "target_pair"]
    for keys, group in anomaly_df.groupby(target_keys, sort=True):
        set_name, aggregation, injection_type, severity, seed, target_pair = keys
        sensitivity.append(dict(zip(target_keys, keys)) | metric_values(
            normal_eval[(set_name, aggregation)], group["score"].to_numpy(), normal_calibration[(set_name, aggregation)]
        ))
    seed_keys = ["relation_set", "aggregation", "injection_type", "severity", "seed"]
    for keys, group in anomaly_df.groupby(seed_keys, sort=True):
        set_name, aggregation, *_ = keys
        seed_summary.append(dict(zip(seed_keys, keys)) | metric_values(
            normal_eval[(set_name, aggregation)], group["score"].to_numpy(), normal_calibration[(set_name, aggregation)]
        ))
    relation_keys = ["relation_set", "aggregation", "injection_type", "severity"]
    for keys, group in anomaly_df.groupby(relation_keys, sort=True):
        set_name, aggregation, *_ = keys
        relation_summary.append(dict(zip(relation_keys, keys)) | metric_values(
            normal_eval[(set_name, aggregation)], group["score"].to_numpy(), normal_calibration[(set_name, aggregation)]
        ))
    for (set_name, aggregation), group in anomaly_df.groupby(["relation_set", "aggregation"], sort=True):
        relation_summary.append({
            "relation_set": set_name, "aggregation": aggregation,
            "injection_type": "all", "severity": "all",
            **metric_values(normal_eval[(set_name, aggregation)], group["score"].to_numpy(), normal_calibration[(set_name, aggregation)]),
        })
    return sensitivity, seed_summary, relation_summary


def summarize_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    keys = ["relation_set", "injection_type", "severity", "seed"]
    for values, group in frame.groupby(keys, sort=True):
        valid = group["valid_injection"].astype(bool)
        output.append(dict(zip(keys, values)) | {
            "attempted_count": len(group), "valid_count": int(valid.sum()),
            "valid_rate": float(valid.mean()), "skipped_count": int((~valid).sum()),
            "median_corr_change_valid": float(group.loc[valid, "corr_change"].median()) if valid.any() else float("nan"),
        })
    return output


def summarize_localization(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    keys = ["relation_set", "injection_type", "severity", "seed", "target_pair"]
    for values, group in frame.groupby(keys, sort=True):
        output.append(dict(zip(keys, values)) | {
            "sample_count": len(group), "top1_hit_rate": float(group["top1_hit"].mean()),
            "top3_hit_rate": float(group["top3_hit"].mean()), "top5_hit_rate": float(group["top5_hit"].mean()),
            "target_pair_mean_rank": float(group["target_pair_rank"].mean()),
        })
    for set_name, group in frame.groupby("relation_set", sort=True):
        output.append({
            "relation_set": set_name, "injection_type": "all", "severity": "all", "seed": "all",
            "target_pair": "all", "sample_count": len(group),
            "top1_hit_rate": float(group["top1_hit"].mean()), "top3_hit_rate": float(group["top3_hit"].mean()),
            "top5_hit_rate": float(group["top5_hit"].mean()), "target_pair_mean_rank": float(group["target_pair_rank"].mean()),
        })
    return output


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    frame = pd.DataFrame(list(rows))
    if fields is not None:
        frame = frame.reindex(columns=fields)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _best_overall(relation_summary: list[dict[str, Any]], set_name: str) -> dict[str, Any]:
    rows = [row for row in relation_summary if row["relation_set"] == set_name and row["injection_type"] == "all"]
    return max(rows, key=lambda row: float(row["auroc"])) if rows else {}


def build_diagnosis(
    set_summaries: dict[str, dict[str, Any]],
    quality_summary: list[dict[str, Any]],
    relation_summary: list[dict[str, Any]],
    localization_summary: list[dict[str, Any]],
    scale_floor: float,
) -> tuple[str, dict[str, Any]]:
    best = {name: _best_overall(relation_summary, name) for name in set_summaries}
    strict = best.get("strict_stable", {})
    diverse = best.get("diversity_capped", {})
    expanded = best.get("expanded_strong", {})
    strict_sensitive = bool(strict and strict.get("auroc", 0) >= 0.80)
    diversity_drop = float(strict.get("auroc", np.nan)) - float(diverse.get("auroc", np.nan)) if strict and diverse else float("nan")
    diversity_degraded = bool(np.isfinite(diversity_drop) and diversity_drop > 0.05)
    expanded_gain = float(expanded.get("auroc", np.nan)) - float(strict.get("auroc", np.nan)) if strict and expanded else float("nan")
    expanded_noise = bool(expanded and strict and float(expanded.get("normal_score_q99", 0)) > 1.20 * float(strict.get("normal_score_q99", 0)) and expanded_gain <= 0.02)
    best_all = max((row for row in best.values() if row), key=lambda row: float(row["auroc"]), default={})
    loc = {
        row["relation_set"]: row for row in localization_summary
        if row["injection_type"] == "all" and row["target_pair"] == "all"
    }
    fixed_supported = bool(
        best_all and float(best_all.get("auroc", 0)) >= 0.80
        and float(best_all.get("relation_only_recall_at_normal_q99", 0)) >= 0.50
        and float(loc.get(best_all.get("relation_set"), {}).get("top3_hit_rate", 0)) >= 0.60
    )
    quality_by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in quality_summary:
        counts = quality_by_type[str(row["injection_type"])]
        counts[0] += int(row["valid_count"])
        counts[1] += int(row["attempted_count"])
    quality_text = ", ".join(
        f"{name}={counts[0] / max(1, counts[1]):.3f}"
        for name, counts in sorted(quality_by_type.items())
    )
    diagnosis = {
        "strict_stable_sensitive": strict_sensitive,
        "diversity_capped_significant_degradation": diversity_degraded,
        "diversity_auroc_drop": diversity_drop,
        "expanded_strong_mainly_adds_noise": expanded_noise,
        "expanded_auroc_gain_vs_strict": expanded_gain,
        "best_relation_set": best_all.get("relation_set"),
        "best_aggregation": best_all.get("aggregation"),
        "supports_fixed_candidate_relation_set": fixed_supported,
        "supports_next_swat_independent_structure_evaluation": fixed_supported,
        "existing_long_structure_score_included": False,
        "existing_long_structure_score_note": "Omitted to avoid expanding this step into checkpoint reconstruction inference code.",
    }
    markdown = f"""# Step 4.3A Final Diagnosis

This diagnosis uses only the SWaT normal training split. No SWaT test data or attack labels were read.

## Candidate sets

| Set | Pairs | Channels | Max degree | Max occurrence ratio |
|---|---:|---:|---:|---:|
"""
    for name, summary in set_summaries.items():
        markdown += f"| {name} | {summary['pair_count']} | {summary['involved_channel_count']} | {summary['maximum_channel_degree']} | {summary['channel_occurrence_ratio']:.3f} |\n"
    markdown += f"""

## Answers

1. **strict_stable sensitivity:** {'supported' if strict_sensitive else 'not established'}; best overall AUROC={strict.get('auroc', float('nan')):.4f}, class-balanced AUPR={strict.get('aupr', float('nan')):.4f}.
2. **Diversity cap:** {'significant degradation' if diversity_degraded else 'no significant degradation by the predefined AUROC > 0.05 rule'}; AUROC change versus strict={-diversity_drop if np.isfinite(diversity_drop) else float('nan'):.4f}.
3. **Expanded set:** {'mainly adds normal noise' if expanded_noise else 'does not meet the predefined mainly-noise rule'}; AUROC gain versus strict={expanded_gain:.4f}.
4. **Best aggregation:** {best_all.get('aggregation', 'unavailable')} in {best_all.get('relation_set', 'unavailable')}.
5. **Injection quality:** mean valid rates by type: {quality_text}.
6. **Localization:** see `relation_localization_metrics.csv`; overall top-k rows use injection_type=`all`.
7. **Fix a candidate set now:** {'supported' if fixed_supported else 'not yet supported'} under AUROC>=0.80, q99 recall>=0.50, and top3 hit rate>=0.60.
8. **Proceed to SWaT comprehensive independent-structure evaluation:** {'supported' if fixed_supported else 'not yet supported'}.

The actual scale floor was {scale_floor:.6f}. The existing reconstruction-based long structure score was not included because doing so would expand this controlled relation-only step into checkpoint reconstruction inference code.
"""
    return markdown, diagnosis


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    output_dir = project_root / (
        "outputs/stage4_3a_candidate_relation_sensitivity_smoke" if args.smoke else args.output_dir
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    required = [
        project_root / "scripts/stage4_2a_analyze_relation_stability.py",
        project_root / "scripts/stage4_2b_diagnose_relation_validity.py",
        project_root / "scripts/stage4_2c_analyze_local_relation_candidates.py",
        project_root / "configs/swat_improve_v4_struct_finetune.yaml",
        project_root / "Swat/normal.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    seeds = parse_seeds(args.seeds)
    print(f"Step 4.3A: window={WINDOW_SIZE}, evaluation_stride={args.evaluation_stride}, seeds={seeds}")
    print("Data: SWaT normal training split only; chronological reference 80%, held-out 20%")
    print("Not used: SWaT test data, attack labels, X_hat, checkpoints, model training, score fusion")
    print(f"Output: {output_dir}")
    if missing:
        print("Missing paths:")
        for path in missing:
            print(f"- {path}")
        if args.dry_run:
            return {"dry_run": True, "missing_paths": missing}
        raise FileNotFoundError(missing[0])
    if args.dry_run:
        return {"dry_run": True, "missing_paths": []}

    values, feature_names, preprocessing = load_stage_data(project_root, args.smoke)
    reference_end = int(len(values) * 0.80)
    reference = values[:reference_end]
    held_out = values[reference_end:]
    raw_activity, _ = channel_activity_rows(
        reference, list(range(len(feature_names))), feature_names, (WINDOW_SIZE,), VARIANCE_EPSILON
    )
    activity_rows = _activity_output_rows(raw_activity)
    print(f"Recomputing 1275 relation candidates on normal_reference rows={len(reference)}")
    pair_rows = _analyze_window(reference, feature_names, WINDOW_SIZE, activity_rows)
    candidate_sets = build_candidate_sets(pair_rows)
    expanded_rows = candidate_sets["expanded_strong"]
    if not expanded_rows:
        raise RuntimeError("No expanded_strong candidates were found on normal_reference")
    reference_starts = np.arange(0, len(reference) - WINDOW_SIZE + 1, dtype=np.int64)
    models, _, scale_floor = fit_normal_relation_models(reference, reference_starts, expanded_rows)
    model_lookup = {str(row["pair_id"]): row for row in models}
    indices_by_set = _set_indices(candidate_sets, models)

    all_held_starts = np.arange(0, len(held_out) - WINDOW_SIZE + 1, args.evaluation_stride, dtype=np.int64)
    split = len(all_held_starts) // 2
    calibration_starts = all_held_starts[:split]
    evaluation_starts = all_held_starts[split:]
    pairs = [_pair_tuple(row) for row in models]
    calibration_corr = correlations_for_starts(held_out, calibration_starts, pairs)
    evaluation_corr = correlations_for_starts(held_out, evaluation_starts, pairs)
    calibration_pair_scores = score_matrix(calibration_corr, models, scale_floor)
    evaluation_pair_scores = score_matrix(evaluation_corr, models, scale_floor)
    calibration_scores = aggregate_normal_scores(calibration_pair_scores, indices_by_set)
    evaluation_scores = aggregate_normal_scores(evaluation_pair_scores, indices_by_set)

    max_samples = min(args.max_samples_per_cell, 5) if args.smoke else args.max_samples_per_cell
    print(f"Generating controlled injections, max valid samples per cell={max_samples}")
    quality_rows, anomaly_rows, localization_raw = generate_injections(
        held_out, evaluation_starts, evaluation_corr, evaluation_pair_scores,
        candidate_sets, models, scale_floor, feature_names, seeds, max_samples,
    )

    normal_raw: list[dict[str, Any]] = []
    for seed in seeds:
        for (set_name, aggregation), scores in evaluation_scores.items():
            normal_raw.extend({
                "sample_id": f"normal:{seed}:{set_name}:{aggregation}:{index}",
                "sample_kind": "held_out_normal", "seed": seed, "relation_set": set_name,
                "aggregation": aggregation, "score": float(score), "target_pair": "",
                "injection_type": "none", "severity": "none",
            } for index, score in enumerate(scores))
    raw_rows = normal_raw + anomaly_rows
    anomaly_df = pd.DataFrame(anomaly_rows)
    sensitivity, seed_summary, relation_summary = build_metric_tables(
        anomaly_df, evaluation_scores, calibration_scores
    )
    quality_summary = summarize_quality(quality_rows)
    localization_summary = summarize_localization(localization_raw)

    set_summaries = {name: relation_set_summary(rows) for name, rows in candidate_sets.items()}
    set_json: dict[str, Any] = {}
    set_csv_rows: list[dict[str, Any]] = []
    for name, rows in candidate_sets.items():
        set_json[name] = {"summary": set_summaries[name], "relations": []}
        for row in rows:
            relation = {
                "pair_id": row["pair_id"], "channel_i": row["channel_i"], "channel_j": row["channel_j"],
                "channel_i_index": row["channel_i_index"], "channel_j_index": row["channel_j_index"],
                "valid_ratio": row["valid_ratio"], "abs_median_corr": row["abs_median_corr"],
                "relation_instability": row["relation_instability"],
            }
            set_json[name]["relations"].append(relation)
            set_csv_rows.append({"relation_set": name, **set_summaries[name], **relation})
    diagnosis_markdown, diagnosis = build_diagnosis(
        set_summaries, quality_summary, relation_summary, localization_summary, scale_floor
    )
    summary = {
        "scope": {
            **preprocessing, "window_size": WINDOW_SIZE, "evaluation_stride": args.evaluation_stride,
            "reference_ratio": 0.80, "reference_rows": len(reference), "held_out_rows": len(held_out),
            "calibration_window_count": len(calibration_starts), "evaluation_window_count": len(evaluation_starts),
            "seeds": seeds, "max_valid_samples_per_pair_type_severity": max_samples,
            "aupr_positive_class_prior": 0.5,
            "test_data_read": False, "attack_labels_read": False, "checkpoint_read": False,
            "model_or_training_modified": False, "current_structure_pairs_modified": False,
        },
        "candidate_relation_sets": set_summaries,
        "actual_scale_floor": scale_floor,
        "injection_attempt_count": len(quality_rows),
        "valid_injection_count": int(sum(bool(row["valid_injection"]) for row in quality_rows)),
        "quality_by_type": quality_summary,
        "best_overall_metrics": {
            name: _best_overall(relation_summary, name) for name in candidate_sets
        },
        "overall_localization": [
            row for row in localization_summary if row["injection_type"] == "all"
        ],
        "diagnosis": diagnosis,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "candidate_relation_sets.json").open("x", encoding="utf-8") as handle:
        json.dump(_safe_json(set_json), handle, ensure_ascii=False, indent=2, allow_nan=False)
    write_csv(output_dir / "candidate_relation_sets.csv", set_csv_rows)
    normal_model_rows = []
    membership = {str(row["pair_id"]): [name for name, rows in candidate_sets.items() if row in rows] for row in expanded_rows}
    for row in models:
        normal_model_rows.append({**row, "relation_sets": "|".join(membership[str(row["pair_id"])])})
    write_csv(output_dir / "normal_relation_models.csv", normal_model_rows)
    write_csv(output_dir / "injection_quality.csv", quality_rows, QUALITY_FIELDS)
    write_csv(output_dir / "injection_quality_summary.csv", quality_summary)
    write_csv(output_dir / "raw_scores.csv", raw_rows)
    write_csv(output_dir / "sensitivity_metrics.csv", sensitivity)
    write_csv(output_dir / "seed_summary.csv", seed_summary)
    write_csv(output_dir / "relation_set_summary.csv", relation_summary)
    write_csv(output_dir / "relation_localization_metrics.csv", localization_summary)
    with (output_dir / "candidate_relation_sensitivity_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(_safe_json(summary), handle, ensure_ascii=False, indent=2, allow_nan=False)
    (output_dir / "final_diagnosis.md").write_text(diagnosis_markdown, encoding="utf-8")
    print(json.dumps(_safe_json({
        "candidate_relation_sets": set_summaries,
        "actual_scale_floor": scale_floor,
        "valid_injections": summary["valid_injection_count"],
        "injection_attempts": summary["injection_attempt_count"],
        "diagnosis": diagnosis,
    }), ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 4.3A controlled local-relation sensitivity evaluation")
    parser.add_argument("--project-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output-dir", default="outputs/stage4_3a_candidate_relation_sensitivity")
    parser.add_argument("--evaluation-stride", type=int, default=16)
    parser.add_argument("--max-samples-per-cell", type=int, default=100)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.evaluation_stride <= 0 or args.max_samples_per_cell <= 0:
        parser.error("evaluation stride and max samples must be positive")
    return args


if __name__ == "__main__":
    analyze(parse_args())
