from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hai_structure_v2a1r_cache_backfill_and_grouped_aggregation import grouped_scores

V2A = ROOT / "outputs" / "hai_structure_v2a_lagged_dependency"
A1R = ROOT / "outputs" / "hai_structure_v2a1r_cache_backfill_and_grouped_aggregation"
A1S = ROOT / "outputs" / "hai_structure_v2a1s_full_clean_stability"
A1D = ROOT / "outputs" / "hai_structure_v2a1d_normal_drift_attribution"
A1M = ROOT / "outputs" / "hai_structure_v2a1m_mode_conditioned_reference"
OUT = ROOT / "outputs" / "hai_structure_v2a1c_causal_adaptive_reference"
SMOKE_OUT = ROOT / "outputs" / "hai_structure_v2a1c_causal_adaptive_reference_smoke"

HALF_LIVES = (512, 2048)
FAMILIES = ("R1", "R2", "R3", "P1", "P2", "C1", "C2")
WARMUP = 1024
CLIP_MULTIPLIER = 3.0
LOCATION_STEP_CAP = 0.05
SCALE_RATIO_LOW = 0.98
SCALE_RATIO_HIGH = 1.02
SCALE_FLOOR_RATIO = 0.25
EPS = 1e-6
CHECKPOINT_STEPS = (1, 16, 64, 256, 1024)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def historical_hashes() -> dict[str, str]:
    result = {}
    for prefix, directory in (("v2a", V2A), ("a1r", A1R), ("a1s", A1S), ("a1d", A1D), ("a1m", A1M)):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                result[f"{prefix}/{path.relative_to(directory)}"] = sha256(path)
    return result


def cache_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def alpha_from_half_life(half_life: int) -> float:
    if half_life not in HALF_LIVES:
        raise ValueError(f"Only half-lives {HALF_LIVES} are allowed")
    return 1.0 - math.exp(math.log(0.5) / half_life)


def edge_scores(values: np.ndarray, location: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, (values - location) / scale)


def adaptive_update(
    values: np.ndarray,
    location: np.ndarray,
    abs_deviation: np.ndarray,
    scale: np.ndarray,
    global_scale: np.ndarray,
    alpha: float,
    update_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    innovation = np.clip(values - location, -CLIP_MULTIPLIER * scale, CLIP_MULTIPLIER * scale)
    location_delta = np.clip(alpha * innovation, -LOCATION_STEP_CAP * scale, LOCATION_STEP_CAP * scale)
    location_candidate = location + location_delta
    abs_dev = np.minimum(np.abs(values - location), CLIP_MULTIPLIER * scale)
    abs_candidate = (1.0 - alpha) * abs_deviation + alpha * abs_dev
    scale_candidate = np.maximum(1.4826 * abs_candidate, np.maximum(SCALE_FLOOR_RATIO * global_scale, EPS))
    scale_candidate = np.clip(scale_candidate, SCALE_RATIO_LOW * scale, SCALE_RATIO_HIGH * scale)
    return (
        np.where(update_mask, location_candidate, location),
        np.where(update_mask, abs_candidate, abs_deviation),
        np.where(update_mask, scale_candidate, scale),
    )


def sequence_slices(sequence_ids: np.ndarray):
    ids = np.asarray(sequence_ids).astype(str)
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(ids)]
    return [(ids[start], slice(int(start), int(end))) for start, end in zip(starts, ends)]


def run_adaptive_stream(
    values: np.ndarray,
    sequence_ids: np.ndarray,
    global_location: np.ndarray,
    global_scale: np.ndarray,
    target_ids: np.ndarray,
    alpha: float,
    global_freeze_threshold: float,
    edge_freeze_thresholds: np.ndarray,
    trace: bool = False,
) -> dict[str, np.ndarray | list[dict]]:
    values = np.asarray(values, dtype=np.float64)
    n, edge_count = values.shape
    grouped_score = np.empty(n, dtype=np.float64)
    edge_score = np.empty((n, edge_count), dtype=np.float32) if trace else None
    locations_before = np.empty((n, edge_count), dtype=np.float64) if trace else None
    scales_before = np.empty((n, edge_count), dtype=np.float64) if trace else None
    updated = np.empty((n, edge_count), dtype=bool) if trace else None
    global_frozen = np.empty(n, dtype=bool) if trace else None
    diagnostics = []
    for sequence, section in sequence_slices(sequence_ids):
        location = global_location.astype(np.float64).copy()
        scale = global_scale.astype(np.float64).copy()
        abs_deviation = scale / 1.4826
        update_count = np.zeros(edge_count, dtype=np.int64)
        edge_freeze_count = np.zeros(edge_count, dtype=np.int64)
        floor_streak = np.zeros(edge_count, dtype=np.int64)
        max_floor_streak = np.zeros(edge_count, dtype=np.int64)
        initial_location = location.copy()
        initial_scale = scale.copy()
        sequence_global_freeze = 0
        start = section.start
        for position in range(section.start, section.stop):
            x = values[position]
            if trace:
                locations_before[position] = location
                scales_before[position] = scale
            current_edges = edge_scores(x, location, scale)
            current_global = float(grouped_scores(current_edges[None, :], target_ids)[0][0])
            grouped_score[position] = current_global
            if trace:
                edge_score[position] = current_edges
            freeze_all = current_global > global_freeze_threshold
            edge_frozen = current_edges > edge_freeze_thresholds
            update_mask = ~(edge_frozen | freeze_all)
            if freeze_all:
                sequence_global_freeze += 1
            edge_freeze_count += edge_frozen.astype(np.int64)
            update_count += update_mask.astype(np.int64)
            if trace:
                updated[position] = update_mask
                global_frozen[position] = freeze_all
            location, abs_deviation, scale = adaptive_update(
                x, location, abs_deviation, scale, global_scale, alpha, update_mask
            )
            at_floor = scale <= np.maximum(SCALE_FLOOR_RATIO * global_scale, EPS) * (1.0 + 1e-12)
            floor_streak = np.where(at_floor, floor_streak + 1, 0)
            max_floor_streak = np.maximum(max_floor_streak, floor_streak)
        diagnostics.append(
            {
                "sequence_id": sequence,
                "window_count": section.stop - section.start,
                "global_freeze_count": sequence_global_freeze,
                "edge_freeze_count": edge_freeze_count,
                "update_count": update_count,
                "max_floor_streak": max_floor_streak,
                "initial_location": initial_location,
                "final_location": location.copy(),
                "initial_scale": initial_scale,
                "final_scale": scale.copy(),
                "location_absolute_change": np.abs(location - initial_location),
                "scale_ratio": scale / initial_scale,
                "start_position": start,
                "end_position": section.stop,
            }
        )
    result = {"grouped_score": grouped_score, "diagnostics": diagnostics}
    if trace:
        result.update(
            {
                "edge_score": edge_score,
                "location_before": locations_before,
                "scale_before": scales_before,
                "updated": updated,
                "global_frozen": global_frozen,
            }
        )
    return result


def fit_update_gates(train_values, global_location, global_scale, target_ids):
    static_edges = edge_scores(train_values, global_location, global_scale)
    static_grouped = grouped_scores(static_edges, target_ids)[0]
    return float(np.quantile(static_grouped, 0.995)), np.quantile(static_edges, 0.995, axis=0), static_grouped


def candidate_protocol(scores, sequence_ids, diagnostics, half_life):
    calibration = np.zeros(len(scores), dtype=bool)
    evaluation = np.zeros(len(scores), dtype=bool)
    parts = []
    for sequence, section in sequence_slices(sequence_ids):
        usable_start = section.start + WARMUP
        usable_count = section.stop - usable_start
        midpoint = usable_start + usable_count // 2
        calibration[usable_start:midpoint] = True
        evaluation[midpoint:section.stop] = True
        parts.append((sequence, slice(usable_start, midpoint), slice(midpoint, section.stop)))
    threshold = float(np.quantile(scores[calibration], 0.99))
    rows = []
    for sequence, cal, eva in parts:
        cal_fpr = float(np.mean(scores[cal] > threshold))
        eval_fpr = float(np.mean(scores[eva] > threshold))
        half = eva.start + (eva.stop - eva.start) // 2
        rows.append(
            {
                "half_life": half_life,
                "sequence_id": sequence,
                "calibration_count": cal.stop - cal.start,
                "evaluation_count": eva.stop - eva.start,
                "calibration_fpr": cal_fpr,
                "evaluation_fpr": eval_fpr,
                "evaluation_first_fpr": float(np.mean(scores[eva.start:half] > threshold)),
                "evaluation_second_fpr": float(np.mean(scores[half:eva.stop] > threshold)),
                "temporal_fpr_change": float(np.mean(scores[half:eva.stop] > threshold) - np.mean(scores[eva.start:half] > threshold)),
            }
        )
    by_sequence = pd.DataFrame(rows)
    eval_fpr = float(np.mean(scores[evaluation] > threshold))
    worst = float(by_sequence.evaluation_fpr.max())
    passing_sequences = int((by_sequence.evaluation_fpr <= 0.05).sum())
    max_floor_streak = max(int(np.max(item["max_floor_streak"])) for item in diagnostics)
    all_finite = bool(np.isfinite(scores).all())
    scale_collapse = any(np.any(np.asarray(item["final_scale"]) < EPS) for item in diagnostics)
    valid = bool(eval_fpr <= 0.02 and worst <= 0.05 and passing_sequences >= 5 and all_finite and not scale_collapse and max_floor_streak < WARMUP)
    row = {
        "half_life": half_life,
        "alpha": alpha_from_half_life(half_life),
        "train_calibration_q99": threshold,
        "train_evaluation_overall_fpr": eval_fpr,
        "train_evaluation_worst_sequence_fpr": worst,
        "sequences_fpr_le_005": passing_sequences,
        "reference_update_count": int(sum(np.asarray(x["update_count"]).sum() for x in diagnostics)),
        "global_freeze_count": int(sum(x["global_freeze_count"] for x in diagnostics)),
        "edge_freeze_count": int(sum(np.asarray(x["edge_freeze_count"]).sum() for x in diagnostics)),
        "maximum_floor_streak": max_floor_streak,
        "all_references_finite": all_finite,
        "scale_collapse": scale_collapse,
        "valid": valid,
    }
    return row, by_sequence, threshold


def select_candidate(candidate_metrics: pd.DataFrame) -> int | None:
    valid = candidate_metrics[candidate_metrics.valid.astype(bool)].copy()
    if valid.empty:
        return None
    valid = valid.sort_values(["train_evaluation_worst_sequence_fpr", "half_life"])
    if len(valid) == 2 and abs(valid.iloc[0].train_evaluation_worst_sequence_fpr - valid.iloc[1].train_evaluation_worst_sequence_fpr) < 0.005:
        return 2048
    return int(valid.iloc[0].half_life)


def alarm_segments_frame(sequence_ids, endpoint_rows, timestamps, alarms):
    rows = []
    for sequence, section in sequence_slices(sequence_ids):
        active = np.asarray(alarms[section], dtype=bool)
        padded = np.r_[False, active, False].astype(np.int8)
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        for segment, (start, end) in enumerate(zip(starts, ends), 1):
            left, right = section.start + start, section.start + end - 1
            rows.append(
                {
                    "sequence_id": sequence,
                    "segment_id": segment,
                    "start_endpoint_row": int(endpoint_rows[left]),
                    "end_endpoint_row": int(endpoint_rows[right]),
                    "start_timestamp": str(timestamps[left]),
                    "end_timestamp": str(timestamps[right]),
                    "length": int(end - start),
                }
            )
    return pd.DataFrame(rows)


def validation_metrics(scores, sequence_ids, threshold):
    alarms = scores > threshold
    rows = []
    for sequence, section in sequence_slices(sequence_ids):
        midpoint = section.start + (section.stop - section.start) // 2
        rows.append(
            {
                "sequence_id": sequence,
                "window_count": section.stop - section.start,
                "alarm_count": int(alarms[section].sum()),
                "fpr": float(alarms[section].mean()),
                "first_half_fpr": float(alarms[section.start:midpoint].mean()),
                "second_half_fpr": float(alarms[midpoint:section.stop].mean()),
            }
        )
    by_sequence = pd.DataFrame(rows)
    train1 = by_sequence[by_sequence.sequence_id.eq("validation1")].iloc[0]
    return {
        "overall_fpr": float(alarms.mean()),
        "worst_sequence_fpr": float(by_sequence.fpr.max()),
        "train1_first_half_fpr": float(train1.first_half_fpr),
        "train1_second_half_fpr": float(train1.second_half_fpr),
        "alarm_count": int(alarms.sum()),
    }, by_sequence, alarms


def source_window_ids(sequence_ids, endpoint_rows):
    result = []
    for sequence, endpoint in zip(np.asarray(sequence_ids).astype(str), endpoint_rows):
        number = sequence.removeprefix("validation")
        result.append(f"train{number}_val_row_{int(endpoint)}")
    return np.asarray(result)


def binary_metrics(clean, positive, threshold):
    clean = np.asarray(clean, float)
    positive = np.asarray(positive, float)
    labels = np.r_[np.zeros(len(clean), int), np.ones(len(positive), int)]
    scores = np.r_[clean, positive]
    predictions = scores > threshold
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "q99_tpr": float(np.mean(positive > threshold)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "median_score_ratio": float(np.median(positive) / max(np.median(clean), 1e-12)),
    }


def evaluate_controlled(controlled, grouped, clean_source, threshold):
    valid = controlled["valid_mask"].astype(bool)
    families = controlled["family"].astype(str)
    sequences = controlled["sequence_id"].astype(str)
    family_rows, sequence_rows = [], []
    for family_index, family in enumerate(families):
        mask = valid[family_index]
        positive = grouped[family_index][mask]
        clean = np.repeat(clean_source, mask.sum(axis=1))
        family_rows.append({"family": family, "sample_count": len(positive), **binary_metrics(clean, positive, threshold)})
        for sequence in np.unique(sequences):
            source_mask = sequences == sequence
            local_mask = mask[source_mask]
            sequence_rows.append(
                {
                    "family": family,
                    "sequence_id": sequence,
                    "sample_count": int(local_mask.sum()),
                    **binary_metrics(
                        np.repeat(clean_source[source_mask], local_mask.sum(axis=1)),
                        grouped[family_index][source_mask][local_mask],
                        threshold,
                    ),
                }
            )
    return pd.DataFrame(family_rows), pd.DataFrame(sequence_rows)


def controlled_scores(controlled, source_positions, validation_trace, target_ids):
    locations = validation_trace["location_before"][source_positions]
    scales = validation_trace["scale_before"][source_positions]
    rolling = controlled["rolling_dependency_loss"].astype(np.float64)
    scores = np.maximum(0.0, (rolling - locations[None, :, None, :]) / scales[None, :, None, :])
    grouped = grouped_scores(scores, target_ids)[0]
    return scores.astype(np.float32), grouped.astype(np.float32)


def select_persistent_samples(valid_mask, sequences, minimum=100):
    pairs = np.argwhere(valid_mask)
    chosen = []
    for sequence in np.unique(sequences):
        for edge in range(valid_mask.shape[1]):
            candidates = pairs[(sequences[pairs[:, 0]] == sequence) & (pairs[:, 1] == edge)]
            if len(candidates):
                chosen.append(tuple(candidates[0]))
    used = set(chosen)
    for pair in map(tuple, pairs):
        if len(chosen) >= minimum:
            break
        if pair not in used:
            chosen.append(pair)
            used.add(pair)
    if len(chosen) < minimum:
        raise RuntimeError("Fewer than 100 valid persistent-corruption samples")
    return np.asarray(chosen, dtype=np.int64)


def persistent_absorption(
    controlled, source_positions, validation_trace, target_ids, global_scale, alpha,
    global_freeze_threshold, edge_freeze_thresholds, threshold,
):
    families = controlled["family"].astype(str)
    sequences = controlled["sequence_id"].astype(str)
    trace_rows, metric_rows = [], []
    trace_payload = {"steps": np.asarray(CHECKPOINT_STEPS), "families": np.asarray(("R1", "R2", "R3"))}
    for family in ("R1", "R2", "R3"):
        family_index = int(np.flatnonzero(families == family)[0])
        selected = select_persistent_samples(controlled["valid_mask"][family_index], sequences, 100)
        windows, targets = selected[:, 0], selected[:, 1]
        positions = source_positions[windows]
        location = validation_trace["location_before"][positions].copy()
        scale = validation_trace["scale_before"][positions].copy()
        abs_deviation = scale / 1.4826
        values = controlled["rolling_dependency_loss"][family_index, windows, targets].astype(np.float64)
        target_edges = controlled["target_edge_id"][targets].astype(int)
        initial_score = None
        initial_detection = None
        checkpoint_scores, checkpoint_target, checkpoint_update, checkpoint_global_freeze = [], [], [], []
        initial_location = location.copy()
        initial_scale = scale.copy()
        for step in range(1, max(CHECKPOINT_STEPS) + 1):
            scores = np.maximum(0.0, (values - location) / scale)
            global_scores = grouped_scores(scores, target_ids)[0]
            target_scores = scores[np.arange(len(scores)), target_edges]
            freeze_all = global_scores > global_freeze_threshold
            edge_frozen = scores > edge_freeze_thresholds
            update_mask = ~(edge_frozen | freeze_all[:, None])
            if initial_score is None:
                initial_score = global_scores.copy()
                initial_detection = initial_score > threshold
            if step in CHECKPOINT_STEPS:
                detection = global_scores > threshold
                retention = global_scores / np.maximum(initial_score, 1e-12)
                checkpoint_scores.append(global_scores.copy())
                checkpoint_target.append(target_scores.copy())
                checkpoint_update.append(update_mask.copy())
                checkpoint_global_freeze.append(freeze_all.copy())
                metric_rows.append(
                    {
                        "family": family,
                        "step": step,
                        "sample_count": len(selected),
                        "median_score_retention": float(np.median(retention)),
                        "initial_detection_rate": float(initial_detection.mean()),
                        "detection_rate": float(detection.mean()),
                        "detection_retention_change": float(detection.mean() - initial_detection.mean()),
                        "reference_absorption_rate": float(np.median(1.0 - np.minimum(retention, 1.0))),
                        "global_freeze_rate": float(freeze_all.mean()),
                        "edge_freeze_rate": float(edge_frozen.mean()),
                        "median_location_change": float(np.median(np.abs(location - initial_location))),
                        "median_scale_ratio": float(np.median(scale / initial_scale)),
                        "maximum_scale_ratio": float(np.max(scale / initial_scale)),
                    }
                )
            innovation = np.clip(values - location, -CLIP_MULTIPLIER * scale, CLIP_MULTIPLIER * scale)
            location_candidate = location + np.clip(alpha * innovation, -LOCATION_STEP_CAP * scale, LOCATION_STEP_CAP * scale)
            abs_dev = np.minimum(np.abs(values - location), CLIP_MULTIPLIER * scale)
            abs_candidate = (1.0 - alpha) * abs_deviation + alpha * abs_dev
            scale_candidate = np.maximum(1.4826 * abs_candidate, np.maximum(SCALE_FLOOR_RATIO * global_scale, EPS))
            scale_candidate = np.clip(scale_candidate, SCALE_RATIO_LOW * scale, SCALE_RATIO_HIGH * scale)
            location = np.where(update_mask, location_candidate, location)
            abs_deviation = np.where(update_mask, abs_candidate, abs_deviation)
            scale = np.where(update_mask, scale_candidate, scale)
        trace_payload[f"{family}_source_window_index"] = windows
        trace_payload[f"{family}_target_position"] = targets
        trace_payload[f"{family}_global_score"] = np.asarray(checkpoint_scores)
        trace_payload[f"{family}_target_edge_score"] = np.asarray(checkpoint_target)
        trace_payload[f"{family}_updated"] = np.asarray(checkpoint_update)
        trace_payload[f"{family}_global_frozen"] = np.asarray(checkpoint_global_freeze)
    metrics = pd.DataFrame(metric_rows)
    return trace_payload, metrics


def absorption_gates(metrics):
    step256 = metrics[metrics.step.eq(256)]
    step1024 = metrics[metrics.step.eq(1024)]
    return {
        "step256_median_retention": bool(step256.median_score_retention.median() >= 0.70),
        "step256_detection_retention": bool(step256.detection_retention_change.mean() >= -0.05),
        "step1024_median_retention": bool(step1024.median_score_retention.median() >= 0.50),
        "worst_family_step256_detection": bool(step256.detection_retention_change.min() >= -0.10),
    }


def write_dry_run():
    print("REUSE:", flush=True)
    for path in (
        A1M / "train_raw_dynamic_dependency_cache.npz",
        A1M / "validation_raw_dynamic_dependency_cache.npz",
        A1M / "controlled_raw_dynamic_dependency_cache.npz",
        V2A / "selected_dynamic_relation_pool.csv",
        V2A / "dynamic_dependency_reference_registry.csv",
        A1S / "full_clean_crossfit_window_manifest.csv",
        A1D / "drift_classification_evidence.json",
    ):
        print(f"  {'OK' if path.exists() else 'MISSING'} {path}", flush=True)
    print("COMPUTE_NEW: train-only update gates; half-life 512/2048 candidate audit; frozen validation adaptive trace; controlled instant scores; persistent absorption", flush=True)
    print("FORBIDDEN: raw CSV, HAI test, attack labels, Ridge/AE/checkpoint/rec, relation screening, Ridge fit, corruption regeneration", flush=True)
    return 0


def validate_cache(cache, required):
    missing = [key for key in required if key not in cache]
    finite = all(np.isfinite(cache[key]).all() for key in required if key in cache and np.issubdtype(cache[key].dtype, np.number))
    return missing, finite


def json_default(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def finalize_existing_outputs(out, before, status):
    """Finish reporting from completed CSV/NPZ artifacts without rerunning a stream."""
    clean = pd.read_csv(out / "validation_adaptive_clean_metrics.csv").iloc[0].to_dict()
    family = pd.read_csv(out / "controlled_adaptive_metrics_by_family.csv")
    sequence = pd.read_csv(out / "controlled_adaptive_metrics_by_sequence.csv")
    absorption = pd.read_csv(out / "persistent_corruption_absorption_metrics.csv")
    drift = pd.read_csv(out / "validation_reference_drift_by_edge.csv")
    selected = yaml.safe_load((out / "selected_adaptive_reference.yaml").read_text(encoding="utf-8"))
    breaking = family[family.family.isin(("R1", "R2", "R3"))]
    breaking_sequence = sequence[sequence.family.isin(("R1", "R2", "R3"))]
    c2 = pd.read_csv(out / "c2_adaptive_preserving_metrics.csv").iloc[0]
    p1_fpr = float(family.loc[family.family.eq("P1"), "q99_tpr"].iloc[0])
    absorption_pass = absorption_gates(absorption)
    gates = {
        "validation_overall_fpr": clean["overall_fpr"] <= 0.02,
        "validation_worst_sequence_fpr": clean["worst_sequence_fpr"] <= 0.05,
        "train1_second_half_reduction": clean["train1_second_half_fpr"] <= 0.0785694,
        "macro_auroc": breaking.auroc.mean() >= 0.82,
        "macro_aupr": breaking.aupr.mean() >= 0.84,
        "macro_q99_tpr": breaking.q99_tpr.mean() >= 0.40,
        "auroc_baseline_preservation": breaking.auroc.mean() >= 0.818228,
        "q99_tpr_baseline_preservation": breaking.q99_tpr.mean() >= 0.401204,
        "two_family_auroc": int((breaking.auroc >= 0.85).sum()) >= 2,
        "two_family_q99_tpr": int((breaking.q99_tpr >= 0.40).sum()) >= 2,
        "five_sequence_auroc": int((breaking_sequence.groupby("sequence_id").auroc.mean() >= 0.70).sum()) >= 5,
        "p1_fpr": p1_fpr <= 0.05,
        "c2_fpr": float(c2.fpr) <= 0.03,
        "c2_excess_fpr": float(c2.excess_fpr) <= 0.015,
        "persistent_absorption": all(absorption_pass.values()),
        "all_results_finite": all(
            np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
            for frame in (family, sequence, absorption, drift)
        ),
        "parameter_count": True,
    }
    after = historical_hashes()
    hash_rows = [{"artifact": key, "sha256_before": value, "sha256_after": after.get(key), "unchanged": after.get(key) == value} for key, value in before.items()]
    hash_audit = pd.DataFrame(hash_rows)
    hash_audit.to_csv(out / "artifact_reuse_audit.csv", index=False)
    gates["historical_hashes"] = bool(hash_audit.unchanged.all() and set(before) == set(after))
    all_passed = bool(all(gates.values()))
    reduction = 1.0 - float(clean["train1_second_half_fpr"]) / 0.112242
    if all_passed:
        classification = "causal_adaptive_dependency_reference_is_valid"
    elif float(clean["overall_fpr"]) < 0.009078 and not all(absorption_pass.values()):
        classification = "adaptive_reference_absorbs_collaborative_anomalies"
    elif reduction > 0 and (float(clean["worst_sequence_fpr"]) < 0.056603 or float(clean["train1_second_half_fpr"]) < 0.112242):
        classification = "adaptive_reference_reduces_drift_but_not_enough"
    else:
        classification = "adaptive_reference_does_not_reduce_drift"
    final_status = "frozen_for_hai_development_evaluation" if classification == "causal_adaptive_dependency_reference_is_valid" else "not_frozen"
    summary = {
        "classification": classification, "status": final_status, "all_gates_passed": all_passed,
        "gates": gates, "absorption_gates": absorption_pass,
        "selected_half_life": int(selected["selected_half_life"]), "alpha": float(selected["alpha"]),
        "clean": clean, "controlled_macro_auroc": float(breaking.auroc.mean()),
        "controlled_macro_aupr": float(breaking.aupr.mean()), "controlled_macro_q99_tpr": float(breaking.q99_tpr.mean()),
        "p1_fpr": p1_fpr, "c2_fpr": float(c2.fpr), "c2_excess_fpr": float(c2.excess_fpr),
        "test_loaded": False, "attack_label_used": False, "ae_loaded": False, "rec_loaded": False,
        "ridge_loaded": False, "ridge_refit": False,
    }
    (out / "structure_v2a1c_summary.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    (out / "selected_structure_v2a1c_config.yaml").write_text(yaml.safe_dump({
        "status": final_status, "classification": classification,
        "selected_half_life": int(selected["selected_half_life"]), "aggregation": "target_group_max_then_top2_mean",
        "relation_count": 12, "hai_test_used": False, "attack_label_used": False, "ridge_loaded": False, "ridge_refit": False,
    }, sort_keys=False), encoding="utf-8")
    unresolved = {
        "adaptive_reduced_drift": reduction > 0, "train1_second_half_relative_reduction": reduction,
        "persistent_anomaly_absorbed": not all(absorption_pass.values()),
        "largest_edge_updates": drift.groupby("edge_id").update_count.sum().sort_values(ascending=False).head(5).index.astype(int).tolist(),
        "sequence_specific_instability_remains": float(clean["worst_sequence_fpr"]) > 0.05,
        "allow_hai_evaluation": classification == "causal_adaptive_dependency_reference_is_valid",
        "final_structure_branch_limit": None if all_passed else "controlled collaborative mechanism remains detectable, but normal non-stationarity limits practical stability",
        "automatic_next_experiment_started": False,
    }
    (out / "unresolved_issues_delta.json").write_text(json.dumps(unresolved, indent=2, default=json_default), encoding="utf-8")
    report = f"""# HAI Structure V2-A1C causal adaptive reference

## Protocol

- Offline reuse of A1M train, validation, and controlled raw dependency caches only.
- Frozen 12-edge pool, rolling length 16, and target-group max then top2 mean aggregation.
- Train-only candidates: half-life 512 and 2048. Selected: {selected['selected_half_life']}.
- Validation uses the frozen train q99 threshold; no validation tuning or sequence-specific threshold.

## Result

- Classification: `{classification}`
- Status: `{final_status}`
- Validation overall FPR: {clean['overall_fpr']:.6f}
- Worst-sequence FPR: {clean['worst_sequence_fpr']:.6f}
- train1 second-half FPR: {clean['train1_second_half_fpr']:.6f}
- Controlled macro AUROC/AUPR/q99 TPR: {breaking.auroc.mean():.6f} / {breaking.aupr.mean():.6f} / {breaking.q99_tpr.mean():.6f}
- Persistent absorption gates passed: {all(absorption_pass.values())}
- All formal gates passed: {all_passed}

## Boundary

No HAI test, attack label, AE, checkpoint, reconstruction score, rec score, Ridge refit, relation screening, corruption regeneration, fusion, or automatic next structure experiment was used.
"""
    (out / "final_hai_structure_v2a1c_causal_adaptive_reference.md").write_text(report, encoding="utf-8")
    status.update({"persistent_absorption_test_complete": True, "final_classification_complete": True, "final_report_complete": True})
    (out / "stage_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"RESUME_FINALIZE classification={classification} status={final_status} all_gates_passed={all_passed}", flush=True)
    return 0


def run(smoke=False):
    out = SMOKE_OUT if smoke else OUT
    before = historical_hashes()
    dependency_hash = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    status_path = out / "stage_status.json"
    if not smoke and status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("final_report_complete") and status.get("dependency_hash") == dependency_hash:
            print(f"REUSE completed V2-A1C: {out}", flush=True)
            return 0
        completed_files = (
            out / "validation_adaptive_reference_trace.npz",
            out / "controlled_adaptive_scores.npz",
            out / "persistent_corruption_absorption_trace.npz",
            out / "persistent_corruption_absorption_metrics.csv",
            out / "validation_reference_drift_by_edge.csv",
        )
        if status.get("controlled_instant_evaluation_complete") and all(path.exists() for path in completed_files):
            return finalize_existing_outputs(out, before, status)
    out.mkdir(parents=True, exist_ok=True)
    status = {
        "historical_artifacts_verified": False,
        "raw_dependency_cache_verified": False,
        "update_gates_fit": False,
        "train_candidate_a_complete": False,
        "train_candidate_b_complete": False,
        "candidate_selected": False,
        "candidate_frozen_before_validation": False,
        "validation_clean_complete": False,
        "controlled_instant_evaluation_complete": False,
        "persistent_absorption_test_complete": False,
        "final_classification_complete": False,
        "final_report_complete": False,
        "dependency_hash": dependency_hash,
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    pool = pd.read_csv(V2A / "selected_dynamic_relation_pool.csv")
    if len(pool) != 12:
        raise RuntimeError("Frozen relation pool is not 12 edges")
    target_names = list(dict.fromkeys(pool.target.astype(str)))
    target_ids = np.asarray([target_names.index(value) for value in pool.target.astype(str)], dtype=np.int16)
    registry = pd.read_csv(V2A / "dynamic_dependency_reference_registry.csv").sort_values("edge_id")
    global_location = registry["median"].to_numpy(float)
    global_scale = registry["effective_scale"].to_numpy(float)
    train = cache_payload(A1M / "train_raw_dynamic_dependency_cache.npz")
    validation = cache_payload(A1M / "validation_raw_dynamic_dependency_cache.npz")
    controlled = cache_payload(A1M / "controlled_raw_dynamic_dependency_cache.npz")
    required_raw = ["sequence_id", "endpoint_row", "timestamp", "target_edge_ids", "self_residual", "cross_residual", "raw_dependency_loss", "rolling_dependency_loss", "finite_mask"]
    audit_rows = []
    for name, cache in (("train", train), ("validation", validation)):
        missing, finite = validate_cache(cache, required_raw)
        audit_rows.append({"cache": name, "missing_fields": ";".join(missing) or "NONE", "all_finite": finite, "passed": not missing and finite})
    controlled_required = ["source_window_id", "replacement_window_id", "family", "target_edge_id", "fold", "sequence_id", "valid_mask", "rolling_dependency_loss", "finite_mask"]
    missing, finite = validate_cache(controlled, controlled_required)
    audit_rows.append({"cache": "controlled", "missing_fields": ";".join(missing) or "NONE", "all_finite": finite, "passed": not missing and finite})
    cache_audit = pd.DataFrame(audit_rows)
    cache_audit.to_csv(out / "adaptive_cache_integrity_audit.csv", index=False)
    if not cache_audit.passed.all():
        raise RuntimeError("A1M raw dependency cache integrity failed")
    if smoke:
        def subset(cache, count):
            indices = np.concatenate([np.arange(section.start, min(section.stop, section.start + count)) for _, section in sequence_slices(cache["sequence_id"])])
            return {key: (value[indices] if value.ndim and value.shape[0] == len(cache["sequence_id"]) else value) for key, value in cache.items()}
        train = subset(train, 2600)
        validation = subset(validation, 600)
    status["historical_artifacts_verified"] = True
    status["raw_dependency_cache_verified"] = True
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    freeze_threshold, edge_thresholds, static_train_grouped = fit_update_gates(
        train["rolling_dependency_loss"], global_location, global_scale, target_ids
    )
    gate_rows = [{"scope": "global", "edge_id": -1, "quantile": 0.995, "threshold": freeze_threshold, "fit_split": "normal_train"}]
    gate_rows.extend({"scope": "edge", "edge_id": edge, "quantile": 0.995, "threshold": value, "fit_split": "normal_train"} for edge, value in enumerate(edge_thresholds))
    pd.DataFrame(gate_rows).to_csv(out / "adaptive_update_gate_registry.csv", index=False)
    status["update_gates_fit"] = True

    candidate_rows, candidate_sequence_frames, candidate_scores = [], [], {}
    for index, half_life in enumerate(HALF_LIVES):
        print(f"TRAIN_ONLY candidate half_life={half_life}", flush=True)
        result = run_adaptive_stream(
            train["rolling_dependency_loss"], train["sequence_id"], global_location, global_scale,
            target_ids, alpha_from_half_life(half_life), freeze_threshold, edge_thresholds, False,
        )
        row, sequence_frame, _ = candidate_protocol(result["grouped_score"], train["sequence_id"], result["diagnostics"], half_life)
        candidate_rows.append(row)
        candidate_sequence_frames.append(sequence_frame)
        candidate_scores[half_life] = result["grouped_score"]
        status["train_candidate_a_complete" if index == 0 else "train_candidate_b_complete"] = True
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    candidate_metrics = pd.DataFrame(candidate_rows)
    candidate_metrics.to_csv(out / "train_adaptive_candidate_metrics.csv", index=False)
    pd.concat(candidate_sequence_frames, ignore_index=True).to_csv(out / "train_adaptive_metrics_by_sequence.csv", index=False)
    selected_half_life = select_candidate(candidate_metrics)
    if selected_half_life is None:
        summary = {"classification": "no_stable_train_adaptive_reference", "status": "not_frozen", "test_loaded": False}
        (out / "structure_v2a1c_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        status["final_classification_complete"] = True
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print("classification=no_stable_train_adaptive_reference", flush=True)
        return 2
    selected_row = candidate_metrics[candidate_metrics.half_life.eq(selected_half_life)].iloc[0]
    alpha = alpha_from_half_life(selected_half_life)
    train_threshold = float(selected_row.train_calibration_q99)
    config_core = {
        "selected_half_life": selected_half_life,
        "alpha": alpha,
        "clip_multiplier": CLIP_MULTIPLIER,
        "location_step_cap": LOCATION_STEP_CAP,
        "scale_ratio_cap": [SCALE_RATIO_LOW, SCALE_RATIO_HIGH],
        "scale_floor_ratio": SCALE_FLOOR_RATIO,
        "update_freeze_threshold": freeze_threshold,
        "per_edge_update_thresholds": edge_thresholds.tolist(),
        "train_calibration_q99_threshold": train_threshold,
        "train_selection_metrics": {key: (bool(value) if isinstance(value, np.bool_) else float(value) if isinstance(value, (np.floating, float)) else int(value) if isinstance(value, (np.integer, int)) else value) for key, value in selected_row.to_dict().items()},
        "status": "frozen_before_validation_evaluation",
    }
    config_core["config_hash"] = hashlib.sha256(json.dumps(config_core, sort_keys=True).encode()).hexdigest()
    (out / "selected_adaptive_reference.yaml").write_text(yaml.safe_dump(config_core, sort_keys=False), encoding="utf-8")
    status["candidate_selected"] = True
    status["candidate_frozen_before_validation"] = True
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if smoke:
        summary = {"smoke": True, "selected_half_life": selected_half_life, "train_candidates": candidate_rows}
        (out / "structure_v2a1c_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"SMOKE complete selected_half_life={selected_half_life}", flush=True)
        return 0

    print(f"FROZEN half_life={selected_half_life}; starting read-only validation evaluation", flush=True)
    started = time.perf_counter()
    validation_trace = run_adaptive_stream(
        validation["rolling_dependency_loss"], validation["sequence_id"], global_location, global_scale,
        target_ids, alpha, freeze_threshold, edge_thresholds, True,
    )
    elapsed = time.perf_counter() - started
    clean, clean_by_sequence, alarms = validation_metrics(validation_trace["grouped_score"], validation["sequence_id"], train_threshold)
    segments = alarm_segments_frame(validation["sequence_id"], validation["endpoint_row"], validation["timestamp"], alarms)
    clean["alarm_segment_count"] = int(len(segments))
    clean["longest_alarm_segment"] = int(segments.length.max()) if len(segments) else 0
    clean["reference_update_count"] = int(sum(np.asarray(x["update_count"]).sum() for x in validation_trace["diagnostics"]))
    clean["global_freeze_count"] = int(sum(x["global_freeze_count"] for x in validation_trace["diagnostics"]))
    clean["edge_freeze_count"] = int(sum(np.asarray(x["edge_freeze_count"]).sum() for x in validation_trace["diagnostics"]))
    pd.DataFrame([clean]).to_csv(out / "validation_adaptive_clean_metrics.csv", index=False)
    clean_by_sequence.to_csv(out / "validation_adaptive_metrics_by_sequence.csv", index=False)
    segments.to_csv(out / "validation_adaptive_alarm_segments.csv", index=False)
    drift_rows = []
    for item in validation_trace["diagnostics"]:
        for edge, relation in enumerate(pool.itertuples()):
            drift_rows.append(
                {
                    "sequence_id": item["sequence_id"], "edge_id": edge, "source": relation.source, "target": relation.target,
                    "initial_location": item["initial_location"][edge], "final_location": item["final_location"][edge],
                    "location_absolute_change": item["location_absolute_change"][edge], "initial_scale": item["initial_scale"][edge],
                    "final_scale": item["final_scale"][edge], "scale_ratio": item["scale_ratio"][edge],
                    "update_count": item["update_count"][edge], "edge_freeze_count": item["edge_freeze_count"][edge],
                    "max_floor_streak": item["max_floor_streak"][edge],
                }
            )
    pd.DataFrame(drift_rows).to_csv(out / "validation_reference_drift_by_edge.csv", index=False)
    np.savez_compressed(
        out / "validation_adaptive_reference_trace.npz",
        sequence_id=validation["sequence_id"], endpoint_row=validation["endpoint_row"], timestamp=validation["timestamp"],
        grouped_score=validation_trace["grouped_score"], edge_score=validation_trace["edge_score"],
        location_before=validation_trace["location_before"], scale_before=validation_trace["scale_before"],
        updated=validation_trace["updated"], global_frozen=validation_trace["global_frozen"], threshold=np.asarray(train_threshold),
    )
    status["validation_clean_complete"] = True
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    ids = source_window_ids(validation["sequence_id"], validation["endpoint_row"])
    lookup = {value: index for index, value in enumerate(ids)}
    source_positions = np.asarray([lookup[value] for value in controlled["source_window_id"].astype(str)], dtype=np.int64)
    controlled_edges, controlled_grouped = controlled_scores(controlled, source_positions, validation_trace, target_ids)
    clean_source = validation_trace["grouped_score"][source_positions]
    family_metrics, sequence_metrics = evaluate_controlled(controlled, controlled_grouped, clean_source, train_threshold)
    family_metrics.to_csv(out / "controlled_adaptive_metrics_by_family.csv", index=False)
    sequence_metrics.to_csv(out / "controlled_adaptive_metrics_by_sequence.csv", index=False)
    family_metrics.to_csv(out / "controlled_adaptive_metrics.csv", index=False)
    np.savez_compressed(
        out / "controlled_adaptive_scores.npz", family=controlled["family"], source_window_id=controlled["source_window_id"],
        sequence_id=controlled["sequence_id"], fold=controlled["fold"], target_edge_id=controlled["target_edge_id"],
        valid_mask=controlled["valid_mask"], clean_source_score=clean_source, edge_scores=controlled_edges,
        grouped_score=controlled_grouped, threshold=np.asarray(train_threshold), source_position=source_positions,
    )
    family_metrics[family_metrics.family.eq("P1")].to_csv(out / "p1_adaptive_boundary_metrics.csv", index=False)
    family_metrics[family_metrics.family.eq("P2")].to_csv(out / "p2_adaptive_mixed_diagnostics.csv", index=False)
    family_metrics[family_metrics.family.eq("C1")].to_csv(out / "c1_adaptive_collateral_diagnostics.csv", index=False)
    c2 = family_metrics[family_metrics.family.eq("C2")].copy()
    c2["fpr"] = c2.q99_tpr
    c2["clean_source_fpr"] = float(np.mean(clean_source > train_threshold))
    c2["excess_fpr"] = c2.fpr - c2.clean_source_fpr
    c2.to_csv(out / "c2_adaptive_preserving_metrics.csv", index=False)
    status["controlled_instant_evaluation_complete"] = True
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    absorption_trace, absorption_metrics = persistent_absorption(
        controlled, source_positions, validation_trace, target_ids, global_scale, alpha,
        freeze_threshold, edge_thresholds, train_threshold,
    )
    np.savez_compressed(out / "persistent_corruption_absorption_trace.npz", **absorption_trace)
    absorption_metrics.to_csv(out / "persistent_corruption_absorption_metrics.csv", index=False)
    absorption_pass = absorption_gates(absorption_metrics)
    status["persistent_absorption_test_complete"] = True

    breaking = family_metrics[family_metrics.family.isin(("R1", "R2", "R3"))]
    breaking_sequence = sequence_metrics[sequence_metrics.family.isin(("R1", "R2", "R3"))]
    p1_fpr = float(family_metrics.loc[family_metrics.family.eq("P1"), "q99_tpr"].iloc[0])
    c2_fpr = float(c2.fpr.iloc[0])
    c2_excess = float(c2.excess_fpr.iloc[0])
    gates = {
        "validation_overall_fpr": clean["overall_fpr"] <= 0.02,
        "validation_worst_sequence_fpr": clean["worst_sequence_fpr"] <= 0.05,
        "train1_second_half_reduction": clean["train1_second_half_fpr"] <= 0.0785694,
        "macro_auroc": breaking.auroc.mean() >= 0.82,
        "macro_aupr": breaking.aupr.mean() >= 0.84,
        "macro_q99_tpr": breaking.q99_tpr.mean() >= 0.40,
        "auroc_baseline_preservation": breaking.auroc.mean() >= 0.838228 - 0.02,
        "q99_tpr_baseline_preservation": breaking.q99_tpr.mean() >= 0.441204 - 0.04,
        "two_family_auroc": int((breaking.auroc >= 0.85).sum()) >= 2,
        "two_family_q99_tpr": int((breaking.q99_tpr >= 0.40).sum()) >= 2,
        "five_sequence_auroc": int((breaking_sequence.groupby("sequence_id").auroc.mean() >= 0.70).sum()) >= 5,
        "p1_fpr": p1_fpr <= 0.05,
        "c2_fpr": c2_fpr <= 0.03,
        "c2_excess_fpr": c2_excess <= 0.015,
        "persistent_absorption": all(absorption_pass.values()),
    }
    all_numeric_finite = all(
        np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
        for frame in (candidate_metrics, clean_by_sequence, family_metrics, sequence_metrics, absorption_metrics, pd.DataFrame(drift_rows))
    )
    after = historical_hashes()
    hash_rows = [{"artifact": key, "sha256_before": value, "sha256_after": after.get(key), "unchanged": after.get(key) == value} for key, value in before.items()]
    hash_audit = pd.DataFrame(hash_rows)
    hash_audit.to_csv(out / "artifact_reuse_audit.csv", index=False)
    gates["all_results_finite"] = all_numeric_finite
    gates["historical_hashes"] = bool(hash_audit.unchanged.all() and set(before) == set(after))
    gates["parameter_count"] = 36 < 10000
    all_passed = bool(all(gates.values()))
    drift_reduction = 1.0 - clean["train1_second_half_fpr"] / 0.112242
    if all_passed:
        classification = "causal_adaptive_dependency_reference_is_valid"
    elif clean["overall_fpr"] < 0.009078 and not all(absorption_pass.values()):
        classification = "adaptive_reference_absorbs_collaborative_anomalies"
    elif drift_reduction > 0 and (clean["worst_sequence_fpr"] < 0.056603 or clean["train1_second_half_fpr"] < 0.112242):
        classification = "adaptive_reference_reduces_drift_but_not_enough"
    else:
        classification = "adaptive_reference_does_not_reduce_drift"
    final_status = "frozen_for_hai_development_evaluation" if classification == "causal_adaptive_dependency_reference_is_valid" else "not_frozen"

    static_sequence = pd.read_csv(A1S / "full_clean_fpr_by_sequence.csv")
    comparison_rows = [
        {"scope": "clean", "metric": "overall_fpr", "static": 0.009118, "adaptive": clean["overall_fpr"]},
        {"scope": "clean", "metric": "worst_sequence_fpr", "static": 0.054300, "adaptive": clean["worst_sequence_fpr"]},
        {"scope": "clean", "metric": "train1_second_half_fpr", "static": 0.107529, "adaptive": clean["train1_second_half_fpr"]},
        {"scope": "controlled", "metric": "macro_auroc", "static": 0.838228, "adaptive": float(breaking.auroc.mean())},
        {"scope": "controlled", "metric": "macro_aupr", "static": 0.866872, "adaptive": float(breaking.aupr.mean())},
        {"scope": "controlled", "metric": "macro_q99_tpr", "static": 0.441204, "adaptive": float(breaking.q99_tpr.mean())},
        {"scope": "controlled", "metric": "p1_fpr", "static": 0.016667, "adaptive": p1_fpr},
        {"scope": "controlled", "metric": "c2_fpr", "static": 0.0, "adaptive": c2_fpr},
    ]
    del static_sequence
    pd.DataFrame(comparison_rows).assign(delta=lambda x: x.adaptive - x.static).to_csv(out / "adaptive_vs_static_comparison.csv", index=False)
    state_count = len(pool) * 3
    pd.DataFrame([{
        "adaptive_state_scalars": state_count, "state_scalars_per_edge": 3, "total_state_count": state_count,
        "serialized_trace_bytes": (out / "validation_adaptive_reference_trace.npz").stat().st_size,
        "seconds_per_window_update_and_score": elapsed / len(validation["sequence_id"]),
        "static_state_scalars": len(pool) * 2, "incremental_state_scalars": len(pool), "below_10000_parameters": state_count < 10000,
    }]).to_csv(out / "lightweight_cost_audit.csv", index=False)
    selected_config = {
        "status": final_status, "classification": classification, "selected_half_life": selected_half_life,
        "aggregation": "target_group_max_then_top2_mean", "relation_count": 12, "hai_test_used": False,
        "attack_label_used": False, "ridge_loaded": False, "ridge_refit": False,
    }
    (out / "selected_structure_v2a1c_config.yaml").write_text(yaml.safe_dump(selected_config, sort_keys=False), encoding="utf-8")
    summary = {
        "classification": classification, "status": final_status, "all_gates_passed": all_passed, "gates": gates,
        "absorption_gates": absorption_pass, "selected_half_life": selected_half_life, "alpha": alpha,
        "clean": clean, "controlled_macro_auroc": float(breaking.auroc.mean()),
        "controlled_macro_aupr": float(breaking.aupr.mean()), "controlled_macro_q99_tpr": float(breaking.q99_tpr.mean()),
        "p1_fpr": p1_fpr, "c2_fpr": c2_fpr, "c2_excess_fpr": c2_excess,
        "test_loaded": False, "attack_label_used": False, "ae_loaded": False, "rec_loaded": False,
        "ridge_loaded": False, "ridge_refit": False,
    }
    (out / "structure_v2a1c_summary.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    unresolved = {
        "adaptive_reduced_drift": drift_reduction > 0, "train1_second_half_relative_reduction": drift_reduction,
        "persistent_anomaly_absorbed": not all(absorption_pass.values()),
        "largest_edge_updates": pd.DataFrame(drift_rows).groupby("edge_id").update_count.sum().sort_values(ascending=False).head(5).index.tolist(),
        "sequence_specific_instability_remains": clean["worst_sequence_fpr"] > 0.05,
        "allow_hai_evaluation": classification == "causal_adaptive_dependency_reference_is_valid",
        "final_structure_branch_limit": None if all_passed else "controlled collaborative mechanism remains detectable, but normal non-stationarity limits practical stability",
        "automatic_next_experiment_started": False,
    }
    (out / "unresolved_issues_delta.json").write_text(json.dumps(unresolved, indent=2), encoding="utf-8")
    report = f"""# HAI Structure V2-A1C causal adaptive reference

## Protocol

- Offline reuse of A1M train, validation, and controlled raw dependency caches only.
- Frozen 12-edge pool, rolling length 16, and target-group max then top2 mean aggregation.
- Train-only candidates: half-life 512 and 2048. Selected: {selected_half_life}.
- Validation uses the frozen train q99 threshold; no validation tuning or sequence-specific threshold.

## Result

- Classification: `{classification}`
- Status: `{final_status}`
- Validation overall FPR: {clean['overall_fpr']:.6f}
- Worst-sequence FPR: {clean['worst_sequence_fpr']:.6f}
- train1 second-half FPR: {clean['train1_second_half_fpr']:.6f}
- Controlled macro AUROC/AUPR/q99 TPR: {breaking.auroc.mean():.6f} / {breaking.aupr.mean():.6f} / {breaking.q99_tpr.mean():.6f}
- Persistent absorption gates passed: {all(absorption_pass.values())}
- All formal gates passed: {all_passed}

## Boundary

No HAI test, attack label, AE, checkpoint, reconstruction score, rec score, Ridge refit, relation screening, corruption regeneration, fusion, or automatic next structure experiment was used.
"""
    (out / "final_hai_structure_v2a1c_causal_adaptive_reference.md").write_text(report, encoding="utf-8")
    status.update({
        "final_classification_complete": True, "final_report_complete": True,
        "candidate_selected": True, "candidate_frozen_before_validation": True,
    })
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"classification={classification} status={final_status} all_gates_passed={all_passed}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description="HAI Structure V2-A1C causal adaptive dependency reference")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        return write_dry_run()
    return run(args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
