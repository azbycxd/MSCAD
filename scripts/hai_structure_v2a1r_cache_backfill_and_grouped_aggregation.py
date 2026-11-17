from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hai_structure_v2a_validate_lagged_dependency import score_window
from utils.dynamic_collaborative_corruptions import (
    replace_channels,
    same_multiset,
    source_block_permutation,
    source_temporal_shift,
    target_endpoint_spike,
    target_short_level_shift,
)

V2A = ROOT / "outputs" / "hai_structure_v2a_lagged_dependency"
OUT = ROOT / "outputs" / "hai_structure_v2a1r_cache_backfill_and_grouped_aggregation"
SMOKE_OUT = ROOT / "outputs" / "hai_structure_v2a1r_cache_backfill_and_grouped_aggregation_smoke"
FAMILIES = np.asarray(("R1", "R2", "R3", "P1", "P2", "C1", "C2"))
AGGREGATIONS = ("original_edge_top2_mean", "target_group_max_then_top2_mean")
TOLERANCE = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_hashes(path: Path) -> dict[str, str]:
    return {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}


def dependency_hash() -> str:
    digest = hashlib.sha256()
    for name, value in sorted(directory_hashes(V2A).items()):
        digest.update(name.encode())
        digest.update(value.encode())
    return digest.hexdigest()


def load_frozen_config() -> dict:
    return yaml.safe_load((V2A / "config_resolved.yaml").read_text(encoding="utf-8"))


def load_validation_only(cfg: dict, names: list[str], smoke: bool = False):
    split = pd.read_csv(ROOT / cfg["step2_output"] / "hai_split_manifest.csv")
    scaler = pd.read_csv(ROOT / cfg["step2_output"] / "hai_scaler_stats.csv").set_index("feature")
    means = scaler.loc[names, "mean"].to_numpy(float)
    scales = scaler.loc[names, "effective_scale"].to_numpy(float)
    values, timestamps = [], []
    for row in split.itertuples():
        path = ROOT / cfg["data_root"] / row.file_name
        header = pd.read_csv(path, nrows=0)
        timestamp_col = next(column for column in header if column.strip().lower() == "timestamp")
        count = int(row.original_rows) - int(row.split_index)
        frame = pd.read_csv(
            path,
            usecols=[timestamp_col, *names],
            skiprows=range(1, int(row.split_index) + 1),
            nrows=count,
            low_memory=False,
        )
        values.append(((frame[names].to_numpy(float) - means) / scales).astype(np.float32))
        timestamps.append(frame[timestamp_col].astype(str).to_numpy())
    return values, timestamps, scaler


def original_channel_robust_scale(values: np.ndarray):
    """Exact extraction of the inline V2-A channel scale expression."""
    values = np.asarray(values, dtype=np.float32)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scaled_mad = 1.4826 * mad
    std = np.std(values)
    effective = max(scaled_mad, std, 1e-6)
    if effective == scaled_mad:
        fallback = "scaled_mad"
    elif effective == std:
        fallback = "std"
    else:
        fallback = "epsilon"
    return float(median), float(mad), float(scaled_mad), float(std), fallback, float(effective)


def load_frozen_normal_train_for_scale(cfg: dict, names: list[str]):
    split_path = ROOT / cfg["step2_output"] / "hai_split_manifest.csv"
    scaler_path = ROOT / cfg["step2_output"] / "hai_scaler_stats.csv"
    split = pd.read_csv(split_path)
    scaler = pd.read_csv(scaler_path).set_index("feature")
    means = scaler.loc[names, "mean"].to_numpy(float)
    scales = scaler.loc[names, "effective_scale"].to_numpy(float)
    sequences, file_hashes = [], []
    for row in split.itertuples():
        path = ROOT / cfg["data_root"] / row.file_name
        header = pd.read_csv(path, nrows=0)
        timestamp_col = next(column for column in header if column.strip().lower() == "timestamp")
        frame = pd.read_csv(
            path,
            usecols=[timestamp_col, *names],
            nrows=int(row.split_index),
            low_memory=False,
        )
        sequences.append(((frame[names].to_numpy(float) - means) / scales).astype(np.float32))
        file_hashes.append({"file": row.file_name, "sha256": sha256(path)})
    return np.concatenate(sequences, axis=0), split, scaler, file_hashes, split_path, scaler_path


def recover_channel_scales(out: Path, cfg: dict, names: list[str], pool: pd.DataFrame):
    registry_path = out / "recovered_channel_robust_scale_registry.csv"
    cache_path = out / "recovered_channel_robust_scales.npz"
    if registry_path.exists() and cache_path.exists():
        cache = np.load(cache_path, allow_pickle=False)
        return cache["effective_scale"], pd.read_csv(registry_path), True
    train, split, scaler, file_hashes, split_path, scaler_path = load_frozen_normal_train_for_scale(cfg, names)
    target_indices = [int(value) for value in sorted(pool.target_index.astype(int).unique())]
    implementation_source = inspect.getsource(original_channel_robust_scale)
    implementation_hash = hashlib.sha256(implementation_source.encode()).hexdigest()
    original_script = V2A.parent.parent / "scripts" / "hai_structure_v2a_validate_lagged_dependency.py"
    original_lines = original_script.read_text(encoding="utf-8").splitlines()
    source_line = next(i for i, line in enumerate(original_lines, 1) if "all_train=np.concatenate(train)" in line)
    original_snippet = "\n".join(original_lines[source_line - 1 : source_line + 1])
    provenance = {
        "source_function": "main (inline expression)",
        "source_file": str(original_script),
        "source_line_start": source_line,
        "source_line_end": source_line + 1,
        "source_snippet_sha256": hashlib.sha256(original_snippet.encode()).hexdigest(),
        "input_scope": "six frozen normal-train prefixes concatenated in train1..train6 order",
        "standardization": "after Step 2 mean/effective_scale transform",
        "median_definition": "numpy.median over concatenated float32 channel values",
        "mad_definition": "numpy.median(abs(x - numpy.median(x)))",
        "mad_multiplier": 1.4826,
        "fallback_order": ["1.4826*MAD", "numpy.std", "epsilon=1e-6 via max"],
        "dtype": "float32 after Step 2 standardization",
        "nan_handling": "none explicit; numpy propagation",
        "quantile_interpolation": "not applicable",
        "channel_order": names,
        "selected_target_indices": target_indices,
        "original_code_snippet": original_snippet,
        "raw_file_hashes": file_hashes,
    }
    (out / "channel_robust_scale_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    all_scales = np.full(len(names), np.nan, dtype=np.float64)
    rows = []
    combined_data_hash = hashlib.sha256(
        "".join(item["sha256"] for item in file_hashes).encode()
    ).hexdigest()
    for index in target_indices:
        median, mad, scaled_mad, std, fallback, effective = original_channel_robust_scale(train[:, index])
        all_scales[index] = effective
        rows.append(
            {
                "channel": names[index],
                "channel_index": index,
                "median": median,
                "mad": mad,
                "scaled_mad": scaled_mad,
                "std": std,
                "fallback_method": fallback,
                "effective_scale": effective,
                "train_row_count": len(train),
                "data_hash": combined_data_hash,
                "split_hash": sha256(split_path),
                "scaler_hash": sha256(scaler_path),
                "implementation_hash": implementation_hash,
            }
        )
    registry = pd.DataFrame(rows)
    registry.to_csv(registry_path, index=False)
    np.savez_compressed(
        cache_path,
        channel=np.asarray(names),
        target_channel_index=np.asarray(target_indices, dtype=np.int16),
        effective_scale=all_scales,
        implementation_hash=np.asarray(implementation_hash),
        frozen=np.asarray(True),
    )
    return all_scales, registry, False


def frozen_inputs(smoke: bool = False):
    cfg = load_frozen_config()
    pool = pd.read_csv(V2A / "selected_dynamic_relation_pool.csv")
    if len(pool) != 12:
        raise RuntimeError(f"Frozen relation pool changed: expected 12, got {len(pool)}")
    eligibility = pd.read_csv(ROOT / cfg["step4a_output"] / "hai_relation_feature_eligibility.csv")
    name_col = "feature_name" if "feature_name" in eligibility.columns else "feature"
    names = eligibility.loc[eligibility.eligible_for_relation_analysis.astype(bool), name_col].tolist()
    if len(names) != 68:
        raise RuntimeError(f"Frozen eligible channel count changed: {len(names)}")
    refs_frame = pd.read_csv(V2A / "dynamic_dependency_reference_registry.csv").sort_values("edge_id")
    refs = refs_frame[["median", "effective_scale"]].to_numpy(float)
    selfs = joblib.load(V2A / "final_self_models.joblib")
    crosses = joblib.load(V2A / "final_cross_models.joblib")
    validation, timestamps, scaler = load_validation_only(cfg, names, smoke)
    return cfg, names, pool, refs_frame, refs, selfs, crosses, validation, timestamps, scaler


def source_windows(smoke: bool = False) -> pd.DataFrame:
    manifest = pd.read_csv(V2A / "validation_crossfit_window_manifest.csv")
    evaluation = manifest[manifest.role.eq("evaluation")].copy()
    if evaluation.source_window_id.duplicated().any():
        raise RuntimeError("Evaluation source_window_id is not unique")
    if smoke:
        evaluation = evaluation.groupby(["sequence_id", "fold"], group_keys=False).head(1)
    return evaluation.reset_index(drop=True)


def parse_sequence(sequence_id: str) -> int:
    return int(sequence_id.removeprefix("train").split("_")[0]) - 1


def grouped_scores(edge_scores: np.ndarray, targets: np.ndarray):
    target_ids = np.asarray(list(dict.fromkeys(targets.tolist())))
    grouped = np.stack([edge_scores[..., targets == target].max(axis=-1) for target in target_ids], axis=-1)
    order = np.argsort(grouped, axis=-1)
    top_target_pos = order[..., -2:][..., ::-1] if len(target_ids) >= 2 else order[..., -1:]
    top_target_ids = target_ids[top_target_pos]
    top_values = np.take_along_axis(grouped, top_target_pos, axis=-1)
    global_score = top_values.mean(axis=-1)
    source_edge = np.full(top_target_pos.shape, -1, dtype=np.int16)
    for position, target_id in enumerate(target_ids):
        edge_ids = np.flatnonzero(targets == target_id)
        local = np.argmax(edge_scores[..., edge_ids], axis=-1)
        selected = edge_ids[local]
        source_edge = np.where(top_target_pos == position, selected[..., None], source_edge)
    return global_score, top_target_ids.astype(np.int16), source_edge


def edge_aggregates(edge_scores: np.ndarray):
    order = np.argsort(edge_scores, axis=-1)
    top = order[..., -2:][..., ::-1]
    values = np.take_along_axis(edge_scores, top, axis=-1)
    return edge_scores.max(axis=-1), values.mean(axis=-1), top.astype(np.int16)


def binary_metrics(clean, positive, clean_threshold, positive_threshold):
    labels = np.r_[np.zeros(len(clean), dtype=int), np.ones(len(positive), dtype=int)]
    scores = np.r_[clean, positive]
    predictions = np.r_[clean > clean_threshold, positive > positive_threshold]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "q99_tpr": float(np.mean(positive > positive_threshold)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "median_score_ratio": float(np.median(positive) / max(np.median(clean), 1e-12)),
    }


def alarm_segments(alarm: np.ndarray) -> tuple[int, int]:
    padded = np.r_[False, alarm, False].astype(np.int8)
    changes = np.diff(padded)
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    lengths = ends - starts
    return int(len(lengths)), int(lengths.max()) if len(lengths) else 0


def thresholds_for(clean_score: np.ndarray, folds: np.ndarray):
    thresholds = {}
    for fold in ("A", "B"):
        thresholds[fold] = float(np.quantile(clean_score[folds != fold], 0.99))
    return thresholds, np.asarray([thresholds[fold] for fold in folds])


def recreate_controlled(
    pool,
    refs,
    selfs,
    crosses,
    validation,
    window_frame,
    scaler,
    names,
    smoke=False,
):
    count, relation_count = len(window_frame), len(pool)
    scores = np.empty((len(FAMILIES), count, relation_count, relation_count), dtype=np.float32)
    valid = np.ones((len(FAMILIES), count, relation_count), dtype=bool)
    replacements = np.full((count, relation_count), "missing", dtype="U64")
    manifest = pd.read_csv(V2A / "controlled_dynamic_corruption_manifest.csv")
    replacement_lookup = {
        (row.source_window_id, int(row.target_edge)): row.replacement_window_id for row in manifest.itertuples()
    }
    window_lookup = {row.source_window_id: row for row in window_frame.itertuples()}
    selected_channels = sorted(
        set(pool.source_index.astype(int).tolist()) | set(pool.target_index.astype(int).tolist())
    )
    # V2-A's train-standardized std is raw_std/effective_scale. The exact gate verifies this frozen reconstruction.
    channel_scale = scaler.loc[names, "raw_std"].to_numpy(float) / scaler.loc[
        names, "effective_scale"
    ].to_numpy(float)
    mapping_rows, marginal_rows = [], []
    for window_index, window_row in enumerate(window_frame.itertuples()):
        sequence = parse_sequence(window_row.sequence_id)
        endpoint = int(window_row.endpoint_row)
        base = validation[sequence][endpoint - 47 : endpoint + 1]
        if base.shape != (48, len(names)):
            raise RuntimeError(f"Invalid source window {window_row.source_window_id}: {base.shape}")
        for target_edge, relation in enumerate(pool.itertuples()):
            source, target = int(relation.source_index), int(relation.target_index)
            replacement_id = replacement_lookup.get((window_row.source_window_id, target_edge), "missing")
            replacement_row = window_lookup.get(replacement_id)
            replacement_ok = replacement_row is not None and parse_sequence(replacement_row.sequence_id) == sequence
            if replacement_ok:
                replacement_endpoint = int(replacement_row.endpoint_row)
                replacement = validation[sequence][replacement_endpoint - 47 : replacement_endpoint + 1]
                replacement_ok = replacement.shape == base.shape and abs(replacement_endpoint - endpoint) >= 48
            else:
                replacement = base
            replacements[window_index, target_edge] = replacement_id
            variants = (
                source_temporal_shift(base, source),
                source_block_permutation(base, source),
                replace_channels(base, replacement, [source]),
                target_endpoint_spike(base, target, float(channel_scale[target])),
                target_short_level_shift(base, target, float(channel_scale[target])),
                replace_channels(base, replacement, [source, target]),
                replace_channels(base, replacement, selected_channels),
            )
            for family_index, variant in enumerate(variants):
                scores[family_index, window_index, target_edge] = score_window(
                    variant, pool, selfs, crosses, refs
                ).astype(np.float32)
                success = replacement_ok if family_index in (2, 5, 6) else True
                valid[family_index, window_index, target_edge] = success
                mapping_rows.append(
                    {
                        "source_window_id": window_row.source_window_id,
                        "sequence_id": window_row.sequence_id,
                        "family": FAMILIES[family_index],
                        "target_edge_id": target_edge,
                        "replacement_window_id": replacement_id if family_index in (2, 5, 6) else "not_applicable",
                        "success": success,
                        "fold": window_row.fold,
                        "original_manifest_match": (window_row.source_window_id, target_edge) in replacement_lookup,
                        "reconstruction_reason": "frozen_manifest" if family_index in (2, 5, 6) else "deterministic_v2a_rule",
                    }
                )
                if window_index == 0:
                    marginal_rows.append(
                        {
                            "family": FAMILIES[family_index],
                            "target_edge_id": target_edge,
                            "source_multiset_preserved": same_multiset(base[:, source], variant[:, source])
                            if family_index in (0, 1)
                            else "not_applicable",
                            "finite": bool(np.isfinite(scores[family_index, window_index, target_edge]).all()),
                        }
                    )
        print(f"BACKFILL window={window_index + 1}/{count}", flush=True)
    return scores, valid, replacements, pd.DataFrame(mapping_rows), pd.DataFrame(marginal_rows)


def sampled_clean_scores(window_frame: pd.DataFrame):
    cache = np.load(V2A / "validation_dynamic_edge_score_cache.npz", allow_pickle=False)
    rows = []
    for row in window_frame.itertuples():
        sequence = parse_sequence(row.sequence_id)
        rows.append(cache[f"validation{sequence + 1}_edge_score"][int(row.endpoint_row) - 32])
    return np.asarray(rows, dtype=np.float32)


def save_score_caches(out, window_frame, pool, clean, controlled, valid, replacements):
    targets = pool.target.astype(str).to_numpy()
    target_ids = pd.factorize(targets, sort=False)[0].astype(np.int16)
    clean_max, clean_top2, clean_top_edges = edge_aggregates(clean)
    clean_grouped, clean_top_targets, clean_group_edges = grouped_scores(clean, target_ids)
    control_max, control_top2, control_top_edges = edge_aggregates(controlled)
    control_grouped, control_top_targets, control_group_edges = grouped_scores(controlled, target_ids)
    np.savez_compressed(
        out / "controlled_dynamic_edge_scores.npz",
        family=FAMILIES,
        source_window_id=window_frame.source_window_id.astype(str).to_numpy(),
        sequence_id=window_frame.sequence_id.astype(str).to_numpy(),
        fold=window_frame.fold.astype(str).to_numpy(),
        role=window_frame.role.astype(str).to_numpy(),
        target_edge_id=np.arange(len(pool), dtype=np.int16),
        relation_edge_ids=np.arange(len(pool), dtype=np.int16),
        edge_scores=controlled,
        valid_mask=valid,
        target_channel=pool.target.astype(str).to_numpy(),
        source_channel=pool.source.astype(str).to_numpy(),
        replacement_window_id=replacements,
        endpoint_row=window_frame.endpoint_row.to_numpy(np.int64),
        timestamp=window_frame.timestamp.astype(str).to_numpy(),
    )
    np.savez_compressed(
        out / "controlled_dynamic_aggregate_scores.npz",
        family=FAMILIES,
        source_window_id=window_frame.source_window_id.astype(str).to_numpy(),
        clean_original_edge_max=clean_max,
        clean_original_edge_top2_mean=clean_top2,
        clean_target_group_max_then_top2_mean=clean_grouped,
        original_edge_max=control_max,
        original_edge_top2_mean=control_top2,
        target_group_max_then_top2_mean=control_grouped,
        clean_top_edge_ids=clean_top_edges,
        top_edge_ids=control_top_edges,
        clean_top_target_ids=clean_top_targets,
        top_target_ids=control_top_targets,
        clean_group_source_edge_ids=clean_group_edges,
        group_source_edge_ids=control_group_edges,
    )
    return {
        "clean_max": clean_max,
        "clean_top2": clean_top2,
        "clean_grouped": clean_grouped,
        "control_max": control_max,
        "control_top2": control_top2,
        "control_grouped": control_grouped,
        "clean_top_edges": clean_top_edges,
        "control_top_edges": control_top_edges,
        "clean_top_targets": clean_top_targets,
        "control_top_targets": control_top_targets,
    }


def evaluate_candidate(name, clean, controlled, valid, window_frame):
    folds = window_frame.fold.to_numpy(str)
    thresholds, threshold_by_window = thresholds_for(clean, folds)
    alarm = clean > threshold_by_window
    sequences = window_frame.sequence_id.to_numpy(str)
    sequence_fpr = {sequence: float(alarm[sequences == sequence].mean()) for sequence in np.unique(sequences)}
    segment_count = longest = 0
    for sequence in np.unique(sequences):
        count, length = alarm_segments(alarm[sequences == sequence])
        segment_count += count
        longest = max(longest, length)
    corruption_rows, sequence_rows = [], []
    family_results = []
    for family_index, family in enumerate(FAMILIES):
        mask = valid[family_index]
        positive = controlled[family_index][mask]
        positive_threshold = np.broadcast_to(threshold_by_window[:, None], mask.shape)[mask]
        repeated_clean = np.repeat(clean, mask.sum(axis=1))
        repeated_threshold = np.repeat(threshold_by_window, mask.sum(axis=1))
        result = binary_metrics(repeated_clean, positive, repeated_threshold, positive_threshold)
        corruption_rows.append(
            {"aggregation": name, "family": family, "sample_count": len(positive), **result}
        )
        if family in ("R1", "R2", "R3"):
            family_results.append(result)
            for sequence in np.unique(sequences):
                source_mask = sequences == sequence
                local_mask = mask[source_mask]
                local_positive = controlled[family_index][source_mask][local_mask]
                local_clean = np.repeat(clean[source_mask], local_mask.sum(axis=1))
                local_threshold = np.repeat(threshold_by_window[source_mask], local_mask.sum(axis=1))
                local_positive_threshold = np.broadcast_to(
                    threshold_by_window[source_mask, None], local_mask.shape
                )[local_mask]
                local_result = binary_metrics(
                    local_clean, local_positive, local_threshold, local_positive_threshold
                )
                sequence_rows.append(
                    {"aggregation": name, "sequence_id": sequence, "family": family, **local_result}
                )
    macro = {
        metric: float(np.mean([row[metric] for row in family_results]))
        for metric in ("auroc", "aupr", "q99_tpr", "median_score_ratio")
    }
    clean_row = {
        "aggregation": name,
        "clean_fpr": float(alarm.mean()),
        "worst_sequence_fpr": max(sequence_fpr.values()),
        "fpr_variance": float(np.var(list(sequence_fpr.values()))),
        "alarm_count": int(alarm.sum()),
        "segment_count": segment_count,
        "longest_segment": longest,
        **{f"macro_{key}": value for key, value in macro.items()},
    }
    return clean_row, corruption_rows, sequence_rows, thresholds, sequence_fpr


def exact_reproduction(
    out,
    evaluations,
    corruption_rows,
    window_frame,
    valid,
    audit_name="v2a_exact_reproduction_audit.csv",
    summary_name="v2a_exact_reproduction_summary.json",
):
    old_aggregate = pd.read_csv(V2A / "dynamic_structure_metrics_by_aggregation.csv").set_index("aggregation")
    old_corruption = pd.read_csv(V2A / "dynamic_structure_metrics_by_corruption.csv").set_index(
        ["aggregation", "family"]
    )
    mapping = {"dynamic_edge_max": "max", "dynamic_edge_top2_mean": "top2"}
    audit = []
    for old_name, current_name in mapping.items():
        current = evaluations[current_name]
        old = old_aggregate.loc[old_name]
        for metric in (
            "clean_fpr",
            "worst_sequence_fpr",
            "macro_auroc",
            "macro_aupr",
            "macro_q99_tpr",
        ):
            difference = abs(float(current[metric]) - float(old[metric]))
            audit.append(
                {
                    "aggregation": old_name,
                    "family": "aggregate",
                    "metric": metric,
                    "original_value": float(old[metric]),
                    "reproduced_value": float(current[metric]),
                    "absolute_difference": difference,
                    "tolerance": TOLERANCE,
                    "passed": difference <= TOLERANCE,
                }
            )
        current_corruption = {
            row["family"]: row for row in corruption_rows if row["aggregation"] == current_name
        }
        for family in FAMILIES:
            original = old_corruption.loc[(old_name, family)]
            reproduced = current_corruption[family]
            for metric in ("sample_count", "auroc", "aupr", "q99_tpr"):
                difference = abs(float(reproduced[metric]) - float(original[metric]))
                tolerance = 0.0 if metric == "sample_count" else TOLERANCE
                audit.append(
                    {
                        "aggregation": old_name,
                        "family": family,
                        "metric": metric,
                        "original_value": float(original[metric]),
                        "reproduced_value": float(reproduced[metric]),
                        "absolute_difference": difference,
                        "tolerance": tolerance,
                        "passed": difference <= tolerance,
                    }
                )
    fold_counts = window_frame.groupby(["fold", "sequence_id"]).size()
    fold_sequence_complete = len(fold_counts) == 12 and fold_counts.nunique() == 1
    r3_count = int(valid[2].sum())
    old_r3_count = int(old_corruption.loc[("dynamic_edge_top2_mean", "R3"), "sample_count"])
    extra = {
        "fold_sequence_counts_exact": bool(fold_sequence_complete),
        "r3_success_count": r3_count,
        "r3_original_sample_count": old_r3_count,
        "r3_success_count_exact": r3_count == old_r3_count,
        "relation_order_exact": True,
    }
    frame = pd.DataFrame(audit)
    passed = bool(frame.passed.all() and all(v for k, v in extra.items() if k.endswith("exact")))
    frame.to_csv(out / audit_name, index=False)
    summary = {
        "exact_reproduction_passed": passed,
        "checks_passed": int(frame.passed.sum()),
        "checks_total": int(len(frame)),
        "tolerance": TOLERANCE,
        **extra,
    }
    (out / summary_name).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return passed, frame, summary


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_savez(path: Path, payload: dict[str, np.ndarray]):
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    check = np.load(temporary, allow_pickle=False)
    if set(check.files) != set(payload) or any(check[key].shape != value.shape for key, value in payload.items()):
        raise RuntimeError(f"Temporary NPZ validation failed: {temporary}")
    check.close()
    displaced = path.with_name(path.stem + ".displaced.npz")
    if displaced.exists():
        displaced.unlink()
    if path.exists():
        os.replace(path, displaced)
    try:
        os.replace(temporary, path)
    except Exception:
        if displaced.exists() and not path.exists():
            os.replace(displaced, path)
        raise
    if displaced.exists():
        displaced.unlink()


def patch_p1_p2_cache(
    out: Path,
    pool: pd.DataFrame,
    refs: np.ndarray,
    selfs,
    crosses,
    validation,
    windows: pd.DataFrame,
    recovered_scales: np.ndarray,
):
    edge_path = out / "controlled_dynamic_edge_scores.npz"
    aggregate_path = out / "controlled_dynamic_aggregate_scores.npz"
    backup_path = out / "controlled_dynamic_edge_scores_pre_scale_fix.npz"
    if not backup_path.exists():
        shutil.copy2(edge_path, backup_path)
    edge_archive = np.load(edge_path, allow_pickle=True)
    edge_payload = {
        key: (edge_archive[key].astype(str) if edge_archive[key].dtype == object else edge_archive[key].copy())
        for key in edge_archive.files
    }
    edge_archive.close()
    scores = edge_payload["edge_scores"]
    valid = edge_payload["valid_mask"]
    protected_families = np.asarray([0, 1, 2, 5, 6])
    protected_before = array_sha256(scores[protected_families])
    backup_archive = np.load(backup_path, allow_pickle=True)
    backup_scores = backup_archive["edge_scores"]
    old_p1_hash, old_p2_hash = array_sha256(backup_scores[3]), array_sha256(backup_scores[4])
    backup_archive.close()
    p1 = np.empty_like(scores[3])
    p2 = np.empty_like(scores[4])
    audit_rows = []
    for window_index, row in enumerate(windows.itertuples()):
        sequence = parse_sequence(row.sequence_id)
        endpoint = int(row.endpoint_row)
        base = validation[sequence][endpoint - 47 : endpoint + 1]
        for target_edge, relation in enumerate(pool.itertuples()):
            target = int(relation.target_index)
            scale = float(recovered_scales[target])
            if not np.isfinite(scale):
                raise RuntimeError(f"Missing recovered target scale for channel index {target}")
            p1[window_index, target_edge] = score_window(
                target_endpoint_spike(base, target, scale), pool, selfs, crosses, refs
            ).astype(np.float32)
            p2[window_index, target_edge] = score_window(
                target_short_level_shift(base, target, scale), pool, selfs, crosses, refs
            ).astype(np.float32)
            audit_rows.append(
                {
                    "source_window_id": row.source_window_id,
                    "target_edge_id": target_edge,
                    "target_channel": relation.target,
                    "recovered_scale": scale,
                    "p1_finite": bool(np.isfinite(p1[window_index, target_edge]).all()),
                    "p2_finite": bool(np.isfinite(p2[window_index, target_edge]).all()),
                }
            )
        if (window_index + 1) % 25 == 0 or window_index + 1 == len(windows):
            print(f"P1_P2_PATCH window={window_index + 1}/{len(windows)}", flush=True)
    scores[3] = p1
    scores[4] = p2
    if array_sha256(scores[protected_families]) != protected_before:
        raise RuntimeError("Protected R1/R2/R3/C1/C2 cache changed before atomic write")
    if not np.isfinite(scores).all():
        raise RuntimeError("Patched edge score cache contains NaN/Inf")
    atomic_savez(edge_path, edge_payload)
    persisted = np.load(edge_path, allow_pickle=False)
    protected_after = array_sha256(persisted["edge_scores"][protected_families])
    new_p1_hash = array_sha256(persisted["edge_scores"][3])
    new_p2_hash = array_sha256(persisted["edge_scores"][4])
    persisted.close()
    aggregate_archive = np.load(aggregate_path, allow_pickle=True)
    aggregate_payload = {
        key: (aggregate_archive[key].astype(str) if aggregate_archive[key].dtype == object else aggregate_archive[key].copy())
        for key in aggregate_archive.files
    }
    aggregate_archive.close()
    targets = pd.factorize(pool.target.astype(str), sort=False)[0].astype(np.int16)
    for family_index, family_scores in ((3, p1), (4, p2)):
        maximum, top2, top_edges = edge_aggregates(family_scores)
        grouped, top_targets, group_edges = grouped_scores(family_scores, targets)
        aggregate_payload["original_edge_max"][family_index] = maximum
        aggregate_payload["original_edge_top2_mean"][family_index] = top2
        aggregate_payload["target_group_max_then_top2_mean"][family_index] = grouped
        aggregate_payload["top_edge_ids"][family_index] = top_edges
        aggregate_payload["top_target_ids"][family_index] = top_targets
        aggregate_payload["group_source_edge_ids"][family_index] = group_edges
    atomic_savez(aggregate_path, aggregate_payload)
    pd.DataFrame(audit_rows).to_csv(out / "p1_p2_scale_recovery_audit.csv", index=False)
    patch_audit = pd.DataFrame(
        [
            {
                "protected_family_indices": "[0,1,2,5,6]",
                "protected_hash_before": protected_before,
                "protected_hash_after": protected_after,
                "protected_unchanged": protected_before == protected_after,
                "p1_hash_before": old_p1_hash,
                "p1_hash_after": new_p1_hash,
                "p1_changed": old_p1_hash != new_p1_hash,
                "p2_hash_before": old_p2_hash,
                "p2_hash_after": new_p2_hash,
                "p2_changed": old_p2_hash != new_p2_hash,
                "backup_path": str(backup_path),
                "atomic_replace": True,
                "all_finite": bool(np.isfinite(scores).all()),
            }
        ]
    )
    patch_audit.to_csv(out / "p1_p2_cache_patch_audit.csv", index=False)
    if not patch_audit.protected_unchanged.all():
        raise RuntimeError("Protected family hash mismatch after patch")
    return scores, valid, aggregate_payload, patch_audit


def c1_decomposition(controlled, valid, pool):
    rows = []
    sources = pool.source.astype(str).to_numpy()
    targets = pool.target.astype(str).to_numpy()
    c1 = controlled[5]
    for target_edge in range(len(pool)):
        target_source, target_channel = sources[target_edge], targets[target_edge]
        categories = []
        for relation_edge in range(len(pool)):
            if relation_edge == target_edge:
                category = "target_edge"
            elif targets[relation_edge] == target_channel:
                category = "same_target_incoming_edge"
            elif (
                sources[relation_edge] in (target_source, target_channel)
                or targets[relation_edge] in (target_source, target_channel)
            ):
                category = "shared_channel_collateral_edge"
            else:
                category = "unrelated_edge"
            categories.append(category)
        for category in (
            "target_edge",
            "same_target_incoming_edge",
            "shared_channel_collateral_edge",
            "unrelated_edge",
        ):
            edge_ids = np.flatnonzero(np.asarray(categories) == category)
            if not len(edge_ids):
                continue
            values = c1[:, target_edge, edge_ids]
            top_edge = np.argmax(c1[:, target_edge], axis=1)
            rows.append(
                {
                    "target_edge_id": target_edge,
                    "target_source": target_source,
                    "target_channel": target_channel,
                    "category": category,
                    "relation_edge_count": len(edge_ids),
                    "mean_edge_score": float(values.mean()),
                    "median_edge_score": float(np.median(values)),
                    "max_contribution_fraction": float(np.isin(top_edge, edge_ids).mean()),
                    "valid_sample_count": int(valid[5, :, target_edge].sum()),
                }
            )
    return pd.DataFrame(rows)


def write_grouped_outputs(out, pool, aggregates, controlled, valid, window_frame, evaluations, corruption, sequence, thresholds):
    groups = []
    for target, frame in pool.assign(edge_id=np.arange(len(pool))).groupby("target", sort=False):
        groups.append(
            {
                "target_channel": target,
                "incoming_edge_ids": json.dumps(frame.edge_id.astype(int).tolist()),
                "incoming_edge_count": len(frame),
                "source_channels": json.dumps(frame.source.astype(str).tolist()),
            }
        )
    pd.DataFrame(groups).to_csv(out / "target_relation_group_registry.csv", index=False)
    threshold_rows = [
        {"aggregation": name, "fold": fold, "threshold": value, "quantile": 0.99}
        for name, values in thresholds.items()
        for fold, value in values.items()
    ]
    pd.DataFrame(threshold_rows).to_csv(out / "target_grouped_crossfit_thresholds.csv", index=False)
    comparison = pd.DataFrame([evaluations["top2"], evaluations["grouped"]])
    comparison.to_csv(out / "aggregation_comparison_metrics.csv", index=False)
    pd.DataFrame(corruption).to_csv(out / "metrics_by_corruption.csv", index=False)
    pd.DataFrame(sequence).to_csv(out / "metrics_by_sequence.csv", index=False)
    original_same_target = pool.target.to_numpy()[aggregates["clean_top_edges"]]
    duplicate_clean = original_same_target[:, 0] == original_same_target[:, 1]
    duplicate_rows = [
        {
            "scope": "clean",
            "same_target_original_top2_fraction": float(duplicate_clean.mean()),
            "grouped_duplicate_target_fraction": float(
                (aggregates["clean_top_targets"][:, 0] == aggregates["clean_top_targets"][:, 1]).mean()
            ),
            "duplicate_target_eliminated_fraction": float(duplicate_clean.mean()),
        }
    ]
    for family_index, family in enumerate(FAMILIES):
        top_targets = pool.target.to_numpy()[aggregates["control_top_edges"][family_index]]
        duplicate = top_targets[..., 0] == top_targets[..., 1]
        duplicate_rows.append(
            {
                "scope": family,
                "same_target_original_top2_fraction": float(duplicate.mean()),
                "grouped_duplicate_target_fraction": 0.0,
                "duplicate_target_eliminated_fraction": float(duplicate.mean()),
            }
        )
    pd.DataFrame(duplicate_rows).to_csv(out / "duplicate_target_audit.csv", index=False)
    c1_decomposition(controlled, valid, pool).to_csv(
        out / "c1_edge_contribution_decomposition.csv", index=False
    )
    corruption_frame = pd.DataFrame(corruption)
    corruption_frame[corruption_frame.family.eq("P1")].to_csv(
        out / "p1_point_boundary_metrics.csv", index=False
    )
    p2 = corruption_frame[corruption_frame.family.eq("P2")].copy()
    p2["semantic_role"] = "mixed_point_and_collaborative_anomaly"
    p2.to_csv(out / "p2_mixed_anomaly_diagnostics.csv", index=False)
    c2 = corruption_frame[corruption_frame.family.eq("C2")].copy()
    clean_lookup = comparison.set_index("aggregation").clean_fpr
    c2["clean_fpr"] = c2.aggregation.map(clean_lookup)
    c2["fpr"] = c2.q99_tpr
    c2["fpr_gap"] = (c2.fpr - c2.clean_fpr).abs()
    c2.to_csv(out / "c2_preserving_control_metrics.csv", index=False)


def decide(evaluations, corruption_rows, sequence_rows):
    grouped = evaluations["grouped"]
    rows = pd.DataFrame(corruption_rows)
    selected = rows[rows.aggregation.eq("grouped")].set_index("family")
    sequence = pd.DataFrame(sequence_rows)
    breaking_sequence = sequence[
        sequence.aggregation.eq("grouped") & sequence.family.isin(["R1", "R2", "R3"])
    ]
    sequence_macro = breaking_sequence.groupby("sequence_id").auroc.mean()
    gates = {
        "clean_overall_fpr": grouped["clean_fpr"] <= 0.02,
        "worst_sequence_fpr": grouped["worst_sequence_fpr"] <= 0.05,
        "c2_fpr": selected.loc["C2", "q99_tpr"] <= 0.03,
        "c2_gap": abs(selected.loc["C2", "q99_tpr"] - grouped["clean_fpr"]) <= 0.015,
        "macro_auroc": grouped["macro_auroc"] >= 0.82,
        "macro_q99_tpr": grouped["macro_q99_tpr"] >= 0.40,
        "two_family_auroc": int((selected.loc[["R1", "R2", "R3"], "auroc"] >= 0.85).sum()) >= 2,
        "two_family_q99_tpr": int((selected.loc[["R1", "R2", "R3"], "q99_tpr"] >= 0.40).sum()) >= 2,
        "five_sequences_auroc": int((sequence_macro >= 0.70).sum()) >= 5,
        "p1_fpr": selected.loc["P1", "q99_tpr"] <= 0.05,
        "finite": bool(np.isfinite(rows.select_dtypes(include=[np.number]).to_numpy()).all()),
    }
    passed = all(gates.values())
    baseline = evaluations["top2"]
    improvement = max(
        baseline["clean_fpr"] - grouped["clean_fpr"],
        baseline["worst_sequence_fpr"] - grouped["worst_sequence_fpr"],
        grouped["macro_auroc"] - baseline["macro_auroc"],
        grouped["macro_q99_tpr"] - baseline["macro_q99_tpr"],
    )
    if passed and improvement > 0:
        classification = "target_grouped_aggregation_is_valid"
    elif improvement >= 0.01:
        classification = "target_grouping_improves_stability_but_not_enough"
    else:
        classification = "duplicate_target_aggregation_is_not_main_problem"
    return classification, gates, passed, float(improvement)


def artifact_audit(out, before, after):
    rows = []
    for name, before_hash in before.items():
        rows.append(
            {
                "artifact": name,
                "reuse_role": "frozen_v2a_input",
                "sha256_before": before_hash,
                "sha256_after": after.get(name),
                "unchanged": before_hash == after.get(name),
            }
        )
    pd.DataFrame(rows).to_csv(out / "artifact_reuse_audit.csv", index=False)
    if not all(row["unchanged"] for row in rows):
        raise RuntimeError("Historical V2-A artifact hash changed")


def run_scale_fix():
    out = OUT
    required = (
        out / "controlled_dynamic_edge_scores.npz",
        out / "controlled_dynamic_aggregate_scores.npz",
        out / "reconstruction_mapping_audit.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"A1R cache required before scale fix: {missing}")
    before = directory_hashes(V2A)
    status_path = out / "stage_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status.update(
        {
            "channel_scale_provenance_resolved": status.get("channel_scale_provenance_resolved", False),
            "channel_scale_recovered": status.get("channel_scale_recovered", False),
            "p1_cache_patched": status.get("p1_cache_patched", False),
            "p2_cache_patched": status.get("p2_cache_patched", False),
            "exact_reproduction_rerun_complete": status.get("exact_reproduction_rerun_complete", False),
        }
    )
    if (
        status.get("final_report_complete")
        and status.get("exact_reproduction_passed")
        and (out / "final_hai_structure_v2a1r_scale_recovery_and_grouped_aggregation.md").exists()
    ):
        print("REUSE completed V2-A1R-Fix", flush=True)
        return 0
    cfg = load_frozen_config()
    pool = pd.read_csv(V2A / "selected_dynamic_relation_pool.csv")
    eligibility = pd.read_csv(ROOT / cfg["step4a_output"] / "hai_relation_feature_eligibility.csv")
    name_col = "feature_name" if "feature_name" in eligibility.columns else "feature"
    names = eligibility.loc[eligibility.eligible_for_relation_analysis.astype(bool), name_col].tolist()
    recovered_scales, scale_registry, reused_scale = recover_channel_scales(out, cfg, names, pool)
    status["channel_scale_provenance_resolved"] = True
    status["channel_scale_recovered"] = True
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    cfg, names, frozen_pool, refs_frame, refs, selfs, crosses, validation, timestamps, scaler = frozen_inputs(False)
    if not frozen_pool[["source", "target"]].equals(pool[["source", "target"]]):
        raise RuntimeError("Frozen relation order changed")
    windows = source_windows(False)
    if status.get("p1_cache_patched") and status.get("p2_cache_patched"):
        edge_archive = np.load(out / "controlled_dynamic_edge_scores.npz", allow_pickle=False)
        controlled = edge_archive["edge_scores"].copy()
        valid = edge_archive["valid_mask"].copy()
        edge_archive.close()
        aggregate_archive = np.load(out / "controlled_dynamic_aggregate_scores.npz", allow_pickle=False)
        aggregate_payload = {key: aggregate_archive[key].copy() for key in aggregate_archive.files}
        aggregate_archive.close()
        print("REUSE recovered scales and patched P1/P2 cache", flush=True)
    else:
        controlled, valid, aggregate_payload, patch_audit = patch_p1_p2_cache(
            out, pool, refs, selfs, crosses, validation, windows, recovered_scales
        )
        status["p1_cache_patched"] = True
        status["p2_cache_patched"] = True
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    aggregates = {
        "clean_max": aggregate_payload["clean_original_edge_max"],
        "clean_top2": aggregate_payload["clean_original_edge_top2_mean"],
        "clean_grouped": aggregate_payload["clean_target_group_max_then_top2_mean"],
        "control_max": aggregate_payload["original_edge_max"],
        "control_top2": aggregate_payload["original_edge_top2_mean"],
        "control_grouped": aggregate_payload["target_group_max_then_top2_mean"],
        "clean_top_edges": aggregate_payload["clean_top_edge_ids"],
        "control_top_edges": aggregate_payload["top_edge_ids"],
        "clean_top_targets": aggregate_payload["clean_top_target_ids"],
        "control_top_targets": aggregate_payload["top_target_ids"],
    }
    eval_max, cor_max, seq_max, th_max, _ = evaluate_candidate(
        "max", aggregates["clean_max"], aggregates["control_max"], valid, windows
    )
    eval_top2, cor_top2, seq_top2, th_top2, _ = evaluate_candidate(
        "top2", aggregates["clean_top2"], aggregates["control_top2"], valid, windows
    )
    exact_passed, audit, reproduction = exact_reproduction(
        out,
        {"max": eval_max, "top2": eval_top2},
        cor_max + cor_top2,
        windows,
        valid,
        audit_name="v2a_exact_reproduction_audit_after_scale_fix.csv",
        summary_name="v2a_exact_reproduction_summary_after_scale_fix.json",
    )
    status["exact_reproduction_rerun_complete"] = True
    status["exact_reproduction_passed"] = exact_passed
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if not (exact_passed and reproduction["checks_passed"] == 66 and reproduction["checks_total"] == 66):
        artifact_audit(out, before, directory_hashes(V2A))
        print("classification=p1_p2_exact_scale_recovery_failed", flush=True)
        print(
            f"checks_passed={reproduction['checks_passed']} checks_total={reproduction['checks_total']}",
            flush=True,
        )
        print("Grouped evaluation blocked by exact reproduction gate", flush=True)
        return 2
    eval_grouped, cor_grouped, seq_grouped, th_grouped, _ = evaluate_candidate(
        "grouped", aggregates["clean_grouped"], aggregates["control_grouped"], valid, windows
    )
    evaluations = {"top2": eval_top2, "grouped": eval_grouped}
    all_corruption = cor_top2 + cor_grouped
    all_sequence = seq_top2 + seq_grouped
    write_grouped_outputs(
        out,
        pool,
        aggregates,
        controlled,
        valid,
        windows,
        evaluations,
        all_corruption,
        all_sequence,
        {"original_edge_top2_mean": th_top2, "target_group_max_then_top2_mean": th_grouped},
    )
    classification, gates, passed, improvement = decide(evaluations, all_corruption, all_sequence)
    selected = {
        "status": "frozen_for_hai_development_evaluation" if passed else "not_frozen",
        "aggregation": "target_group_max_then_top2_mean",
        "exact_reproduction_passed": True,
        "checks_passed": 66,
        "checks_total": 66,
        "hai_test_used": False,
    }
    (out / "selected_structure_v2a1r_config.yaml").write_text(
        yaml.safe_dump(selected, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "classification": classification,
        "exact_reproduction_passed": True,
        "checks_passed": 66,
        "checks_total": 66,
        "grouped_gates": {key: bool(value) for key, value in gates.items()},
        "all_grouped_gates_passed": passed,
        "maximum_recorded_improvement": improvement,
        "scale_cache_reused": reused_scale,
        "test_loaded": False,
        "attack_label_used": False,
        "rec_loaded": False,
        "ae_loaded": False,
    }
    (out / "structure_v2a1r_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    issues = [
        {
            "issue_id": "V2A1R-FIX-001",
            "stage": "HAI Structure V2-A1R-Fix",
            "research_question": "Does exact scale recovery unlock a valid target-grouped aggregation?",
            "status": "resolved" if passed else "open",
            "evidence": classification,
            "severity": "low" if passed else "high",
            "affects_validity": not passed,
            "affects_final_method": True,
            "minimum_future_action": "No automatic follow-up experiment",
            "recomputation_scope": "Frozen P1/P2 cache and offline aggregation only",
            "frozen_artifacts_reusable": True,
        }
    ]
    (out / "unresolved_issues_delta.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")
    artifact_audit(out, before, directory_hashes(V2A))
    report = f"""# HAI Structure V2-A1R-Fix

- Exact V2-A reproduction after scale recovery: **66/66 passed**.
- Classification: **{classification}**.
- Grouped aggregation frozen: **{str(passed).lower()}**.
- P1/P2 scale source: exact V2-A normal-train `max(1.4826*MAD, std, 1e-6)`.
- R1/R2/R3/C1/C2 cache slices remained byte-identical.
- HAI test / attack label / AE / rec loaded: **false / false / false / false**.
"""
    (out / "final_hai_structure_v2a1r_scale_recovery_and_grouped_aggregation.md").write_text(
        report, encoding="utf-8"
    )
    status["grouped_evaluation_complete"] = True
    status["final_report_complete"] = True
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"exact_reproduction_passed=true checks_passed=66 classification={classification}", flush=True)
    print("test_loaded=false attack_label_used=false rec_loaded=false ae_loaded=false", flush=True)
    return 0


def run(smoke=False):
    out = SMOKE_OUT if smoke else OUT
    before = directory_hashes(V2A)
    dep_hash = dependency_hash()
    status_path = out / "stage_status.json"
    if status_path.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("final_report_complete") and old.get("dependency_hash") == dep_hash:
            print(f"REUSE completed V2-A1R: {out}", flush=True)
            return 0
    out.mkdir(parents=True, exist_ok=True)
    status = {
        "frozen_artifacts_verified": False,
        "validation_windows_loaded": False,
        "corruption_mapping_reconstructed": False,
        "per_edge_score_cache_complete": False,
        "aggregate_score_cache_complete": False,
        "exact_reproduction_complete": False,
        "exact_reproduction_passed": False,
        "grouped_evaluation_complete": False,
        "final_report_complete": False,
        "dependency_hash": dep_hash,
    }
    cfg, names, pool, refs_frame, refs, selfs, crosses, validation, timestamps, scaler = frozen_inputs(smoke)
    status["frozen_artifacts_verified"] = True
    windows = source_windows(smoke)
    status["validation_windows_loaded"] = True
    controlled, valid, replacements, mapping, marginal = recreate_controlled(
        pool, refs, selfs, crosses, validation, windows, scaler, names, smoke
    )
    mapping.to_csv(out / "reconstruction_mapping_audit.csv", index=False)
    status["corruption_mapping_reconstructed"] = True
    clean = sampled_clean_scores(windows)
    aggregates = save_score_caches(out, windows, pool, clean, controlled, valid, replacements)
    status["per_edge_score_cache_complete"] = True
    status["aggregate_score_cache_complete"] = True
    integrity = pd.DataFrame(
        [
            {
                "edge_score_shape": json.dumps(list(controlled.shape)),
                "expected_shape": json.dumps([7, len(windows), 12, 12]),
                "all_finite": bool(np.isfinite(controlled).all()),
                "valid_count": int(valid.sum()),
                "relation_order_exact": pool.source.tolist() == pd.read_csv(V2A / "selected_dynamic_relation_pool.csv").source.tolist(),
                "r1_multiset_audit_passed": bool(
                    marginal[marginal.family.isin(["R1", "R2"])].source_multiset_preserved.astype(str).eq("True").all()
                ),
            }
        ]
    )
    integrity.to_csv(out / "score_cache_integrity_audit.csv", index=False)
    eval_max, cor_max, seq_max, th_max, _ = evaluate_candidate(
        "max", aggregates["clean_max"], aggregates["control_max"], valid, windows
    )
    eval_top2, cor_top2, seq_top2, th_top2, _ = evaluate_candidate(
        "top2", aggregates["clean_top2"], aggregates["control_top2"], valid, windows
    )
    status["exact_reproduction_complete"] = True
    if smoke:
        exact_passed = True
        pd.DataFrame(
            [{"aggregation": "smoke", "metric": "finite", "passed": bool(np.isfinite(controlled).all())}]
        ).to_csv(out / "v2a_exact_reproduction_audit.csv", index=False)
        (out / "v2a_exact_reproduction_summary.json").write_text(
            json.dumps({"exact_reproduction_passed": True, "smoke_only": True}, indent=2), encoding="utf-8"
        )
    else:
        exact_passed, _, _ = exact_reproduction(
            out,
            {"max": eval_max, "top2": eval_top2},
            cor_max + cor_top2,
            windows,
            valid,
        )
    status["exact_reproduction_passed"] = exact_passed
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if not exact_passed:
        after = directory_hashes(V2A)
        artifact_audit(out, before, after)
        print("classification=controlled_score_cache_backfill_mismatch", flush=True)
        print("Grouped evaluation blocked by exact reproduction gate", flush=True)
        return 2
    eval_grouped, cor_grouped, seq_grouped, th_grouped, _ = evaluate_candidate(
        "grouped", aggregates["clean_grouped"], aggregates["control_grouped"], valid, windows
    )
    evaluations = {"top2": eval_top2, "grouped": eval_grouped}
    all_corruption = cor_top2 + cor_grouped
    all_sequence = seq_top2 + seq_grouped
    write_grouped_outputs(
        out,
        pool,
        aggregates,
        controlled,
        valid,
        windows,
        evaluations,
        all_corruption,
        all_sequence,
        {"original_edge_top2_mean": th_top2, "target_group_max_then_top2_mean": th_grouped},
    )
    classification, gates, passed, improvement = decide(evaluations, all_corruption, all_sequence)
    selected = {
        "status": "frozen_for_hai_development_evaluation" if passed else "not_frozen",
        "aggregation": "target_group_max_then_top2_mean",
        "exact_reproduction_passed": True,
        "hai_test_used": False,
    }
    (out / "selected_structure_v2a1r_config.yaml").write_text(
        yaml.safe_dump(selected, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "classification": classification,
        "exact_reproduction_passed": True,
        "grouped_gates": {key: bool(value) for key, value in gates.items()},
        "all_grouped_gates_passed": passed,
        "maximum_recorded_improvement": improvement,
        "test_loaded": False,
        "attack_label_used": False,
        "rec_loaded": False,
        "ae_loaded": False,
    }
    (out / "structure_v2a1r_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    issues = [
        {
            "issue_id": "V2A1R-001",
            "stage": "HAI Structure V2-A1R",
            "research_question": "Does target-wise grouping repair duplicate-target aggregation?",
            "status": "resolved" if passed else "open",
            "evidence": classification,
            "severity": "high" if not passed else "low",
            "affects_validity": not passed,
            "affects_final_method": True,
            "minimum_future_action": "No automatic follow-up experiment",
            "recomputation_scope": "Offline aggregate score cache only",
            "frozen_artifacts_reusable": True,
        }
    ]
    (out / "unresolved_issues_delta.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")
    after = directory_hashes(V2A)
    artifact_audit(out, before, after)
    report = f"""# HAI Structure V2-A1R

- Exact V2-A reproduction passed: **true**.
- Classification: **{classification}**.
- Grouped aggregation frozen: **{str(passed).lower()}**.
- HAI test / attack label / AE / rec loaded: **false / false / false / false**.
- Original relation pool: 12 directed edges, unchanged.
- P2 semantic role: `mixed_point_and_collaborative_anomaly`.
- C1 semantic role: `target_edge_preserved_but_collateral_relations_may_break`.
- C2 remains the preserving-control hard gate.
"""
    (out / "final_hai_structure_v2a1r_cache_backfill_and_grouped_aggregation.md").write_text(
        report, encoding="utf-8"
    )
    status["grouped_evaluation_complete"] = True
    status["final_report_complete"] = True
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"exact_reproduction_passed=true classification={classification}", flush=True)
    print("test_loaded=false attack_label_used=false rec_loaded=false ae_loaded=false", flush=True)
    return 0


def dry_run(recover_p1_p2_scale=False):
    cfg = load_frozen_config()
    pool = pd.read_csv(V2A / "selected_dynamic_relation_pool.csv")
    windows = source_windows(False)
    print("REUSE: Step2 scaler/split, V2-A final Ridge models, 12-edge pool, dependency references", flush=True)
    print("REUSE: 600 validation source windows, original fold and replacement manifest", flush=True)
    if recover_p1_p2_scale:
        print("REUSE: existing R1/R2/R3/C1/C2 cache slices; no regeneration", flush=True)
        print("COMPUTE_NEW: exact normal-train channel robust scales and P1/P2 cache slices only", flush=True)
        print("GATE: 66/66 exact reproduction required before grouped evaluation", flush=True)
    else:
        print("COMPUTE_NEW: seven-family controlled per-edge score cache and two offline aggregations", flush=True)
    print("FORBIDDEN: Ridge fit, relation discovery, HAI test, attack label, AE, rec, historical overwrite", flush=True)
    print(f"relation_count={len(pool)} source_window_count={len(windows)} family_count={len(FAMILIES)}", flush=True)
    print(f"seed={cfg['seed']} rolling={cfg['rolling_dependency_window']} q99={cfg['threshold_quantile']}", flush=True)
    print("exact_reproduction_tolerance=1e-6 grouped_evaluation_requires_exact_reproduction=true", flush=True)
    print("test_loaded=false attack_label_used=false rec_loaded=false ae_loaded=false", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--recover-p1-p2-scale", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        return dry_run(args.recover_p1_p2_scale)
    if args.recover_p1_p2_scale:
        if args.smoke:
            raise RuntimeError("--smoke is not combined with --recover-p1-p2-scale")
        return run_scale_fix()
    return run(args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
