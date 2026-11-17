from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    silhouette_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hai_structure_v2a1r_cache_backfill_and_grouped_aggregation import grouped_scores
from utils.dynamic_collaborative_corruptions import (
    replace_channels,
    source_block_permutation,
    source_temporal_shift,
    target_endpoint_spike,
    target_short_level_shift,
)
from utils.lagged_conditional_dependency import (
    causal_rolling_mean,
    cross_matrix,
    dependency_loss,
    lag_matrix,
    robust_reference,
)

V2A = ROOT / "outputs" / "hai_structure_v2a_lagged_dependency"
A1R = ROOT / "outputs" / "hai_structure_v2a1r_cache_backfill_and_grouped_aggregation"
A1S = ROOT / "outputs" / "hai_structure_v2a1s_full_clean_stability"
A1D = ROOT / "outputs" / "hai_structure_v2a1d_normal_drift_attribution"
OUT = ROOT / "outputs" / "hai_structure_v2a1m_mode_conditioned_reference"
SMOKE_OUT = ROOT / "outputs" / "hai_structure_v2a1m_mode_conditioned_reference_smoke"
FAMILIES = np.asarray(("R1", "R2", "R3", "P1", "P2", "C1", "C2"))
CONTEXT_LAGS = (8, 12, 16, 24, 32)
K_CANDIDATES = (2, 3)
K_SEEDS = (1, 2, 3)
TOLERANCE = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def historical_hashes():
    result = {}
    for prefix, directory in (("v2a", V2A), ("a1r", A1R), ("a1s", A1S), ("a1d", A1D)):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                result[f"{prefix}/{path.relative_to(directory)}"] = sha256(path)
    return result


def frozen_config():
    return yaml.safe_load((V2A / "config_resolved.yaml").read_text(encoding="utf-8"))


def load_frozen_sequences(smoke=False):
    cfg = frozen_config()
    split = pd.read_csv(ROOT / cfg["step2_output"] / "hai_split_manifest.csv")
    scaler = pd.read_csv(ROOT / cfg["step2_output"] / "hai_scaler_stats.csv").set_index("feature")
    eligibility = pd.read_csv(ROOT / cfg["step4a_output"] / "hai_relation_feature_eligibility.csv")
    name_col = "feature_name" if "feature_name" in eligibility.columns else "feature"
    names = eligibility.loc[eligibility.eligible_for_relation_analysis.astype(bool), name_col].tolist()
    if len(names) != 68:
        raise RuntimeError(f"Expected 68 eligible channels, got {len(names)}")
    mean = scaler.loc[names, "mean"].to_numpy(float)
    scale = scaler.loc[names, "effective_scale"].to_numpy(float)
    train, validation, train_time, validation_time = [], [], [], []
    for row in split.itertuples():
        path = ROOT / cfg["data_root"] / row.file_name
        header = pd.read_csv(path, nrows=0)
        timestamp_col = next(column for column in header if column.strip().lower() == "timestamp")
        frame = pd.read_csv(path, usecols=[timestamp_col, *names], nrows=int(row.original_rows), low_memory=False)
        values = ((frame[names].to_numpy(float) - mean) / scale).astype(np.float32)
        timestamps = frame[timestamp_col].astype(str).to_numpy()
        cut = int(row.split_index)
        tr, va = values[:cut], values[cut:]
        tt, vt = timestamps[:cut], timestamps[cut:]
        if smoke:
            tr, va, tt, vt = tr[:4000], va[:2000], tt[:4000], vt[:2000]
        train.append(tr)
        validation.append(va)
        train_time.append(tt)
        validation_time.append(vt)
    return cfg, names, split, train, validation, train_time, validation_time


def raw_dependency_for_sequences(sequences, timestamps, pool, selfs, crosses, prefix):
    sequence_ids, endpoint_rows, time_rows = [], [], []
    self_rows, cross_rows, dependency_rows, rolling_rows = [], [], [], []
    for sequence, values in enumerate(sequences):
        self_edge, cross_edge, dependency_edge, rolling_edge = [], [], [], []
        for relation in pool.itertuples():
            source, target = int(relation.source_index), int(relation.target_index)
            observed = values[32:, target]
            self_residual = observed - selfs[target].predict(lag_matrix(values, target))
            cross_residual = observed - crosses[(source, target)].predict(cross_matrix(values, source, target))
            raw = dependency_loss(self_residual, cross_residual)
            rolling = causal_rolling_mean(raw, 16)
            self_edge.append(self_residual[15:])
            cross_edge.append(cross_residual[15:])
            dependency_edge.append(raw[15:])
            rolling_edge.append(rolling[15:])
        count = len(values) - 47
        sequence_ids.append(np.full(count, f"{prefix}{sequence + 1}", dtype="U24"))
        endpoint_rows.append(np.arange(47, len(values), dtype=np.int64))
        time_rows.append(np.asarray(timestamps[sequence][47:], dtype="U32"))
        self_rows.append(np.column_stack(self_edge).astype(np.float32))
        cross_rows.append(np.column_stack(cross_edge).astype(np.float32))
        dependency_rows.append(np.column_stack(dependency_edge).astype(np.float32))
        # V2-A keeps the causal rolling result in float64 until reference normalization.
        rolling_rows.append(np.column_stack(rolling_edge).astype(np.float64))
    payload = {
        "sequence_id": np.concatenate(sequence_ids),
        "endpoint_row": np.concatenate(endpoint_rows),
        "timestamp": np.concatenate(time_rows),
        "target_edge_ids": np.arange(len(pool), dtype=np.int16),
        "self_residual": np.vstack(self_rows),
        "cross_residual": np.vstack(cross_rows),
        "raw_dependency_loss": np.vstack(dependency_rows),
        "rolling_dependency_loss": np.vstack(rolling_rows),
    }
    payload["finite_mask"] = np.isfinite(payload["rolling_dependency_loss"])
    # Keep explicit protocol names alongside the established cache field names.
    payload["endpoint"] = payload["endpoint_row"]
    payload["edge_id"] = payload["target_edge_ids"]
    payload["finite"] = payload["finite_mask"]
    if not all(np.isfinite(payload[key]).all() for key in ("self_residual", "cross_residual", "raw_dependency_loss", "rolling_dependency_loss")):
        raise RuntimeError("Non-finite raw dependency cache")
    return payload


def ensure_raw_cache_schema(path):
    """Backfill metadata aliases without recomputing any dependency score."""
    with np.load(path, allow_pickle=False) as archive:
        if all(key in archive.files for key in ("endpoint", "edge_id", "finite")):
            return False
        payload = {key: archive[key] for key in archive.files}
    payload.setdefault("endpoint", payload["endpoint_row"])
    payload.setdefault("edge_id", payload["target_edge_ids"])
    payload.setdefault("finite", payload["finite_mask"])
    np.savez_compressed(path, **payload)
    return True


def context_for_sequences(sequences, timestamps, channel_indices, channel_names, prefix):
    feature_rows, sequence_rows, endpoint_rows, timestamp_rows = [], [], [], []
    for sequence, values in enumerate(sequences):
        endpoints = np.arange(47, len(values))
        columns = [values[endpoints - lag, channel] for channel in channel_indices for lag in CONTEXT_LAGS]
        feature_rows.append(np.column_stack(columns).astype(np.float32))
        sequence_rows.append(np.full(len(endpoints), f"{prefix}{sequence + 1}", dtype="U24"))
        endpoint_rows.append(endpoints.astype(np.int64))
        timestamp_rows.append(np.asarray(timestamps[sequence][47:], dtype="U32"))
    return {
        "features": np.vstack(feature_rows),
        "sequence_id": np.concatenate(sequence_rows),
        "endpoint_row": np.concatenate(endpoint_rows),
        "timestamp": np.concatenate(timestamp_rows),
        "channel_indices": np.asarray(channel_indices, dtype=np.int16),
        "channel_names": np.asarray(channel_names),
        "context_lags": np.asarray(CONTEXT_LAGS, dtype=np.int16),
    }


def raw_window(window, pool, selfs, crosses):
    self_last, cross_last, raw_last, rolling = [], [], [], []
    for relation in pool.itertuples():
        source, target = int(relation.source_index), int(relation.target_index)
        observed = window[32:, target]
        self_residual = observed - selfs[target].predict(lag_matrix(window, target))
        cross_residual = observed - crosses[(source, target)].predict(cross_matrix(window, source, target))
        raw = dependency_loss(self_residual, cross_residual)
        self_last.append(self_residual[-1])
        cross_last.append(cross_residual[-1])
        raw_last.append(raw[-1])
        rolling.append(np.mean(raw[-16:]))
    return (
        np.asarray(self_last, np.float32),
        np.asarray(cross_last, np.float32),
        np.asarray(raw_last, np.float32),
        np.asarray(rolling, np.float32),
    )


def controlled_raw_cache(validation, pool, selfs, crosses, channel_indices, scales, smoke=False):
    old = np.load(A1R / "controlled_dynamic_edge_scores.npz", allow_pickle=False)
    source_ids = old["source_window_id"].astype(str)
    sequence_ids = old["sequence_id"].astype(str)
    endpoint_rows = old["endpoint_row"].astype(int)
    folds = old["fold"].astype(str)
    replacements = old["replacement_window_id"].astype(str)
    valid = old["valid_mask"].copy()
    old.close()
    if smoke:
        source_ids, sequence_ids, endpoint_rows, folds, replacements = (
            source_ids[:12], sequence_ids[:12], endpoint_rows[:12], folds[:12], replacements[:12]
        )
        valid = valid[:, :12]
    count, edge_count = len(source_ids), len(pool)
    shape = (len(FAMILIES), count, edge_count, edge_count)
    self_residual = np.empty(shape, np.float32)
    cross_residual = np.empty(shape, np.float32)
    raw_dependency = np.empty(shape, np.float32)
    rolling = np.empty(shape, np.float32)
    observed_context = np.empty((len(FAMILIES), count, edge_count, len(channel_indices) * len(CONTEXT_LAGS)), np.float32)
    source_lookup = {value: index for index, value in enumerate(source_ids)}
    selected_channels = sorted(set(pool.source_index.astype(int)) | set(pool.target_index.astype(int)))
    for window_index, (source_id, sequence_id, endpoint) in enumerate(zip(source_ids, sequence_ids, endpoint_rows)):
        sequence = int(sequence_id.removeprefix("train").split("_")[0]) - 1
        base = validation[sequence][endpoint - 47 : endpoint + 1]
        for target_edge, relation in enumerate(pool.itertuples()):
            source, target = int(relation.source_index), int(relation.target_index)
            replacement_id = replacements[window_index, target_edge]
            if replacement_id in source_lookup:
                replacement_endpoint = endpoint_rows[source_lookup[replacement_id]]
                replacement = validation[sequence][replacement_endpoint - 47 : replacement_endpoint + 1]
            else:
                replacement = base
            variants = (
                source_temporal_shift(base, source),
                source_block_permutation(base, source),
                replace_channels(base, replacement, [source]),
                target_endpoint_spike(base, target, float(scales[target])),
                target_short_level_shift(base, target, float(scales[target])),
                replace_channels(base, replacement, [source, target]),
                replace_channels(base, replacement, selected_channels),
            )
            for family, variant in enumerate(variants):
                a, b, c, d = raw_window(variant, pool, selfs, crosses)
                self_residual[family, window_index, target_edge] = a
                cross_residual[family, window_index, target_edge] = b
                raw_dependency[family, window_index, target_edge] = c
                rolling[family, window_index, target_edge] = d
                observed_context[family, window_index, target_edge] = np.asarray(
                    [variant[-1 - lag, channel] for channel in channel_indices for lag in CONTEXT_LAGS],
                    dtype=np.float32,
                )
        if (window_index + 1) % 25 == 0 or window_index + 1 == count:
            print(f"CONTROLLED_RAW window={window_index + 1}/{count}", flush=True)
    return {
        "family": FAMILIES,
        "source_window_id": source_ids,
        "sequence_id": sequence_ids,
        "fold": folds,
        "target_edge_id": np.arange(edge_count, dtype=np.int16),
        "self_residual": self_residual,
        "cross_residual": cross_residual,
        "raw_dependency_loss": raw_dependency,
        "rolling_dependency_loss": rolling,
        "finite_mask": np.isfinite(rolling),
        "valid_mask": valid,
        "replacement_window_id": replacements,
        "observed_context": observed_context,
    }


def global_edge_scores(rolling, references):
    median = references[:, 0]
    scale = references[:, 1]
    return np.maximum(0.0, (rolling - median) / scale).astype(np.float32)


def deterministic_sample(sequence_ids, maximum):
    indices = []
    for sequence in np.unique(sequence_ids):
        available = np.flatnonzero(sequence_ids == sequence)
        count = min(len(available), max(1, maximum // len(np.unique(sequence_ids))))
        indices.extend(available[np.linspace(0, len(available) - 1, count, dtype=int)].tolist())
    return np.asarray(sorted(set(indices)), dtype=np.int64)


def fit_train_only_modes(features, sequence_ids, smoke=False):
    fit_limit = 12000 if smoke else 120000
    fit_index = deterministic_sample(sequence_ids, fit_limit)
    fit_features = features[fit_index]
    pca_full = PCA(n_components=min(8, features.shape[1]), whiten=False, svd_solver="full", random_state=1)
    pca_full.fit(features)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    reached = np.flatnonzero(cumulative >= 0.90)
    components = int(reached[0] + 1) if len(reached) else min(8, features.shape[1])
    pca = PCA(n_components=components, whiten=False, svd_solver="full", random_state=1)
    pca.fit(features)
    projected_all = pca.transform(features).astype(np.float32)
    projected_fit = projected_all[fit_index]
    candidate_rows, stability_rows, models = [], [], {}
    min_count = 20 if smoke else 5000
    silhouette_index = deterministic_sample(sequence_ids, 1000 if smoke else 5000)
    for k in K_CANDIDATES:
        labels_by_seed, seed_models = {}, {}
        for seed in K_SEEDS:
            model = KMeans(n_clusters=k, n_init=20, random_state=seed, algorithm="lloyd")
            model.fit(projected_fit)
            labels = model.predict(projected_all)
            labels_by_seed[seed] = labels
            seed_models[seed] = model
        ari_values = []
        for left, right in ((1, 2), (1, 3), (2, 3)):
            value = float(adjusted_rand_score(labels_by_seed[left], labels_by_seed[right]))
            ari_values.append(value)
            stability_rows.append({"k": k, "seed_left": left, "seed_right": right, "ari": value})
        labels = labels_by_seed[1]
        counts = np.bincount(labels, minlength=k)
        occupancy = counts / len(labels)
        coverage = [int(sum(np.any(labels[sequence_ids == seq] == mode) for seq in np.unique(sequence_ids))) for mode in range(k)]
        silhouette = float(silhouette_score(projected_all[silhouette_index], labels[silhouette_index]))
        valid = bool(
            np.all(occupancy >= 0.05)
            and np.all(counts >= min_count)
            and min(coverage) >= 3
            and np.median(ari_values) >= 0.90
            and silhouette >= 0.10
        )
        candidate_rows.append(
            {
                "k": k,
                "silhouette": silhouette,
                "ari_median": float(np.median(ari_values)),
                "minimum_occupancy": float(occupancy.min()),
                "minimum_count": int(counts.min()),
                "minimum_sequence_coverage": int(min(coverage)),
                "valid": valid,
                "fit_window_count": len(fit_index),
            }
        )
        models[k] = (seed_models[1], labels, counts, occupancy, coverage)
    candidates = pd.DataFrame(candidate_rows)
    valid_candidates = candidates[candidates.valid].sort_values(["silhouette", "k"], ascending=[False, True])
    if len(valid_candidates) == 2 and abs(valid_candidates.iloc[0].silhouette - valid_candidates.iloc[1].silhouette) < 0.02:
        selected_k = int(valid_candidates.k.min())
    elif len(valid_candidates):
        selected_k = int(valid_candidates.iloc[0].k)
    else:
        selected_k = None
    return pca, projected_all, candidates, pd.DataFrame(stability_rows), models, selected_k, fit_index


def assign_modes(projected, model, radii):
    labels = model.predict(projected)
    distance_matrix = np.linalg.norm(projected[:, None, :] - model.cluster_centers_[None, :, :], axis=2)
    distances = distance_matrix[np.arange(len(labels)), labels]
    ood = distances > radii[labels]
    return labels.astype(np.int16), distances.astype(np.float32), ood


def build_mode_references(train_rolling, train_labels, pool, selected_k, global_refs, smoke=False):
    rows, fallback_rows = [], []
    medians = np.empty((selected_k, len(pool)), float)
    scales = np.empty_like(medians)
    minimum = 20 if smoke else 5000
    for mode in range(selected_k):
        mode_mask = train_labels == mode
        for edge, relation in enumerate(pool.itertuples()):
            values = train_rolling[mode_mask, edge]
            if len(values) < minimum:
                median, scale = global_refs[edge]
                mad, scaled_mad, fallback, iqr_scale, std = 0.0, 0.0, "global_reference", 0.0, 0.0
            else:
                median, scale, fallback, iqr_scale, std = robust_reference(values)
                mad = float(np.median(np.abs(values - np.median(values))))
                scaled_mad = float(1.4826 * mad)
            medians[mode, edge], scales[mode, edge] = median, scale
            row = {
                "mode": mode,
                "edge_id": edge,
                "source": relation.source,
                "target": relation.target,
                "sample_count": len(values),
                "median": median,
                "mad": mad,
                "scaled_mad": scaled_mad,
                "iqr_scale": iqr_scale,
                "std": std,
                "effective_scale": scale,
                "fallback": fallback,
                "used_global_reference": fallback == "global_reference",
            }
            rows.append(row)
            fallback_rows.append(row.copy())
    return medians, scales, pd.DataFrame(rows), pd.DataFrame(fallback_rows)


def conditioned_edge_scores(rolling, labels, ood, mode_medians, mode_scales, global_refs):
    medians = mode_medians[labels].copy()
    scales = mode_scales[labels].copy()
    medians[ood] = global_refs[:, 0]
    scales[ood] = global_refs[:, 1]
    return np.maximum(0.0, (rolling - medians) / scales).astype(np.float32)


def thresholds_from_a1s_manifest(scores, sequence_id, endpoint_row, manifest, quantile=0.99):
    lookup = {(seq, int(endpoint)): index for index, (seq, endpoint) in enumerate(zip(sequence_id, endpoint_row))}
    unique = manifest[manifest.fold.eq("A")][["source_window_id", "sequence_id", "endpoint_row", "half"]].copy()
    def cache_sequence_id(value):
        if value.startswith("train") and value.endswith("_validation"):
            return f"validation{value.removeprefix('train').split('_')[0]}"
        return value
    positions = np.asarray(
        [lookup[(cache_sequence_id(row.sequence_id), int(row.endpoint_row))] for row in unique.itertuples()]
    )
    unique["position"] = positions
    unique["score"] = scores[positions]
    thresholds, rows = {}, []
    for fold, half in (("A", "first"), ("B", "second")):
        values = unique.loc[unique.half.eq(half), "score"].to_numpy(float)
        threshold = float(np.quantile(values, quantile))
        thresholds[fold] = threshold
        rows.append(
            {
                "fold": fold,
                "calibration_half": half,
                "calibration_window_count": len(values),
                "threshold": threshold,
                "quantile": quantile,
                "synthetic_used": False,
                "mode_specific_threshold": False,
            }
        )
    unique["evaluation_fold"] = np.where(unique.half.eq("first"), "B", "A")
    unique["threshold"] = unique.evaluation_fold.map(thresholds)
    unique["alarm"] = unique.score > unique.threshold
    return thresholds, pd.DataFrame(rows), unique


def alarm_segments(active):
    padded = np.r_[False, np.asarray(active, bool), False].astype(np.int8)
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    lengths = ends - starts
    return int(len(lengths)), int(lengths.max()) if len(lengths) else 0


def clean_metrics(unique, assignment_frame):
    merged = unique.merge(
        assignment_frame[["sequence_id", "endpoint_row", "mode", "mode_ood"]],
        on=["sequence_id", "endpoint_row"],
        how="left",
        validate="one_to_one",
    )
    rows = []
    for sequence, frame in merged.groupby("sequence_id"):
        for half, part in frame.groupby("half"):
            rows.append(
                {
                    "sequence_id": sequence,
                    "half": half,
                    "window_count": len(part),
                    "alarm_count": int(part.alarm.sum()),
                    "fpr": float(part.alarm.mean()),
                    "ood_rate": float(part.mode_ood.mean()),
                }
            )
        rows.append(
            {
                "sequence_id": sequence,
                "half": "all",
                "window_count": len(frame),
                "alarm_count": int(frame.alarm.sum()),
                "fpr": float(frame.alarm.mean()),
                "ood_rate": float(frame.mode_ood.mean()),
            }
        )
    sequence_metrics = pd.DataFrame(rows)
    segment_count = longest = 0
    for _, frame in merged.groupby(["sequence_id", "half"]):
        count, length = alarm_segments(frame.sort_values("endpoint_row").alarm)
        segment_count += count
        longest = max(longest, length)
    all_sequence = sequence_metrics[sequence_metrics.half.eq("all")]
    overall = {
        "overall_fpr": float(merged.alarm.mean()),
        "worst_sequence_fpr": float(all_sequence.fpr.max()),
        "alarm_count": int(merged.alarm.sum()),
        "segment_count": segment_count,
        "longest_segment": longest,
        "ood_fpr": float(merged.loc[merged.mode_ood, "alarm"].mean()) if merged.mode_ood.any() else 0.0,
        "non_ood_fpr": float(merged.loc[~merged.mode_ood, "alarm"].mean()),
        "train1_first_fpr": float(sequence_metrics.query("sequence_id == 'train1_validation' and half == 'first'").fpr.iloc[0]),
        "train1_second_fpr": float(sequence_metrics.query("sequence_id == 'train1_validation' and half == 'second'").fpr.iloc[0]),
    }
    return overall, sequence_metrics, merged


def binary_metrics(clean, positive, clean_threshold, positive_threshold):
    labels = np.r_[np.zeros(len(clean), int), np.ones(len(positive), int)]
    scores = np.r_[clean, positive]
    prediction = np.r_[clean > clean_threshold, positive > positive_threshold]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, prediction, average="binary", zero_division=0)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "q99_tpr": float(np.mean(positive > positive_threshold)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "median_score_ratio": float(np.median(positive) / max(np.median(clean), 1e-12)),
    }


def controlled_conditioned_scores(raw, source_labels, source_ood, mode_medians, mode_scales, global_refs, target_ids):
    rolling = raw["rolling_dependency_loss"]
    family_count, window_count, target_edge_count, edge_count = rolling.shape
    scores = np.empty_like(rolling, dtype=np.float32)
    for window in range(window_count):
        median = global_refs[:, 0] if source_ood[window] else mode_medians[source_labels[window]]
        scale = global_refs[:, 1] if source_ood[window] else mode_scales[source_labels[window]]
        scores[:, window] = np.maximum(0.0, (rolling[:, window] - median) / scale)
    grouped, _, _ = grouped_scores(scores, target_ids)
    return scores, grouped


def evaluate_controlled(grouped_controlled, raw, clean_source_scores, source_thresholds, source_labels, source_sequences):
    valid = raw["valid_mask"]
    family_rows, mode_rows, sequence_rows = [], [], []
    for family_index, family in enumerate(FAMILIES):
        mask = valid[family_index]
        positive = grouped_controlled[family_index][mask]
        positive_threshold = np.broadcast_to(source_thresholds[:, None], mask.shape)[mask]
        clean = np.repeat(clean_source_scores, mask.sum(axis=1))
        clean_threshold = np.repeat(source_thresholds, mask.sum(axis=1))
        result = binary_metrics(clean, positive, clean_threshold, positive_threshold)
        family_rows.append({"family": family, "sample_count": len(positive), **result})
        for mode in np.unique(source_labels):
            source_mask = source_labels == mode
            local_mask = mask[source_mask]
            mode_rows.append(
                {
                    "family": family,
                    "mode": int(mode),
                    **binary_metrics(
                        np.repeat(clean_source_scores[source_mask], local_mask.sum(axis=1)),
                        grouped_controlled[family_index][source_mask][local_mask],
                        np.repeat(source_thresholds[source_mask], local_mask.sum(axis=1)),
                        np.broadcast_to(source_thresholds[source_mask, None], local_mask.shape)[local_mask],
                    ),
                }
            )
        if family in ("R1", "R2", "R3"):
            for sequence in np.unique(source_sequences):
                source_mask = source_sequences == sequence
                local_mask = mask[source_mask]
                sequence_rows.append(
                    {
                        "family": family,
                        "sequence_id": sequence,
                        **binary_metrics(
                            np.repeat(clean_source_scores[source_mask], local_mask.sum(axis=1)),
                            grouped_controlled[family_index][source_mask][local_mask],
                            np.repeat(source_thresholds[source_mask], local_mask.sum(axis=1)),
                            np.broadcast_to(source_thresholds[source_mask, None], local_mask.shape)[local_mask],
                        ),
                    }
                )
    return pd.DataFrame(family_rows), pd.DataFrame(mode_rows), pd.DataFrame(sequence_rows)


def reproduce_global(out, validation_raw, controlled_raw, refs, target_ids):
    validation_edges = global_edge_scores(validation_raw["rolling_dependency_loss"], refs)
    validation_grouped, _, _ = grouped_scores(validation_edges, target_ids)
    manifest = pd.read_csv(A1S / "full_clean_crossfit_window_manifest.csv")
    old_thresholds = pd.read_csv(A1S / "full_clean_crossfit_thresholds.csv").set_index("fold").threshold.to_dict()
    _, _, unique = thresholds_from_a1s_manifest(
        validation_grouped,
        validation_raw["sequence_id"],
        validation_raw["endpoint_row"],
        manifest,
    )
    unique["threshold"] = unique.evaluation_fold.map(old_thresholds)
    unique["alarm"] = unique.score > unique.threshold
    old_controlled = np.load(A1R / "controlled_dynamic_edge_scores.npz", allow_pickle=False)
    reconstructed_controlled = global_edge_scores(controlled_raw["rolling_dependency_loss"], refs)
    max_edge_difference = float(np.max(np.abs(reconstructed_controlled - old_controlled["edge_scores"])))
    old_controlled.close()
    reconstructed_grouped, _, _ = grouped_scores(reconstructed_controlled, target_ids)
    source_ids = controlled_raw["source_window_id"].astype(str)
    lookup = unique.set_index("source_window_id")
    source_clean = lookup.loc[source_ids, "score"].to_numpy(float)
    source_threshold = lookup.loc[source_ids, "threshold"].to_numpy(float)
    family, _, _ = evaluate_controlled(
        reconstructed_grouped,
        controlled_raw,
        source_clean,
        source_threshold,
        np.zeros(len(source_ids), int),
        controlled_raw["sequence_id"].astype(str),
    )
    breaking = family[family.family.isin(["R1", "R2", "R3"])]
    train1 = unique[unique.sequence_id.eq("train1_validation")]
    reproduced = {
        "overall_fpr": float(unique.alarm.mean()),
        "worst_sequence_fpr": float(unique.groupby("sequence_id").alarm.mean().max()),
        "train1_alarm": int(train1.alarm.sum()),
        "train1_first_alarm": int(train1[train1.half.eq("first")].alarm.sum()),
        "train1_second_alarm": int(train1[train1.half.eq("second")].alarm.sum()),
        "macro_auroc": float(breaking.auroc.mean()),
        "macro_aupr": float(breaking.aupr.mean()),
        "macro_q99_tpr": float(breaking.q99_tpr.mean()),
        "p1_fpr": float(family[family.family.eq("P1")].q99_tpr.iloc[0]),
        "c2_fpr": float(family[family.family.eq("C2")].q99_tpr.iloc[0]),
        "max_controlled_edge_score_difference": max_edge_difference,
    }
    expected = {
        "overall_fpr": 0.009118,
        "worst_sequence_fpr": 0.054300,
        "train1_alarm": 1014,
        "train1_first_alarm": 10,
        "train1_second_alarm": 1004,
        "macro_auroc": 0.838228,
        "macro_aupr": 0.866872,
        "macro_q99_tpr": 0.441204,
        "p1_fpr": 0.016667,
        "c2_fpr": 0.0,
        "max_controlled_edge_score_difference": 0.0,
    }
    rows = []
    for metric, expected_value in expected.items():
        difference = abs(float(reproduced[metric]) - expected_value)
        tolerance = 0.0 if "alarm" in metric else TOLERANCE
        rows.append(
            {
                "metric": metric,
                "expected": expected_value,
                "reproduced": reproduced[metric],
                "absolute_difference": difference,
                "tolerance": tolerance,
                "passed": difference <= tolerance,
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(out / "raw_dependency_reproduction_audit.csv", index=False)
    return bool(audit.passed.all()), validation_edges, validation_grouped, unique, family


def cache_payload(path):
    archive = np.load(path, allow_pickle=False)
    payload = {key: archive[key] for key in archive.files}
    archive.close()
    return payload


def validation_assignment_frame(validation_context, labels, distances, ood):
    return pd.DataFrame(
        {
            "sequence_id": validation_context["sequence_id"].astype(str),
            "endpoint_row": validation_context["endpoint_row"].astype(int),
            "timestamp": validation_context["timestamp"].astype(str),
            "mode": labels,
            "mode_distance": distances,
            "mode_ood": ood,
        }
    )


def classification_and_gates(clean, family, sequence_family, assignment, baseline_family, hashes_unchanged):
    breaking = family[family.family.isin(["R1", "R2", "R3"])]
    baseline_breaking = baseline_family[baseline_family.family.isin(["R1", "R2", "R3"])]
    sequence_macro = sequence_family.groupby("sequence_id").auroc.mean()
    c2_fpr = float(family[family.family.eq("C2")].q99_tpr.iloc[0])
    c2_excess = max(0.0, c2_fpr - clean["overall_fpr"])
    macro_auroc = float(breaking.auroc.mean())
    macro_aupr = float(breaking.aupr.mean())
    macro_tpr = float(breaking.q99_tpr.mean())
    baseline_auroc = float(baseline_breaking.auroc.mean())
    baseline_tpr = float(baseline_breaking.q99_tpr.mean())
    overall_ood = float(assignment.mode_ood.mean())
    gates = {
        "overall_fpr": clean["overall_fpr"] <= 0.02,
        "worst_sequence_fpr": clean["worst_sequence_fpr"] <= 0.05,
        "train1_second_reduction": clean["train1_second_fpr"] <= 0.107529 * 0.70,
        "c2_fpr": c2_fpr <= 0.03,
        "c2_excess_fpr": c2_excess <= 0.015,
        "macro_auroc": macro_auroc >= 0.82,
        "macro_aupr": macro_aupr >= 0.84,
        "macro_q99_tpr": macro_tpr >= 0.40,
        "two_family_auroc": int((breaking.auroc >= 0.85).sum()) >= 2,
        "two_family_q99_tpr": int((breaking.q99_tpr >= 0.40).sum()) >= 2,
        "five_sequence_auroc": int((sequence_macro >= 0.70).sum()) >= 5,
        "p1_fpr": float(family[family.family.eq("P1")].q99_tpr.iloc[0]) <= 0.05,
        "auroc_preservation": baseline_auroc - macro_auroc <= 0.02,
        "tpr_preservation": baseline_tpr - macro_tpr <= 0.04,
        "overall_ood": overall_ood <= 0.10,
        "finite": bool(np.isfinite(family.select_dtypes(include=[np.number]).to_numpy()).all()),
        "historical_hashes": hashes_unchanged,
    }
    all_passed = all(gates.values())
    train1_second_assignment = assignment[
        assignment.sequence_id.eq("validation1")
        & (assignment.endpoint_row >= assignment[assignment.sequence_id.eq("validation1")].endpoint_row.median())
    ]
    train1_second_ood = float(train1_second_assignment.mode_ood.mean())
    if all_passed:
        classification = "mode_conditioned_dependency_reference_is_valid"
    elif train1_second_ood > 0.20:
        classification = "unseen_normal_operating_mode_not_covered_by_train"
    elif not gates["auroc_preservation"] or not gates["tpr_preservation"]:
        classification = "mode_conditioning_hurts_collaborative_detection"
    elif clean["train1_second_fpr"] < 0.107529 and clean["worst_sequence_fpr"] < 0.054300:
        classification = "represented_modes_reduce_drift_but_not_enough"
    else:
        classification = "represented_modes_do_not_reduce_drift"
    diagnostics = {
        "macro_auroc": macro_auroc,
        "macro_aupr": macro_aupr,
        "macro_q99_tpr": macro_tpr,
        "baseline_macro_auroc": baseline_auroc,
        "baseline_macro_q99_tpr": baseline_tpr,
        "c2_fpr": c2_fpr,
        "c2_excess_fpr": c2_excess,
        "overall_ood_rate": overall_ood,
        "train1_second_ood_rate": train1_second_ood,
    }
    return classification, gates, all_passed, diagnostics


def run(smoke=False):
    if smoke:
        return run_smoke()
    out = OUT
    before = historical_hashes()
    dependency_hash = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    status_path = out / "stage_status.json"
    schema_backfilled = False
    for cache_name in ("train_raw_dynamic_dependency_cache.npz", "validation_raw_dynamic_dependency_cache.npz"):
        cache_path = out / cache_name
        if cache_path.exists():
            schema_backfilled = ensure_raw_cache_schema(cache_path) or schema_backfilled
    if schema_backfilled:
        print("CACHE_SCHEMA_BACKFILL metadata aliases added; scores unchanged", flush=True)
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("final_report_complete") and status.get("dependency_hash") == dependency_hash:
            print(f"REUSE completed V2-A1M: {out}", flush=True)
            return 0
    out.mkdir(parents=True, exist_ok=True)
    status = {
        "frozen_artifacts_verified": False,
        "raw_dependency_cache_complete": False,
        "global_reproduction_complete": False,
        "global_reproduction_passed": False,
        "context_cache_complete": False,
        "train_mode_selection_complete": False,
        "mode_model_frozen_before_validation": False,
        "validation_assignment_complete": False,
        "mode_reference_complete": False,
        "candidate_evaluation_complete": False,
        "final_report_complete": False,
        "dependency_hash": dependency_hash,
    }
    pool = pd.read_csv(V2A / "selected_dynamic_relation_pool.csv")
    if len(pool) != 12:
        raise RuntimeError("Frozen relation pool is not 12")
    target_names = list(dict.fromkeys(pool.target.astype(str).tolist()))
    target_ids = np.asarray([target_names.index(value) for value in pool.target.astype(str)], dtype=np.int16)
    global_registry = pd.read_csv(V2A / "dynamic_dependency_reference_registry.csv").sort_values("edge_id")
    global_refs = global_registry[["median", "effective_scale"]].to_numpy(float)
    train_raw_path = out / "train_raw_dynamic_dependency_cache.npz"
    validation_raw_path = out / "validation_raw_dynamic_dependency_cache.npz"
    controlled_raw_path = out / "controlled_raw_dynamic_dependency_cache.npz"
    train_context_path = out / "train_operating_context_features.npz"
    validation_context_path = out / "validation_operating_context_features.npz"
    raw_precision_invalid = False
    if train_raw_path.exists() and validation_raw_path.exists():
        train_probe = np.load(train_raw_path, allow_pickle=False)
        validation_probe = np.load(validation_raw_path, allow_pickle=False)
        raw_precision_invalid = (
            train_probe["rolling_dependency_loss"].dtype != np.float64
            or validation_probe["rolling_dependency_loss"].dtype != np.float64
        )
        train_probe.close()
        validation_probe.close()
    need_forward = raw_precision_invalid or not all(
        path.exists() for path in (train_raw_path, validation_raw_path, controlled_raw_path, train_context_path, validation_context_path)
    )
    if need_forward:
        cfg, names, split, train, validation, train_time, validation_time = load_frozen_sequences(False)
        selfs = joblib.load(V2A / "final_self_models.joblib")
        crosses = joblib.load(V2A / "final_cross_models.joblib")
        if raw_precision_invalid or not train_raw_path.exists() or not validation_raw_path.exists():
            train_raw = raw_dependency_for_sequences(train, train_time, pool, selfs, crosses, "train")
            validation_raw = raw_dependency_for_sequences(validation, validation_time, pool, selfs, crosses, "validation")
            np.savez_compressed(train_raw_path, **train_raw)
            np.savez_compressed(validation_raw_path, **validation_raw)
        channel_indices = sorted(set(pool.source_index.astype(int)) | set(pool.target_index.astype(int)))
        channel_names = [names[index] for index in channel_indices]
        if not train_context_path.exists() or not validation_context_path.exists():
            train_context = context_for_sequences(train, train_time, channel_indices, channel_names, "train")
            validation_context = context_for_sequences(validation, validation_time, channel_indices, channel_names, "validation")
            np.savez_compressed(train_context_path, **train_context)
            np.savez_compressed(validation_context_path, **validation_context)
        if not controlled_raw_path.exists():
            scale_archive = np.load(A1R / "recovered_channel_robust_scales.npz", allow_pickle=False)
            scales = scale_archive["effective_scale"]
            scale_archive.close()
            controlled_raw = controlled_raw_cache(validation, pool, selfs, crosses, channel_indices, scales, False)
            np.savez_compressed(controlled_raw_path, **controlled_raw)
        del train, validation, train_time, validation_time, selfs, crosses
        print("RAW_AND_CONTEXT_CACHE saved", flush=True)
    status["raw_dependency_cache_complete"] = True
    status["context_cache_complete"] = True
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    train_raw = cache_payload(train_raw_path)
    validation_raw = cache_payload(validation_raw_path)
    controlled_raw = cache_payload(controlled_raw_path)
    train_context = cache_payload(train_context_path)
    validation_context = cache_payload(validation_context_path)
    reproduced, global_validation_edges, global_validation_grouped, global_unique, global_family = reproduce_global(
        out, validation_raw, controlled_raw, global_refs, target_ids
    )
    status["global_reproduction_complete"] = True
    status["global_reproduction_passed"] = reproduced
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if not reproduced:
        print("classification=raw_dependency_cache_reproduction_failed", flush=True)
        return 2
    feature_registry = []
    for channel, channel_index in zip(train_context["channel_names"], train_context["channel_indices"]):
        for lag in CONTEXT_LAGS:
            feature_registry.append(
                {
                    "feature_index": len(feature_registry),
                    "channel": str(channel),
                    "channel_index": int(channel_index),
                    "lag": lag,
                    "causal": True,
                    "uses_current_t": False,
                    "uses_structure_score": False,
                }
            )
    pd.DataFrame(feature_registry).to_csv(out / "operating_mode_feature_registry.csv", index=False)
    pca, train_projected, candidate_metrics, stability, models, selected_k, fit_index = fit_train_only_modes(
        train_context["features"], train_context["sequence_id"].astype(str), False
    )
    candidate_metrics.to_csv(out / "mode_candidate_metrics.csv", index=False)
    stability.to_csv(out / "mode_seed_stability.csv", index=False)
    status["train_mode_selection_complete"] = True
    if selected_k is None:
        (out / "structure_v2a1m_summary.json").write_text(
            json.dumps({"classification": "no_stable_train_operating_modes"}, indent=2), encoding="utf-8"
        )
        print("classification=no_stable_train_operating_modes", flush=True)
        return 2
    mode_model, train_labels, counts, occupancy, coverage = models[selected_k]
    distances = np.linalg.norm(train_projected - mode_model.cluster_centers_[train_labels], axis=1)
    radii = np.asarray([np.quantile(distances[train_labels == mode], 0.995) for mode in range(selected_k)])
    selected_payload = {
        "selected_k": selected_k,
        "pca_components": int(pca.n_components_),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "centroids": mode_model.cluster_centers_.tolist(),
        "train_occupancy": occupancy.tolist(),
        "train_sequence_coverage": coverage,
        "silhouette": float(candidate_metrics.set_index("k").loc[selected_k, "silhouette"]),
        "ari_stability": float(candidate_metrics.set_index("k").loc[selected_k, "ari_median"]),
        "config_hash": hashlib.sha256(
            json.dumps({"lags": CONTEXT_LAGS, "k": K_CANDIDATES, "seeds": K_SEEDS, "n_init": 20}, sort_keys=True).encode()
        ).hexdigest(),
        "status": "frozen_before_validation_evaluation",
    }
    (out / "selected_mode_model.yaml").write_text(yaml.safe_dump(selected_payload, sort_keys=False), encoding="utf-8")
    joblib.dump({"pca": pca, "kmeans": mode_model, "radii": radii}, out / "selected_mode_pipeline.joblib")
    status["mode_model_frozen_before_validation"] = True
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    validation_projected = pca.transform(validation_context["features"]).astype(np.float32)
    validation_labels, validation_distances, validation_ood = assign_modes(validation_projected, mode_model, radii)
    assignment = validation_assignment_frame(validation_context, validation_labels, validation_distances, validation_ood)
    assignment.to_csv(out / "validation_mode_assignment.csv", index=False)
    ood_rows = [
        {"scope": "overall", "window_count": len(assignment), "ood_count": int(assignment.mode_ood.sum()), "ood_rate": float(assignment.mode_ood.mean())}
    ]
    for sequence, frame in assignment.groupby("sequence_id"):
        ood_rows.append(
            {"scope": sequence, "window_count": len(frame), "ood_count": int(frame.mode_ood.sum()), "ood_rate": float(frame.mode_ood.mean())}
        )
        if sequence == "validation1":
            midpoint = len(frame) // 2
            for half, part in (("train1_first", frame.iloc[:midpoint]), ("train1_second", frame.iloc[midpoint:])):
                ood_rows.append(
                    {"scope": half, "window_count": len(part), "ood_count": int(part.mode_ood.sum()), "ood_rate": float(part.mode_ood.mean())}
                )
    ood_frame = pd.DataFrame(ood_rows)
    ood_frame.to_csv(out / "validation_mode_ood_audit.csv", index=False)
    registry_rows = []
    for mode in range(selected_k):
        registry_rows.append(
            {
                "mode": mode,
                "train_count": int(counts[mode]),
                "train_occupancy": float(occupancy[mode]),
                "sequence_coverage": int(coverage[mode]),
                "support_radius_q995": float(radii[mode]),
            }
        )
    pd.DataFrame(registry_rows).to_csv(out / "operating_mode_registry.csv", index=False)
    status["validation_assignment_complete"] = True
    mode_medians, mode_scales, reference_frame, fallback_frame = build_mode_references(
        train_raw["rolling_dependency_loss"], train_labels, pool, selected_k, global_refs, False
    )
    reference_frame.to_csv(out / "mode_conditioned_dependency_reference.csv", index=False)
    fallback_frame.to_csv(out / "mode_reference_fallback_audit.csv", index=False)
    status["mode_reference_complete"] = True
    validation_mode_edges = conditioned_edge_scores(
        validation_raw["rolling_dependency_loss"], validation_labels, validation_ood, mode_medians, mode_scales, global_refs
    )
    validation_mode_grouped, _, _ = grouped_scores(validation_mode_edges, target_ids)
    a1s_manifest = pd.read_csv(A1S / "full_clean_crossfit_window_manifest.csv")
    mode_thresholds, threshold_frame, mode_unique = thresholds_from_a1s_manifest(
        validation_mode_grouped,
        validation_raw["sequence_id"].astype(str),
        validation_raw["endpoint_row"],
        a1s_manifest,
    )
    threshold_frame.to_csv(out / "mode_conditioned_crossfit_thresholds.csv", index=False)
    mode_assignment_for_merge = assignment.copy()
    mode_assignment_for_merge["sequence_id"] = mode_assignment_for_merge.sequence_id.str.replace("validation", "train", regex=False)
    # Restore A1S sequence naming exactly.
    mode_assignment_for_merge["sequence_id"] = assignment.sequence_id.str.extract(r"(\d+)")[0].map(lambda value: f"train{value}_validation")
    clean_overall, clean_sequence, clean_merged = clean_metrics(mode_unique, mode_assignment_for_merge)
    target_comparison_rows = []
    validation_sequence = validation_raw["sequence_id"].astype(str)
    train1_positions = np.flatnonzero(validation_sequence == "validation1")
    train1_second_positions = train1_positions[len(train1_positions) // 2 :]
    global_fold_a_threshold = float(
        pd.read_csv(A1S / "full_clean_crossfit_thresholds.csv").set_index("fold").loc["A", "threshold"]
    )
    mode_fold_a_threshold = float(mode_thresholds["A"])
    for target, target_name in enumerate(target_names):
        edge_ids = np.flatnonzero(target_ids == target)
        global_target_score = global_validation_edges[train1_second_positions][:, edge_ids].max(axis=1)
        mode_target_score = validation_mode_edges[train1_second_positions][:, edge_ids].max(axis=1)
        global_count = int((global_target_score > global_fold_a_threshold).sum())
        mode_count = int((mode_target_score > mode_fold_a_threshold).sum())
        target_comparison_rows.append(
            {
                "target_channel": target_name,
                "global_reference_exceed_count_train1_second": global_count,
                "mode_conditioned_exceed_count_train1_second": mode_count,
                "exceed_count_reduction": global_count - mode_count,
                "global_q99": float(np.quantile(global_target_score, 0.99)),
                "mode_conditioned_q99": float(np.quantile(mode_target_score, 0.99)),
            }
        )
    pd.DataFrame(target_comparison_rows).to_csv(out / "mode_conditioning_target_comparison.csv", index=False)
    train1_second_alarm = clean_merged[
        clean_merged.sequence_id.eq("train1_validation")
        & clean_merged.half.eq("second")
        & clean_merged.alarm
    ]
    ood_frame = pd.concat(
        [
            ood_frame,
            pd.DataFrame(
                [
                    {
                        "scope": "train1_second_alarm_windows",
                        "window_count": len(train1_second_alarm),
                        "ood_count": int(train1_second_alarm.mode_ood.sum()),
                        "ood_rate": float(train1_second_alarm.mode_ood.mean()) if len(train1_second_alarm) else 0.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ood_frame.to_csv(out / "validation_mode_ood_audit.csv", index=False)
    pd.DataFrame([clean_overall]).to_csv(out / "clean_mode_conditioned_metrics.csv", index=False)
    clean_sequence.to_csv(out / "clean_metrics_by_sequence_and_half.csv", index=False)
    source_ids = controlled_raw["source_window_id"].astype(str)
    source_lookup = mode_unique.set_index("source_window_id")
    source_clean_score = source_lookup.loc[source_ids, "score"].to_numpy(float)
    source_threshold = source_lookup.loc[source_ids, "threshold"].to_numpy(float)
    validation_lookup = {
        (sequence, int(endpoint)): index
        for index, (sequence, endpoint) in enumerate(
            zip(validation_raw["sequence_id"].astype(str), validation_raw["endpoint_row"])
        )
    }
    source_meta_archive = np.load(A1R / "controlled_dynamic_edge_scores.npz", allow_pickle=False)
    controlled_endpoints = source_meta_archive["endpoint_row"].astype(int)
    source_meta_archive.close()
    source_positions = np.asarray(
        [
            validation_lookup[(f"validation{int(sequence.removeprefix('train').split('_')[0])}", int(endpoint))]
            for sequence, endpoint in zip(controlled_raw["sequence_id"].astype(str), controlled_endpoints)
        ]
    )
    source_labels = validation_labels[source_positions]
    source_ood = validation_ood[source_positions]
    controlled_edges, controlled_grouped = controlled_conditioned_scores(
        controlled_raw, source_labels, source_ood, mode_medians, mode_scales, global_refs, target_ids
    )
    family_metrics, mode_metrics, sequence_metrics = evaluate_controlled(
        controlled_grouped,
        controlled_raw,
        source_clean_score,
        source_threshold,
        source_labels,
        controlled_raw["sequence_id"].astype(str),
    )
    family_metrics.to_csv(out / "controlled_mode_conditioned_metrics.csv", index=False)
    family_metrics.to_csv(out / "controlled_metrics_by_family.csv", index=False)
    mode_metrics.to_csv(out / "controlled_metrics_by_mode.csv", index=False)
    sequence_metrics.to_csv(out / "controlled_metrics_by_sequence.csv", index=False)
    baseline_clean = pd.read_csv(A1S / "full_clean_fpr_overall.csv").query("scope == 'combined'").iloc[0]
    baseline_family = pd.read_csv(A1S / "controlled_metrics_by_family.csv")
    after = historical_hashes()
    audit = pd.DataFrame(
        [
            {"artifact": name, "sha256_before": value, "sha256_after": after.get(name), "unchanged": value == after.get(name)}
            for name, value in before.items()
        ]
    )
    audit.to_csv(out / "artifact_reuse_audit.csv", index=False)
    hashes_unchanged = bool(audit.unchanged.all())
    classification, gates, passed, diagnostics = classification_and_gates(
        clean_overall, family_metrics, sequence_metrics, assignment, baseline_family, hashes_unchanged
    )
    comparison_rows = [
        {
            "metric": "overall_fpr",
            "global_reference": float(baseline_clean.overall_fpr),
            "mode_conditioned": clean_overall["overall_fpr"],
            "difference": clean_overall["overall_fpr"] - float(baseline_clean.overall_fpr),
        },
        {
            "metric": "worst_sequence_fpr",
            "global_reference": float(baseline_clean.worst_sequence_fpr),
            "mode_conditioned": clean_overall["worst_sequence_fpr"],
            "difference": clean_overall["worst_sequence_fpr"] - float(baseline_clean.worst_sequence_fpr),
        },
        {
            "metric": "train1_second_fpr",
            "global_reference": 0.107529,
            "mode_conditioned": clean_overall["train1_second_fpr"],
            "difference": clean_overall["train1_second_fpr"] - 0.107529,
        },
        {
            "metric": "controlled_macro_auroc",
            "global_reference": diagnostics["baseline_macro_auroc"],
            "mode_conditioned": diagnostics["macro_auroc"],
            "difference": diagnostics["macro_auroc"] - diagnostics["baseline_macro_auroc"],
        },
        {
            "metric": "controlled_macro_q99_tpr",
            "global_reference": diagnostics["baseline_macro_q99_tpr"],
            "mode_conditioned": diagnostics["macro_q99_tpr"],
            "difference": diagnostics["macro_q99_tpr"] - diagnostics["baseline_macro_q99_tpr"],
        },
    ]
    pd.DataFrame(comparison_rows).to_csv(out / "mode_conditioning_comparison.csv", index=False)
    # Corrupted observed mode is diagnostic only; formal assignment remains the source clean mode.
    observed_flat = controlled_raw["observed_context"].reshape(-1, controlled_raw["observed_context"].shape[-1])
    observed_projected = pca.transform(observed_flat)
    observed_labels_flat, observed_distances_flat, observed_ood_flat = assign_modes(
        observed_projected, mode_model, radii
    )
    observed_labels = observed_labels_flat.reshape(controlled_raw["observed_context"].shape[:-1])
    observed_distances = observed_distances_flat.reshape(controlled_raw["observed_context"].shape[:-1])
    observed_ood = observed_ood_flat.reshape(controlled_raw["observed_context"].shape[:-1])
    observed_disagreement = float(np.mean(observed_labels != source_labels[None, :, None]))
    controlled_raw["source_clean_mode"] = source_labels.astype(np.int16)
    controlled_raw["source_mode_distance"] = validation_distances[source_positions].astype(np.float32)
    controlled_raw["source_mode_ood"] = source_ood
    controlled_raw["corrupted_observed_mode"] = observed_labels.astype(np.int16)
    controlled_raw["corrupted_observed_mode_distance"] = observed_distances.astype(np.float32)
    controlled_raw["corrupted_observed_mode_ood"] = observed_ood
    np.savez_compressed(controlled_raw_path, **controlled_raw)
    assignment["source_mode_formal_for_controlled"] = True
    assignment.to_csv(out / "validation_mode_assignment.csv", index=False)
    start = time.perf_counter()
    _ = mode_model.predict(pca.transform(validation_context["features"][:1000]))
    assignment_latency = (time.perf_counter() - start) / 1000
    start = time.perf_counter()
    _ = conditioned_edge_scores(
        validation_raw["rolling_dependency_loss"][:1000],
        validation_labels[:1000],
        validation_ood[:1000],
        mode_medians,
        mode_scales,
        global_refs,
    )
    scoring_latency = (time.perf_counter() - start) / 1000
    pca_parameters = int(pca.components_.size + pca.mean_.size)
    centroid_parameters = int(mode_model.cluster_centers_.size)
    reference_statistics = int(mode_medians.size + mode_scales.size)
    pd.DataFrame(
        [
            {
                "pca_parameter_count": pca_parameters,
                "kmeans_centroid_parameter_count": centroid_parameters,
                "reference_statistic_count": reference_statistics,
                "total_parameter_count": pca_parameters + centroid_parameters + reference_statistics,
                "pipeline_file_size_bytes": (out / "selected_mode_pipeline.joblib").stat().st_size,
                "mode_assignment_seconds_per_window": assignment_latency,
                "structure_scoring_seconds_per_window": scoring_latency,
                "below_10000_parameters": pca_parameters + centroid_parameters + reference_statistics < 10000,
            }
        ]
    ).to_csv(out / "lightweight_cost_audit.csv", index=False)
    selected = {
        "status": "frozen_for_hai_development_evaluation" if passed else "not_frozen",
        "aggregation": "target_group_max_then_top2_mean",
        "selected_k": selected_k,
        "mode_model_status": "frozen_before_validation_evaluation",
        "hai_test_used": False,
    }
    (out / "selected_structure_v2a1m_config.yaml").write_text(yaml.safe_dump(selected, sort_keys=False), encoding="utf-8")
    summary = {
        "classification": classification,
        "all_gates_passed": passed,
        "gates": {key: bool(value) for key, value in gates.items()},
        "clean": clean_overall,
        **diagnostics,
        "selected_k": selected_k,
        "controlled_observed_mode_disagreement_rate_diagnostic_only": observed_disagreement,
        "test_loaded": False,
        "attack_label_used": False,
        "rec_loaded": False,
        "ae_loaded": False,
        "ridge_refit": False,
    }
    (out / "structure_v2a1m_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    issues = [
        {
            "issue_id": "V2A1M-001",
            "stage": "HAI Structure V2-A1M",
            "research_question": "Are validation normal modes covered by train?",
            "status": "resolved" if diagnostics["overall_ood_rate"] <= 0.10 else "open",
            "evidence": diagnostics,
            "severity": "high",
            "affects_validity": not passed,
            "affects_final_method": True,
            "minimum_future_action": "No automatic follow-up",
            "recomputation_scope": "Frozen caches reusable",
            "frozen_artifacts_reusable": True,
        }
    ]
    (out / "unresolved_issues_delta.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")
    report = f"""# HAI Structure V2-A1M

- Train-only selected K: **{selected_k}**.
- Classification: **{classification}**.
- Mode-conditioned overall / worst FPR: **{clean_overall['overall_fpr']:.6f} / {clean_overall['worst_sequence_fpr']:.6f}**.
- Train1 second-half FPR: **{clean_overall['train1_second_fpr']:.6f}**.
- Validation overall OOD rate: **{diagnostics['overall_ood_rate']:.6f}**.
- Controlled macro AUROC / AUPR / q99 TPR: **{diagnostics['macro_auroc']:.6f} / {diagnostics['macro_aupr']:.6f} / {diagnostics['macro_q99_tpr']:.6f}**.
- Frozen for HAI development evaluation: **{str(passed).lower()}**.
- HAI test / attack label / AE / rec loaded: **false / false / false / false**.
"""
    (out / "final_hai_structure_v2a1m_mode_conditioned_reference.md").write_text(report, encoding="utf-8")
    status["candidate_evaluation_complete"] = True
    status["frozen_artifacts_verified"] = hashes_unchanged
    status["final_report_complete"] = True
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"classification={classification} selected_k={selected_k} all_gates_passed={str(passed).lower()}", flush=True)
    print("test_loaded=false attack_label_used=false rec_loaded=false ae_loaded=false ridge_refit=false", flush=True)
    return 0


def run_smoke():
    out = SMOKE_OUT
    out.mkdir(parents=True, exist_ok=True)
    cfg, names, split, train, validation, train_time, validation_time = load_frozen_sequences(True)
    pool = pd.read_csv(V2A / "selected_dynamic_relation_pool.csv")
    selfs = joblib.load(V2A / "final_self_models.joblib")
    crosses = joblib.load(V2A / "final_cross_models.joblib")
    raw = raw_dependency_for_sequences(train, train_time, pool, selfs, crosses, "train")
    channel_indices = sorted(set(pool.source_index.astype(int)) | set(pool.target_index.astype(int)))
    context = context_for_sequences(train, train_time, channel_indices, [names[index] for index in channel_indices], "train")
    pca, projected, candidates, stability, models, selected_k, _ = fit_train_only_modes(
        context["features"], context["sequence_id"].astype(str), True
    )
    candidates.to_csv(out / "mode_candidate_metrics.csv", index=False)
    stability.to_csv(out / "mode_seed_stability.csv", index=False)
    summary = {
        "smoke": True,
        "raw_shape": list(raw["rolling_dependency_loss"].shape),
        "context_shape": list(context["features"].shape),
        "selected_k": selected_k,
        "finite": bool(np.isfinite(raw["rolling_dependency_loss"]).all()),
        "test_loaded": False,
        "ridge_refit": False,
    }
    (out / "structure_v2a1m_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"SMOKE selected_k={selected_k} raw_shape={summary['raw_shape']} context_shape={summary['context_shape']}", flush=True)
    print("test_loaded=false attack_label_used=false rec_loaded=false ae_loaded=false ridge_refit=false", flush=True)
    return 0


def dry_run():
    print("REUSE: Step2 scaler/split, 68 eligible channels, frozen self/cross Ridge, 12 relations, A1S folds", flush=True)
    print("COMPUTE_NEW_ONCE: train/validation/controlled raw dependency and causal context caches", flush=True)
    print("TRAIN_ONLY_FIT: PCA <=8 components; KMeans K in {2,3}, n_init=20, seeds={1,2,3}", flush=True)
    print("VALIDATION_AFTER_FREEZE: mode assignment, q99 crossfit, controlled reuse evaluation", flush=True)
    print("FORBIDDEN: Ridge fit, relation selection, new aggregation, mode-specific threshold, HAI test, AE, rec", flush=True)
    print(f"context_lags={list(CONTEXT_LAGS)} rolling=16 aggregation=target_group_max_then_top2_mean quantile=0.99", flush=True)
    print("test_loaded=false attack_label_used=false rec_loaded=false ae_loaded=false ridge_refit=false", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        return dry_run()
    return run(args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
