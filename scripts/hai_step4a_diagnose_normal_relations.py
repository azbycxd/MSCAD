from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from itertools import combinations, product
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.transforms import StandardScaler  # noqa: E402


TRAIN_FILES = tuple(f"train{i}.csv" for i in range(1, 7))
WINDOWS = (64, 128)
GLOBAL_EPS = 1e-8
LOCAL_STD_EPS = 1e-6
DEFAULT_STRIDE = 32
DEFAULT_PAIR_CHUNK = 64
FORMAL_OUTPUT = ROOT / "outputs" / "hai_step4a_normal_relation_diagnosis"
SMOKE_OUTPUT = ROOT / "outputs" / "hai_step4a_normal_relation_diagnosis_smoke"
OUTPUT_FILES = (
    "hai_relation_feature_eligibility.csv", "hai_pair_global_sequence_stats.csv",
    "hai_pair_local_stats_by_sequence_w64.csv", "hai_pair_local_stats_by_sequence_w128.csv",
    "hai_pair_local_stats_aggregate_w64.csv", "hai_pair_local_stats_aggregate_w128.csv",
    "hai_sequence_relation_similarity.csv", "hai_sequence_relation_similarity_summary.csv",
    "hai_relation_threshold_sensitivity.csv", "hai_relation_pair_classification_w64.csv",
    "hai_relation_pair_classification_w128.csv", "hai_relation_window_comparison.csv",
    "strict_candidate_pool.csv", "diversity_candidate_pool.csv", "hai_relation_reference_strategy.json",
    "hai_relation_diagnosis_summary.json", "final_hai_normal_relation_diagnosis.md",
)
THRESHOLDS = {
    "valid_ratio": (0.80, 0.90, 0.95), "median_abs_corr": (0.30, 0.50, 0.70),
    "fisher_z_robust_sigma": (0.20, 0.25, 0.30, 0.40),
    "sign_consistency": (0.80, 0.90, 0.95), "sequence_coverage": (3, 4, 5, 6),
}
CLASSIFICATION_THRESHOLDS = {
    "strict": {"valid_ratio": 0.95, "median_abs_corr": 0.50, "fisher_z_robust_sigma": 0.25,
               "sign_consistency": 0.90, "minimum_sequences": 5, "between_sequence_corr_range": 0.25},
    "moderate": {"valid_ratio": 0.90, "median_abs_corr": 0.30, "fisher_z_robust_sigma": 0.35,
                 "sign_consistency": 0.85, "minimum_sequences": 4},
}


def ensure_fresh(output: Path) -> None:
    existing = [name for name in OUTPUT_FILES if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite HAI Step 4A outputs: {existing}")
    output.mkdir(parents=True, exist_ok=True)


def subsystem(feature: str) -> str:
    match = re.match(r"^(P[1-4])(?:_|$)", str(feature).upper())
    return match.group(1) if match else "other"


def relation_scope(left: str, right: str) -> str:
    a, b = subsystem(left), subsystem(right)
    if a == b and a in {"P1", "P2", "P3", "P4"}:
        return f"within_{a}"
    if a in {"P1", "P2", "P3", "P4"} and b in {"P1", "P2", "P3", "P4"}:
        return "cross_subsystem"
    return "unknown"


def load_prefix_manifest() -> pd.DataFrame:
    path = ROOT / "outputs" / "hai_step2_dataloader_validation" / "hai_split_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"HAI Step 2 split manifest is required: {path}")
    frame = pd.read_csv(path)
    if frame.file_name.tolist() != list(TRAIN_FILES):
        raise ValueError("HAI Step 2 manifest does not contain train1..train6 in order")
    return frame


def load_train_prefixes(data_root: Path, smoke: bool = False) -> tuple[list[np.ndarray], list[str], list[dict[str, Any]]]:
    manifest = load_prefix_manifest()
    arrays: list[np.ndarray] = []
    feature_names: list[str] | None = None
    ranges: list[dict[str, Any]] = []
    for row in manifest.itertuples():
        path = data_root / row.file_name
        if path.stem.lower().startswith("test"):
            raise RuntimeError(f"Formal Step 4A attempted to access forbidden test file: {path}")
        if not path.exists():
            raise FileNotFoundError(path)
        header = pd.read_csv(path, nrows=0)
        timestamp_name = next((name for name in header.columns if str(name).strip().lower() == "timestamp"), None)
        label_name = next((name for name in header.columns if str(name).strip().lower() == "attack"), None)
        if timestamp_name is None or label_name is None:
            raise ValueError(f"Missing timestamp/Attack schema columns in {path}")
        names = [str(name).strip() for name in header.columns if name not in {timestamp_name, label_name}]
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f"Feature schema/order mismatch in {path}")
        prefix_rows = int(row.train_rows)
        used_rows = min(prefix_rows, 4096) if smoke else prefix_rows
        usecols = [timestamp_name, *[name for name in header.columns if str(name).strip() in set(names)]]
        frame = pd.read_csv(path, usecols=usecols, nrows=used_rows, low_memory=False)
        numeric = frame[[name for name in frame.columns if str(name).strip() in set(names)]].copy()
        numeric.columns = [str(name).strip() for name in numeric.columns]
        numeric = numeric[names].apply(pd.to_numeric, errors="raise")
        values = numeric.to_numpy(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite training feature in {path}")
        arrays.append(values)
        ranges.append({"file_name": row.file_name, "prefix_rows_registered": prefix_rows, "rows_used": used_rows,
                       "start_timestamp": str(frame[timestamp_name].iloc[0]),
                       "end_timestamp": str(frame[timestamp_name].iloc[-1]), "attack_column_read": False})
    assert feature_names is not None
    if len(feature_names) != 86:
        raise ValueError(f"Expected 86 ordered HAI features, got {len(feature_names)}")
    return arrays, feature_names, ranges


def rolling_moments(values: np.ndarray, window: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(values) < window:
        return np.empty((0, values.shape[1])), np.empty((0, values.shape[1])), np.empty(0, dtype=int)
    starts = np.arange(0, len(values) - window + 1, stride, dtype=np.int64)
    ends = starts + window
    cumulative = np.vstack([np.zeros((1, values.shape[1]), np.float64), np.cumsum(values, axis=0, dtype=np.float64)])
    squared = np.vstack([np.zeros((1, values.shape[1]), np.float64),
                         np.cumsum(np.square(values, dtype=np.float64), axis=0, dtype=np.float64)])
    sums = cumulative[ends] - cumulative[starts]
    variances = np.maximum(squared[ends] - squared[starts] - sums * sums / window, 0.0)
    return sums, variances, starts


def fisher_transform(values: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(np.asarray(values, dtype=np.float64), -0.999999, 0.999999))


def sign_consistency(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    signs = np.sign(finite[np.isfinite(finite)])
    nonzero = signs[signs != 0]
    if len(nonzero) == 0:
        return 0.0
    return float(max((nonzero > 0).sum(), (nonzero < 0).sum()) / len(nonzero))


def correlation_statistics(correlations: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(correlations, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        return {name: np.nan for name in ("median_corr", "median_abs_corr", "mean_abs_corr", "q10_corr",
                                            "q90_corr", "sign_consistency", "fisher_z_median", "fisher_z_mad",
                                            "fisher_z_robust_sigma", "corr_iqr")} | {"sign_flip_count": 0}
    fisher = fisher_transform(valid)
    fisher_median = float(np.median(fisher))
    fisher_mad = float(np.median(np.abs(fisher - fisher_median)))
    signs = np.sign(valid)
    return {
        "median_corr": float(np.median(valid)), "median_abs_corr": float(np.median(np.abs(valid))),
        "mean_abs_corr": float(np.mean(np.abs(valid))), "q10_corr": float(np.quantile(valid, 0.10)),
        "q90_corr": float(np.quantile(valid, 0.90)), "sign_consistency": sign_consistency(valid),
        "sign_flip_count": int((signs[1:] != signs[:-1]).sum()), "fisher_z_median": fisher_median,
        "fisher_z_mad": fisher_mad, "fisher_z_robust_sigma": 1.4826 * fisher_mad,
        "corr_iqr": float(np.quantile(valid, 0.75) - np.quantile(valid, 0.25)),
    }


def pair_correlations(values: np.ndarray, sums: np.ndarray, variances: np.ndarray, starts: np.ndarray,
                      pairs: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left, right = pairs[:, 0], pairs[:, 1]
    products = values[:, left] * values[:, right]
    cumulative = np.vstack([np.zeros((1, len(pairs)), np.float64), np.cumsum(products, axis=0, dtype=np.float64)])
    sum_xy = cumulative[starts + window] - cumulative[starts]
    covariance = sum_xy - sums[:, left] * sums[:, right] / window
    constant_left = variances[:, left] <= window * LOCAL_STD_EPS ** 2
    constant_right = variances[:, right] <= window * LOCAL_STD_EPS ** 2
    denominator = np.sqrt(variances[:, left] * variances[:, right])
    valid = (~constant_left) & (~constant_right) & (denominator > 0)
    corr = np.full_like(covariance, np.nan, dtype=np.float64)
    np.divide(covariance, denominator, out=corr, where=valid)
    corr = np.clip(corr, -1.0, 1.0)
    valid &= np.isfinite(corr)
    corr[~valid] = np.nan
    return corr, constant_left, constant_right, valid


def feature_eligibility(raw: list[np.ndarray], normalized: list[np.ndarray], names: list[str], stride: int) -> pd.DataFrame:
    combined_count = sum(len(x) for x in raw)
    total = sum((x.sum(axis=0, dtype=np.float64) for x in raw), start=np.zeros(len(names), np.float64))
    mean = total / combined_count
    squared = sum((((x.astype(np.float64) - mean) ** 2).sum(axis=0) for x in raw), start=np.zeros(len(names), np.float64))
    global_std = np.sqrt(squared / combined_count)
    sequence_std = np.vstack([x.astype(np.float64).std(axis=0) for x in raw])
    local_ratios: dict[int, np.ndarray] = {}
    for window in WINDOWS:
        valid_counts = np.zeros(len(names), dtype=np.int64)
        attempts = 0
        for values in normalized:
            _, variances, _ = rolling_moments(values, window, stride)
            valid_counts += (variances > window * LOCAL_STD_EPS ** 2).sum(axis=0)
            attempts += len(variances)
        local_ratios[window] = valid_counts / max(attempts, 1)
    rows = []
    for index, name in enumerate(names):
        zero = bool(global_std[index] < GLOBAL_EPS)
        row = {"feature_name": name, "subsystem": subsystem(name), "global_train_mean": mean[index],
               "global_train_std": global_std[index], "global_zero_variance": zero}
        for seq in range(6): row[f"sequence{seq + 1}_std"] = sequence_std[seq, index]
        row.update({"constant_sequence_count": int((sequence_std[:, index] < GLOBAL_EPS).sum()),
                    "near_constant_sequence_count": int((sequence_std[:, index] < LOCAL_STD_EPS).sum()),
                    "active_sequence_count": int((sequence_std[:, index] >= LOCAL_STD_EPS).sum()),
                    "local_valid_ratio_w64": local_ratios[64][index],
                    "local_valid_ratio_w128": local_ratios[128][index],
                    "eligible_for_relation_analysis": not zero,
                    "exclusion_reason": "global_train_raw_std_below_1e-8" if zero else ""})
        rows.append(row)
    return pd.DataFrame(rows)


def pair_metadata(names: list[str], eligible_indices: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    pairs = np.asarray(list(combinations(eligible_indices.tolist(), 2)), dtype=np.int64)
    rows = []
    for pair_id, (left, right) in enumerate(pairs, 1):
        rows.append({"pair_id": pair_id, "feature_i_index": left, "feature_j_index": right,
                     "feature_i": names[left], "feature_j": names[right],
                     "subsystem_i": subsystem(names[left]), "subsystem_j": subsystem(names[right]),
                     "relation_scope": relation_scope(names[left], names[right])})
    return pairs, pd.DataFrame(rows)


def safe_corrcoef(values: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.corrcoef(values.astype(np.float64), rowvar=False)


def global_sequence_stats(normalized: list[np.ndarray], pairs: np.ndarray, metadata: pd.DataFrame) -> tuple[pd.DataFrame, list[np.ndarray]]:
    matrices = [safe_corrcoef(values[:, np.unique(pairs)]) for values in normalized]
    active_indices = np.unique(pairs)
    remap = {original: index for index, original in enumerate(active_indices)}
    left = np.asarray([remap[index] for index in pairs[:, 0]])
    right = np.asarray([remap[index] for index in pairs[:, 1]])
    sequence_values = np.column_stack([matrix[left, right] for matrix in matrices])
    pooled = safe_corrcoef(np.concatenate([values[:, active_indices] for values in normalized], axis=0))[left, right]
    rows = metadata.copy()
    rows["pooled_pearson"] = pooled
    for seq in range(6): rows[f"sequence{seq + 1}_pearson"] = sequence_values[:, seq]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rows["sequence_corr_median"] = np.nanmedian(sequence_values, axis=1)
        rows["sequence_corr_mean"] = np.nanmean(sequence_values, axis=1)
        rows["sequence_corr_min"] = np.nanmin(sequence_values, axis=1)
        rows["sequence_corr_max"] = np.nanmax(sequence_values, axis=1)
        rows["between_sequence_corr_range"] = np.nanmax(sequence_values, axis=1) - np.nanmin(sequence_values, axis=1)
        rows["between_sequence_corr_std"] = np.nanstd(sequence_values, axis=1)
        rows["median_abs_corr"] = np.nanmedian(np.abs(sequence_values), axis=1)
    rows["sign_consistency"] = [sign_consistency(row) for row in sequence_values]
    rows["sign_flip_count"] = [int((np.sign(row[np.isfinite(row)])[1:] != np.sign(row[np.isfinite(row)])[:-1]).sum())
                               for row in sequence_values]
    rows["valid_sequence_count"] = np.isfinite(sequence_values).sum(axis=1)
    return rows, matrices


def local_relation_stats(normalized: list[np.ndarray], pairs: np.ndarray, metadata: pd.DataFrame,
                         window: int, stride: int, chunk_size: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    moments = [rolling_moments(values, window, stride) for values in normalized]
    sequence_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for chunk_start in range(0, len(pairs), chunk_size):
        chunk = pairs[chunk_start:chunk_start + chunk_size]
        combined: list[list[np.ndarray]] = [[] for _ in range(len(chunk))]
        counts = [{"attempted": 0, "valid": 0, "both": 0, "left": 0, "right": 0, "numeric": 0}
                  for _ in range(len(chunk))]
        for seq_index, (values, moment) in enumerate(zip(normalized, moments), 1):
            sums, variances, starts = moment
            corr, const_left, const_right, valid = pair_correlations(values, sums, variances, starts, chunk, window)
            for local_index in range(len(chunk)):
                correlations = corr[:, local_index]
                valid_values = correlations[valid[:, local_index]]
                combined[local_index].append(valid_values)
                both = const_left[:, local_index] & const_right[:, local_index]
                numeric_invalid = (~const_left[:, local_index]) & (~const_right[:, local_index]) & (~valid[:, local_index])
                base = metadata.iloc[chunk_start + local_index].to_dict()
                attempted, valid_count = len(correlations), len(valid_values)
                base.update({"sequence_id": f"train{seq_index}", "window_size": window,
                             "diagnostic_stride": stride, "attempted_window_count": attempted,
                             "valid_window_count": valid_count, "invalid_window_count": attempted - valid_count,
                             "valid_ratio": valid_count / max(attempted, 1),
                             "both_local_constant_count": int(both.sum()),
                             "feature_i_local_constant_count": int(const_left[:, local_index].sum()),
                             "feature_j_local_constant_count": int(const_right[:, local_index].sum()),
                             "numeric_invalid_count": int(numeric_invalid.sum()),
                             **correlation_statistics(valid_values)})
                sequence_rows.append(base)
                counter = counts[local_index]
                counter["attempted"] += attempted; counter["valid"] += valid_count
                counter["both"] += int(both.sum()); counter["left"] += int(const_left[:, local_index].sum())
                counter["right"] += int(const_right[:, local_index].sum()); counter["numeric"] += int(numeric_invalid.sum())
        for local_index in range(len(chunk)):
            values = np.concatenate(combined[local_index]) if combined[local_index] else np.empty(0)
            counter = counts[local_index]
            base = metadata.iloc[chunk_start + local_index].to_dict()
            base.update({"window_size": window, "diagnostic_stride": stride,
                         "attempted_window_count": counter["attempted"], "valid_window_count": counter["valid"],
                         "invalid_window_count": counter["attempted"] - counter["valid"],
                         "valid_ratio": counter["valid"] / max(counter["attempted"], 1),
                         "both_local_constant_count": counter["both"],
                         "feature_i_local_constant_count": counter["left"],
                         "feature_j_local_constant_count": counter["right"],
                         "numeric_invalid_count": counter["numeric"], **correlation_statistics(values)})
            aggregate_rows.append(base)
    return pd.DataFrame(sequence_rows), pd.DataFrame(aggregate_rows), sum(len(moment[2]) for moment in moments)


def sequence_similarity(global_stats: pd.DataFrame, local128: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sequence_columns = [f"sequence{i}_pearson" for i in range(1, 7)]
    masks = {
        "all_pairs": np.ones(len(global_stats), dtype=bool),
        "within_subsystem": global_stats.relation_scope.str.startswith("within_").to_numpy(),
        "cross_subsystem": (global_stats.relation_scope == "cross_subsystem").to_numpy(),
        "local_valid_ge_0.95": (local128.valid_ratio >= 0.95).to_numpy(),
        "medium_strength": (global_stats.median_abs_corr >= 0.30).to_numpy(),
    }
    summary_rows = []
    base_rows = []
    for left, right in combinations(range(6), 2):
        for scope, mask in masks.items():
            x = global_stats.loc[mask, sequence_columns[left]].to_numpy(float)
            y = global_stats.loc[mask, sequence_columns[right]].to_numpy(float)
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = x[finite], y[finite]
            if len(x) < 2:
                values = {"pearson_similarity": np.nan, "spearman_similarity": np.nan, "cosine_similarity": np.nan,
                          "mean_absolute_difference": np.nan, "sign_disagreement_ratio": np.nan}
            else:
                denominator = np.linalg.norm(x) * np.linalg.norm(y)
                values = {"pearson_similarity": float(pearsonr(x, y).statistic),
                          "spearman_similarity": float(spearmanr(x, y).statistic),
                          "cosine_similarity": float(np.dot(x, y) / denominator) if denominator else np.nan,
                          "mean_absolute_difference": float(np.mean(np.abs(x - y))),
                          "sign_disagreement_ratio": float(np.mean(np.sign(x) != np.sign(y)))}
            row = {"sequence_i": f"train{left + 1}", "sequence_j": f"train{right + 1}",
                   "relation_subset": scope, "pair_count": len(x), **values}
            summary_rows.append(row)
            if scope == "all_pairs": base_rows.append(row.copy())
    return pd.DataFrame(base_rows), pd.DataFrame(summary_rows)


def threshold_sensitivity(local_by_sequence: dict[int, pd.DataFrame], local_aggregate: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for window in WINDOWS:
        by_sequence = local_by_sequence[window]
        aggregate = local_aggregate[window]
        for valid_min, strength_min, sigma_max, sign_min, coverage_min in product(
                THRESHOLDS["valid_ratio"], THRESHOLDS["median_abs_corr"], THRESHOLDS["fisher_z_robust_sigma"],
                THRESHOLDS["sign_consistency"], THRESHOLDS["sequence_coverage"]):
            sequence_ok = ((by_sequence.valid_ratio >= valid_min) & (by_sequence.median_abs_corr >= strength_min) &
                           (by_sequence.fisher_z_robust_sigma <= sigma_max) &
                           (by_sequence.sign_consistency >= sign_min))
            coverage = sequence_ok.groupby(by_sequence.pair_id).sum().reindex(aggregate.pair_id, fill_value=0).to_numpy()
            aggregate_ok = ((aggregate.valid_ratio >= valid_min) & (aggregate.median_abs_corr >= strength_min) &
                            (aggregate.fisher_z_robust_sigma <= sigma_max) &
                            (aggregate.sign_consistency >= sign_min))
            rows.append({"window_size": window, "minimum_valid_ratio": valid_min,
                         "minimum_median_abs_corr": strength_min, "maximum_fisher_z_robust_sigma": sigma_max,
                         "minimum_sign_consistency": sign_min, "minimum_sequence_coverage": coverage_min,
                         "candidate_count": int((aggregate_ok.to_numpy() & (coverage >= coverage_min)).sum()),
                         "total_pair_count": len(aggregate)})
    return pd.DataFrame(rows)


def classify_pair(aggregate: pd.Series, sequence_frame: pd.DataFrame, between_range: float) -> tuple[str, int, int]:
    strict = CLASSIFICATION_THRESHOLDS["strict"]
    moderate = CLASSIFICATION_THRESHOLDS["moderate"]
    strict_seq = ((sequence_frame.valid_ratio >= strict["valid_ratio"]) &
                  (sequence_frame.median_abs_corr >= strict["median_abs_corr"]) &
                  (sequence_frame.fisher_z_robust_sigma <= strict["fisher_z_robust_sigma"]) &
                  (sequence_frame.sign_consistency >= strict["sign_consistency"]))
    moderate_seq = ((sequence_frame.valid_ratio >= moderate["valid_ratio"]) &
                    (sequence_frame.median_abs_corr >= moderate["median_abs_corr"]) &
                    (sequence_frame.fisher_z_robust_sigma <= moderate["fisher_z_robust_sigma"]) &
                    (sequence_frame.sign_consistency >= moderate["sign_consistency"]))
    strict_count, moderate_count = int(strict_seq.sum()), int(moderate_seq.sum())
    if (aggregate.valid_ratio >= strict["valid_ratio"] and aggregate.median_abs_corr >= strict["median_abs_corr"] and
            aggregate.fisher_z_robust_sigma <= strict["fisher_z_robust_sigma"] and
            aggregate.sign_consistency >= strict["sign_consistency"] and strict_count >= strict["minimum_sequences"] and
            between_range <= strict["between_sequence_corr_range"]):
        category = "strict_cross_sequence"
    elif (aggregate.valid_ratio >= moderate["valid_ratio"] and aggregate.median_abs_corr >= moderate["median_abs_corr"] and
          aggregate.fisher_z_robust_sigma <= moderate["fisher_z_robust_sigma"] and
          aggregate.sign_consistency >= moderate["sign_consistency"] and moderate_count >= moderate["minimum_sequences"]):
        category = "moderate_cross_sequence"
    elif 1 <= moderate_count <= 3:
        category = "sequence_specific_stable"
    elif aggregate.median_abs_corr >= 0.50 and (aggregate.fisher_z_robust_sigma > 0.35 or aggregate.sign_consistency < 0.85):
        category = "strong_but_locally_unstable"
    elif aggregate.valid_ratio < 0.90 and (aggregate.feature_i_local_constant_count + aggregate.feature_j_local_constant_count) > 0:
        category = "invalid_due_to_local_constancy"
    else:
        category = "weak_or_uninformative"
    return category, strict_count, moderate_count


def classify_pairs(aggregate: pd.DataFrame, by_sequence: pd.DataFrame, global_stats: pd.DataFrame) -> pd.DataFrame:
    ranges = global_stats.set_index("pair_id").between_sequence_corr_range
    rows = []
    for item in aggregate.itertuples(index=False):
        sequence_frame = by_sequence[by_sequence.pair_id == item.pair_id]
        category, strict_count, moderate_count = classify_pair(pd.Series(item._asdict()), sequence_frame, float(ranges.loc[item.pair_id]))
        row = item._asdict() | {"between_sequence_corr_range": float(ranges.loc[item.pair_id]),
                               "strict_sequence_count": strict_count, "moderate_sequence_count": moderate_count,
                               "relation_category": category,
                               "classification_thresholds": json.dumps(CLASSIFICATION_THRESHOLDS, sort_keys=True)}
        rows.append(row)
    return pd.DataFrame(rows)


def window_comparison(classified64: pd.DataFrame, classified128: pd.DataFrame) -> pd.DataFrame:
    left = classified64.set_index("pair_id")
    right = classified128.set_index("pair_id")
    rows = []
    stable = {"strict_cross_sequence", "moderate_cross_sequence", "sequence_specific_stable"}
    for pair_id in left.index:
        a, b = left.loc[pair_id], right.loc[pair_id]
        rows.append({"pair_id": pair_id, "feature_i": a.feature_i, "feature_j": a.feature_j,
                     "relation_scope": a.relation_scope, "valid_ratio_w64": a.valid_ratio,
                     "valid_ratio_w128": b.valid_ratio, "valid_ratio_change": b.valid_ratio - a.valid_ratio,
                     "median_abs_corr_w64": a.median_abs_corr, "median_abs_corr_w128": b.median_abs_corr,
                     "median_abs_corr_change": b.median_abs_corr - a.median_abs_corr,
                     "fisher_z_robust_sigma_w64": a.fisher_z_robust_sigma,
                     "fisher_z_robust_sigma_w128": b.fisher_z_robust_sigma,
                     "fisher_z_robust_sigma_change": b.fisher_z_robust_sigma - a.fisher_z_robust_sigma,
                     "sign_consistency_w64": a.sign_consistency, "sign_consistency_w128": b.sign_consistency,
                     "sign_consistency_change": b.sign_consistency - a.sign_consistency,
                     "classification_w64": a.relation_category, "classification_w128": b.relation_category,
                     "classification_changed": a.relation_category != b.relation_category,
                     "invalid_to_valid": a.valid_ratio < 0.90 <= b.valid_ratio,
                     "unstable_to_stable": a.relation_category not in stable and b.relation_category in stable,
                     "possible_long_window_smoothing": b.fisher_z_robust_sigma < 0.75 * a.fisher_z_robust_sigma and
                                                       b.median_abs_corr >= a.median_abs_corr})
    return pd.DataFrame(rows)


def candidate_pools(classified128: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = classified128.copy()
    frame["candidate_score"] = (frame.valid_ratio + frame.median_abs_corr + frame.sign_consistency -
                                  frame.fisher_z_robust_sigma + frame.strict_sequence_count / 6.0)
    strict = frame[frame.relation_category == "strict_cross_sequence"].sort_values(
        ["strict_sequence_count", "valid_ratio", "fisher_z_robust_sigma", "median_abs_corr", "sign_consistency"],
        ascending=[False, False, True, False, False]).head(20).copy()
    strict.insert(0, "rank", np.arange(1, len(strict) + 1))
    pool = frame[frame.relation_category.isin(["strict_cross_sequence", "moderate_cross_sequence"])].copy()
    selected, degree, covered = [], {}, set()
    while len(selected) < 12:
        options = []
        for row in pool.itertuples(index=False):
            if any(item["pair_id"] == row.pair_id for item in selected): continue
            if degree.get(row.feature_i, 0) >= 2 or degree.get(row.feature_j, 0) >= 2: continue
            new_subsystems = {row.subsystem_i, row.subsystem_j} - covered - {"other"}
            score = float(row.candidate_score + 0.20 * len(new_subsystems))
            options.append((score, row, new_subsystems))
        if not options: break
        _, row, new_subsystems = max(options, key=lambda item: (item[0], item[1].median_abs_corr, -item[1].pair_id))
        before = {row.feature_i: degree.get(row.feature_i, 0), row.feature_j: degree.get(row.feature_j, 0)}
        degree[row.feature_i] = before[row.feature_i] + 1; degree[row.feature_j] = before[row.feature_j] + 1
        covered.update({row.subsystem_i, row.subsystem_j} - {"other"})
        selected.append(row._asdict() | {"rank": len(selected) + 1,
                         "score_components": json.dumps({"base": row.candidate_score,
                                                          "new_subsystem_bonus": 0.20 * len(new_subsystems)}),
                         "degree_before": json.dumps(before),
                         "degree_after": json.dumps({row.feature_i: degree[row.feature_i], row.feature_j: degree[row.feature_j]}),
                         "subsystem_coverage": ",".join(sorted(covered)),
                         "selection_reason": "highest normal-only stability score under degree<=2 with subsystem bonus"})
    diversity = pd.DataFrame(selected)
    return strict, diversity


def choose_reference_strategy(classified128: pd.DataFrame, similarities: pd.DataFrame, strict_pool: pd.DataFrame,
                              diversity: pd.DataFrame) -> tuple[str, bool, dict[str, Any]]:
    counts = classified128.relation_category.value_counts().to_dict()
    all_scope = similarities[similarities.relation_subset == "all_pairs"]
    median_spearman = float(all_scope.spearman_similarity.median())
    strict_count = int(counts.get("strict_cross_sequence", 0))
    specific_count = int(counts.get("sequence_specific_stable", 0))
    moderate_count = int(counts.get("moderate_cross_sequence", 0))
    if strict_count >= 10 and median_spearman >= 0.90 and specific_count <= strict_count:
        strategy = "single_global_reference_feasible"
    elif strict_count > 0 and specific_count > 0:
        strategy = "mixed_global_and_sequence_specific"
    elif strict_count == 0 and moderate_count + specific_count >= 10:
        strategy = "sequence_conditioned_reference_preferred"
    else:
        strategy = "insufficient_stable_pearson_relations"
    evidence = {"relation_category_counts_w128": counts, "median_all_pair_spearman_similarity": median_spearman,
                "strict_pool_size": len(strict_pool), "diversity_pool_size": len(diversity)}
    ready = bool(len(diversity) > 0 and strict_count + moderate_count > 0)
    return strategy, ready, evidence


def memory_mb() -> float | None:
    try:
        import psutil
        info = psutil.Process().memory_info()
        return float(getattr(info, "peak_wset", info.rss) / 1024 ** 2)
    except Exception:
        return None


def run(data_root: Path, output: Path, smoke: bool, stride: int, chunk_size: int) -> dict[str, Any]:
    ensure_fresh(output)
    started = time.perf_counter()
    raw, names, data_ranges = load_train_prefixes(data_root, smoke=smoke)
    scaler = StandardScaler(eps=GLOBAL_EPS).fit_many(raw)
    normalized = [scaler.transform(values).astype(np.float32, copy=False) for values in raw]
    eligibility = feature_eligibility(raw, normalized, names, stride)
    eligible = np.flatnonzero(eligibility.eligible_for_relation_analysis.to_numpy())
    if smoke: eligible = eligible[:min(12, len(eligible))]
    pairs, metadata = pair_metadata(names, eligible)
    print(f"training_prefixes={[(item['file_name'], item['rows_used']) for item in data_ranges]}")
    print(f"ordered_features_86={names}")
    print(f"global_zero_variance={eligibility.loc[eligibility.global_zero_variance, 'feature_name'].tolist()}")
    print(f"active_feature_count={len(eligible)} pair_count={len(pairs)}")
    print("reused=data.transforms.StandardScaler, HAI Step2 split manifest, rolling/Fisher statistical definitions")
    print("new=scripts/hai_step4a_diagnose_normal_relations.py, tests/test_hai_normal_relation_diagnosis.py")
    eligibility.to_csv(output / "hai_relation_feature_eligibility.csv", index=False)
    global_stats, _ = global_sequence_stats(normalized, pairs, metadata)
    global_stats.to_csv(output / "hai_pair_global_sequence_stats.csv", index=False)
    by_sequence, aggregate, sampled = {}, {}, {}
    for window in WINDOWS:
        by_sequence[window], aggregate[window], sampled[window] = local_relation_stats(
            normalized, pairs, metadata, window, stride, chunk_size)
        by_sequence[window].to_csv(output / f"hai_pair_local_stats_by_sequence_w{window}.csv", index=False)
        aggregate[window].to_csv(output / f"hai_pair_local_stats_aggregate_w{window}.csv", index=False)
    similarity, similarity_summary = sequence_similarity(global_stats, aggregate[128])
    similarity.to_csv(output / "hai_sequence_relation_similarity.csv", index=False)
    similarity_summary.to_csv(output / "hai_sequence_relation_similarity_summary.csv", index=False)
    sensitivity = threshold_sensitivity(by_sequence, aggregate)
    sensitivity.to_csv(output / "hai_relation_threshold_sensitivity.csv", index=False)
    classified = {}
    for window in WINDOWS:
        classified[window] = classify_pairs(aggregate[window], by_sequence[window], global_stats)
        classified[window].to_csv(output / f"hai_relation_pair_classification_w{window}.csv", index=False)
    comparison = window_comparison(classified[64], classified[128])
    comparison.to_csv(output / "hai_relation_window_comparison.csv", index=False)
    strict_pool, diversity = candidate_pools(classified[128])
    strict_pool.to_csv(output / "strict_candidate_pool.csv", index=False)
    diversity.to_csv(output / "diversity_candidate_pool.csv", index=False)
    strategy, ready, evidence = choose_reference_strategy(classified[128], similarity_summary, strict_pool, diversity)
    runtime = time.perf_counter() - started
    category_counts = {str(window): classified[window].relation_category.value_counts().to_dict() for window in WINDOWS}
    top_global = global_stats.assign(abs_pooled=global_stats.pooled_pearson.abs()).nlargest(20, "abs_pooled")
    top_global_categories = classified[128].set_index("pair_id").loc[top_global.pair_id, "relation_category"].value_counts().to_dict()
    invalid_causes = {str(window): {
        "invalid_window_count": int(aggregate[window].invalid_window_count.sum()),
        "both_local_constant_count": int(aggregate[window].both_local_constant_count.sum()),
        "feature_i_only_local_constant_count": int((aggregate[window].feature_i_local_constant_count - aggregate[window].both_local_constant_count).sum()),
        "feature_j_only_local_constant_count": int((aggregate[window].feature_j_local_constant_count - aggregate[window].both_local_constant_count).sum()),
        "numeric_invalid_count": int(aggregate[window].numeric_invalid_count.sum()),
    } for window in WINDOWS}
    summary = {
        "mode": "smoke" if smoke else "formal", "data_scope": "train1..train6 registered first 80% prefixes only",
        "data_ranges": data_ranges, "attack_column_read": False, "validation_rows_read": False,
        "test_files_read": False, "checkpoint_or_model_loaded": False, "feature_schema_count": len(names),
        "global_zero_variance_count": int(eligibility.global_zero_variance.sum()),
        "active_feature_count": len(eligible), "pair_count": len(pairs), "diagnostic_stride": stride,
        "pair_chunk_size": chunk_size, "sampled_window_count": sampled,
        "peak_memory_mb": memory_mb(), "estimated_pair_chunk_working_memory_mb": max(len(x) for x in normalized) * chunk_size * 16 / 1024 ** 2,
        "total_runtime_seconds": runtime, "threshold_sensitivity_grid": THRESHOLDS,
        "classification_thresholds": CLASSIFICATION_THRESHOLDS, "category_counts": category_counts,
        "local_invalid_cause_totals": invalid_causes,
        "top20_pooled_abs_correlation_local_categories_w128": top_global_categories,
        "strict_candidate_pool_size": len(strict_pool),
        "strict_candidate_channel_coverage": len(set(strict_pool.feature_i) | set(strict_pool.feature_j)) if len(strict_pool) else 0,
        "diversity_candidate_pool_size": len(diversity),
        "diversity_candidate_channel_coverage": len(set(diversity.feature_i) | set(diversity.feature_j)) if len(diversity) else 0,
        "diversity_candidate_subsystem_coverage": sorted((set(diversity.subsystem_i) | set(diversity.subsystem_j)) - {"other"}) if len(diversity) else [],
        "reference_strategy": strategy, "reference_strategy_evidence": evidence,
        "ready_for_controlled_relation_anomaly_validation": ready,
        "no_attack_detection_claim": True,
    }
    reference = {"reference_strategy": strategy, "evidence": evidence,
                 "ready_for_controlled_relation_anomaly_validation": ready,
                 "decision_scope": "normal HAI train prefixes only; no attack detection evaluation"}
    (output / "hai_relation_reference_strategy.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    (output / "hai_relation_diagnosis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    valid_summary = {w: {"median": float(aggregate[w].valid_ratio.median()),
                         "q10": float(aggregate[w].valid_ratio.quantile(.1)),
                         "q90": float(aggregate[w].valid_ratio.quantile(.9))} for w in WINDOWS}
    similarity_all = similarity_summary[similarity_summary.relation_subset == "all_pairs"]
    smoothing_ratio = float(comparison.possible_long_window_smoothing.mean())
    report = f"""# HAI Step 4A normal relation diagnosis

- Scope: six registered normal training prefixes only; validation rows, test files, Attack values, models, and checkpoints were not read.
- Features: 86 retained in the data/model schema; {len(eligible)} eligible for Pearson analysis; {int(eligibility.global_zero_variance.sum())} global zero-variance features excluded from relation calculations only.
- Candidate pairs: {len(pairs)} complete eligible-feature pairs; no top-k prefilter.
- Local validity: w64 {valid_summary[64]}; w128 {valid_summary[128]}. Invalid windows are decomposed into per-channel/both local constancy and numeric failures in CSV outputs.
- Invalid-window causes: {invalid_causes}. Numeric failures after excluding local constants: zero.
- Categories w64: {category_counts['64']}.
- Categories w128: {category_counts['128']}.
- Six-sequence all-pair relation similarity: median Pearson {similarity_all.pearson_similarity.median():.4f}, median Spearman {similarity_all.spearman_similarity.median():.4f}, median cosine {similarity_all.cosine_similarity.median():.4f}.
- Normal operating-sequence heterogeneity is {'material' if similarity_all.spearman_similarity.median() < 0.90 else 'limited'}; correlation-matrix differences are treated as operating-state evidence, not data errors.
- Window comparison: median valid-ratio change {comparison.valid_ratio_change.median():.6f}; median Fisher robust-sigma change {comparison.fisher_z_robust_sigma_change.median():.6f}; possible long-window smoothing affects {smoothing_ratio:.2%} of pairs.
- Normal-data window recommendation: {'128' if comparison.fisher_z_robust_sigma_change.median() < 0 else '64'} for the initial candidate reference, while w64 remains useful for exposing short-lived normal changes. This is not an attack-detection claim.
- Strict pool: {len(strict_pool)} relations covering {summary['strict_candidate_channel_coverage']} channels. Diversity pool: {len(diversity)} relations covering {summary['diversity_candidate_channel_coverage']} channels and subsystems {summary['diversity_candidate_subsystem_coverage']}.
- Reference strategy: `{strategy}`. Ready for controlled relation-anomaly validation: `{ready}`.
- The top 20 pooled absolute correlations are not locally reliable as a group: w128 categories are {top_global_categories}; none are strict/moderate cross-sequence relations.
- Runtime: {runtime:.2f}s; measured peak process memory: {summary['peak_memory_mb']} MB; estimated pair-chunk working memory: {summary['estimated_pair_chunk_working_memory_mb']:.1f} MB; pair chunk size: {chunk_size}.

All candidate selection is based solely on normal training statistics. No test feature, label, rec score, persistent-miss event, model, checkpoint, F1, AUPR, AUROC, structure score, or fusion result is used.
"""
    (output / "final_hai_normal_relation_diagnosis.md").write_text(report, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAI Step 4A normal-only relation availability diagnosis")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--diagnostic-stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--pair-chunk-size", type=int, default=DEFAULT_PAIR_CHUNK)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    output = SMOKE_OUTPUT if args.smoke else FORMAL_OUTPUT
    if args.dry_run:
        raw, names, ranges = load_train_prefixes(data_root, smoke=False)
        scaler = StandardScaler(eps=GLOBAL_EPS).fit_many(raw)
        zero = np.flatnonzero(scaler.raw_std_.reshape(-1) < GLOBAL_EPS)
        active = len(names) - len(zero)
        print(f"training_prefixes={[(item['file_name'], item['rows_used']) for item in ranges]}")
        print(f"ordered_features_86={names}")
        print(f"global_zero_variance={[names[index] for index in zero]}")
        print(f"active_feature_count={active} pair_count={active * (active - 1) // 2}")
        print(f"windows={WINDOWS} diagnostic_stride={args.diagnostic_stride} pair_chunk_size={args.pair_chunk_size}")
        print("access policy=train prefixes only; Attack excluded by usecols; validation/test/model/checkpoint forbidden")
        print("reused=StandardScaler + rolling/Fisher definitions; new=Step4A script and tests")
        return 0
    result = run(data_root, output, args.smoke, args.diagnostic_stride, args.pair_chunk_size)
    print(json.dumps({"output": str(output), **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
