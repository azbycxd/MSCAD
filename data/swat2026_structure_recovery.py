from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from data.swat2026_structure import rolling_pair_correlations, stable_pairs, window_batch_pair_correlations
from data.swat2026_training import EXPECTED_SOURCE_SHA256, sha256_file


TRAIN_START = pd.Timestamp("2026-03-11 09:00:00")
TRAIN_END = pd.Timestamp("2026-03-11 12:10:18")
VALIDATION_START = TRAIN_END
VALIDATION_END = pd.Timestamp("2026-03-11 12:57:53")
TEST_CONTEXT_START = VALIDATION_END
TEST_START = pd.Timestamp("2026-03-11 13:00:00")
TEST_END = pd.Timestamp("2026-03-11 17:00:00")
LEGACY_EXPANDED3 = (
    "PIT501.Pv__PIT503.Pv", "FIT501.Pv__PIT503.Pv", "FIT501.Pv__PIT501.Pv",
)
REPRESENTATIONS = ("level_pearson", "delta_pearson", "delta_spearman")


@dataclass
class RecoveryData:
    data_path: Path
    source_sha256: str
    timestamps: pd.DatetimeIndex
    feature_names: list[str]
    raw_features: np.ndarray
    scaled_features: np.ndarray
    valid_points: np.ndarray
    scaler_metadata: dict[str, Any]


def load_recovery_data(data_path: str | Path, m1_dir: str | Path) -> RecoveryData:
    path, m1 = Path(data_path), Path(m1_dir)
    source_hash = sha256_file(path)
    scaler = json.loads((m1 / "scaler_metadata.json").read_text(encoding="utf-8-sig"))
    if source_hash != EXPECTED_SOURCE_SHA256 or source_hash != scaler["source_csv_sha256"]:
        raise ValueError("Official source CSV hash changed")
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = frame.columns.astype(str).str.strip()
    timestamp_column = next(name for name in frame if name.strip().lower() in {"t_stamp", "timestamp", "datetime", "time", "date"})
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame[timestamp_column], format="%d-%b-%Y %H:%M:%S", errors="raise"))
    names = list(scaler["feature_order"])
    raw = frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(raw).all(axis=1)
    scaled = ((raw - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])).astype(np.float32)
    return RecoveryData(path, source_hash, timestamps, names, raw, scaled, valid, scaler)


def train_relation_split(timestamps: pd.DatetimeIndex, window_size: int = 128,
                         discovery_ratio: float = .70) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train = np.flatnonzero(np.asarray((timestamps >= TRAIN_START) & (timestamps < TRAIN_END)))
    discovery_count = int(np.floor(len(train) * discovery_ratio))
    guard_count = window_size - 1
    discovery = train[:discovery_count]
    guard = train[discovery_count : discovery_count + guard_count]
    confirmation = train[discovery_count + guard_count :]
    if set(discovery) & set(confirmation):
        raise ValueError("Discovery and confirmation timestamps overlap")
    discovery_ends = discovery[window_size - 1 :]
    confirmation_ends = confirmation[window_size - 1 :]
    manifest = {
        "train_raw_points": len(train), "discovery_ratio": discovery_ratio,
        "relation_discovery": {"start": timestamps[discovery[0]].isoformat(),
                               "end_exclusive": (timestamps[discovery[-1]] + pd.Timedelta(seconds=1)).isoformat(),
                               "raw_points": len(discovery), "candidate_windows": len(discovery_ends)},
        "boundary_guard": {"start": timestamps[guard[0]].isoformat(),
                           "end_exclusive": (timestamps[guard[-1]] + pd.Timedelta(seconds=1)).isoformat(),
                           "raw_points": len(guard)},
        "relation_confirmation": {"start": timestamps[confirmation[0]].isoformat(),
                                  "end_exclusive": (timestamps[confirmation[-1]] + pd.Timedelta(seconds=1)).isoformat(),
                                  "raw_points": len(confirmation), "candidate_windows": len(confirmation_ends)},
        "raw_timestamp_overlap": False, "windows_cross_boundary": False, "windows_cross_csv": False,
    }
    return manifest, {"discovery": discovery, "guard": guard, "confirmation": confirmation,
                      "discovery_ends": discovery_ends, "confirmation_ends": confirmation_ends}


def validation_usage_split(timestamps: pd.DatetimeIndex, window_size: int = 128) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    validation = np.flatnonzero(np.asarray((timestamps >= VALIDATION_START) & (timestamps < VALIDATION_END)))
    usable_raw = len(validation) - (window_size - 1)
    part_raw_count = usable_raw // 2
    threshold_raw = validation[:part_raw_count]
    guard = validation[part_raw_count : part_raw_count + window_size - 1]
    controlled_raw = validation[part_raw_count + window_size - 1 : part_raw_count + window_size - 1 + part_raw_count]
    threshold_ends = threshold_raw[window_size - 1 :]
    controlled_ends = controlled_raw[window_size - 1 :]
    if set(threshold_raw) & set(controlled_raw):
        raise ValueError("Validation threshold and controlled raw timestamps overlap")
    manifest = {
        "validation_raw_points": len(validation), "threshold_raw_points": len(threshold_raw),
        "boundary_guard_points": len(guard), "controlled_raw_points": len(controlled_raw),
        "threshold_window_count": len(threshold_ends), "controlled_window_count": len(controlled_ends),
        "threshold_start": timestamps[threshold_raw[0]].isoformat(),
        "threshold_end_exclusive": (timestamps[threshold_raw[-1]] + pd.Timedelta(seconds=1)).isoformat(),
        "guard_start": timestamps[guard[0]].isoformat(),
        "guard_end_exclusive": (timestamps[guard[-1]] + pd.Timedelta(seconds=1)).isoformat(),
        "controlled_start": timestamps[controlled_raw[0]].isoformat(),
        "controlled_end_exclusive": (timestamps[controlled_raw[-1]] + pd.Timedelta(seconds=1)).isoformat(),
        "raw_timestamp_overlap": False, "injected_samples_used_for_threshold": False,
    }
    return manifest, {"threshold_raw": threshold_raw, "guard": guard, "controlled_raw": controlled_raw,
                      "threshold_ends": threshold_ends, "controlled_ends": controlled_ends}


def _windows(values: np.ndarray, ends: np.ndarray, window_size: int = 128) -> np.ndarray:
    return np.stack([values[end - window_size + 1 : end + 1] for end in ends]).astype(np.float64)


def representation_correlations(values: np.ndarray, validity: np.ndarray, ends: np.ndarray,
                                pairs: pd.DataFrame, representation: str, window_size: int = 128,
                                batch_size: int = 256) -> np.ndarray:
    ends = np.asarray(ends, dtype=np.int64)
    if representation == "level_pearson":
        return rolling_pair_correlations(values, ends, pairs, window_size, validity)[0]
    if representation == "delta_pearson":
        delta = np.diff(values.astype(np.float64), axis=0)
        delta_valid = np.asarray(validity[1:] & validity[:-1], dtype=bool)
        return rolling_pair_correlations(delta, ends - 1, pairs, window_size - 1, delta_valid)[0]
    if representation != "delta_spearman":
        raise ValueError(f"Unsupported representation: {representation}")
    output = []
    for start in range(0, len(ends), batch_size):
        current_ends = ends[start : start + batch_size]
        windows = _windows(values, current_ends, window_size)
        valid = np.array([validity[end - window_size + 1 : end + 1].all() for end in current_ends])
        delta = np.diff(windows, axis=1)
        ranks = rankdata(delta, axis=1, method="average")
        correlations = window_batch_pair_correlations(ranks, pairs)
        correlations[~valid] = np.nan
        output.append(correlations)
    return np.concatenate(output, axis=0)


def batch_representation_correlations(windows: np.ndarray, pairs: pd.DataFrame, representation: str) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float64)
    if representation == "level_pearson":
        transformed = values
    elif representation == "delta_pearson":
        transformed = np.diff(values, axis=1)
    elif representation == "delta_spearman":
        transformed = rankdata(np.diff(values, axis=1), axis=1, method="average")
    else:
        raise ValueError(f"Unsupported representation: {representation}")
    return window_batch_pair_correlations(transformed, pairs)


def _stats(values: np.ndarray, candidate_windows: int) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"valid_window_count": 0, "valid_window_ratio": 0.0, "median_corr": np.nan,
                "median_abs_corr": np.nan, "corr_mad": np.nan, "corr_iqr": np.nan,
                "sign_consistency": 0.0, "q25": np.nan, "q75": np.nan}
    median = float(np.median(finite))
    return {"valid_window_count": len(finite), "valid_window_ratio": len(finite) / max(candidate_windows, 1),
            "median_corr": median, "median_abs_corr": float(np.median(np.abs(finite))),
            "corr_mad": float(np.median(np.abs(finite - median))),
            "corr_iqr": float(np.quantile(finite, .75) - np.quantile(finite, .25)),
            "sign_consistency": float(max((finite >= 0).mean(), (finite <= 0).mean())),
            "q25": float(np.quantile(finite, .25)), "q75": float(np.quantile(finite, .75))}


def reproducibility_diagnostics(pairs: pd.DataFrame, discovery: np.ndarray, confirmation: np.ndarray,
                                representation: str, cfg: dict[str, float]) -> pd.DataFrame:
    rows = []
    for index, pair in enumerate(pairs.itertuples(index=False)):
        first = _stats(discovery[:, index], len(discovery))
        second = _stats(confirmation[:, index], len(confirmation))
        finite = np.concatenate([discovery[np.isfinite(discovery[:, index]), index],
                                 confirmation[np.isfinite(confirmation[:, index]), index]])
        pooled_scale = float(1.4826 * max(first["corr_mad"], second["corr_mad"])) if len(finite) else np.nan
        gap = abs(first["median_corr"] - second["median_corr"]) if np.isfinite(first["median_corr"] + second["median_corr"]) else np.inf
        sign_match = bool(np.sign(first["median_corr"]) == np.sign(second["median_corr"]) and
                          np.sign(first["median_corr"]) != 0)
        overlap = bool(max(first["q25"], second["q25"]) <= min(first["q75"], second["q75"])) if len(finite) else False
        activity = min(first["valid_window_ratio"], second["valid_window_ratio"])
        reproducible = bool(
            first["valid_window_ratio"] >= cfg["valid_window_ratio"] and
            second["valid_window_ratio"] >= cfg["valid_window_ratio"] and
            first["median_abs_corr"] >= cfg["median_abs_corr"] and
            second["median_abs_corr"] >= cfg["median_abs_corr"] and
            first["sign_consistency"] >= cfg["sign_consistency"] and
            second["sign_consistency"] >= cfg["sign_consistency"] and
            first["corr_iqr"] <= cfg["corr_iqr"] and second["corr_iqr"] <= cfg["corr_iqr"] and
            sign_match and gap <= max(cfg["absolute_median_gap"], cfg["pooled_scale_multiplier"] * pooled_scale) and
            activity >= cfg["activity_valid_ratio"]
        )
        row = {**pair._asdict(), "representation": representation,
               **{f"discovery_{key}": value for key, value in first.items()},
               **{f"confirmation_{key}": value for key, value in second.items()},
               "activity_valid_ratio": activity, "discovery_confirmation_median_gap": gap,
               "pooled_robust_scale": pooled_scale, "discovery_confirmation_sign_match": sign_match,
               "reference_interval_overlap": overlap, "reproducible_pass": reproducible,
               "legacy_expanded3": pair.pair_name in LEGACY_EXPANDED3}
        rows.append(row)
    return pd.DataFrame(rows)


def select_reproducible_sets(diagnostics: pd.DataFrame, max_pairs: int = 20,
                             diverse_max_pairs: int = 10, max_degree: int = 2) -> dict[str, pd.DataFrame]:
    selected = diagnostics[diagnostics.reproducible_pass].copy()
    selected["minimum_median_abs_corr"] = selected[["discovery_median_abs_corr", "confirmation_median_abs_corr"]].min(axis=1)
    selected["maximum_corr_iqr"] = selected[["discovery_corr_iqr", "confirmation_corr_iqr"]].max(axis=1)
    selected = selected.sort_values(
        ["confirmation_valid_window_ratio", "minimum_median_abs_corr", "discovery_confirmation_median_gap",
         "maximum_corr_iqr", "pair_name"], ascending=[False, False, True, True, True], kind="mergesort",
    ).head(max_pairs).reset_index(drop=True)
    degree: Counter[str] = Counter(); diverse_rows = []
    for _, row in selected.iterrows():
        if degree[row.channel_i] >= max_degree or degree[row.channel_j] >= max_degree:
            continue
        diverse_rows.append(row.to_dict()); degree[row.channel_i] += 1; degree[row.channel_j] += 1
        if len(diverse_rows) >= diverse_max_pairs:
            break
    diverse = pd.DataFrame(diverse_rows, columns=selected.columns)
    status = "general_structure_candidate" if len(selected) >= 3 else (
        "narrow_relation_diagnostic" if len(selected) else "no_reproducible_relations")
    return {"reproducible_set": selected, "diverse_set": diverse, "status": status}


def select_primary(controlled: pd.DataFrame, representation_counts: dict[str, int],
                   minimum_auroc: float = .80) -> dict[str, Any]:
    if controlled.empty:
        return {"primary_frozen": False, "representation": None, "selection_used_test": False}
    eligible = controlled[(controlled.score_name == "representation_max") &
                          controlled.representation.map(representation_counts).ge(3) &
                          controlled.marginal_statistics_preserved &
                          controlled.reference_valid &
                          controlled.auroc.ge(minimum_auroc)].copy()
    if eligible.empty:
        return {"primary_frozen": False, "representation": None, "selection_used_test": False}
    best = float(eligible.auroc.max())
    shortlist = eligible[eligible.auroc >= best - .01].copy()
    simplicity = {"level_pearson": 0, "delta_pearson": 1, "delta_spearman": 2}
    shortlist["simplicity_rank"] = shortlist.representation.map(simplicity)
    row = shortlist.sort_values(["simplicity_rank", "representation"]).iloc[0]
    return {"primary_frozen": True, "representation": row.representation,
            "controlled_auroc": float(row.auroc), "selection_used_test": False,
            "selection_rule": "highest controlled AUROC; within 0.01 prefer level > delta Pearson > delta Spearman"}
