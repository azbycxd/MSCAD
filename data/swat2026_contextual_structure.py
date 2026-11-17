from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from data.swat2026_structure import FISHER_CLIP


LAG_SECONDS = (-30, -20, -10, -5, 0, 5, 10, 20, 30)
REPRESENTATIONS = ("lagged_level", "lagged_delta", "conditioned_level", "conditioned_delta")
LEGACY_NARROW = "PIT501.Pv__PIT503.Pv"


def _rolling_sum(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate([[0.0], np.cumsum(np.asarray(values, dtype=np.float64))])
    return cumulative[ends + 1] - cumulative[starts]


def lagged_pair_correlations(values: np.ndarray, validity: np.ndarray, ends: np.ndarray,
                             pairs: pd.DataFrame, lag: int, representation: str,
                             window_size: int = 128) -> np.ndarray:
    """Frozen-lag Pearson correlation. Positive lag means Xi leads Xj."""
    if int(lag) not in LAG_SECONDS:
        raise ValueError("lag must belong to the prespecified lag set")
    if representation not in {"lagged_level", "lagged_delta"}:
        raise ValueError(f"unsupported lagged representation: {representation}")
    x = np.asarray(values, dtype=np.float64)
    valid = np.asarray(validity, dtype=bool)
    requested_ends = np.asarray(ends, dtype=np.int64)
    if representation == "lagged_delta":
        x = np.diff(x, axis=0)
        valid = valid[1:] & valid[:-1]
        requested_ends = requested_ends - 1
        base_window = window_size - 1
    else:
        base_window = window_size
    distance = abs(int(lag))
    effective = base_window - distance
    if effective < 2:
        raise ValueError("lag leaves fewer than two aligned observations")
    if lag > 0:
        left, right, aligned_ends = x[:-distance], x[distance:], requested_ends - distance
        left_valid, right_valid = valid[:-distance], valid[distance:]
    elif lag < 0:
        left, right, aligned_ends = x[distance:], x[:-distance], requested_ends - distance
        left_valid, right_valid = valid[distance:], valid[:-distance]
    else:
        left = right = x
        left_valid = right_valid = valid
        aligned_ends = requested_ends
    starts = aligned_ends - effective + 1
    if (starts < 0).any() or (aligned_ends >= len(left)).any():
        raise ValueError("invalid lagged window boundary")
    invalid = ~(left_valid & right_valid)
    invalid_cumulative = np.concatenate([[0], np.cumsum(invalid)])
    window_valid = (invalid_cumulative[aligned_ends + 1] - invalid_cumulative[starts]) == 0
    result = np.full((len(ends), len(pairs)), np.nan, dtype=np.float64)
    for column, pair in enumerate(pairs.itertuples(index=False)):
        a = left[:, int(pair.channel_i_index)]
        b = right[:, int(pair.channel_j_index)]
        sa, sb = _rolling_sum(a, starts, aligned_ends), _rolling_sum(b, starts, aligned_ends)
        saa = _rolling_sum(a * a, starts, aligned_ends)
        sbb = _rolling_sum(b * b, starts, aligned_ends)
        sab = _rolling_sum(a * b, starts, aligned_ends)
        numerator = sab - sa * sb / effective
        denominator = np.sqrt(np.maximum(saa - sa * sa / effective, 0.0) *
                              np.maximum(sbb - sb * sb / effective, 0.0))
        usable = window_valid & (denominator > 1e-12)
        result[usable, column] = np.clip(numerator[usable] / denominator[usable], -1.0, 1.0)
    return result


def relation_stats(values: np.ndarray, candidate_count: int) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"valid_window_count": 0, "valid_window_ratio": 0.0, "median_corr": np.nan,
                "median_abs_corr": np.nan, "corr_iqr": np.nan, "corr_mad": np.nan,
                "sign_consistency": 0.0}
    median = float(np.median(finite))
    return {"valid_window_count": int(len(finite)), "valid_window_ratio": float(len(finite) / max(candidate_count, 1)),
            "median_corr": median, "median_abs_corr": float(np.median(np.abs(finite))),
            "corr_iqr": float(np.quantile(finite, .75) - np.quantile(finite, .25)),
            "corr_mad": float(np.median(np.abs(finite - median))),
            "sign_consistency": float(max((finite >= 0).mean(), (finite <= 0).mean()))}


def discovery_selected_lags(pairs: pd.DataFrame, discovery_by_lag: dict[int, np.ndarray]) -> pd.DataFrame:
    """Select lags using Discovery statistics only; Confirmation is intentionally absent."""
    rows: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs.itertuples(index=False)):
        candidates = []
        for lag in LAG_SECONDS:
            stats = relation_stats(discovery_by_lag[lag][:, pair_index], len(discovery_by_lag[lag]))
            candidates.append({"selected_lag": lag, **stats})
        chosen = sorted(candidates, key=lambda row: (-np.nan_to_num(row["median_abs_corr"], nan=-1.0),
                                                     np.nan_to_num(row["corr_iqr"], nan=np.inf),
                                                     abs(row["selected_lag"]), row["selected_lag"]))[0]
        rows.append({**pair._asdict(), **chosen})
    return pd.DataFrame(rows)


def reproducibility_row(pair: dict[str, Any], representation: str, discovery: np.ndarray,
                        confirmation: np.ndarray, cfg: dict[str, float], **context: Any) -> dict[str, Any]:
    first, second = relation_stats(discovery, len(discovery)), relation_stats(confirmation, len(confirmation))
    gap = abs(first["median_corr"] - second["median_corr"]) if np.isfinite(first["median_corr"] + second["median_corr"]) else np.inf
    pooled = 1.4826 * max(first["corr_mad"], second["corr_mad"])
    sign = bool(np.sign(first["median_corr"]) == np.sign(second["median_corr"]) and np.sign(first["median_corr"]) != 0)
    passed = bool(first["valid_window_ratio"] >= cfg["valid_window_ratio"] and
                  second["valid_window_ratio"] >= cfg["valid_window_ratio"] and
                  first["median_abs_corr"] >= cfg["median_abs_corr"] and
                  second["median_abs_corr"] >= cfg["median_abs_corr"] and
                  first["sign_consistency"] >= cfg["sign_consistency"] and
                  second["sign_consistency"] >= cfg["sign_consistency"] and
                  first["corr_iqr"] <= cfg["corr_iqr"] and second["corr_iqr"] <= cfg["corr_iqr"] and
                  sign and gap <= max(cfg["absolute_median_gap"], cfg["pooled_scale_multiplier"] * pooled))
    return {**pair, "representation": representation, **context,
            **{f"discovery_{k}": v for k, v in first.items()},
            **{f"confirmation_{k}": v for k, v in second.items()},
            "median_gap": float(gap), "pooled_robust_scale": float(pooled),
            "direction_consistent": sign, "reproducible": passed}


def lagged_diagnostics(pairs: pd.DataFrame, selected: pd.DataFrame,
                       discovery_by_lag: dict[int, np.ndarray], confirmation_by_lag: dict[int, np.ndarray],
                       representation: str, cfg: dict[str, float]) -> pd.DataFrame:
    rows = []
    for pair_index, pair in enumerate(pairs.itertuples(index=False)):
        lag = int(selected.iloc[pair_index].selected_lag)
        direction = "synchronous" if lag == 0 else (f"{pair.channel_i} leads {pair.channel_j}" if lag > 0
                                                    else f"{pair.channel_j} leads {pair.channel_i}")
        rows.append(reproducibility_row(pair._asdict(), representation,
                                        discovery_by_lag[lag][:, pair_index], confirmation_by_lag[lag][:, pair_index], cfg,
                                        selected_lag=lag, lag_direction=direction,
                                        lag_selected_on_discovery_only=True, confirmation_reselected_lag=False))
    return pd.DataFrame(rows)


def parse_binary_state(series: pd.Series, mapping: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if not mapping:
        return np.full(len(series), -1, dtype=np.int8), np.zeros(len(series), dtype=bool)
    normalized = {str(key).strip().lower(): value for key, value in mapping.items()}
    output = np.full(len(series), -1, dtype=np.int8)
    known = np.zeros(len(series), dtype=bool)
    for index, raw in enumerate(series):
        key = str(raw).strip().lower()
        if key not in normalized:
            continue
        value = normalized[key]
        if value in (0, False, "inactive", "closed", "stopped", "off"):
            output[index] = 0
        elif value in (1, True, "active", "open", "running", "on"):
            output[index] = 1
        else:
            raise ValueError(f"mapping for {raw!r} is not explicitly binary")
        known[index] = True
    return output, known


def stable_state_window_mask(states: np.ndarray, ends: np.ndarray, value: int,
                             window_size: int = 128) -> np.ndarray:
    state = np.asarray(states)
    return np.array([bool(np.all(state[end - window_size + 1:end + 1] == value)) for end in ends])


def audit_state_channels(frame: pd.DataFrame, discovery: np.ndarray, confirmation: np.ndarray,
                         state_cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows, parsed = [], {}
    allowed = tuple(state_cfg["allowed_suffixes"])
    forbidden = tuple(state_cfg["forbidden_fragments"])
    for channel in frame.columns:
        if not channel.endswith(allowed) or any(token in channel for token in forbidden):
            continue
        mapping = state_cfg["explicit_binary_mappings"].get(channel, {})
        values, known = parse_binary_state(frame[channel], mapping)
        reasons = []
        if not mapping:
            reasons.append("no_explicit_binary_mapping")
        for name, indices in (("discovery", discovery), ("confirmation", confirmation)):
            section = values[indices]
            section_known = known[indices]
            unknown_ratio = 1.0 - float(section_known.mean())
            counts = {str(value): int((section == value).sum()) for value in (0, 1)}
            transitions = int(np.sum(np.diff(section[section_known]) != 0)) if section_known.any() else 0
            if unknown_ratio >= state_cfg["maximum_unknown_ratio"]:
                reasons.append(f"{name}_unknown_ratio")
            if min(counts.values()) < state_cfg["minimum_state_points"]:
                reasons.append(f"{name}_insufficient_state_points")
            if transitions < 1:
                reasons.append(f"{name}_no_transition")
        d_ratio = float((values[discovery] == 1).mean())
        c_ratio = float((values[confirmation] == 1).mean())
        eligible = not reasons
        rows.append({"state_channel": channel, "eligible": eligible, "invalid_reasons": ";".join(sorted(set(reasons))),
                     "discovery_active_ratio": d_ratio, "confirmation_active_ratio": c_ratio,
                     "occupancy_ratio_gap": abs(d_ratio - c_ratio), "qualification_uses_test": False})
        if eligible:
            parsed[channel] = values
    audit = pd.DataFrame(rows)
    eligible = audit[audit.eligible].sort_values(["occupancy_ratio_gap", "state_channel"]).head(state_cfg["maximum_channels"])
    keep = set(eligible.state_channel)
    return audit, {name: value for name, value in parsed.items() if name in keep}


def select_contextual_set(diagnostics: pd.DataFrame, max_pairs: int = 20, max_degree: int = 3) -> dict[str, Any]:
    selected = diagnostics[diagnostics.reproducible].copy()
    if selected.empty:
        return {"status": "none", "relations": selected}
    selected["minimum_abs_corr"] = selected[["discovery_median_abs_corr", "confirmation_median_abs_corr"]].min(axis=1)
    selected["maximum_iqr"] = selected[["discovery_corr_iqr", "confirmation_corr_iqr"]].max(axis=1)
    tie = "absolute_lag" if "selected_lag" in selected else "condition_name"
    if "selected_lag" in selected:
        selected["absolute_lag"] = selected["selected_lag"].abs()
    selected = selected.sort_values(["confirmation_valid_window_ratio", "minimum_abs_corr", "median_gap", "maximum_iqr", tie, "pair_name"],
                                    ascending=[False, False, True, True, True, True], kind="mergesort")
    degree: Counter[str] = Counter(); rows = []; seen = set()
    for _, row in selected.iterrows():
        if row.pair_name in seen or degree[row.channel_i] >= max_degree or degree[row.channel_j] >= max_degree:
            continue
        rows.append(row.to_dict()); seen.add(row.pair_name)
        degree[row.channel_i] += 1; degree[row.channel_j] += 1
        if len(rows) >= max_pairs:
            break
    relations = pd.DataFrame(rows, columns=selected.columns)
    status = "general" if len(relations) >= 3 else "narrow"
    return {"status": status, "relations": relations}


def fit_relation_reference(correlations: np.ndarray, floor: float) -> dict[str, np.ndarray]:
    z = np.arctanh(np.clip(correlations, -FISHER_CLIP, FISHER_CLIP))
    center = np.nanmedian(z, axis=0)
    raw = 1.4826 * np.nanmedian(np.abs(z - center), axis=0)
    scale = np.where(np.isfinite(raw) & (raw >= floor), raw, floor)
    return {"center": center, "scale": scale, "raw_scale": raw}


def score_correlations(correlations: np.ndarray, reference: dict[str, np.ndarray],
                       applicable: np.ndarray | None = None) -> dict[str, np.ndarray]:
    z = np.arctanh(np.clip(correlations, -FISHER_CLIP, FISHER_CLIP))
    deviation = np.abs(z - reference["center"]) / reference["scale"]
    mask = np.isfinite(deviation) if applicable is None else np.isfinite(deviation) & applicable
    count = mask.sum(axis=1)
    scores = np.full(len(deviation), np.nan); top3 = np.full(len(deviation), np.nan)
    top_relation = np.full(len(deviation), -1, dtype=int)
    for row in range(len(deviation)):
        usable = np.flatnonzero(mask[row])
        if not len(usable):
            continue
        values = deviation[row, usable]
        order = np.argsort(values)
        scores[row] = values[order[-1]]
        top_relation[row] = usable[order[-1]]
        top3[row] = np.mean(values[order[-min(3, len(order)):]])
    return {"score": scores, "top3_mean": top3, "applicable_relation_count": count,
            "top_relation_index": top_relation}


def circular_shift_window(window: np.ndarray, channel_index: int, shift: int) -> np.ndarray:
    result = np.asarray(window).copy()
    result[:, channel_index] = np.roll(result[:, channel_index], int(shift))
    return result


def binary_metrics(normal: np.ndarray, anomaly: np.ndarray, threshold_values: np.ndarray,
                   quantile: float = .99) -> dict[str, float]:
    normal, anomaly = np.asarray(normal), np.asarray(anomaly)
    normal, anomaly = normal[np.isfinite(normal)], anomaly[np.isfinite(anomaly)]
    threshold_finite = np.asarray(threshold_values)[np.isfinite(threshold_values)]
    if not len(normal) or not len(anomaly) or not len(threshold_finite):
        return {"auroc": np.nan, "aupr": np.nan, "threshold": np.nan, "precision": np.nan,
                "recall": np.nan, "f1": np.nan, "fpr": np.nan}
    threshold = float(np.quantile(threshold_finite, quantile))
    labels = np.r_[np.zeros(len(normal), dtype=int), np.ones(len(anomaly), dtype=int)]
    scores = np.r_[normal, anomaly]
    alarm = scores > threshold
    tp, fp = int((alarm & (labels == 1)).sum()), int((alarm & (labels == 0)).sum())
    fn, tn = int((~alarm & (labels == 1)).sum()), int((~alarm & (labels == 0)).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"auroc": float(roc_auc_score(labels, scores)), "aupr": float(average_precision_score(labels, scores)),
            "threshold": threshold, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12), "fpr": fp / max(fp + tn, 1)}


def select_primary(controlled: pd.DataFrame, relation_counts: dict[str, int], coverage: dict[str, float],
                   minimum_auroc: float = .80, minimum_coverage: float = .80) -> dict[str, Any]:
    if controlled.empty:
        return {"primary_frozen": False, "representation": None, "selection_used_test": False}
    eligible = controlled[(controlled.score_name == "structure_max") & controlled.representation.map(relation_counts).ge(3) &
                          controlled.auroc.ge(minimum_auroc) & controlled.marginal_statistics_preserved &
                          controlled.reference_valid & controlled.representation.map(coverage).ge(minimum_coverage)].copy()
    if eligible.empty:
        return {"primary_frozen": False, "representation": None, "selection_used_test": False}
    best = eligible.auroc.max(); eligible = eligible[eligible.auroc >= best - .01].copy()
    order = {"lagged_level": 0, "lagged_delta": 1, "conditioned_level": 2, "conditioned_delta": 3}
    eligible["simplicity"] = eligible.representation.map(order)
    eligible["relation_count"] = eligible.representation.map(relation_counts)
    eligible["coverage"] = eligible.representation.map(coverage)
    row = eligible.sort_values(["simplicity", "relation_count", "coverage"], ascending=[True, False, False]).iloc[0]
    return {"primary_frozen": True, "representation": row.representation, "controlled_auroc": float(row.auroc),
            "selection_used_test": False, "relation_count": int(row.relation_count), "coverage": float(row.coverage)}
