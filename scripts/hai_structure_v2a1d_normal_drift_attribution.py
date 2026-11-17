from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hai_structure_v2a1r_cache_backfill_and_grouped_aggregation import grouped_scores

V2A = ROOT / "outputs" / "hai_structure_v2a_lagged_dependency"
A1R = ROOT / "outputs" / "hai_structure_v2a1r_cache_backfill_and_grouped_aggregation"
A1S = ROOT / "outputs" / "hai_structure_v2a1s_full_clean_stability"
OUT = ROOT / "outputs" / "hai_structure_v2a1d_normal_drift_attribution"
FAMILIES = ("R1", "R2", "R3")
TOLERANCE = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def historical_hashes() -> dict[str, str]:
    result = {}
    for prefix, directory in (("v2a", V2A), ("a1r", A1R), ("a1s", A1S)):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                result[f"{prefix}/{path.relative_to(directory)}"] = sha256(path)
    return result


def load_full_clean():
    pool = pd.read_csv(V2A / "selected_dynamic_relation_pool.csv")
    if len(pool) != 12:
        raise RuntimeError(f"Frozen relation pool changed: {len(pool)}")
    target_names = list(dict.fromkeys(pool.target.astype(str).tolist()))
    target_ids = np.asarray([target_names.index(value) for value in pool.target.astype(str)], dtype=np.int16)
    manifest = pd.read_csv(A1S / "full_clean_crossfit_window_manifest.csv")
    source = manifest[manifest.fold.eq("A")].copy()
    source = source[["source_window_id", "sequence_id", "timestamp", "endpoint_row", "half"]]
    source = source.sort_values(["sequence_id", "endpoint_row"]).reset_index(drop=True)
    thresholds = pd.read_csv(A1S / "full_clean_crossfit_thresholds.csv").set_index("fold").threshold.to_dict()
    archive = np.load(V2A / "validation_dynamic_edge_score_cache.npz", allow_pickle=False)
    edge_parts = []
    for sequence in range(1, 7):
        values = archive[f"validation{sequence}_edge_score"][15:].astype(np.float32)
        expected = source.sequence_id.eq(f"train{sequence}_validation").sum()
        if len(values) != expected or values.shape[1] != 12 or not np.isfinite(values).all():
            raise RuntimeError(f"Clean cache alignment failed for validation{sequence}")
        edge_parts.append(values)
    archive.close()
    edge_scores = np.vstack(edge_parts)
    target_scores = np.stack(
        [edge_scores[:, target_ids == target].max(axis=1) for target in range(len(target_names))], axis=1
    )
    target_source_edge = np.empty_like(target_scores, dtype=np.int16)
    for target in range(len(target_names)):
        edge_ids = np.flatnonzero(target_ids == target)
        target_source_edge[:, target] = edge_ids[np.argmax(edge_scores[:, edge_ids], axis=1)]
    target_order = np.argsort(target_scores, axis=1)[:, -2:][:, ::-1]
    global_scores = np.take_along_axis(target_scores, target_order, axis=1).mean(axis=1)
    top_edges = np.take_along_axis(target_source_edge, target_order, axis=1)
    evaluation_fold = np.where(source.half.eq("first"), "B", "A")
    threshold = np.asarray([thresholds[fold] for fold in evaluation_fold])
    alarm = global_scores > threshold
    source["evaluation_fold"] = evaluation_fold
    source["threshold"] = threshold
    source["global_score"] = global_scores
    source["alarm"] = alarm
    return pool, target_names, target_ids, source, edge_scores, target_scores, target_source_edge, target_order, top_edges, thresholds


def reproduce_a1s(out: Path, source: pd.DataFrame, thresholds: dict[str, float]):
    fpr_by_sequence = source.groupby("sequence_id").alarm.mean()
    train1 = source[source.sequence_id.eq("train1_validation")]
    actual = {
        "overall_fpr": float(source.alarm.mean()),
        "worst_sequence_fpr": float(fpr_by_sequence.max()),
        "train1_total_alarm": int(train1.alarm.sum()),
        "train1_first_alarm": int(train1[train1.half.eq("first")].alarm.sum()),
        "train1_second_alarm": int(train1[train1.half.eq("second")].alarm.sum()),
        "fold_a_threshold": float(thresholds["A"]),
        "fold_b_threshold": float(thresholds["B"]),
    }
    expected = {
        "overall_fpr": 0.009118,
        "worst_sequence_fpr": 0.054300,
        "train1_total_alarm": 1014,
        "train1_first_alarm": 10,
        "train1_second_alarm": 1004,
        "fold_a_threshold": 2.139370,
        "fold_b_threshold": 2.415423,
    }
    rows = []
    for metric, expected_value in expected.items():
        difference = abs(float(actual[metric]) - float(expected_value))
        tolerance = 0.0 if "alarm" in metric else TOLERANCE
        rows.append(
            {
                "metric": metric,
                "expected": expected_value,
                "reproduced": actual[metric],
                "absolute_difference": difference,
                "tolerance": tolerance,
                "passed": difference <= tolerance,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "a1s_exact_reproduction_audit.csv", index=False)
    return bool(frame.passed.all()), actual


def quantiles(values):
    return {
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def run_lengths(active: np.ndarray):
    padded = np.r_[False, np.asarray(active, bool), False].astype(np.int8)
    changes = np.diff(padded)
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    lengths = ends - starts
    return int(len(lengths)), int(lengths.max()) if len(lengths) else 0


def attribution_scopes(source: pd.DataFrame):
    scopes = {
        "train1_first": source.sequence_id.eq("train1_validation") & source.half.eq("first"),
        "train1_second": source.sequence_id.eq("train1_validation") & source.half.eq("second"),
    }
    for sequence in range(2, 7):
        scopes[f"train{sequence}_all"] = source.sequence_id.eq(f"train{sequence}_validation")
    return scopes


def target_attribution(source, target_names, target_scores, target_order):
    scopes = attribution_scopes(source)
    first = scopes["train1_first"]
    second = scopes["train1_second"]
    first_threshold = source.loc[first, "threshold"].to_numpy(float)
    second_threshold = source.loc[second, "threshold"].to_numpy(float)
    rows = []
    for target, name in enumerate(target_names):
        first_values, second_values = target_scores[first, target], target_scores[second, target]
        first_q, second_q = quantiles(first_values), quantiles(second_values)
        for scope, mask in scopes.items():
            if not mask.any():
                continue
            values = target_scores[mask, target]
            local_alarm = source.loc[mask, "alarm"].to_numpy(bool)
            top1 = target_order[mask, 0] == target
            top2 = np.any(target_order[mask] == target, axis=1)
            alarm_participation = local_alarm & top2
            primary_alarm_contribution = local_alarm & top1
            high = values > source.loc[mask, "threshold"].to_numpy(float)
            segment_count, longest = run_lengths(high)
            rows.append(
                {
                    "scope": scope,
                    "target_id": target,
                    "target_channel": name,
                    "window_count": int(mask.sum()),
                    "top1_count": int(top1.sum()),
                    "top2_count": int(top2.sum()),
                    "alarm_participation_count": int(alarm_participation.sum()),
                    "alarm_contribution_ratio": float(alarm_participation.sum() / max(local_alarm.sum(), 1)),
                    "primary_alarm_contribution_count": int(primary_alarm_contribution.sum()),
                    "primary_alarm_contribution_ratio": float(
                        primary_alarm_contribution.sum() / max(local_alarm.sum(), 1)
                    ),
                    **quantiles(values),
                    "train1_median_shift_second_minus_first": second_q["median"] - first_q["median"],
                    "train1_q95_shift_second_minus_first": second_q["q95"] - first_q["q95"],
                    "train1_q99_shift_second_minus_first": second_q["q99"] - first_q["q99"],
                    "train1_exceed_probability_first": float(np.mean(first_values > first_threshold)),
                    "train1_exceed_probability_second": float(np.mean(second_values > second_threshold)),
                    "train1_exceed_probability_change": float(
                        np.mean(second_values > second_threshold) - np.mean(first_values > first_threshold)
                    ),
                    "high_score_segment_count": segment_count,
                    "longest_high_score_segment": longest,
                }
            )
    return pd.DataFrame(rows)


def safe_correlation(left, right):
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def edge_attribution(source, pool, target_ids, edge_scores, target_source_edge, top_edges):
    scopes = attribution_scopes(source)
    first, second = scopes["train1_first"], scopes["train1_second"]
    rows = []
    for edge, relation in enumerate(pool.itertuples()):
        first_q, second_q = quantiles(edge_scores[first, edge]), quantiles(edge_scores[second, edge])
        peers = np.flatnonzero((target_ids == target_ids[edge]) & (np.arange(len(pool)) != edge))
        peer_corr = max((abs(safe_correlation(edge_scores[second, edge], edge_scores[second, peer])) for peer in peers), default=0.0)
        for scope, mask in scopes.items():
            if not mask.any():
                continue
            values = edge_scores[mask, edge]
            alarm = source.loc[mask, "alarm"].to_numpy(bool)
            target_max = target_source_edge[mask, target_ids[edge]] == edge
            global_top = np.any(top_edges[mask] == edge, axis=1)
            global_top1 = top_edges[mask, 0] == edge
            contribution = alarm & global_top
            segments, longest = run_lengths(contribution)
            rows.append(
                {
                    "scope": scope,
                    "edge_id": edge,
                    "source_channel": relation.source,
                    "target_channel": relation.target,
                    "window_count": int(mask.sum()),
                    **quantiles(values),
                    "train1_median_shift_second_minus_first": second_q["median"] - first_q["median"],
                    "train1_q95_shift_second_minus_first": second_q["q95"] - first_q["q95"],
                    "train1_q99_shift_second_minus_first": second_q["q99"] - first_q["q99"],
                    "threshold_exceed_fraction": float(
                        np.mean(values > source.loc[mask, "threshold"].to_numpy(float))
                    ),
                    "target_group_max_fraction": float(target_max.mean()),
                    "global_top1_fraction": float(global_top1.mean()),
                    "global_top2_fraction": float(global_top.mean()),
                    "alarm_contribution_count": int(contribution.sum()),
                    "alarm_segment_contribution_count": segments,
                    "longest_alarm_contribution_segment": longest,
                    "incoming_peer_count": len(peers),
                    "max_abs_incoming_peer_correlation_train1_second": peer_corr,
                    "train1_second_false_alarm_contribution_ratio": float(
                        contribution.sum() / max(alarm.sum(), 1)
                    ) if scope == "train1_second" else 0.0,
                }
            )
    return pd.DataFrame(rows)


def concentration(source, target_names, pool, target_order, top_edges):
    mask = source.sequence_id.eq("train1_validation") & source.half.eq("second") & source.alarm
    top_target = target_order[mask, 0]
    top_edge = top_edges[mask, 0]
    target_counts = np.bincount(top_target, minlength=len(target_names))
    edge_counts = np.bincount(top_edge, minlength=len(pool))
    target_share = np.sort(target_counts / max(target_counts.sum(), 1))[::-1]
    edge_share = np.sort(edge_counts / max(edge_counts.sum(), 1))[::-1]
    rows = [
        {"metric": "top1_target_contribution", "value": float(target_share[0])},
        {"metric": "top2_target_cumulative_contribution", "value": float(target_share[:2].sum())},
        {"metric": "top3_target_cumulative_contribution", "value": float(target_share[:3].sum())},
        {"metric": "top1_edge_contribution", "value": float(edge_share[0])},
        {"metric": "top2_edge_cumulative_contribution", "value": float(edge_share[:2].sum())},
        {"metric": "top3_edge_cumulative_contribution", "value": float(edge_share[:3].sum())},
        {"metric": "target_herfindahl_index", "value": float(np.square(target_share).sum())},
        {"metric": "edge_herfindahl_index", "value": float(np.square(edge_share).sum())},
        {"metric": "fixed_edge_persistent_driver", "value": float(edge_share[0] >= 0.80)},
        {"metric": "multiple_edge_switching", "value": float(edge_share[0] < 0.50)},
    ]
    return pd.DataFrame(rows), target_counts, edge_counts


def psi(reference, current, bins=10):
    boundaries = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(boundaries) < 3:
        return 0.0
    boundaries[0], boundaries[-1] = -np.inf, np.inf
    expected = np.histogram(reference, bins=boundaries)[0] / len(reference)
    actual = np.histogram(current, bins=boundaries)[0] / len(current)
    expected, actual = np.maximum(expected, 1e-6), np.maximum(actual, 1e-6)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def operating_state_shift(source, target_names, target_scores):
    first = source.sequence_id.eq("train1_validation") & source.half.eq("first")
    second = source.sequence_id.eq("train1_validation") & source.half.eq("second")
    rows, drifted = [], 0
    unavailable = ("current_value", "first_difference", "self_residual", "cross_residual", "dependency_loss")
    for target, name in enumerate(target_names):
        left, right = target_scores[first, target], target_scores[second, target]
        ks = float(ks_2samp(left, right).statistic)
        value_psi = psi(left, right)
        q95_shift = float(np.quantile(right, 0.95) - np.quantile(left, 0.95))
        q99_shift = float(np.quantile(right, 0.99) - np.quantile(left, 0.99))
        significant = value_psi >= 0.25 or ks >= 0.20 or q95_shift >= 0.25 or q99_shift >= 0.50
        drifted += int(significant)
        left_iqr = float(np.quantile(left, 0.75) - np.quantile(left, 0.25))
        right_iqr = float(np.quantile(right, 0.75) - np.quantile(right, 0.25))
        rows.append(
            {
                "target_channel": name,
                "measure": "target_group_score",
                "available": True,
                "unavailable_reason": "none",
                "psi": value_psi,
                "wasserstein_distance": float(wasserstein_distance(left, right)),
                "ks_statistic": ks,
                "median_shift": float(np.median(right) - np.median(left)),
                "iqr_shift": right_iqr - left_iqr,
                "q95_shift": q95_shift,
                "q99_shift": q99_shift,
                "significant_drift": significant,
            }
        )
        for measure in unavailable:
            rows.append(
                {
                    "target_channel": name,
                    "measure": measure,
                    "available": False,
                    "unavailable_reason": "unavailable_without_edge_score_recompute_or_model_loading",
                    "psi": 0.0,
                    "wasserstein_distance": 0.0,
                    "ks_statistic": 0.0,
                    "median_shift": 0.0,
                    "iqr_shift": 0.0,
                    "q95_shift": 0.0,
                    "q99_shift": 0.0,
                    "significant_drift": False,
                }
            )
    frame = pd.DataFrame(rows)
    frame["multi_target_significant_drift_count"] = drifted
    return frame, drifted


def controlled_value(source, pool, edge_scores, edge_counts):
    edge_archive = np.load(A1R / "controlled_dynamic_edge_scores.npz", allow_pickle=False)
    aggregate_archive = np.load(A1R / "controlled_dynamic_aggregate_scores.npz", allow_pickle=False)
    controlled = edge_archive["edge_scores"]
    valid = edge_archive["valid_mask"]
    source_ids = edge_archive["source_window_id"].astype(str)
    aggregate_edges = aggregate_archive["group_source_edge_ids"]
    lookup = source.set_index("source_window_id")
    clean_indices = lookup.loc[source_ids].index
    position_lookup = {value: index for index, value in enumerate(source.source_window_id)}
    positions = np.asarray([position_lookup[value] for value in source_ids])
    thresholds = source.threshold.to_numpy(float)[positions]
    rows = []
    total_alarms = max(edge_counts.sum(), 1)
    for edge, relation in enumerate(pool.itertuples()):
        family_metrics, global_frequency = [], []
        for family_index, family in enumerate(FAMILIES):
            mask = valid[family_index, :, edge]
            positive = controlled[family_index, :, edge, edge][mask]
            clean = edge_scores[positions, edge][mask]
            labels = np.r_[np.zeros(len(clean), int), np.ones(len(positive), int)]
            values = np.r_[clean, positive]
            family_metrics.append(
                {
                    "family": family,
                    "auroc": float(roc_auc_score(labels, values)),
                    "aupr": float(average_precision_score(labels, values)),
                    "q99_tpr": float(np.mean(positive > thresholds[mask])),
                    "median_contribution": float(np.median(positive)),
                }
            )
            global_frequency.append(float(np.any(aggregate_edges[family_index, :, edge] == edge, axis=1).mean()))
        rows.append(
            {
                "edge_id": edge,
                "source_channel": relation.source,
                "target_channel": relation.target,
                "target_edge_auroc": float(np.mean([item["auroc"] for item in family_metrics])),
                "target_edge_aupr": float(np.mean([item["aupr"] for item in family_metrics])),
                "target_edge_q99_tpr": float(np.mean([item["q99_tpr"] for item in family_metrics])),
                "global_top1_top2_fraction": float(np.mean(global_frequency)),
                "mean_corruption_contribution": float(np.mean([item["median_contribution"] for item in family_metrics])),
                "normal_drift_cost": float(edge_counts[edge] / total_alarms),
                "controlled_detection_value": float(
                    np.mean([item["auroc"] for item in family_metrics])
                    * np.mean([item["q99_tpr"] for item in family_metrics])
                ),
                "controlled_detection_key_relation": bool(
                    np.mean([item["auroc"] for item in family_metrics]) >= 0.80
                    and np.mean([item["q99_tpr"] for item in family_metrics]) >= 0.40
                ),
            }
        )
    edge_archive.close()
    aggregate_archive.close()
    return pd.DataFrame(rows)


def segment_attribution(source, target_names, pool, target_order, top_edges):
    frame = source[source.sequence_id.eq("train1_validation") & source.half.eq("second")].copy()
    active = frame.alarm.to_numpy(bool)
    padded = np.r_[False, active, False].astype(np.int8)
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    rows = []
    original_positions = np.flatnonzero(source.sequence_id.eq("train1_validation") & source.half.eq("second"))
    for segment, (start, end) in enumerate(zip(starts, ends)):
        positions = original_positions[start:end]
        target_count = np.bincount(target_order[positions, 0], minlength=len(target_names))
        edge_count = np.bincount(top_edges[positions, 0], minlength=len(pool))
        target = int(np.argmax(target_count))
        edge = int(np.argmax(edge_count))
        rows.append(
            {
                "segment_id": segment,
                "start_timestamp": frame.iloc[start].timestamp,
                "end_timestamp": frame.iloc[end - 1].timestamp,
                "length": int(end - start),
                "dominant_target": target_names[target],
                "dominant_target_share": float(target_count[target] / (end - start)),
                "dominant_edge_id": edge,
                "dominant_edge": f"{pool.iloc[edge].source}->{pool.iloc[edge].target}",
                "dominant_edge_share": float(edge_count[edge] / (end - start)),
            }
        )
    return pd.DataFrame(rows)


def classify(concentration_frame, drift_count, edge_frame):
    values = concentration_frame.set_index("metric").value
    top2_target = float(values["top2_target_cumulative_contribution"])
    top3_edge = float(values["top3_edge_cumulative_contribution"])
    drifting_edges = int(
        (
            edge_frame[edge_frame.scope.eq("train1_second")]
            .train1_q95_shift_second_minus_first.abs()
            > 0.25
        ).sum()
    )
    localized = top2_target >= 0.80 or top3_edge >= 0.80
    broad = drift_count >= 3 and top2_target < 0.80 and drifting_edges >= 3
    if localized and drift_count < 3:
        classification = "localized_relation_instability"
        recommendation = "stability_constrained_relation_pool_repair"
    elif broad:
        classification = "broad_operating_mode_dependency_shift"
        recommendation = "train_only_mode_conditioned_dependency_reference"
    elif localized and drift_count >= 3:
        classification = "mixed_local_and_mode_shift"
        recommendation = "diagnose_local_relations_and_train_only_modes_separately"
    else:
        classification = "unstable_unexplained_normal_dependency"
        recommendation = "no_automatic_repair"
    evidence = {
        "classification": classification,
        "recommended_future_direction": recommendation,
        "top2_target_contribution": top2_target,
        "top3_edge_contribution": top3_edge,
        "significantly_drifting_target_count": drift_count,
        "drifting_edge_count": drifting_edges,
        "automatic_followup_started": False,
    }
    return classification, evidence


def run():
    before = historical_hashes()
    dependency_hash = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    status_path = OUT / "stage_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("final_report_complete") and status.get("dependency_hash") == dependency_hash:
            print(f"REUSE completed V2-A1D: {OUT}", flush=True)
            return 0
    OUT.mkdir(parents=True, exist_ok=True)
    pool, target_names, target_ids, source, edge_scores, target_scores, target_source_edge, target_order, top_edges, thresholds = load_full_clean()
    reproduced, reproduction = reproduce_a1s(OUT, source, thresholds)
    if not reproduced:
        summary = {"classification": "drift_attribution_reproduction_failed", "reproduction": reproduction}
        (OUT / "structure_v2a1d_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("classification=drift_attribution_reproduction_failed", flush=True)
        return 2
    targets = target_attribution(source, target_names, target_scores, target_order)
    edges = edge_attribution(source, pool, target_ids, edge_scores, target_source_edge, top_edges)
    concentration_frame, target_counts, edge_counts = concentration(source, target_names, pool, target_order, top_edges)
    operating, drift_count = operating_state_shift(source, target_names, target_scores)
    value = controlled_value(source, pool, edge_scores, edge_counts)
    segments = segment_attribution(source, target_names, pool, target_order, top_edges)
    classification, evidence = classify(concentration_frame, drift_count, edges)
    targets.to_csv(OUT / "target_drift_attribution.csv", index=False)
    edges.to_csv(OUT / "edge_drift_attribution.csv", index=False)
    concentration_frame.to_csv(OUT / "false_alarm_concentration.csv", index=False)
    value.to_csv(OUT / "relation_drift_vs_detection_value.csv", index=False)
    operating.to_csv(OUT / "train1_half_operating_state_shift.csv", index=False)
    segments.to_csv(OUT / "train1_false_alarm_segments_attribution.csv", index=False)
    (OUT / "drift_classification_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    after = historical_hashes()
    audit = pd.DataFrame(
        [
            {
                "artifact": name,
                "sha256_before": value_hash,
                "sha256_after": after.get(name),
                "unchanged": value_hash == after.get(name),
            }
            for name, value_hash in before.items()
        ]
    )
    audit.to_csv(OUT / "artifact_reuse_audit.csv", index=False)
    if not audit.unchanged.all():
        raise RuntimeError("Historical artifact hash changed")
    summary = {
        "classification": classification,
        "a1s_exact_reproduction_passed": True,
        **evidence,
        "standardized_channel_value_cache_available": False,
        "models_loaded": False,
        "edge_scores_recomputed": False,
        "test_loaded": False,
        "attack_label_used": False,
        "rec_loaded": False,
        "ae_loaded": False,
    }
    (OUT / "structure_v2a1d_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    issues = [
        {
            "issue_id": "V2A1D-001",
            "stage": "HAI Structure V2-A1D",
            "research_question": "Are train1 normal false alarms localized or mode-wide?",
            "status": "diagnosed",
            "evidence": classification,
            "severity": "high",
            "affects_validity": True,
            "affects_final_method": True,
            "minimum_future_action": evidence["recommended_future_direction"],
            "recomputation_scope": "No automatic action",
            "frozen_artifacts_reusable": True,
        },
        {
            "issue_id": "V2A1D-002",
            "stage": "HAI Structure V2-A1D",
            "research_question": "Can channel/residual operating-state drift be audited offline?",
            "status": "open",
            "evidence": "standardized channel/residual/dependency-loss caches were not persisted",
            "severity": "medium",
            "affects_validity": False,
            "affects_final_method": False,
            "minimum_future_action": "Persist these arrays only in a separately authorized stage",
            "recomputation_scope": "Would require prohibited historical recomputation",
            "frozen_artifacts_reusable": True,
        },
    ]
    (OUT / "unresolved_issues_delta.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")
    report = f"""# HAI Structure V2-A1D

- A1S exact reproduction: **passed**.
- Classification: **{classification}**.
- Top-2 target false-alarm contribution: **{evidence['top2_target_contribution']:.6f}**.
- Top-3 edge false-alarm contribution: **{evidence['top3_edge_contribution']:.6f}**.
- Significantly drifting target-group scores: **{drift_count}**.
- Recommended future direction (not executed): `{evidence['recommended_future_direction']}`.
- Standardized channel values/residuals were not cached; unavailable fields are explicitly marked without recomputation.
- HAI test / attack label / AE / rec loaded: **false / false / false / false**.
"""
    (OUT / "final_hai_structure_v2a1d_normal_drift_attribution.md").write_text(report, encoding="utf-8")
    status = {
        "historical_artifacts_verified": True,
        "a1s_reproduction_complete": True,
        "a1s_reproduction_passed": True,
        "target_attribution_complete": True,
        "edge_attribution_complete": True,
        "concentration_complete": True,
        "controlled_value_complete": True,
        "operating_state_available_cache_audit_complete": True,
        "final_report_complete": True,
        "dependency_hash": dependency_hash,
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"classification={classification}", flush=True)
    print("test_loaded=false attack_label_used=false rec_loaded=false ae_loaded=false", flush=True)
    return 0


def dry_run():
    print("REUSE: V2-A full-clean 12-edge scores, A1S manifest/thresholds/alarms, A1R controlled scores", flush=True)
    print("COMPUTE_NEW: target/edge drift attribution, concentration, controlled value, cached-score shift statistics", flush=True)
    print("UNAVAILABLE: standardized channel values, self/cross residuals, raw dependency loss were not cached", flush=True)
    print("FORBIDDEN: HAI test, attack label, AE, rec, model load/fit, edge-score recompute, threshold change, edge deletion", flush=True)
    print("candidate_unchanged=target_group_max_then_top2_mean threshold_reused=true new_aggregation=false", flush=True)
    print("test_loaded=false attack_label_used=false rec_loaded=false ae_loaded=false", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        return dry_run()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
