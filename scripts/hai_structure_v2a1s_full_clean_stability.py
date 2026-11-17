from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
OUT = ROOT / "outputs" / "hai_structure_v2a1s_full_clean_stability"
SMOKE_OUT = ROOT / "outputs" / "hai_structure_v2a1s_full_clean_stability_smoke"
FAMILIES = np.asarray(("R1", "R2", "R3", "P1", "P2", "C1", "C2"))
QUANTILE = 0.99


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    rows = {}
    for prefix, directory in (("v2a", V2A), ("a1r", A1R)):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                rows[f"{prefix}/{path.relative_to(directory)}"] = sha256(path)
    return rows


def load_config() -> dict:
    return yaml.safe_load((V2A / "config_resolved.yaml").read_text(encoding="utf-8"))


def read_validation_timestamps_only(cfg: dict):
    split = pd.read_csv(ROOT / cfg["step2_output"] / "hai_split_manifest.csv")
    timestamps = []
    for row in split.itertuples():
        path = ROOT / cfg["data_root"] / row.file_name
        header = pd.read_csv(path, nrows=0)
        timestamp_column = next(column for column in header if column.strip().lower() == "timestamp")
        frame = pd.read_csv(
            path,
            usecols=[timestamp_column],
            skiprows=range(1, int(row.split_index) + 1),
            nrows=int(row.original_rows) - int(row.split_index),
        )
        timestamps.append(frame[timestamp_column].astype(str).to_numpy())
    return timestamps, split


def load_full_clean_grouped(smoke: bool = False):
    pool = pd.read_csv(V2A / "selected_dynamic_relation_pool.csv")
    if len(pool) != 12:
        raise RuntimeError(f"Expected frozen 12-edge pool, got {len(pool)}")
    targets = pd.factorize(pool.target.astype(str), sort=False)[0].astype(np.int16)
    archive = np.load(V2A / "validation_dynamic_edge_score_cache.npz", allow_pickle=False)
    expected = [f"validation{i}_edge_score" for i in range(1, 7)]
    missing = [key for key in expected if key not in archive.files]
    if missing:
        raise RuntimeError(f"Incomplete full-clean edge-score cache: {missing}")
    grouped, edges = [], []
    for key in expected:
        matrix = archive[key]
        if matrix.ndim != 2 or matrix.shape[1] != 12:
            raise RuntimeError(f"Invalid clean cache shape for {key}: {matrix.shape}")
        valid_matrix = matrix[15:]
        if not np.isfinite(valid_matrix).all():
            raise RuntimeError(f"Non-finite clean score in {key}")
        score, _, _ = grouped_scores(valid_matrix, targets)
        grouped.append(score.astype(np.float32))
        edges.append(valid_matrix)
    archive.close()
    return pool, targets, grouped, edges


def build_manifest(grouped, timestamps, smoke=False):
    rows, score_rows = [], []
    for sequence, score in enumerate(grouped):
        full_timestamp = timestamps[sequence][47:]
        sequence_timestamps = full_timestamp
        endpoint_rows = np.arange(47, 47 + len(score))
        if len(sequence_timestamps) != len(score):
            raise RuntimeError(f"Timestamp/cache length mismatch for validation{sequence + 1}")
        half_boundary = len(score) // 2
        halves = np.where(np.arange(len(score)) < half_boundary, "first", "second")
        sequence_id = f"train{sequence + 1}_validation"
        for local_index, (timestamp, endpoint, half, value) in enumerate(
            zip(sequence_timestamps, endpoint_rows, halves, score)
        ):
            source_id = f"train{sequence + 1}_val_row_{int(endpoint)}"
            score_rows.append(
                {
                    "source_window_id": source_id,
                    "sequence_id": sequence_id,
                    "timestamp": timestamp,
                    "endpoint_row": int(endpoint),
                    "half": half,
                    "score": float(value),
                }
            )
            for fold in ("A", "B"):
                calibration = (fold == "A" and half == "first") or (fold == "B" and half == "second")
                rows.append(
                    {
                        "source_window_id": source_id,
                        "sequence_id": sequence_id,
                        "timestamp": timestamp,
                        "endpoint_row": int(endpoint),
                        "fold": fold,
                        "role": "calibration" if calibration else "evaluation",
                        "half": half,
                        "valid": True,
                        "exclusion_reason": "none",
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(score_rows)


def fit_full_clean_thresholds(score_frame: pd.DataFrame):
    rows, thresholds = [], {}
    for fold, calibration_half in (("A", "first"), ("B", "second")):
        values = score_frame.loc[score_frame.half.eq(calibration_half), "score"].to_numpy(float)
        threshold = float(np.quantile(values, QUANTILE))
        thresholds[fold] = threshold
        rows.append(
            {
                "fold": fold,
                "calibration_window_count": len(values),
                "threshold": threshold,
                "quantile": QUANTILE,
                "min": float(values.min()),
                "median": float(np.median(values)),
                "q95": float(np.quantile(values, 0.95)),
                "q99": threshold,
                "max": float(values.max()),
            }
        )
    return thresholds, pd.DataFrame(rows)


def evaluation_fold(half: str) -> str:
    return "B" if half == "first" else "A"


def alarm_runs(frame: pd.DataFrame):
    rows = []
    for (fold, sequence), group in frame.groupby(["evaluation_fold", "sequence_id"], sort=False):
        ordered = group.sort_values("endpoint_row").reset_index(drop=True)
        alarm = ordered.alarm.to_numpy(bool)
        endpoints = ordered.endpoint_row.to_numpy(int)
        start = None
        for index, active in enumerate(alarm):
            if active and start is None:
                start = index
            ends = start is not None and (not active or index == len(alarm) - 1)
            if ends:
                end = index if active and index == len(alarm) - 1 else index - 1
                rows.append(
                    {
                        "fold": fold,
                        "sequence_id": sequence,
                        "segment_id": len(rows),
                        "start_endpoint_row": int(endpoints[start]),
                        "end_endpoint_row": int(endpoints[end]),
                        "start_timestamp": ordered.iloc[start].timestamp,
                        "end_timestamp": ordered.iloc[end].timestamp,
                        "length": int(end - start + 1),
                    }
                )
                start = None
    return pd.DataFrame(rows)


def evaluate_full_clean(score_frame: pd.DataFrame, thresholds: dict[str, float]):
    frame = score_frame.copy()
    frame["evaluation_fold"] = frame.half.map(evaluation_fold)
    frame["threshold"] = frame.evaluation_fold.map(thresholds)
    frame["alarm"] = frame.score > frame.threshold
    sequence_rows = []
    for scope, scoped in [("combined", frame), *[(fold, frame[frame.evaluation_fold.eq(fold)]) for fold in ("A", "B")]]:
        for sequence, group in scoped.groupby("sequence_id"):
            sequence_rows.append(
                {
                    "scope": scope,
                    "sequence_id": sequence,
                    "evaluation_window_count": len(group),
                    "alarm_count": int(group.alarm.sum()),
                    "fpr": float(group.alarm.mean()),
                }
            )
    by_sequence = pd.DataFrame(sequence_rows)
    overall_rows = []
    for scope, scoped in [("combined", frame), ("A", frame[frame.evaluation_fold.eq("A")]), ("B", frame[frame.evaluation_fold.eq("B")])]:
        local_fpr = by_sequence[by_sequence.scope.eq(scope)].fpr
        segments = alarm_runs(scoped.rename(columns={"evaluation_fold": "_fold"}).assign(evaluation_fold=scope)) if False else None
        overall_rows.append(
            {
                "scope": scope,
                "overall_fpr": float(scoped.alarm.mean()),
                "worst_sequence_fpr": float(local_fpr.max()),
                "best_sequence_fpr": float(local_fpr.min()),
                "fpr_variance": float(local_fpr.var(ddof=0)),
                "evaluation_window_count": len(scoped),
                "alarm_count": int(scoped.alarm.sum()),
            }
        )
    segments = alarm_runs(frame)
    if len(segments):
        bounds = frame.groupby("sequence_id").endpoint_row.agg(["min", "max"])
        segments["distance_to_sequence_start"] = segments.apply(
            lambda row: int(row.start_endpoint_row - bounds.loc[row.sequence_id, "min"]), axis=1
        )
        segments["distance_to_sequence_end"] = segments.apply(
            lambda row: int(bounds.loc[row.sequence_id, "max"] - row.end_endpoint_row), axis=1
        )
        segments["near_sequence_boundary_100"] = (
            segments[["distance_to_sequence_start", "distance_to_sequence_end"]].min(axis=1) <= 100
        )
    segment_summary = segments.groupby("fold").length.agg(["count", "max"]) if len(segments) else pd.DataFrame()
    overall = pd.DataFrame(overall_rows)
    for index, row in overall.iterrows():
        subset = segments if row.scope == "combined" else segments[segments.fold.eq(row.scope)]
        overall.loc[index, "alarm_segment_count"] = len(subset)
        overall.loc[index, "longest_alarm_segment"] = int(subset.length.max()) if len(subset) else 0
    return frame, overall, by_sequence, segments


def binary_metrics(clean, positive, clean_threshold, positive_threshold):
    labels = np.r_[np.zeros(len(clean), dtype=int), np.ones(len(positive), dtype=int)]
    scores = np.r_[clean, positive]
    prediction = np.r_[clean > clean_threshold, positive > positive_threshold]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, prediction, average="binary", zero_division=0
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


def evaluate_controlled(thresholds: dict[str, float], full_score_frame: pd.DataFrame, smoke=False):
    edge_archive = np.load(A1R / "controlled_dynamic_edge_scores.npz", allow_pickle=False)
    aggregate_archive = np.load(A1R / "controlled_dynamic_aggregate_scores.npz", allow_pickle=False)
    controlled = aggregate_archive["target_group_max_then_top2_mean"]
    clean_source = aggregate_archive["clean_target_group_max_then_top2_mean"]
    valid = edge_archive["valid_mask"]
    source_ids = edge_archive["source_window_id"].astype(str)
    sequences = edge_archive["sequence_id"].astype(str)
    endpoints = edge_archive["endpoint_row"].astype(int)
    if smoke:
        controlled, clean_source, valid = controlled[:, :24], clean_source[:24], valid[:, :24]
        source_ids, sequences, endpoints = source_ids[:24], sequences[:24], endpoints[:24]
    half_lookup = full_score_frame.set_index("source_window_id").half.to_dict()
    halves = np.asarray([half_lookup[source_id] for source_id in source_ids])
    folds = np.asarray([evaluation_fold(half) for half in halves])
    threshold_by_source = np.asarray([thresholds[fold] for fold in folds])
    family_rows, sequence_rows = [], []
    for family_index, family in enumerate(FAMILIES):
        mask = valid[family_index]
        positive = controlled[family_index][mask]
        positive_threshold = np.broadcast_to(threshold_by_source[:, None], mask.shape)[mask]
        clean = np.repeat(clean_source, mask.sum(axis=1))
        clean_threshold = np.repeat(threshold_by_source, mask.sum(axis=1))
        result = binary_metrics(clean, positive, clean_threshold, positive_threshold)
        family_rows.append({"family": family, "sample_count": len(positive), **result})
        if family in ("R1", "R2", "R3"):
            for sequence in np.unique(sequences):
                source_mask = sequences == sequence
                local_mask = mask[source_mask]
                local_positive = controlled[family_index][source_mask][local_mask]
                local_positive_threshold = np.broadcast_to(
                    threshold_by_source[source_mask, None], local_mask.shape
                )[local_mask]
                local_clean = np.repeat(clean_source[source_mask], local_mask.sum(axis=1))
                local_clean_threshold = np.repeat(threshold_by_source[source_mask], local_mask.sum(axis=1))
                sequence_rows.append(
                    {
                        "sequence_id": sequence,
                        "family": family,
                        **binary_metrics(
                            local_clean, local_positive, local_clean_threshold, local_positive_threshold
                        ),
                    }
                )
    edge_archive.close()
    aggregate_archive.close()
    return pd.DataFrame(family_rows), pd.DataFrame(sequence_rows)


def sampled_comparison(full_overall, full_sequence):
    archive = np.load(A1R / "controlled_dynamic_aggregate_scores.npz", allow_pickle=False)
    edge = np.load(A1R / "controlled_dynamic_edge_scores.npz", allow_pickle=False)
    scores = archive["clean_target_group_max_then_top2_mean"]
    sequences = edge["sequence_id"].astype(str)
    folds = edge["fold"].astype(str)
    threshold_frame = pd.read_csv(A1R / "target_grouped_crossfit_thresholds.csv")
    threshold_lookup = threshold_frame[
        threshold_frame.aggregation.eq("target_group_max_then_top2_mean")
    ].set_index("fold").threshold.to_dict()
    alarm = scores > np.asarray([threshold_lookup[fold] for fold in folds])
    rows = []
    full_lookup = full_sequence[full_sequence.scope.eq("combined")].set_index("sequence_id")
    for sequence in np.unique(sequences):
        sampled_fpr = float(alarm[sequences == sequence].mean())
        full_fpr = float(full_lookup.loc[sequence, "fpr"])
        rows.append(
            {
                "sequence_id": sequence,
                "sampled_window_count": int((sequences == sequence).sum()),
                "sampled_alarm_count": int(alarm[sequences == sequence].sum()),
                "sampled_fpr": sampled_fpr,
                "full_clean_window_count": int(full_lookup.loc[sequence, "evaluation_window_count"]),
                "full_clean_alarm_count": int(full_lookup.loc[sequence, "alarm_count"]),
                "full_clean_fpr": full_fpr,
                "fpr_difference_full_minus_sampled": full_fpr - sampled_fpr,
                "sampled_0_06_is_6_of_100": bool((sequences == sequence).sum() == 100 and alarm[sequences == sequence].sum() == 6),
            }
        )
    rows.append(
        {
            "sequence_id": "overall",
            "sampled_window_count": len(scores),
            "sampled_alarm_count": int(alarm.sum()),
            "sampled_fpr": float(alarm.mean()),
            "full_clean_window_count": int(full_overall.loc[full_overall.scope.eq("combined"), "evaluation_window_count"].iloc[0]),
            "full_clean_alarm_count": int(full_overall.loc[full_overall.scope.eq("combined"), "alarm_count"].iloc[0]),
            "full_clean_fpr": float(full_overall.loc[full_overall.scope.eq("combined"), "overall_fpr"].iloc[0]),
            "fpr_difference_full_minus_sampled": float(full_overall.loc[full_overall.scope.eq("combined"), "overall_fpr"].iloc[0]) - float(alarm.mean()),
            "sampled_0_06_is_6_of_100": False,
        }
    )
    archive.close()
    edge.close()
    return pd.DataFrame(rows)


def decide(full_overall, controlled_family, controlled_sequence, pool_count, hashes_unchanged):
    combined = full_overall[full_overall.scope.eq("combined")].iloc[0]
    family = controlled_family.set_index("family")
    breaking = family.loc[["R1", "R2", "R3"]]
    sequence_macro = controlled_sequence.groupby("sequence_id").auroc.mean()
    clean_fpr = float(combined.overall_fpr)
    c2_fpr = float(family.loc["C2", "q99_tpr"])
    c2_excess = max(0.0, c2_fpr - clean_fpr)
    gates = {
        "full_clean_overall_fpr": clean_fpr <= 0.02,
        "full_clean_worst_sequence_fpr": float(combined.worst_sequence_fpr) <= 0.05,
        "c2_fpr": c2_fpr <= 0.03,
        "c2_excess_fpr": c2_excess <= 0.015,
        "macro_auroc": float(breaking.auroc.mean()) >= 0.82,
        "macro_aupr": float(breaking.aupr.mean()) >= 0.84,
        "macro_q99_tpr": float(breaking.q99_tpr.mean()) >= 0.40,
        "two_family_auroc": int((breaking.auroc >= 0.85).sum()) >= 2,
        "two_family_q99_tpr": int((breaking.q99_tpr >= 0.40).sum()) >= 2,
        "five_sequences_auroc": int((sequence_macro >= 0.70).sum()) >= 5,
        "p1_fpr": float(family.loc["P1", "q99_tpr"]) <= 0.05,
        "relation_pool": pool_count == 12,
        "finite": bool(np.isfinite(controlled_family.select_dtypes(include=[np.number]).to_numpy()).all()),
        "historical_hashes": hashes_unchanged,
    }
    passed = all(gates.values())
    controlled_core = all(
        gates[key]
        for key in (
            "macro_auroc",
            "macro_aupr",
            "macro_q99_tpr",
            "two_family_auroc",
            "two_family_q99_tpr",
            "five_sequences_auroc",
        )
    )
    if passed:
        classification = "dynamic_conditional_structure_is_valid"
    elif not controlled_core:
        classification = "controlled_detection_degrades_under_full_clean_threshold"
    elif float(combined.worst_sequence_fpr) > 0.05:
        classification = "normal_sequence_specific_instability_is_confirmed"
    else:
        classification = "sampled_clean_fpr_was_pessimistic_but_minor_instability_remains"
    diagnostics = {
        "clean_fpr": clean_fpr,
        "worst_sequence_fpr": float(combined.worst_sequence_fpr),
        "c2_fpr": c2_fpr,
        "c2_excess_fpr": c2_excess,
        "c2_absolute_gap_diagnostic_only": abs(c2_fpr - clean_fpr),
    }
    return classification, gates, passed, diagnostics


def run(smoke=False):
    out = SMOKE_OUT if smoke else OUT
    before = frozen_hashes()
    status_path = out / "stage_status.json"
    dependency_hash = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("final_report_complete") and status.get("dependency_hash") == dependency_hash:
            print(f"REUSE completed V2-A1S: {out}", flush=True)
            return 0
    out.mkdir(parents=True, exist_ok=True)
    status = {
        "frozen_artifacts_verified": False,
        "full_clean_cache_verified": False,
        "crossfit_manifest_complete": False,
        "thresholds_complete": False,
        "full_clean_evaluation_complete": False,
        "controlled_reuse_evaluation_complete": False,
        "final_report_complete": False,
        "dependency_hash": dependency_hash,
    }
    cfg = load_config()
    pool, targets, grouped, edge_matrices = load_full_clean_grouped(smoke)
    status["full_clean_cache_verified"] = True
    timestamps, split = read_validation_timestamps_only(cfg)
    manifest, scores = build_manifest(grouped, timestamps, smoke)
    manifest.to_csv(out / "full_clean_crossfit_window_manifest.csv", index=False)
    overlap = manifest.groupby(["source_window_id", "fold"]).role.nunique().max() - 1
    integrity = pd.DataFrame(
        [
            {
                "sequence_count": manifest.sequence_id.nunique(),
                "source_window_count": manifest.source_window_id.nunique(),
                "manifest_row_count": len(manifest),
                "calibration_evaluation_overlap": int(overlap),
                "all_valid": bool(manifest.valid.all()),
                "minimum_evaluation_windows_per_sequence": int(scores.groupby("sequence_id").size().min()),
                "cross_sequence_windows": 0,
            }
        ]
    )
    integrity.to_csv(out / "full_clean_crossfit_integrity_audit.csv", index=False)
    status["crossfit_manifest_complete"] = True
    thresholds, threshold_frame = fit_full_clean_thresholds(scores)
    threshold_frame.to_csv(out / "full_clean_crossfit_thresholds.csv", index=False)
    status["thresholds_complete"] = True
    alarm_frame, overall, by_sequence, segments = evaluate_full_clean(scores, thresholds)
    overall.to_csv(out / "full_clean_fpr_overall.csv", index=False)
    by_sequence.to_csv(out / "full_clean_fpr_by_sequence.csv", index=False)
    segments.to_csv(out / "full_clean_alarm_segments.csv", index=False)
    status["full_clean_evaluation_complete"] = True
    comparison = sampled_comparison(overall, by_sequence)
    comparison.to_csv(out / "sampled_vs_full_clean_comparison.csv", index=False)
    controlled_family, controlled_sequence = evaluate_controlled(thresholds, scores, smoke)
    controlled_family.to_csv(out / "controlled_metrics_full_clean_threshold.csv", index=False)
    controlled_family.to_csv(out / "controlled_metrics_by_family.csv", index=False)
    controlled_sequence.to_csv(out / "controlled_metrics_by_sequence.csv", index=False)
    controlled_family[controlled_family.family.eq("P1")].to_csv(out / "p1_boundary_metrics.csv", index=False)
    p2 = controlled_family[controlled_family.family.eq("P2")].copy()
    p2["semantic_role"] = "mixed_point_and_collaborative_anomaly"
    p2["hard_gate"] = False
    p2.to_csv(out / "p2_mixed_diagnostics.csv", index=False)
    c1_source = pd.read_csv(A1R / "c1_edge_contribution_decomposition.csv")
    c1_source["semantic_role"] = "collateral_relation_diagnostic"
    c1_source["hard_gate"] = False
    c1_source.to_csv(out / "c1_collateral_diagnostics.csv", index=False)
    status["controlled_reuse_evaluation_complete"] = True
    after = frozen_hashes()
    hash_rows = [
        {
            "artifact": name,
            "sha256_before": value,
            "sha256_after": after.get(name),
            "unchanged": value == after.get(name),
        }
        for name, value in before.items()
    ]
    hash_frame = pd.DataFrame(hash_rows)
    hash_frame.to_csv(out / "artifact_reuse_audit.csv", index=False)
    hashes_unchanged = bool(hash_frame.unchanged.all())
    cache_integrity = pd.DataFrame(
        [
            {
                "sequence_count": len(grouped),
                "relation_count": len(pool),
                "all_clean_finite": bool(all(np.isfinite(values).all() for values in grouped)),
                "all_controlled_finite": bool(
                    np.isfinite(np.load(A1R / "controlled_dynamic_edge_scores.npz", allow_pickle=False)["edge_scores"]).all()
                ),
                "clean_window_counts": json.dumps([len(values) for values in grouped]),
                "rolling_warmup_excluded_per_sequence": 15,
                "history_hashes_unchanged": hashes_unchanged,
                "models_loaded": False,
            }
        ]
    )
    cache_integrity.to_csv(out / "full_clean_cache_integrity_audit.csv", index=False)
    classification, gates, passed, diagnostics = decide(
        overall, controlled_family, controlled_sequence, len(pool), hashes_unchanged
    )
    c2 = controlled_family[controlled_family.family.eq("C2")].copy()
    c2["clean_fpr"] = diagnostics["clean_fpr"]
    c2["fpr"] = c2.q99_tpr
    c2["excess_fpr"] = np.maximum(0.0, c2.fpr - c2.clean_fpr)
    c2["absolute_gap_diagnostic_only"] = np.abs(c2.fpr - c2.clean_fpr)
    c2.to_csv(out / "c2_preserving_control_metrics.csv", index=False)
    selected = {
        "status": "frozen_for_hai_development_evaluation" if passed else "not_frozen",
        "aggregation": "target_group_max_then_top2_mean",
        "threshold_protocol": "full_clean_chronological_two_fold_global_q99",
        "hai_test_used": False,
    }
    (out / "selected_structure_v2a1s_config.yaml").write_text(
        yaml.safe_dump(selected, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "classification": classification,
        "all_gates_passed": passed,
        "gates": {key: bool(value) for key, value in gates.items()},
        **diagnostics,
        "test_loaded": False,
        "attack_label_used": False,
        "rec_loaded": False,
        "ae_loaded": False,
    }
    (out / "structure_v2a1s_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    issues = [
        {
            "issue_id": "V2A1S-001",
            "stage": "HAI Structure V2-A1S",
            "research_question": "Is target-grouped clean stability confirmed on all validation windows?",
            "status": "resolved" if passed else "open",
            "evidence": classification,
            "severity": "low" if passed else "high",
            "affects_validity": not passed,
            "affects_final_method": True,
            "minimum_future_action": "No automatic follow-up experiment",
            "recomputation_scope": "Offline full-clean thresholds only",
            "frozen_artifacts_reusable": True,
        }
    ]
    (out / "unresolved_issues_delta.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")
    report = f"""# HAI Structure V2-A1S

- Candidate: `target_group_max_then_top2_mean` (the only candidate).
- Full-clean overall FPR: **{diagnostics['clean_fpr']:.6f}**.
- Full-clean worst-sequence FPR: **{diagnostics['worst_sequence_fpr']:.6f}**.
- C2 FPR / one-sided excess FPR: **{diagnostics['c2_fpr']:.6f} / {diagnostics['c2_excess_fpr']:.6f}**.
- Old absolute C2 gap (diagnostic only): **{diagnostics['c2_absolute_gap_diagnostic_only']:.6f}**.
- Classification: **{classification}**.
- Frozen for HAI development evaluation: **{str(passed).lower()}**.
- HAI test / attack label / AE / rec loaded: **false / false / false / false**.
"""
    (out / "final_hai_structure_v2a1s_full_clean_stability.md").write_text(report, encoding="utf-8")
    status["frozen_artifacts_verified"] = hashes_unchanged
    status["final_report_complete"] = True
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"classification={classification} all_gates_passed={str(passed).lower()}", flush=True)
    print("test_loaded=false attack_label_used=false rec_loaded=false ae_loaded=false", flush=True)
    return 0


def dry_run():
    archive = np.load(V2A / "validation_dynamic_edge_score_cache.npz", allow_pickle=False)
    shapes = {key: list(archive[key].shape) for key in archive.files}
    archive.close()
    print("REUSE: V2-A 12-edge full-clean validation score cache and model/reference hashes", flush=True)
    print("REUSE: A1R-Fix controlled edge/aggregate score cache; no corruption regeneration", flush=True)
    print("COMPUTE_NEW: unique grouped aggregation full-clean chronological two-fold q99 audit", flush=True)
    print("TIMESTAMP_ONLY: read frozen normal-validation timestamp column using Step2 split", flush=True)
    print("FORBIDDEN: model load/fit, edge-score recompute, HAI test, attack label, AE, rec, new aggregation", flush=True)
    print(f"clean_cache_shapes={json.dumps(shapes, sort_keys=True)} rolling_warmup_excluded=15", flush=True)
    print("candidate=target_group_max_then_top2_mean quantile=0.99 random_sampling=false", flush=True)
    print("test_loaded=false attack_label_used=false rec_loaded=false ae_loaded=false", flush=True)
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
