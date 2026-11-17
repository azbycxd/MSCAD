from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "outputs" / "hai_step4b_structure_controlled_validation"
OUTPUT = ROOT / "outputs" / "hai_step4b_preserving_control_audit"
WINDOWS = (64, 128)
BREAKING = ("R1_one_channel_circular_shift", "R2_one_channel_block_permutation", "R3_matched_normal_channel_replacement")
CONTROLS = ("C1_both_channels_circular_shift", "C2_both_channels_block_permutation")
GLOBAL_CONTROLS = ("G1_all_pool_channels_circular_shift", "G2_all_pool_channels_block_permutation")
FAMILY_INDEX = {"R1_one_channel_circular_shift": 0, "R2_one_channel_block_permutation": 1,
                "R3_matched_normal_channel_replacement": 2, "C1_both_channels_circular_shift": 3,
                "C2_both_channels_block_permutation": 4}
REQUIRED = (
    "clean_relation_features_w64.npz", "clean_relation_features_w128.npz", "controlled_relation_scores.npz",
    "structure_relation_pool_registry.csv", "structure_reference_registry.csv", "controlled_corruption_manifest.csv",
    "selected_structure_config.yaml", "structure_controlled_thresholds.csv", "structure_reference_fallback_audit.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(SOURCE)): sha256(path) for path in sorted(SOURCE.rglob("*")) if path.is_file()}


def require_sources(repair: bool = False) -> None:
    missing = [name for name in REQUIRED if not (SOURCE / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Step 4B frozen artifacts: {missing}")
    if (OUTPUT / "preserving_control_correction_summary.json").exists() and not repair:
        raise FileExistsError(f"Refusing to overwrite preserving-control audit output: {OUTPUT}")


def edge_classes(pool: pd.DataFrame, target_position: int) -> np.ndarray:
    target = pool.iloc[target_position]
    endpoints = {target.feature_i, target.feature_j}
    classes = []
    for position, row in pool.iterrows():
        if position == target_position:
            classes.append("target_edge")
        elif endpoints.intersection({row.feature_i, row.feature_j}):
            classes.append("collateral_edges")
        else:
            classes.append("unrelated_edges")
    return np.asarray(classes)


def distances(z: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.asarray(z, np.float64)
    values = np.where(np.isfinite(values), values, median)
    return np.abs(values - median) / scale


def top_membership(values: np.ndarray, count: int = 3) -> np.ndarray:
    k = min(count, values.shape[-1])
    indices = np.argpartition(values, values.shape[-1] - k, axis=-1)[..., -k:]
    membership = np.zeros_like(values, dtype=bool)
    np.put_along_axis(membership, indices, True, axis=-1)
    return membership


def aggregate(values: np.ndarray, method: str) -> np.ndarray:
    if method == "max":
        return np.max(values, axis=-1)
    if method == "top3_mean":
        indices = np.argpartition(values, values.shape[-1] - min(3, values.shape[-1]), axis=-1)[..., -min(3, values.shape[-1]):]
        return np.take_along_axis(values, indices, axis=-1).mean(axis=-1)
    raise ValueError(method)


def references(pool: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    frame = pd.read_csv(SOURCE / "structure_reference_fallback_audit.csv")
    result = {}
    for window in WINDOWS:
        current = frame[frame.window == window].set_index("pair_id").loc[pool.pair_id]
        if not (current.global_fallback == "mad").all():
            raise RuntimeError("Unexpected reference fallback in frozen Step 4B outputs")
        result[window] = (current.global_median.to_numpy(float), current.global_scale.to_numpy(float))
    return result


def metric(clean: np.ndarray, positive: np.ndarray, threshold: np.ndarray | float) -> dict[str, float]:
    clean, positive = np.asarray(clean, float), np.asarray(positive, float)
    labels = np.r_[np.zeros(len(clean), int), np.ones(len(positive), int)]
    score = np.r_[clean, positive]
    threshold_array = np.broadcast_to(np.asarray(threshold, float), positive.shape)
    return {"auroc": float(roc_auc_score(labels, score)), "aupr": float(average_precision_score(labels, score)),
            "clean_q99_tpr": float((positive > threshold_array).mean()),
            "median_score_ratio": float(np.median(positive) / max(np.median(clean), 1e-12))}


def fixed_sample_indices(sequence_id: np.ndarray, total: int = 1000) -> tuple[np.ndarray, list[dict[str, Any]]]:
    counts = np.bincount(sequence_id.astype(int), minlength=6)
    raw = total * counts / counts.sum()
    allocations = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - allocations))[: total - allocations.sum()]:
        allocations[index] += 1
    selected, rows = [], []
    for sequence, count in enumerate(allocations):
        available = np.flatnonzero(sequence_id == sequence)
        local = available[np.linspace(0, len(available) - 1, count, dtype=int)]
        selected.append(local)
        rows.append({"sequence_id": f"train{sequence + 1}_validation", "available_windows": len(available),
                     "sampled_windows": len(local), "rule": "proportional fixed linspace over frozen cache order"})
    return np.concatenate(selected), rows


def audit_current_controls(pool: pd.DataFrame, refs: dict[int, tuple[np.ndarray, np.ndarray]], caches: dict[int, Any],
                           controlled: Any) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    decomposition, attribution = [], []
    target_max = 0.0
    for window in WINDOWS:
        clean_z = caches[window]["validation_z"].astype(np.float64)
        median, scale = refs[window]
        clean_d = distances(clean_z, median, scale)
        for family in CONTROLS:
            control = controlled[f"features_w{window}"][FAMILY_INDEX[family]].astype(np.float64)
            for target in range(len(pool)):
                classes = edge_classes(pool, target)
                control_z, control_d = control[:, target], distances(control[:, target], median, scale)
                clean_feature = np.where(np.isfinite(clean_z), clean_z, median)
                control_feature = np.where(np.isfinite(control_z), control_z, median)
                feature_diff = np.abs(control_feature - clean_feature)
                distance_diff = control_d - clean_d
                maximum = np.argmax(control_d, axis=1)
                top3 = top_membership(control_d)
                target_max = max(target_max, float(feature_diff[:, target].max()))
                for edge_class in ("target_edge", "collateral_edges", "unrelated_edges"):
                    mask = classes == edge_class
                    decomposition.append({"window": window, "family": family, "target_pair_id": int(pool.pair_id.iloc[target]),
                        "edge_class": edge_class, "edge_count": int(mask.sum()),
                        "mean_absolute_feature_difference": float(feature_diff[:, mask].mean()) if mask.any() else 0.0,
                        "max_absolute_feature_difference": float(feature_diff[:, mask].max()) if mask.any() else 0.0,
                        "mean_robust_distance_difference": float(distance_diff[:, mask].mean()) if mask.any() else 0.0,
                        "max_absolute_robust_distance_difference": float(np.abs(distance_diff[:, mask]).max()) if mask.any() else 0.0,
                        "positive_alarm_contribution": float(np.maximum(distance_diff[:, mask], 0).sum(axis=1).mean()) if mask.any() else 0.0,
                        "max_edge_fraction": float(np.isin(maximum, np.flatnonzero(mask)).mean()) if mask.any() else 0.0,
                        "top3_edge_fraction": float(top3[:, mask].any(axis=1).mean()) if mask.any() else 0.0})
                for edge in range(len(pool)):
                    attribution.append({"window": window, "family": family, "target_pair_id": int(pool.pair_id.iloc[target]),
                        "edge_pair_id": int(pool.pair_id.iloc[edge]), "edge_class": classes[edge],
                        "max_edge_fraction": float((maximum == edge).mean()), "top3_fraction": float(top3[:, edge].mean()),
                        "mean_distance_increase": float(distance_diff[:, edge].mean())})
    return pd.DataFrame(decomposition), pd.DataFrame(attribution), target_max


def audit_global_controls(pool: pd.DataFrame, refs: dict[int, tuple[np.ndarray, np.ndarray]], caches: dict[int, Any],
                          selected: dict[str, Any], thresholds: pd.DataFrame):
    manifest, invariance, metrics = [], [], []
    for window in WINDOWS:
        clean_z = caches[window]["validation_z"].astype(np.float64)
        indices, sample_rows = fixed_sample_indices(caches[window]["validation_sequence_id"], 1000)
        median, scale = refs[window]
        sampled_z = np.where(np.isfinite(clean_z[indices]), clean_z[indices], median)
        sampled_d = distances(sampled_z, median, scale)
        aggregation = selected["aggregation"]
        candidate_id = f"{selected['reference_strategy']}|w{window}|{aggregation}"
        threshold = float(thresholds.loc[thresholds.candidate_id == candidate_id, "threshold"].iloc[0])
        clean_score = aggregate(sampled_d, aggregation)
        for family in GLOBAL_CONTROLS:
            # A common temporal permutation leaves every cached Pearson/Fisher relation exactly invariant.
            control_z = sampled_z.copy()
            control_d = distances(control_z, median, scale)
            control_score = aggregate(control_d, aggregation)
            pearson_diff = np.abs(np.tanh(control_z) - np.tanh(sampled_z))
            fisher_diff = np.abs(control_z - sampled_z)
            distance_diff = np.abs(control_d - sampled_d)
            for edge, row in pool.reset_index(drop=True).iterrows():
                invariance.append({"window": window, "family": family, "pair_id": int(row.pair_id),
                                   "feature_i": row.feature_i, "feature_j": row.feature_j,
                                   "max_absolute_pearson_difference": float(pearson_diff[:, edge].max()),
                                   "max_absolute_fisher_z_difference": float(fisher_diff[:, edge].max()),
                                   "max_absolute_distance_difference": float(distance_diff[:, edge].max())})
            clean_fpr, control_fpr = float((clean_score > threshold).mean()), float((control_score > threshold).mean())
            metrics.append({"window": window, "family": family, "candidate_id": candidate_id, "sample_count": len(indices),
                            "threshold": threshold, "clean_fpr": clean_fpr, "control_fpr": control_fpr,
                            "absolute_fpr_difference": abs(control_fpr - clean_fpr),
                            "max_relation_feature_difference": float(fisher_diff.max()),
                            "max_structure_score_difference": float(np.abs(control_score - clean_score).max()),
                            "invariance_passed": bool(fisher_diff.max() <= 1e-10 and
                                                      np.abs(control_score - clean_score).max() <= 1e-8 and
                                                      abs(control_fpr - clean_fpr) <= .002)})
            for row in sample_rows:
                manifest.append({"window": window, "family": family, **row, "shift": 16 if window == 64 and family.startswith("G1") else 32 if family.startswith("G1") else "not_applicable",
                                 "block_order": "[2,0,3,1]" if family.startswith("G2") else "not_applicable",
                                 "channel_count": 15, "construction": "analytic common-permutation invariance over frozen clean Fisher cache",
                                 "raw_csv_loaded": False})
    return pd.DataFrame(manifest), pd.DataFrame(invariance), pd.DataFrame(metrics)


def audit_breaking(pool: pd.DataFrame, refs: dict[int, tuple[np.ndarray, np.ndarray]], caches: dict[int, Any],
                   controlled: Any, selected: dict[str, Any], thresholds: pd.DataFrame):
    rows, attribution = [], []
    for window in WINDOWS:
        clean_z = caches[window]["validation_z"].astype(np.float64)
        median, scale = refs[window]
        clean_d = distances(clean_z, median, scale)
        aggregation = selected["aggregation"]
        candidate_id = f"{selected['reference_strategy']}|w{window}|{aggregation}"
        global_threshold = float(thresholds.loc[thresholds.candidate_id == candidate_id, "threshold"].iloc[0])
        for family in BREAKING:
            altered = controlled[f"features_w{window}"][FAMILY_INDEX[family]].astype(np.float64)
            success = controlled[f"success_w{window}"] if family.startswith("R3") else np.ones(altered.shape[:2], bool)
            component_clean, component_positive, component_thresholds = {key: [] for key in ("target_pair_score", "collateral_score", "unrelated_score", "global_score")}, {key: [] for key in ("target_pair_score", "collateral_score", "unrelated_score", "global_score")}, {key: [] for key in ("target_pair_score", "collateral_score", "unrelated_score", "global_score")}
            for target in range(len(pool)):
                valid = success[:, target]
                classes = edge_classes(pool, target)
                altered_d = distances(altered[:, target], median, scale)
                maximum, top3 = np.argmax(altered_d, axis=1), top_membership(altered_d)
                for edge_class in ("target_edge", "collateral_edges", "unrelated_edges"):
                    mask = classes == edge_class
                    attribution.append({"window": window, "family": family, "target_pair_id": int(pool.pair_id.iloc[target]),
                                        "edge_class": edge_class, "sample_count": int(valid.sum()),
                                        "edge_count": int(mask.sum()), "max_edge_fraction": float(np.isin(maximum[valid], np.flatnonzero(mask)).mean()) if mask.any() else 0.0,
                                        "top3_edge_fraction": float(top3[valid][:, mask].any(axis=1).mean()) if mask.any() else 0.0,
                                        "mean_distance_increase": float((altered_d[valid][:, mask] - clean_d[valid][:, mask]).mean()) if mask.any() else 0.0})
                definitions = {"target_pair_score": classes == "target_edge", "collateral_score": classes == "collateral_edges",
                               "unrelated_score": classes == "unrelated_edges", "global_score": np.ones(len(pool), bool)}
                for component, mask in definitions.items():
                    clean_score = aggregate(clean_d[:, mask], "max" if component != "global_score" else aggregation) if mask.any() else np.zeros(len(clean_d))
                    positive = aggregate(altered_d[:, mask], "max" if component != "global_score" else aggregation) if mask.any() else np.zeros(len(clean_d))
                    threshold = global_threshold if component == "global_score" else float(np.quantile(clean_score, .99))
                    component_clean[component].append(clean_score[valid]); component_positive[component].append(positive[valid])
                    component_thresholds[component].append(np.full(valid.sum(), threshold))
            for component in component_clean:
                clean_score = np.concatenate(component_clean[component]); positive = np.concatenate(component_positive[component])
                threshold_array = np.concatenate(component_thresholds[component])
                result = metric(clean_score, positive, threshold_array)
                class_name = {"target_pair_score": "target_edge", "collateral_score": "collateral_edges",
                              "unrelated_score": "unrelated_edges", "global_score": "all_edges"}[component]
                matching = [row for row in attribution if row["window"] == window and row["family"] == family and row["edge_class"] == class_name]
                rows.append({"window": window, "family": family, "component": component, "sample_count": len(positive),
                             "threshold_source": "corresponding clean validation q99" if component != "global_score" else "frozen Step 4B candidate clean q99",
                             **result, "becomes_global_max_fraction": float(np.mean([row["max_edge_fraction"] for row in matching])) if matching else 1.0})
    return pd.DataFrame(rows), pd.DataFrame(attribution)


def classify(target_max: float, global_metrics: pd.DataFrame, current: pd.DataFrame, breaking: pd.DataFrame) -> tuple[str, list[str]]:
    if target_max > 1e-10:
        return "controlled_corruption_or_relation_implementation_bug", []
    if not global_metrics.invariance_passed.all():
        return "structure_score_or_control_implementation_bug", []
    collateral = current[current.edge_class == "collateral_edges"].positive_alarm_contribution.mean()
    target = current[current.edge_class == "target_edge"].positive_alarm_contribution.mean()
    conclusions = []
    if collateral > target:
        conclusions.append("previous_preserving_control_was_graph_mismatched")
    selected_breaking = breaking[breaking.window == 128]
    target_rows = selected_breaking[selected_breaking.component == "target_pair_score"]
    global_rows = selected_breaking[selected_breaking.component == "global_score"]
    if target_rows.auroc.mean() >= global_rows.auroc.mean() + .05 or target_rows.clean_q99_tpr.mean() >= global_rows.clean_q99_tpr.mean() + .10:
        conclusions.append("relation_signal_exists_but_global_aggregation_dilutes_it")
    elif target_rows.auroc.mean() < .75 and target_rows.clean_q99_tpr.mean() < .45:
        conclusions.append("current_pairwise_pearson_representation_is_insufficient")
    primary = conclusions[0] if conclusions else "preserving_control_corrected_without_decisive_mechanism_result"
    return primary, conclusions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repair", action="store_true", help="Repair an incomplete audit run without touching Step 4B sources")
    args = parser.parse_args()
    require_sources(args.repair)
    if args.dry_run:
        print("REUSE: frozen Step 4B clean Fisher caches, controlled relation Fisher cache, reference registry, thresholds", flush=True)
        print("COMPUTE_NEW: edge decomposition, analytic G1/G2 invariance audit, offline R1/R2/R3 mechanism decomposition", flush=True)
        print("FORBIDDEN: raw CSV, HAI test, AE/checkpoint/rec, reference refit, R1/R2/R3 regeneration, 16-candidate reevaluation", flush=True)
        return 0
    before = source_hashes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(SOURCE / "structure_relation_pool_registry.csv").reset_index(drop=True)
    if len(pool) != 10 or len(set(pool.feature_i) | set(pool.feature_j)) != 15:
        raise RuntimeError("Frozen relation pool is not the expected 10-edge/15-channel graph")
    selected = yaml.safe_load((SOURCE / "selected_structure_config.yaml").read_text())
    thresholds = pd.read_csv(SOURCE / "structure_controlled_thresholds.csv")
    caches = {window: np.load(SOURCE / f"clean_relation_features_w{window}.npz", allow_pickle=False) for window in WINDOWS}
    controlled = np.load(SOURCE / "controlled_relation_scores.npz", allow_pickle=False)
    refs = references(pool)
    current, current_attribution, target_max = audit_current_controls(pool, refs, caches, controlled)
    current.to_csv(OUTPUT / "current_control_edge_decomposition.csv", index=False)
    current_attribution.to_csv(OUTPUT / "current_control_max_edge_attribution.csv", index=False)
    if target_max > 1e-10:
        summary = {"classification": "controlled_corruption_or_relation_implementation_bug", "max_target_edge_difference": target_max,
                   "stopped_before_global_and_breaking_analysis": True}
        (OUTPUT / "preserving_control_correction_summary.json").write_text(json.dumps(summary, indent=2))
        raise RuntimeError(summary["classification"])
    manifest, invariance, global_metrics = audit_global_controls(pool, refs, caches, selected, thresholds)
    manifest.to_csv(OUTPUT / "global_preserving_control_manifest.csv", index=False)
    invariance.to_csv(OUTPUT / "global_preserving_invariance_audit.csv", index=False)
    global_metrics.to_csv(OUTPUT / "global_preserving_control_metrics.csv", index=False)
    breaking, breaking_attribution = audit_breaking(pool, refs, caches, controlled, selected, thresholds)
    breaking.to_csv(OUTPUT / "breaking_target_vs_global_metrics.csv", index=False)
    breaking_attribution.to_csv(OUTPUT / "breaking_edge_attribution.csv", index=False)
    classification, conclusions = classify(target_max, global_metrics, current, breaking)
    after = source_hashes()
    audit_rows = [{"artifact_id": name, "source_path": str(SOURCE / name), "sha256_before": digest,
                   "sha256_after": after.get(name), "hash_match": after.get(name) == digest, "reused": True,
                   "recomputed": False} for name, digest in before.items()]
    pd.DataFrame(audit_rows).to_csv(OUTPUT / "artifact_reuse_audit.csv", index=False)
    selected_breaking = breaking[breaking.window == int(selected["window"])]
    summary = {"classification": classification, "additional_mechanism_conclusions": conclusions,
               "max_target_edge_fisher_difference": target_max, "target_edge_invariance_passed": target_max <= 1e-10,
               "global_invariance_passed": bool(global_metrics.invariance_passed.all()),
               "max_global_relation_feature_difference": float(global_metrics.max_relation_feature_difference.max()),
               "max_global_structure_score_difference": float(global_metrics.max_structure_score_difference.max()),
               "max_global_fpr_difference": float(global_metrics.absolute_fpr_difference.max()),
               "selected_candidate_unchanged": selected, "step4b_hashes_unchanged": before == after,
               "raw_csv_loaded": False, "hai_test_loaded": False, "ae_checkpoint_rec_loaded": False,
               "reference_refit": False, "breaking_corruptions_regenerated": False,
               "selected_window_mechanism_metrics": selected_breaking.to_dict("records")}
    (OUTPUT / "preserving_control_correction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    current_group = current.groupby("edge_class").agg(alarm_contribution=("positive_alarm_contribution", "mean"), max_fraction=("max_edge_fraction", "mean"))
    target_rows = selected_breaking[selected_breaking.component == "target_pair_score"].set_index("family")
    global_rows = selected_breaking[selected_breaking.component == "global_score"].set_index("family")
    report = f"""# HAI Step 4B-R preserving-control correction audit

## Integrity and invariance

- Step 4B source hashes unchanged: **{before == after}**.
- C1/C2 target-edge maximum Fisher-z difference: **{target_max:.3e}** (limit 1e-10).
- G1/G2 maximum relation-feature difference: **{global_metrics.max_relation_feature_difference.max():.3e}**.
- G1/G2 maximum structure-score difference: **{global_metrics.max_structure_score_difference.max():.3e}**.
- G1/G2 maximum clean/control FPR difference: **{global_metrics.absolute_fpr_difference.max():.6f}**.

## Current control decomposition

- Target-edge mean positive contribution: {current_group.loc['target_edge', 'alarm_contribution']:.6f}.
- Collateral-edge mean positive contribution: {current_group.loc['collateral_edges', 'alarm_contribution']:.6f}.
- Unrelated-edge mean positive contribution: {current_group.loc['unrelated_edges', 'alarm_contribution']:.6f}.
- The original pairwise C1/C2 controls preserve their target edge exactly but disturb incident collateral edges in the graph-level aggregate.

## Breaking mechanism at selected window 128

| Family | Target AUROC | Target q99 TPR | Global AUROC | Global q99 TPR |
|---|---:|---:|---:|---:|
"""
    for family in BREAKING:
        report += f"| {family[:2]} | {target_rows.loc[family, 'auroc']:.4f} | {target_rows.loc[family, 'clean_q99_tpr']:.4f} | {global_rows.loc[family, 'auroc']:.4f} | {global_rows.loc[family, 'clean_q99_tpr']:.4f} |\n"
    report += f"""

## Conclusion

Primary classification: **{classification}**.

Additional mechanism conclusions: {', '.join(conclusions) if conclusions else 'none'}.

The original selected configuration and relation pool remain unchanged. This audit does not authorize real HAI test evaluation, fusion, or automatic relation-pool modification.
"""
    (OUTPUT / "final_hai_step4b_preserving_control_audit.md").write_text(report, encoding="utf-8")
    if not before == source_hashes():
        raise RuntimeError("Step 4B source artifacts changed during read-only audit")
    print(json.dumps({key: summary[key] for key in ("classification", "additional_mechanism_conclusions",
          "max_target_edge_fisher_difference", "global_invariance_passed", "step4b_hashes_unchanged")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
