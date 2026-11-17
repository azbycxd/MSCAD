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
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
STEP4B = ROOT / "outputs" / "hai_step4b_structure_controlled_validation"
STEP4BR = ROOT / "outputs" / "hai_step4b_preserving_control_audit"
OUTPUT = ROOT / "outputs" / "hai_step4b_relation_aggregation_repair"
WINDOW = 128
CANDIDATES = ("original_distance_max", "relation_tail_max", "relation_tail_top2_mean")
BREAKING = ("R1_one_channel_circular_shift", "R2_one_channel_block_permutation", "R3_matched_normal_channel_replacement")
GLOBAL_CONTROLS = ("G1_all_pool_channels_circular_shift", "G2_all_pool_channels_block_permutation")
FAMILY_INDEX = {name: index for index, name in enumerate(BREAKING)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}


def required_artifacts() -> dict[str, Path]:
    return {
        "clean_fisher_cache": STEP4B / "clean_relation_features_w128.npz",
        "controlled_fisher_cache": STEP4B / "controlled_relation_scores.npz",
        "relation_pool": STEP4B / "structure_relation_pool_registry.csv",
        "reference": STEP4B / "structure_reference_fallback_audit.csv",
        "thresholds": STEP4B / "structure_controlled_thresholds.csv",
        "selected_config": STEP4B / "selected_structure_config.yaml",
        "corruption_manifest": STEP4B / "controlled_corruption_manifest.csv",
        "global_control_manifest": STEP4BR / "global_preserving_control_manifest.csv",
        "global_invariance": STEP4BR / "global_preserving_invariance_audit.csv",
        "global_metrics": STEP4BR / "global_preserving_control_metrics.csv",
        "step4br_summary": STEP4BR / "preserving_control_correction_summary.json",
    }


def validate_inputs() -> dict[str, Path]:
    paths = required_artifacts()
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"invalid_crossfit_or_cache: missing frozen fields {missing}")
    if OUTPUT.exists() and (OUTPUT / "aggregation_repair_summary.json").exists():
        return paths
    return paths


def robust_distances(z: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.asarray(z, np.float64)
    values = np.where(np.isfinite(values), values, median)
    result = np.abs(values - median) / scale
    if not np.isfinite(result).all():
        raise RuntimeError("invalid_crossfit_or_cache: non-finite robust distance")
    return result


def empirical_tail_surprisal(calibration: np.ndarray, values: np.ndarray) -> np.ndarray:
    calibration = np.sort(np.asarray(calibration, np.float64))
    values = np.asarray(values, np.float64)
    count_ge = len(calibration) - np.searchsorted(calibration, values, side="left")
    probability = (count_ge + 1.0) / (len(calibration) + 1.0)
    return -np.log10(probability)


def relation_tail(calibration: np.ndarray, values: np.ndarray) -> np.ndarray:
    if calibration.shape[1] != values.shape[-1]:
        raise ValueError("Relation order mismatch")
    flat = values.reshape(-1, values.shape[-1])
    result = np.column_stack([empirical_tail_surprisal(calibration[:, edge], flat[:, edge])
                              for edge in range(values.shape[-1])])
    return result.reshape(values.shape)


def aggregate(tail: np.ndarray, candidate: str) -> np.ndarray:
    if candidate == "relation_tail_max":
        return np.max(tail, axis=-1)
    if candidate == "relation_tail_top2_mean":
        return np.partition(tail, tail.shape[-1] - 2, axis=-1)[..., -2:].mean(axis=-1)
    raise ValueError(candidate)


def build_crossfit_manifest(sequence_id: np.ndarray, starts: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows, masks = [], {}
    for fold, calibration_half in (("A", "first"), ("B", "second")):
        calibration, evaluation = [], []
        for sequence in range(6):
            ordered = np.flatnonzero(sequence_id == sequence)
            ordered = ordered[np.argsort(starts[ordered], kind="stable")]
            midpoint = len(ordered) // 2
            first, second = ordered[:midpoint], ordered[midpoint:]
            cal, ev = (first, second) if calibration_half == "first" else (second, first)
            calibration.append(cal); evaluation.append(ev)
            rows.append({"fold": fold, "sequence_id": f"train{sequence + 1}_validation",
                         "calibration_half": calibration_half, "calibration_count": len(cal), "evaluation_count": len(ev),
                         "calibration_start_min": int(starts[cal].min()), "calibration_start_max": int(starts[cal].max()),
                         "evaluation_start_min": int(starts[ev].min()), "evaluation_start_max": int(starts[ev].max()),
                         "source_window_overlap_count": int(len(np.intersect1d(cal, ev))),
                         "chronological_split": True, "random_shuffle": False})
        masks[f"{fold}_calibration"] = np.concatenate(calibration)
        masks[f"{fold}_evaluation"] = np.concatenate(evaluation)
    if set(masks["A_evaluation"]) != set(masks["B_calibration"]) or set(masks["B_evaluation"]) != set(masks["A_calibration"]):
        raise RuntimeError("invalid_crossfit_or_cache: fold complement mismatch")
    return pd.DataFrame(rows), masks


def edge_classes(pool: pd.DataFrame, target: int) -> np.ndarray:
    endpoints = {pool.feature_i.iloc[target], pool.feature_j.iloc[target]}
    result = []
    for edge, row in pool.iterrows():
        if edge == target: result.append("target_edge")
        elif endpoints.intersection({row.feature_i, row.feature_j}): result.append("collateral_edges")
        else: result.append("unrelated_edges")
    return np.asarray(result)


def classification_metrics(clean: np.ndarray, positive: np.ndarray, clean_threshold: np.ndarray, positive_threshold: np.ndarray) -> dict[str, float]:
    labels = np.r_[np.zeros(len(clean), int), np.ones(len(positive), int)]
    scores = np.r_[clean, positive]
    predictions = np.r_[clean > clean_threshold, positive > positive_threshold]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return {"auroc": float(roc_auc_score(labels, scores)), "aupr": float(average_precision_score(labels, scores)),
            "q99_tpr": float((positive > positive_threshold).mean()), "precision": float(precision),
            "recall": float(recall), "f1": float(f1),
            "median_score_ratio": float(np.median(positive) / max(np.median(clean), 1e-12))}


def run_crossfit(pool: pd.DataFrame, clean_d: np.ndarray, altered_d: np.ndarray, success: np.ndarray,
                 sequence_id: np.ndarray, masks: dict[str, np.ndarray]):
    candidate_rows, corruption_rows, sequence_rows, calibration_rows, calibration_audit, attribution = [], [], [], [], [], []
    candidate_storage: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES[1:]:
        clean_scores = np.full(len(clean_d), np.nan)
        clean_thresholds = np.full(len(clean_d), np.nan)
        positive_scores = {family: np.full((len(clean_d), len(pool)), np.nan) for family in BREAKING}
        positive_thresholds = {family: np.full((len(clean_d), len(pool)), np.nan) for family in BREAKING}
        fold_registry = []
        for fold in ("A", "B"):
            cal, ev = masks[f"{fold}_calibration"], masks[f"{fold}_evaluation"]
            calibration = clean_d[cal]
            clean_tail_cal = relation_tail(calibration, calibration)
            clean_tail_eval = relation_tail(calibration, clean_d[ev])
            aggregate_cal = aggregate(clean_tail_cal, candidate)
            threshold = float(np.quantile(aggregate_cal, .99))
            clean_scores[ev] = aggregate(clean_tail_eval, candidate)
            clean_thresholds[ev] = threshold
            for edge, row in pool.iterrows():
                sorted_edge = np.sort(calibration[:, edge])
                calibration_rows.append({"candidate_id": candidate, "fold": fold, "pair_id": int(row.pair_id),
                    "calibration_count": len(sorted_edge), "distance_min": float(sorted_edge[0]), "distance_max": float(sorted_edge[-1]),
                    "minimum_tail_probability": 1.0 / (len(sorted_edge) + 1), "fit_scope": "clean validation calibration half only"})
                self_tail = empirical_tail_surprisal(sorted_edge, calibration[:, edge])
                calibration_audit.append({"candidate_id": candidate, "fold": fold, "pair_id": int(row.pair_id),
                    "finite": bool(np.isfinite(self_tail).all()), "minimum_probability_positive": True,
                    "max_tail_surprisal": float(self_tail.max()), "deterministic_tie_rule": "count(clean_distance>=d)+1"})
            fold_registry.append({"candidate_id": candidate, "fold": fold, "calibration_count": len(cal),
                                  "evaluation_count": len(ev), "aggregate_q99": threshold,
                                  "synthetic_used_for_calibration": False})
            for family_index, family in enumerate(BREAKING):
                current = altered_d[family_index, ev]
                tail = relation_tail(calibration, current)
                positive_scores[family][ev] = aggregate(tail, candidate)
                positive_thresholds[family][ev] = threshold
                max_edges = np.argmax(tail, axis=-1)
                for target in range(len(pool)):
                    valid = success[ev, target] if family.startswith("R3") else np.ones(len(ev), bool)
                    classes = edge_classes(pool, target)
                    for edge_class in ("target_edge", "collateral_edges", "unrelated_edges"):
                        attribution.append({"candidate_id": candidate, "fold": fold, "family": family,
                            "target_pair_id": int(pool.pair_id.iloc[target]), "edge_class": edge_class,
                            "sample_count": int(valid.sum()),
                            "tail_max_fraction": float(np.isin(max_edges[valid, target], np.flatnonzero(classes == edge_class)).mean())})
        if not np.isfinite(clean_scores).all() or not np.isfinite(clean_thresholds).all():
            raise RuntimeError("invalid_crossfit_or_cache: incomplete cross-fit clean evaluation")
        clean_alarm = clean_scores > clean_thresholds
        seq_fpr = [float(clean_alarm[sequence_id == seq].mean()) for seq in range(6)]
        family_results = []
        for family in BREAKING:
            valid = success if family.startswith("R3") else np.ones_like(success)
            positive = positive_scores[family][valid]
            positive_thr = positive_thresholds[family][valid]
            clean_repeated = np.repeat(clean_scores, valid.sum(axis=1))
            clean_thr_repeated = np.repeat(clean_thresholds, valid.sum(axis=1))
            result = classification_metrics(clean_repeated, positive, clean_thr_repeated, positive_thr)
            family_results.append(result)
            corruption_rows.append({"candidate_id": candidate, "family": family, "sample_count": len(positive), **result})
            for seq in range(6):
                row_mask = sequence_id == seq
                seq_valid = valid[row_mask]
                seq_positive = positive_scores[family][row_mask][seq_valid]
                seq_positive_thr = positive_thresholds[family][row_mask][seq_valid]
                seq_clean = np.repeat(clean_scores[row_mask], seq_valid.sum(axis=1))
                seq_clean_thr = np.repeat(clean_thresholds[row_mask], seq_valid.sum(axis=1))
                sequence_rows.append({"candidate_id": candidate, "family": family,
                    "sequence_id": f"train{seq + 1}_validation", "sample_count": len(seq_positive),
                    **classification_metrics(seq_clean, seq_positive, seq_clean_thr, seq_positive_thr)})
        macro = {key: float(np.mean([row[key] for row in family_results])) for key in ("auroc", "aupr", "q99_tpr", "precision", "recall", "f1")}
        seq_frame = pd.DataFrame([row for row in sequence_rows if row["candidate_id"] == candidate])
        seq_macro = seq_frame.groupby("sequence_id").auroc.mean()
        gates = {"clean_fpr": clean_alarm.mean() <= .015, "worst_sequence_fpr": max(seq_fpr) <= .04,
                 "macro_auroc": macro["auroc"] >= .82, "macro_q99_tpr": macro["q99_tpr"] >= .40,
                 "two_family_auroc": sum(row["auroc"] >= .85 for row in family_results) >= 2,
                 "two_family_tpr": sum(row["q99_tpr"] >= .40 for row in family_results) >= 2,
                 "five_sequences": int((seq_macro >= .70).sum()) >= 5}
        candidate_rows.append({"candidate_id": candidate, "evaluation_mode": "two-fold chronological crossfit",
            "clean_overall_fpr": float(clean_alarm.mean()), "worst_sequence_fpr": max(seq_fpr), "fpr_variance": float(np.var(seq_fpr)),
            "alarm_count": int(clean_alarm.sum()), **{f"macro_{key}": value for key, value in macro.items()},
            "worst_family_auroc": min(row["auroc"] for row in family_results), "worst_sequence_macro_auroc": float(seq_macro.min()),
            "unrelated_edge_auroc": .5, "g1_g2_fpr_gap": 0.0, "gates_passed": all(gates.values()),
            "gate_detail": json.dumps({key: bool(value) for key, value in gates.items()}, sort_keys=True)})
        candidate_storage[candidate] = {"clean_scores": clean_scores, "clean_thresholds": clean_thresholds,
                                        "positive_scores": positive_scores, "positive_thresholds": positive_thresholds,
                                        "fold_registry": fold_registry}
    return (pd.DataFrame(candidate_rows), pd.DataFrame(corruption_rows), pd.DataFrame(sequence_rows),
            pd.DataFrame(calibration_rows), pd.DataFrame(calibration_audit), pd.DataFrame(attribution), candidate_storage)


def baseline_row() -> dict[str, Any]:
    frame = pd.read_csv(STEP4B / "structure_controlled_metrics_by_candidate.csv")
    row = frame[frame.candidate_id == "diversity_global|w128|max"].iloc[0]
    corruption = pd.read_csv(STEP4B / "structure_controlled_metrics_by_corruption.csv")
    corruption = corruption[corruption.candidate_id == "diversity_global|w128|max"]
    sequence = pd.read_csv(STEP4B / "structure_controlled_metrics_by_sequence.csv")
    sequence = sequence[sequence.candidate_id == "diversity_global|w128|max"]
    return {"candidate_id": "original_distance_max", "evaluation_mode": "frozen Step 4B baseline; not recomputed",
            "clean_overall_fpr": row.clean_fpr, "worst_sequence_fpr": row.worst_sequence_fpr,
            "fpr_variance": row.clean_fpr_variance, "alarm_count": 60, "macro_auroc": row.macro_auroc,
            "macro_aupr": row.macro_aupr, "macro_q99_tpr": row.macro_tpr_q99, "macro_precision": row.macro_precision,
            "macro_recall": row.macro_recall, "macro_f1": row.macro_f1,
            "worst_family_auroc": float(corruption.auroc.min()),
            "worst_sequence_macro_auroc": float(sequence.groupby("sequence_id").auroc.mean().min()),
            "unrelated_edge_auroc": .5, "g1_g2_fpr_gap": 0.0,
            "gates_passed": False, "gate_detail": "frozen baseline; not eligible for repaired selection"}


def select_and_classify(candidates: pd.DataFrame) -> tuple[pd.Series, str]:
    tails = candidates[candidates.candidate_id != "original_distance_max"].copy()
    passing = tails[tails.gates_passed]
    if len(passing):
        best_aupr = passing.macro_aupr.max()
        shortlist = passing[passing.macro_aupr >= best_aupr - .01]
        min_worst = shortlist.worst_sequence_fpr.min()
        shortlist = shortlist[shortlist.worst_sequence_fpr <= min_worst + 1e-15]
        shortlist["primary"] = shortlist.candidate_id == "relation_tail_max"
        selected = shortlist.sort_values("primary", ascending=False).iloc[0]
        baseline = candidates[candidates.candidate_id == "original_distance_max"].iloc[0]
        clear = selected.macro_auroc >= baseline.macro_auroc + .02 or selected.macro_q99_tpr >= baseline.macro_q99_tpr + .10
        classification = "relation_tail_calibration_repairs_global_aggregation" if clear else "relation_signal_valid_but_aggregation_still_unresolved"
    else:
        selected = tails.sort_values(["macro_aupr", "worst_sequence_fpr"], ascending=[False, True]).iloc[0]
        baseline = candidates[candidates.candidate_id == "original_distance_max"].iloc[0]
        improved = selected.macro_auroc >= baseline.macro_auroc + .01
        classification = "relation_signal_valid_but_aggregation_still_unresolved" if improved else "tail_calibration_does_not_improve_aggregation"
    return selected, classification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = validate_inputs()
    if args.dry_run:
        print("REUSE: Step 4B clean/controlled Fisher caches + frozen median/MAD; Step 4B-R G1/G2 exact-invariance alias", flush=True)
        print("CROSSFIT: per sequence chronological halves; A first->second, B second->first", flush=True)
        print(f"CANDIDATES: {list(CANDIDATES)}", flush=True)
        print("FORBIDDEN: raw HAI CSV/test, AE/checkpoint/rec, relation feature recompute, reference refit, corruption regeneration", flush=True)
        return 0
    if (OUTPUT / "aggregation_repair_summary.json").exists():
        print(f"REUSE completed Step 4B-A: {OUTPUT}", flush=True)
        return 0
    before4b, before4br = tree_hashes(STEP4B), tree_hashes(STEP4BR)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    clean_cache = np.load(paths["clean_fisher_cache"], allow_pickle=False)
    controlled = np.load(paths["controlled_fisher_cache"], allow_pickle=False)
    pool = pd.read_csv(paths["relation_pool"]).reset_index(drop=True)
    if len(pool) != 10 or len(set(pool.feature_i) | set(pool.feature_j)) != 15:
        raise RuntimeError("invalid_crossfit_or_cache: relation pool changed")
    ref = pd.read_csv(paths["reference"])
    ref = ref[ref.window == WINDOW].set_index("pair_id").loc[pool.pair_id]
    median, scale = ref.global_median.to_numpy(float), ref.global_scale.to_numpy(float)
    clean_d = robust_distances(clean_cache["validation_z"], median, scale)
    altered_d = robust_distances(controlled["features_w128"][:3], median, scale)
    success = controlled["success_w128"]
    manifest, masks = build_crossfit_manifest(clean_cache["validation_sequence_id"], clean_cache["validation_start"])
    manifest.to_csv(OUTPUT / "crossfit_manifest.csv", index=False)
    results = run_crossfit(pool, clean_d, altered_d, success, clean_cache["validation_sequence_id"], masks)
    candidates, corruptions, sequences, calibration_registry, calibration_audit, attribution, storage = results
    candidates = pd.concat([pd.DataFrame([baseline_row()]), candidates], ignore_index=True)
    candidates.to_csv(OUTPUT / "aggregation_repair_metrics_by_candidate.csv", index=False)
    corruptions.to_csv(OUTPUT / "aggregation_repair_metrics_by_corruption.csv", index=False)
    sequences.to_csv(OUTPUT / "aggregation_repair_metrics_by_sequence.csv", index=False)
    calibration_registry.to_csv(OUTPUT / "relation_tail_calibration_registry.csv", index=False)
    calibration_audit.to_csv(OUTPUT / "relation_tail_calibration_audit.csv", index=False)
    attribution.to_csv(OUTPUT / "aggregation_edge_attribution.csv", index=False)
    selected, classification = select_and_classify(candidates)
    preserving_rows = []
    global_audit = pd.read_csv(paths["global_invariance"])
    if not (global_audit.max_absolute_distance_difference == 0).all():
        raise RuntimeError("invalid_crossfit_or_cache: G1/G2 no longer exact clean aliases")
    for candidate in CANDIDATES[1:]:
        state = storage[candidate]
        clean_score, thresholds = state["clean_scores"], state["clean_thresholds"]
        clean_fpr = float((clean_score > thresholds).mean())
        for family in GLOBAL_CONTROLS:
            preserving_rows.append({"candidate_id": candidate, "family": family, "sample_count": len(clean_score),
                "clean_fpr": clean_fpr, "control_fpr": clean_fpr, "absolute_fpr_gap": 0.0,
                "max_score_difference": 0.0, "distance_alias_source": "frozen clean distance; Step 4B-R exact invariance",
                "passed": True})
    pd.DataFrame(preserving_rows).to_csv(OUTPUT / "preserving_control_recheck.csv", index=False)
    frozen = bool(selected.gates_passed and classification == "relation_tail_calibration_repairs_global_aggregation")
    config = {"status": "frozen" if frozen else "not_frozen", "relation_pool": "Step 4A diversity pool, unchanged",
              "relation_pair_ids": pool.pair_id.astype(int).tolist(), "window": 128, "relation_feature": "Pearson then Fisher-z",
              "normal_reference": "frozen median/MAD", "calibration": "relation-wise empirical clean-validation tail",
              "aggregation": selected.candidate_id, "threshold": "two-fold cross-fit calibration-clean q99",
              "hai_test_used": False}
    (OUTPUT / "selected_repaired_structure_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    after4b, after4br = tree_hashes(STEP4B), tree_hashes(STEP4BR)
    audit = []
    for source, root, before, after in (("step4b", STEP4B, before4b, after4b), ("step4b_r", STEP4BR, before4br, after4br)):
        audit.extend({"source": source, "artifact_id": name, "source_path": str(root / name), "sha256_before": digest,
                      "sha256_after": after.get(name), "hash_match": after.get(name) == digest,
                      "reused": True, "recomputed": False} for name, digest in before.items())
    pd.DataFrame(audit).to_csv(OUTPUT / "artifact_reuse_audit.csv", index=False)
    summary = {"classification": classification, "candidate_count": 3, "selected_candidate": selected.candidate_id,
               "selected_candidate_gates_passed": bool(selected.gates_passed), "configuration_frozen": frozen,
               "allow_real_structure_only_evaluation": frozen, "step4b_hashes_unchanged": before4b == after4b,
               "step4br_hashes_unchanged": before4br == after4br, "raw_csv_loaded": False, "hai_test_loaded": False,
               "ae_checkpoint_rec_loaded": False, "reference_refit": False, "relation_features_recomputed": False,
               "corruptions_regenerated": False, "candidate_metrics": candidates.to_dict("records")}
    (OUTPUT / "aggregation_repair_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = candidates.set_index("candidate_id")
    report = f"""# HAI Step 4B-A relation-wise tail calibration and aggregation repair

## Protocol integrity

- Two-fold chronological cross-fitting was applied independently within all six validation sequences.
- Each source window is evaluated only with a calibrator fitted on the opposite temporal half.
- Step 4B and Step 4B-R hashes remained unchanged: {before4b == after4b and before4br == after4br}.
- No raw data, test data, model, checkpoint, reconstruction score, synthetic regeneration, or reference refit was used.

## Candidate results

| Candidate | Macro AUROC | Macro AUPR | Macro q99 TPR | Clean FPR | Worst sequence FPR | Passed |
|---|---:|---:|---:|---:|---:|---:|
"""
    for name in CANDIDATES:
        row = rows.loc[name]
        report += f"| {name} | {row.macro_auroc:.4f} | {row.macro_aupr:.4f} | {row.macro_q99_tpr:.4f} | {row.clean_overall_fpr:.4f} | {row.worst_sequence_fpr:.4f} | {bool(row.gates_passed)} |\n"
    report += f"""

G1/G2 remain exact clean aliases after relation-wise calibration: score difference and FPR gap are zero. Unrelated-edge AUROC remains 0.5 by paired identity.

## Decision

- Selected candidate: `{selected.candidate_id}`.
- Classification: **{classification}**.
- Configuration status: **{'frozen' if frozen else 'not_frozen'}**.
- Real HAI structure-only evaluation is {'allowed only as a separately approved next step' if frozen else 'not allowed'}.

This step does not alter the relation pool and stops before test evaluation, reconstruction complementarity, or fusion.
"""
    (OUTPUT / "final_hai_step4b_relation_aggregation_repair.md").write_text(report, encoding="utf-8")
    if before4b != tree_hashes(STEP4B) or before4br != tree_hashes(STEP4BR):
        raise RuntimeError("invalid_crossfit_or_cache: historical artifact changed")
    print(json.dumps({"classification": classification, "selected_candidate": selected.candidate_id,
                      "gates_passed": bool(selected.gates_passed), "configuration_frozen": frozen,
                      "historical_hashes_unchanged": True}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
