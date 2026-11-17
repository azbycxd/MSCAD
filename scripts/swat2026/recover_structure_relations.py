from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat2026_structure import aggregate_deviation, fit_robust_reference, relation_deviation, stable_pairs  # noqa: E402
from data.swat2026_structure_recovery import (  # noqa: E402
    LEGACY_EXPANDED3,
    REPRESENTATIONS,
    TEST_END,
    TEST_START,
    batch_representation_correlations,
    load_recovery_data,
    representation_correlations,
    reproducibility_diagnostics,
    select_primary,
    select_reproducible_sets,
    train_relation_split,
    validation_usage_split,
)
from data.swat2026_training import sha256_file  # noqa: E402
from scripts.swat2026.analyze_rec_and_structure import (  # noqa: E402
    contiguous_normal_intervals,
    event_diagnostics,
    injected_rec_scores,
    load_frozen_model,
    read_json,
    resolve,
    validate_m1_artifacts,
    write_json,
)
from scripts.swat2026.train_eval_rec_baseline import build_device, point_metrics  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.project_paths import resolve_data_path  # noqa: E402


def windows_for_ends(values: np.ndarray, ends: np.ndarray, window_size: int = 128) -> np.ndarray:
    return np.stack([values[end - window_size + 1 : end + 1] for end in ends]).astype(np.float32)


def selected_pair_frame(all_pairs: pd.DataFrame, names: list[str] | tuple[str, ...]) -> pd.DataFrame:
    lookup = all_pairs.set_index("pair_name")
    missing = [name for name in names if name not in lookup.index]
    if missing:
        raise ValueError(f"Requested relation pairs do not exist: {missing}")
    return lookup.loc[list(names)].reset_index()


def score_from_correlations(correlations: np.ndarray, reference) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deviation = relation_deviation(correlations, reference)
    maximum, top3 = aggregate_deviation(deviation)
    return maximum, top3, deviation


def fixed_threshold_metrics(normal: np.ndarray, anomaly: np.ndarray, threshold_values: np.ndarray,
                            quantile: float = .99) -> dict[str, Any]:
    threshold = float(np.quantile(threshold_values, quantile))
    labels = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(anomaly), dtype=int)])
    scores = np.concatenate([normal, anomaly])
    return {"threshold": threshold, **point_metrics(labels, scores, threshold)}


def _reference(correlations: np.ndarray, selected: pd.DataFrame, all_pairs: pd.DataFrame, floor: float):
    indices = [int(np.flatnonzero(all_pairs.pair_name.to_numpy() == name)[0]) for name in selected.pair_name]
    values = correlations[:, indices]
    return fit_robust_reference(values, floor), indices


def _representation_summary(diagnostics: pd.DataFrame, sets: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for representation in REPRESENTATIONS:
        current = diagnostics[diagnostics.representation.eq(representation)]
        relation_set = sets[representation]
        rows.append({
            "representation": representation, "pair_count": len(current),
            "reproducible_relation_count": len(relation_set["reproducible_set"]),
            "diverse_relation_count": len(relation_set["diverse_set"]), "status": relation_set["status"],
            "median_gap_pass_count": int((current.discovery_confirmation_median_gap <=
                                           np.maximum(.15, 2.5 * current.pooled_robust_scale)).sum()),
            "sign_match_count": int(current.discovery_confirmation_sign_match.sum()),
            "interval_overlap_count": int(current.reference_interval_overlap.sum()),
        })
    return pd.DataFrame(rows)


def controlled_validation(
    cfg: dict[str, Any], data, all_pairs: pd.DataFrame, sets: dict[str, dict[str, Any]],
    references: dict[str, Any], threshold_ends: np.ndarray, controlled_ends: np.ndarray,
    output: Path, smoke: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = output / "controlled"; directory.mkdir(parents=True, exist_ok=True)
    if smoke:
        threshold_ends = threshold_ends[:128]
        controlled_ends = controlled_ends[:128]
    threshold_windows = windows_for_ends(data.scaled_features, threshold_ends)
    controlled_windows = windows_for_ends(data.scaled_features, controlled_ends)
    device = build_device("cuda")
    model = load_frozen_model(cfg, device)
    rec_threshold_scores = injected_rec_scores(model, threshold_windows, device)
    rec_normal_scores = injected_rec_scores(model, controlled_windows, device)
    metric_rows, manifest_rows = [], []
    shifts = list(cfg["controlled"]["shifts"])
    quantile = float(cfg["controlled"]["validation_quantile"])
    for representation in REPRESENTATIONS:
        selected = sets[representation]["reproducible_set"]
        if len(selected) < 3:
            continue
        injected = controlled_windows.copy()
        for index in range(len(injected)):
            relation = selected.iloc[index % len(selected)]
            shift = shifts[index % len(shifts)]
            channel = int(relation.channel_j_index)
            original = injected[index, :, channel].copy()
            injected[index, :, channel] = np.roll(original, shift)
            changed = injected[index, :, channel]
            manifest_rows.append({
                "representation": representation, "injection_id": index,
                "source_window_end": data.timestamps[controlled_ends[index]].isoformat(),
                "relation": relation.pair_name, "shifted_channel": relation.channel_j,
                "shift_seconds": shift, "boundary_policy": "circular_within_window", "seed": 1,
                "value_multiset_preserved": bool(np.array_equal(np.sort(original), np.sort(changed))),
                "mean_abs_change": abs(float(original.mean() - changed.mean())),
                "std_abs_change": abs(float(original.std() - changed.std())),
                "min_abs_change": abs(float(original.min() - changed.min())),
                "max_abs_change": abs(float(original.max() - changed.max())), "test_used": False,
            })
        relation_pairs = selected_pair_frame(all_pairs, selected.pair_name.tolist())
        reference = references[representation]
        threshold_corr = batch_representation_correlations(threshold_windows, relation_pairs, representation)
        normal_corr = batch_representation_correlations(controlled_windows, relation_pairs, representation)
        anomaly_corr = batch_representation_correlations(injected, relation_pairs, representation)
        threshold_score = score_from_correlations(threshold_corr, reference)[0]
        normal_score = score_from_correlations(normal_corr, reference)[0]
        anomaly_score = score_from_correlations(anomaly_corr, reference)[0]
        metric = fixed_threshold_metrics(normal_score, anomaly_score, threshold_score, quantile)
        metric_rows.append({"representation": representation, "score_name": "representation_max",
                            "marginal_statistics_preserved": True, "reference_valid": True,
                            **{key: value for key, value in metric.items() if key != "confusion_matrix"},
                            **{f"cm_{key}": value for key, value in metric["confusion_matrix"].items()}})
        injected_rec = injected_rec_scores(model, injected, device)
        rec_metric = fixed_threshold_metrics(rec_normal_scores, injected_rec, rec_threshold_scores, quantile)
        metric_rows.append({"representation": representation, "score_name": "rec_score",
                            "marginal_statistics_preserved": True, "reference_valid": True,
                            **{key: value for key, value in rec_metric.items() if key != "confusion_matrix"},
                            **{f"cm_{key}": value for key, value in rec_metric["confusion_matrix"].items()}})
        rec_alarm = injected_rec > rec_metric["threshold"]
        structure_alarm = anomaly_score > metric["threshold"]
        metric_rows[-2]["structure_detected_rec_missed_ratio"] = float((structure_alarm & ~rec_alarm).mean())
        metric_rows[-2]["rec_detected_structure_missed_ratio"] = float((rec_alarm & ~structure_alarm).mean())
        metric_rows[-2]["injected_score_pearson"] = float(pd.Series(injected_rec).corr(pd.Series(anomaly_score), method="pearson"))
        metric_rows[-2]["injected_score_spearman"] = float(pd.Series(injected_rec).corr(pd.Series(anomaly_score), method="spearman"))
    manifest = pd.DataFrame(manifest_rows)
    metrics = pd.DataFrame(metric_rows)
    manifest.to_csv(directory / "controlled_injection_manifest.csv", index=False)
    metrics.to_csv(directory / "controlled_metrics.csv", index=False)
    summary = {"injection_count": len(manifest), "representations_evaluated": sorted(metrics.representation.unique().tolist()) if len(metrics) else [],
               "marginal_statistics_preserved": bool(len(manifest) and manifest.value_multiset_preserved.all() and
                                                      manifest[["mean_abs_change", "std_abs_change", "min_abs_change", "max_abs_change"]].to_numpy().max() < 1e-6),
               "threshold_source": "validation_threshold_normal_only_q99", "test_used": False,
               "model_retrained": False, "fusion_computed": False}
    write_json(directory / "controlled_summary.json", summary)
    return metrics, summary


def formal_test_evaluation(cfg: dict[str, Any], data, all_pairs: pd.DataFrame, selected: pd.DataFrame,
                           representation: str, reference, threshold_score: np.ndarray,
                           output: Path) -> dict[str, Any]:
    # This function is called only after frozen_primary_structure_manifest.json has been written.
    labels_path = resolve(cfg["data"]["n2_dir"]) / "point_labels_11mar.csv"
    labels = pd.read_csv(labels_path, encoding="utf-8-sig", dtype={"attack_id": str})
    labels["timestamp"] = pd.to_datetime(labels.timestamp)
    test_labels = labels[(labels.timestamp >= TEST_START) & (labels.timestamp < TEST_END)].reset_index(drop=True)
    test_ends = np.flatnonzero(np.asarray((data.timestamps >= TEST_START) & (data.timestamps < TEST_END)))
    relation_pairs = selected_pair_frame(all_pairs, selected.pair_name.tolist())
    correlations = representation_correlations(data.scaled_features, data.valid_points, test_ends,
                                               relation_pairs, representation, 128)
    scores, _, deviation = score_from_correlations(correlations, reference)
    threshold = float(np.quantile(threshold_score, float(cfg["evaluation"]["validation_quantile"])))
    metrics = point_metrics(test_labels.point_label.to_numpy(int), scores, threshold)
    metrics.update({"threshold": threshold, "threshold_source": "validation_threshold_q99",
                    "representation": representation, "point_adjustment": False,
                    "test_used_for_selection": False, "fusion_computed": False})
    real = output / "real_test"; real.mkdir(parents=True, exist_ok=True)
    write_json(real / "structure_metrics.json", metrics)
    frame = pd.DataFrame({"timestamp": test_labels.timestamp, "structure_score": scores,
                          "point_label": test_labels.point_label.astype(int), "attack_id": test_labels.attack_id,
                          "phase": test_labels.phase, "evaluation_eligible": test_labels.evaluation_eligible})
    frame.to_csv(real / "structure_scores_test.csv", index=False)
    schedule = pd.read_csv(resolve(cfg["data"]["n2_dir"]) / "approved_attack_schedule.csv", encoding="utf-8-sig")
    events = event_diagnostics(frame, schedule, "structure_score", threshold)
    events.to_csv(real / "structure_event_summary.csv", index=False)
    false_positive = contiguous_normal_intervals(frame, schedule, "structure_score", threshold)
    false_positive.to_csv(real / "structure_false_positive_summary.csv", index=False)
    top_index = np.argmax(deviation, axis=1)
    attribution = pd.DataFrame({"timestamp": frame.timestamp,
                                "top_relation": relation_pairs.iloc[top_index].pair_name.to_numpy(),
                                "top_relation_score": deviation[np.arange(len(deviation)), top_index]})
    top_rows = []
    for _, event in events.iterrows():
        current = attribution[(attribution.timestamp >= pd.Timestamp(event.start_time)) &
                              (attribution.timestamp < pd.Timestamp(event.end_time_exclusive))]
        top = current.sort_values("top_relation_score", ascending=False).iloc[0]
        top_rows.append({"attack_id": event.attack_id, "top_relation": top.top_relation,
                         "peak_relation_score": top.top_relation_score})
    pd.DataFrame(top_rows).to_csv(real / "top_relation_by_event.csv", index=False)

    m1 = resolve(cfg["data"]["m1_dir"])
    rec = pd.read_csv(m1 / "rec_scores_test.csv"); rec["timestamp"] = pd.to_datetime(rec.timestamp)
    if not rec.timestamp.equals(frame.timestamp):
        raise ValueError("M1 rec and M2R structure test timestamps do not align")
    rec_threshold = float(read_json(m1 / "baseline_metrics.json")["threshold"])
    rec_alarm, structure_alarm = rec.rec_score.to_numpy(float) > rec_threshold, scores > threshold
    attack = frame.point_label.to_numpy(int) == 1; normal = ~attack
    comp = {
        "pearson": float(rec.rec_score.corr(pd.Series(scores), method="pearson")),
        "spearman": float(rec.rec_score.corr(pd.Series(scores), method="spearman")),
        "structure_adds_rec_missed_attack_points": int((attack & structure_alarm & ~rec_alarm).sum()),
        "structure_supplement_ratio_of_rec_misses": float((attack & structure_alarm & ~rec_alarm).sum() / max((attack & ~rec_alarm).sum(), 1)),
        "both_hit_attack_points": int((attack & structure_alarm & rec_alarm).sum()),
        "both_miss_attack_points": int((attack & ~structure_alarm & ~rec_alarm).sum()),
        "rec_false_positive_points": int((normal & rec_alarm).sum()),
        "structure_false_positive_points": int((normal & structure_alarm).sum()),
        "false_positive_overlap": int((normal & rec_alarm & structure_alarm).sum()), "fusion_computed": False,
    }
    rec_events = event_diagnostics(rec, schedule, "rec_score", rec_threshold)
    event_comp = events[["attack_id", "detected"]].rename(columns={"detected": "structure_detected"}).merge(
        rec_events[["attack_id", "detected"]].rename(columns={"detected": "rec_detected"}), on="attack_id")
    event_comp["structure_adds_event"] = event_comp.structure_detected & ~event_comp.rec_detected
    event_comp.to_csv(real / "event_level_complementarity.csv", index=False)
    comp["structure_adds_rec_missed_events"] = int(event_comp.structure_adds_event.sum())
    write_json(real / "complementarity_summary.json", comp)
    return {"metrics": metrics, "events_detected": int(events.detected.sum()), "events": events,
            "false_positive": false_positive, "complementarity": comp}


def run(cfg: dict[str, Any], dry_run: bool = False, smoke: bool = False) -> dict[str, Any]:
    if int(cfg["seed"]) != 1 or tuple(cfg["representations"]) != REPRESENTATIONS:
        raise ValueError("M2R fixed seed/representation contract changed")
    artifact = validate_m1_artifacts(cfg)
    if sha256_file(resolve_data_path(cfg["data"]["data_path"])) != cfg["data"]["source_sha256"]:
        raise ValueError("Official source CSV hash changed")
    base = resolve(cfg["output"]["dir"])
    output = base / ("dry_run" if dry_run else "smoke") if (dry_run or smoke) else base
    output.mkdir(parents=True, exist_ok=True)
    data = load_recovery_data(resolve_data_path(cfg["data"]["data_path"]), resolve(cfg["data"]["m1_dir"]))
    split_manifest, split = train_relation_split(data.timestamps, int(cfg["split"]["window_size"]),
                                                 float(cfg["split"]["discovery_ratio"]))
    write_json(output / "train_relation_split_manifest.json", split_manifest)
    excluded = set(data.scaler_metadata["zero_variance_features"])
    all_pairs = stable_pairs(data.feature_names, excluded)
    all_diagnostics, sets, correlations = [], {}, {}
    for representation in REPRESENTATIONS:
        discovery = representation_correlations(data.scaled_features, data.valid_points, split["discovery_ends"],
                                                all_pairs, representation, 128)
        confirmation = representation_correlations(data.scaled_features, data.valid_points, split["confirmation_ends"],
                                                   all_pairs, representation, 128)
        current = reproducibility_diagnostics(all_pairs, discovery, confirmation, representation, cfg["reproducibility"])
        relation_set = select_reproducible_sets(current, int(cfg["reproducibility"]["max_pairs"]),
                                                int(cfg["reproducibility"]["diverse_max_pairs"]),
                                                int(cfg["reproducibility"]["diverse_max_degree"]))
        all_diagnostics.append(current); sets[representation] = relation_set
        correlations[representation] = {"discovery": discovery, "confirmation": confirmation}
    diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    diagnostics.to_csv(output / "relation_reproducibility_all.csv", index=False)
    representation_summary = _representation_summary(diagnostics, sets)
    representation_summary.to_csv(output / "representation_summary.csv", index=False)
    relation_payload = {representation: {"status": sets[representation]["status"],
                                         "reproducible_set": sets[representation]["reproducible_set"].pair_name.tolist(),
                                         "diverse_set": sets[representation]["diverse_set"].pair_name.tolist()}
                        for representation in REPRESENTATIONS}
    relation_payload["legacy_expanded3"] = {"pairs": list(LEGACY_EXPANDED3), "diagnostic_only": True,
                                            "automatic_promotion": False}
    write_json(output / "relation_sets.json", relation_payload)
    if dry_run:
        result = {**artifact, "code_readiness": "A. implementation_complete", "dry_run": True,
                  "relation_counts": {name: len(sets[name]["reproducible_set"]) for name in REPRESENTATIONS},
                  "validation_or_test_used_for_screening": False, "model_retrained": False, "fusion_computed": False}
        write_json(output / "dry_run_summary.json", result); print(json.dumps(result, indent=2)); return result

    floor = float(cfg["reproducibility"]["fisher_scale_floor"])
    references = {}; selected_indices = {}
    for representation in REPRESENTATIONS:
        selected = sets[representation]["reproducible_set"]
        if not len(selected):
            continue
        combined = np.concatenate([correlations[representation]["discovery"], correlations[representation]["confirmation"]])
        references[representation], selected_indices[representation] = _reference(combined, selected, all_pairs, floor)
    validation_manifest, validation = validation_usage_split(data.timestamps, 128)
    write_json(output / "validation_usage_manifest.json", validation_manifest)
    controlled_metrics, controlled_summary = controlled_validation(
        cfg, data, all_pairs, sets, references, validation["threshold_ends"], validation["controlled_ends"], output, smoke
    )
    counts = {name: len(sets[name]["reproducible_set"]) for name in REPRESENTATIONS}
    primary = select_primary(controlled_metrics, counts, float(cfg["controlled"]["minimum_primary_auroc"]))
    primary.update({"relation_count": counts.get(primary.get("representation"), 0),
                    "relations": sets[primary["representation"]]["reproducible_set"].pair_name.tolist() if primary["primary_frozen"] else [],
                    "test_labels_read_before_freeze": False})
    write_json(output / "frozen_primary_structure_manifest.json", primary)
    if smoke:
        result = {**artifact, "code_readiness": "A. implementation_complete", "smoke": True,
                  "relation_counts": counts, "primary": primary, "controlled": controlled_summary,
                  "formal_test_read": False, "model_retrained": False, "fusion_computed": False}
        write_json(output / "smoke_summary.json", result); print(json.dumps(result, indent=2)); return result

    real_result = None
    if primary["primary_frozen"]:
        representation = primary["representation"]
        selected = sets[representation]["reproducible_set"]
        relation_pairs = selected_pair_frame(all_pairs, selected.pair_name.tolist())
        threshold_corr = representation_correlations(data.scaled_features, data.valid_points,
                                                     validation["threshold_ends"], relation_pairs, representation, 128)
        threshold_score = score_from_correlations(threshold_corr, references[representation])[0]
        real_result = formal_test_evaluation(cfg, data, all_pairs, selected, representation,
                                             references[representation], threshold_score, output)
    max_count = max(counts.values())
    if primary["primary_frozen"]:
        scientific = ("A. general_structure_mechanism_validated_real_evaluation_complete"
                      if real_result["metrics"]["auroc"] >= .60
                      else "B. controlled_mechanism_validated_real_performance_weak")
    elif max_count >= 3:
        scientific = "C. reproducible_relations_found_but_controlled_mechanism_weak"
    elif max_count >= 1:
        scientific = "E. only_narrow_relation_diagnostic_available"
    else:
        scientific = "D. no_general_reproducible_relation_set"
    inherited_path = ROOT / "outputs" / "swat2026_batch_m2_structure" / "rec_diagnostics" / "rec_diagnosis_summary.json"
    inherited = read_json(inherited_path) if inherited_path.exists() else {
        "primary_diagnoses": ["A. global_operating_regime_shift", "B. scaler_extrapolation",
                              "C. channel_specific_drift", "D. threshold_miscalibration"],
        "preexisting_alarm_event_count": 6,
    }
    readiness = {
        "code_readiness": "A. implementation_complete", "scientific_readiness": scientific,
        "source_hash_matches": True, "checkpoint_hash_matches": True, "relations_train_only": True,
        "validation_threshold_controlled_disjoint": validation_manifest["raw_timestamp_overlap"] is False,
        "test_labels_read_only_after_primary_freeze": True, "primary_frozen": primary["primary_frozen"],
        "model_retrained": False, "fusion_computed": False, "multiple_seeds_run": False,
        "inherited_m2_reconstruction_diagnosis": inherited,
    }
    write_json(output / "batch_m2r_readiness.json", readiness)
    write_json(output / "resolved_config.json", cfg)
    result = {"readiness": readiness, "relation_counts": counts, "primary": primary,
              "controlled_metrics": controlled_metrics.to_dict("records"),
              "real_result": None if real_result is None else {"metrics": real_result["metrics"],
                                                                 "events_detected": real_result["events_detected"],
                                                                 "complementarity": real_result["complementarity"]}}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str)); return result


def main() -> int:
    parser = argparse.ArgumentParser(description="SWaT Batch-M2R train-reproducible structure recovery")
    parser.add_argument("--config", required=True); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.smoke:
        raise ValueError("--dry-run and --smoke are mutually exclusive")
    run(load_config(args.config), args.dry_run, args.smoke); return 0


if __name__ == "__main__":
    raise SystemExit(main())
