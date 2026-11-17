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

from data.swat2026_contextual_structure import (  # noqa: E402
    LAG_SECONDS, LEGACY_NARROW, REPRESENTATIONS, audit_state_channels, binary_metrics,
    circular_shift_window, discovery_selected_lags, fit_relation_reference,
    lagged_diagnostics, lagged_pair_correlations, reproducibility_row,
    score_correlations, select_contextual_set, select_primary, stable_state_window_mask,
)
from data.swat2026_structure import stable_pairs  # noqa: E402
from data.swat2026_structure_recovery import (  # noqa: E402
    TEST_END, TEST_START, load_recovery_data, train_relation_split, validation_usage_split,
)
from data.swat2026_training import sha256_file  # noqa: E402
from scripts.swat2026.analyze_rec_and_structure import (  # noqa: E402
    contiguous_normal_intervals, event_diagnostics, injected_rec_scores, load_frozen_model,
    resolve, validate_m1_artifacts, write_json,
)
from scripts.swat2026.train_eval_rec_baseline import build_device, point_metrics  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.project_paths import resolve_data_path  # noqa: E402


def windows_for_ends(values: np.ndarray, ends: np.ndarray, window_size: int = 128) -> np.ndarray:
    return np.stack([values[end - window_size + 1:end + 1] for end in ends]).astype(np.float32)


def _raw_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = frame.columns.astype(str).str.strip()
    return frame


def _selected_pair_frame(all_pairs: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    lookup = all_pairs.set_index("pair_name")
    return lookup.loc[selected.pair_name.tolist()].reset_index()


def correlations_for_relations(values: np.ndarray, validity: np.ndarray, ends: np.ndarray,
                               all_pairs: pd.DataFrame, selected: pd.DataFrame,
                               representation: str, states: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    result = np.full((len(ends), len(selected)), np.nan, dtype=float)
    applicable = np.ones_like(result, dtype=bool)
    base = "lagged_level" if representation.endswith("level") else "lagged_delta"
    pair_lookup = {name: index for index, name in enumerate(all_pairs.pair_name)}
    if representation.startswith("lagged"):
        for lag in sorted(set(selected.selected_lag.astype(int))):
            full = lagged_pair_correlations(values, validity, ends, all_pairs, lag, representation)
            for output_index, row in selected[selected.selected_lag.eq(lag)].iterrows():
                local = selected.index.get_loc(output_index)
                result[:, local] = full[:, pair_lookup[row.pair_name]]
    else:
        full = lagged_pair_correlations(values, validity, ends, all_pairs, 0, base)
        for local, row in enumerate(selected.itertuples(index=False)):
            result[:, local] = full[:, pair_lookup[row.pair_name]]
            if states is None or row.state_channel not in states:
                applicable[:, local] = False
            else:
                applicable[:, local] = stable_state_window_mask(states[row.state_channel], ends, int(row.state_value))
                result[~applicable[:, local], local] = np.nan
    applicable &= np.isfinite(result)
    return result, applicable


def window_relation_correlations(windows: np.ndarray, selected: pd.DataFrame,
                                 representation: str) -> np.ndarray:
    result = np.full((len(windows), len(selected)), np.nan, dtype=float)
    delta = representation.endswith("delta")
    for column, row in enumerate(selected.itertuples(index=False)):
        lag = int(getattr(row, "selected_lag", 0)) if representation.startswith("lagged") else 0
        distance = abs(lag)
        for index, window in enumerate(windows):
            x = np.diff(window, axis=0) if delta else window
            if lag > 0:
                a = x[:-distance, int(row.channel_i_index)]; b = x[distance:, int(row.channel_j_index)]
            elif lag < 0:
                a = x[distance:, int(row.channel_i_index)]; b = x[:-distance, int(row.channel_j_index)]
            else:
                a = x[:, int(row.channel_i_index)]; b = x[:, int(row.channel_j_index)]
            if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                result[index, column] = np.corrcoef(a, b)[0, 1]
    return result


def discover_lagged(data, split: dict[str, np.ndarray], all_pairs: pd.DataFrame,
                     cfg: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    diagnostics, sets = {}, {}
    for representation in ("lagged_level", "lagged_delta"):
        discovery_by_lag, confirmation_by_lag = {}, {}
        for lag in LAG_SECONDS:
            discovery_by_lag[lag] = lagged_pair_correlations(data.scaled_features, data.valid_points,
                                                             split["discovery_ends"], all_pairs, lag, representation)
            confirmation_by_lag[lag] = lagged_pair_correlations(data.scaled_features, data.valid_points,
                                                                split["confirmation_ends"], all_pairs, lag, representation)
        selected = discovery_selected_lags(all_pairs, discovery_by_lag)
        current = lagged_diagnostics(all_pairs, selected, discovery_by_lag, confirmation_by_lag,
                                     representation, cfg["reproducibility"])
        diagnostics[representation] = current
        sets[representation] = select_contextual_set(current, cfg["reproducibility"]["max_pairs"],
                                                     cfg["reproducibility"]["max_degree"])
    return diagnostics, sets


def discover_conditioned(data, split: dict[str, np.ndarray], all_pairs: pd.DataFrame,
                         raw: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, np.ndarray],
                                                                             dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    audit, states = audit_state_channels(raw, split["discovery"], split["confirmation"], cfg["states"])
    diagnostics, sets = {}, {}
    for representation in ("conditioned_level", "conditioned_delta"):
        rows = []
        base = "lagged_level" if representation.endswith("level") else "lagged_delta"
        discovery_corr = lagged_pair_correlations(data.scaled_features, data.valid_points,
                                                  split["discovery_ends"], all_pairs, 0, base)
        confirmation_corr = lagged_pair_correlations(data.scaled_features, data.valid_points,
                                                     split["confirmation_ends"], all_pairs, 0, base)
        for channel, state_values in states.items():
            for state_value in (0, 1):
                dmask = stable_state_window_mask(state_values, split["discovery_ends"], state_value)
                cmask = stable_state_window_mask(state_values, split["confirmation_ends"], state_value)
                if dmask.sum() < cfg["states"]["minimum_stable_windows"] or cmask.sum() < cfg["states"]["minimum_stable_windows"]:
                    continue
                for pair_index, pair in enumerate(all_pairs.itertuples(index=False)):
                    rows.append(reproducibility_row(pair._asdict(), representation,
                                                    discovery_corr[dmask, pair_index], confirmation_corr[cmask, pair_index],
                                                    cfg["reproducibility"], state_channel=channel, state_value=state_value,
                                                    condition_name=f"{channel}={state_value}",
                                                    discovery_window_count=int(dmask.sum()),
                                                    confirmation_window_count=int(cmask.sum()),
                                                    pre_test_coverage_estimate=float((dmask.mean() + cmask.mean()) / 2)))
        current = pd.DataFrame(rows)
        if not current.empty:
            current["minimum_abs_corr"] = current[["discovery_median_abs_corr", "confirmation_median_abs_corr"]].min(axis=1)
            current = current.sort_values(["pair_name", "reproducible", "minimum_abs_corr", "median_gap", "condition_name"],
                                          ascending=[True, False, False, True, True]).groupby("pair_name", as_index=False).head(1)
        else:
            current = pd.DataFrame(columns=["pair_name", "channel_i", "channel_j", "channel_i_index", "channel_j_index",
                                                    "representation", "state_channel", "state_value", "condition_name", "reproducible"])
        diagnostics[representation] = current
        sets[representation] = select_contextual_set(current, cfg["reproducibility"]["max_pairs"],
                                                     cfg["reproducibility"]["max_degree"])
    return audit, states, diagnostics, sets


def _reference_for_set(data, split, all_pairs, selected, representation, states, floor):
    ends = np.r_[split["discovery_ends"], split["confirmation_ends"]]
    corr, applicable = correlations_for_relations(data.scaled_features, data.valid_points, ends, all_pairs,
                                                  selected, representation, states)
    corr[~applicable] = np.nan
    return fit_relation_reference(corr, floor)


def controlled_validation(cfg, data, all_pairs, sets, references, states, validation, output, smoke):
    directory = output / "controlled"; directory.mkdir(parents=True, exist_ok=True)
    threshold_ends = validation["threshold_ends"][:128] if smoke else validation["threshold_ends"]
    controlled_ends = validation["controlled_ends"][:128] if smoke else validation["controlled_ends"]
    threshold_windows = windows_for_ends(data.scaled_features, threshold_ends)
    controlled_windows = windows_for_ends(data.scaled_features, controlled_ends)
    device = build_device("cuda"); model = load_frozen_model(cfg, device)
    rec_threshold = injected_rec_scores(model, threshold_windows, device)
    rec_normal = injected_rec_scores(model, controlled_windows, device)
    metrics, manifest, coverage = [], [], {}
    for representation in REPRESENTATIONS:
        relation_set = sets[representation]
        selected = relation_set["relations"]
        if relation_set["status"] != "general":
            coverage[representation] = 0.0
            continue
        threshold_corr, threshold_app = correlations_for_relations(data.scaled_features, data.valid_points,
                                                                   threshold_ends, all_pairs, selected, representation, states)
        normal_corr, normal_app = correlations_for_relations(data.scaled_features, data.valid_points,
                                                             controlled_ends, all_pairs, selected, representation, states)
        threshold_result = score_correlations(threshold_corr, references[representation], threshold_app)
        normal_result = score_correlations(normal_corr, references[representation], normal_app)
        coverage[representation] = float(np.isfinite(threshold_result["score"]).mean())
        injected = controlled_windows.copy(); injected_app = normal_app.copy()
        for index in range(len(injected)):
            applicable_relations = np.flatnonzero(normal_app[index])
            relation_index = int(applicable_relations[index % len(applicable_relations)]) if len(applicable_relations) else index % len(selected)
            relation = selected.iloc[relation_index]
            frozen_lag = int(relation.get("selected_lag", 0)); shifts = [s for s in cfg["controlled"]["shifts"] if s != abs(frozen_lag)]
            shift = int(shifts[index % len(shifts)])
            before = injected[index, :, int(relation.channel_j_index)].copy()
            injected[index] = circular_shift_window(injected[index], int(relation.channel_j_index), shift)
            after = injected[index, :, int(relation.channel_j_index)]
            manifest.append({"representation": representation, "injection_id": index, "relation": relation.pair_name,
                             "shifted_channel": relation.channel_j, "frozen_lag": frozen_lag, "injection_shift": shift,
                             "state_channel": relation.get("state_channel", ""), "state_value": relation.get("state_value", ""),
                             "state_value_changed": False, "within_window": True, "test_used": False,
                             "value_multiset_preserved": bool(np.array_equal(np.sort(before), np.sort(after))),
                             "mean_difference": abs(float(before.mean() - after.mean())),
                             "std_difference": abs(float(before.std() - after.std())),
                             "range_difference": abs(float(np.ptp(before) - np.ptp(after)))})
        anomaly_corr = window_relation_correlations(injected, selected, representation)
        anomaly_result = score_correlations(anomaly_corr, references[representation], injected_app)
        struct = binary_metrics(normal_result["score"], anomaly_result["score"], threshold_result["score"])
        metrics.append({"representation": representation, "score_name": "structure_max", **struct,
                        "applicable_coverage": coverage[representation], "reference_valid": True,
                        "marginal_statistics_preserved": True})
        injected_rec = injected_rec_scores(model, injected, device)
        rec = binary_metrics(rec_normal, injected_rec, rec_threshold)
        metrics.append({"representation": representation, "score_name": "rec_score", **rec,
                        "applicable_coverage": 1.0, "reference_valid": True,
                        "marginal_statistics_preserved": True})
        struct_alarm = anomaly_result["score"] > struct["threshold"]
        rec_alarm = injected_rec > rec["threshold"]
        metrics[-2]["structure_detected_rec_missed_ratio"] = float((struct_alarm & ~rec_alarm).mean())
        metrics[-2]["score_pearson"] = float(pd.Series(anomaly_result["score"]).corr(pd.Series(injected_rec), method="pearson"))
        metrics[-2]["score_spearman"] = float(pd.Series(anomaly_result["score"]).corr(pd.Series(injected_rec), method="spearman"))
    metrics_frame, manifest_frame = pd.DataFrame(metrics), pd.DataFrame(manifest)
    metrics_frame.to_csv(directory / "controlled_metrics.csv", index=False)
    manifest_frame.to_csv(directory / "controlled_injection_manifest.csv", index=False)
    summary = {"injection_count": len(manifest_frame), "coverage": coverage, "test_used": False,
               "threshold_uses_injected_samples": False, "marginal_statistics_preserved": bool(len(manifest_frame) == 0 or
               (manifest_frame.value_multiset_preserved.all() and manifest_frame[["mean_difference", "std_difference", "range_difference"]].to_numpy().max() < 1e-6)),
               "model_retrained": False, "fusion_computed": False}
    write_json(directory / "controlled_summary.json", summary)
    return metrics_frame, summary, coverage


def _target_match(relation: pd.Series, target: str) -> str:
    channels = [str(relation.channel_i).split(".")[0], str(relation.channel_j).split(".")[0]]
    target_upper = str(target).upper()
    if any(channel.upper() in target_upper for channel in channels):
        return "exact target match"
    stages = {channel[0:2] for channel in channels if len(channel) >= 2}
    target_stages = {token[:2] for token in target_upper.replace(",", " ").split()}
    return "same process stage match" if stages & target_stages else "no obvious match"


def formal_test_evaluation(cfg, data, all_pairs, selected, representation, reference, states,
                           threshold_scores, output, primary_manifest):
    if not primary_manifest["primary_frozen"]:
        raise ValueError("Test labels cannot be read before Primary freeze")
    # All Test-dependent files are first opened below this post-freeze gate.
    labels = pd.read_csv(resolve(cfg["data"]["n2_dir"]) / "point_labels_11mar.csv", encoding="utf-8-sig", dtype={"attack_id": str})
    labels["timestamp"] = pd.to_datetime(labels.timestamp)
    schedule = pd.read_csv(resolve(cfg["data"]["n2_dir"]) / "approved_attack_schedule.csv")
    rec = pd.read_csv(resolve(cfg["data"]["m1_dir"]) / "rec_scores_test.csv"); rec["timestamp"] = pd.to_datetime(rec.timestamp)
    ends = np.flatnonzero(np.asarray((data.timestamps >= TEST_START) & (data.timestamps < TEST_END)))
    if len(ends) != cfg["evaluation"]["test_points"]:
        raise ValueError("formal Test must contain exactly 14,400 points")
    corr, applicable = correlations_for_relations(data.scaled_features, data.valid_points, ends, all_pairs,
                                                  selected, representation, states)
    result = score_correlations(corr, reference, applicable)
    threshold = float(np.quantile(threshold_scores[np.isfinite(threshold_scores)], .99))
    test = pd.DataFrame({"timestamp": data.timestamps[ends], "structure_score": result["score"],
                         "applicable_relation_count": result["applicable_relation_count"],
                         "coverage": result["applicable_relation_count"] / len(selected),
                         "top_relation_index": result["top_relation_index"]})
    test = test.merge(labels[["timestamp", "point_label", "attack_id"]], on="timestamp", how="left", validate="one_to_one")
    finite = test.structure_score.notna(); metrics = point_metrics(test.loc[finite, "point_label"].to_numpy(int),
                                                                  test.loc[finite, "structure_score"].to_numpy(float), threshold)
    test["alarm"] = test.structure_score > threshold
    test["top_relation"] = test.top_relation_index.map(lambda value: selected.iloc[int(value)].pair_name if value >= 0 else "")
    directory = output / "real_test"; directory.mkdir(parents=True, exist_ok=True)
    test.to_csv(directory / "structure_scores_test.csv", index=False)
    events = event_diagnostics(test, schedule, "structure_score", threshold)
    top_rows = []
    for index, event in events.iterrows():
        interval = test[(test.timestamp >= pd.Timestamp(event.start_time)) & (test.timestamp < pd.Timestamp(event.end_time_exclusive))]
        top = interval.loc[interval.structure_score.idxmax()] if interval.structure_score.notna().any() else None
        relation = selected.iloc[int(top.top_relation_index)] if top is not None and top.top_relation_index >= 0 else None
        top_rows.append({"attack_id": event.attack_id, "top_relation": "" if relation is None else relation.pair_name,
                         "official_targets": event.official_targets,
                         "target_interpretation": "no applicable relation" if relation is None else _target_match(relation, event.official_targets),
                         "causal_claimed": False})
    events = events.merge(pd.DataFrame(top_rows), on=["attack_id", "official_targets"], how="left")
    events.to_csv(directory / "event_detection_summary.csv", index=False)
    fp = contiguous_normal_intervals(test, schedule, "structure_score", threshold); fp.to_csv(directory / "false_positive_intervals.csv", index=False)
    rec_threshold = float(pd.read_csv(resolve(cfg["data"]["m1_dir"]) / "rec_scores_validation.csv").rec_score.quantile(.99))
    comparison = test.merge(rec[["timestamp", "rec_score"]], on="timestamp", how="left")
    applicable_rows = comparison.structure_score.notna()
    attack = comparison.point_label.eq(1); struct_alarm = comparison.structure_score > threshold; rec_alarm = comparison.rec_score > rec_threshold
    comp = {"all_test_points": len(comparison), "applicable_test_points": int(applicable_rows.sum()),
            "coverage": float(applicable_rows.mean()),
            "pearson": float(comparison.loc[applicable_rows, ["structure_score", "rec_score"]].corr(method="pearson").iloc[0, 1]),
            "spearman": float(comparison.loc[applicable_rows, ["structure_score", "rec_score"]].corr(method="spearman").iloc[0, 1]),
            "structure_adds_rec_missed_attack_points": int((attack & struct_alarm & ~rec_alarm).sum()),
            "both_hit_attack_points": int((attack & struct_alarm & rec_alarm).sum()),
            "both_miss_attack_points": int((attack & ~struct_alarm & ~rec_alarm).sum()),
            "false_positive_overlap": int((~attack & struct_alarm & rec_alarm).sum()),
            "structure_new_false_positives": int((~attack & struct_alarm & ~rec_alarm).sum()), "fusion_computed": False}
    rec_events = event_diagnostics(comparison, schedule, "rec_score", rec_threshold)
    event_comp = events[["attack_id", "detected"]].rename(columns={"detected": "structure_detected"}).merge(
        rec_events[["attack_id", "detected"]].rename(columns={"detected": "rec_detected"}), on="attack_id")
    comp["structure_adds_rec_missed_events"] = int((event_comp.structure_detected & ~event_comp.rec_detected).sum())
    comp["both_detect_events"] = int((event_comp.structure_detected & event_comp.rec_detected).sum())
    comp["both_miss_events"] = int((~event_comp.structure_detected & ~event_comp.rec_detected).sum())
    write_json(directory / "complementarity_summary.json", comp)
    write_json(directory / "real_test_metrics.json", {"threshold": threshold, "metrics": metrics,
                                                        "events_detected": int(events.detected.sum()),
                                                        "preexisting_alarms": int(events.preexisting_alarm.sum()),
                                                        "tail_normal_false_positives": int(fp[fp.interval_type.eq("trailing_normal")].false_positive_points.sum()),
                                                        "coverage": float(finite.mean()), "zero_applicable_points": int((~finite).sum()),
                                                        "point_adjustment": False})
    return {"metrics": metrics, "events_detected": int(events.detected.sum()), "coverage": float(finite.mean()), "complementarity": comp}


def run(cfg: dict[str, Any], dry_run: bool = False, smoke: bool = False) -> dict[str, Any]:
    if cfg["seed"] != 1 or tuple(cfg["lag_seconds"]) != LAG_SECONDS or tuple(cfg["representations"]) != REPRESENTATIONS:
        raise ValueError("M2S fixed seed, lag set, or representation contract changed")
    artifact = validate_m1_artifacts(cfg)
    if sha256_file(resolve_data_path(cfg["data"]["data_path"])) != cfg["data"]["source_sha256"]:
        raise ValueError("Official source CSV hash changed")
    base = resolve(cfg["output"]["dir"]); output = base / ("dry_run" if dry_run else "smoke") if dry_run or smoke else base
    output.mkdir(parents=True, exist_ok=True)
    data = load_recovery_data(resolve_data_path(cfg["data"]["data_path"]), resolve(cfg["data"]["m1_dir"]))
    split_manifest, split = train_relation_split(data.timestamps, cfg["split"]["window_size"], cfg["split"]["discovery_ratio"])
    write_json(output / "train_relation_split_manifest.json", split_manifest)
    excluded = set(data.scaler_metadata["zero_variance_features"]); all_pairs = stable_pairs(data.feature_names, excluded)
    lag_diagnostics, sets = discover_lagged(data, split, all_pairs, cfg)
    raw = _raw_frame(resolve_data_path(cfg["data"]["data_path"]))
    state_audit, states, state_diagnostics, state_sets = discover_conditioned(data, split, all_pairs, raw, cfg)
    sets.update(state_sets); diagnostics = {**lag_diagnostics, **state_diagnostics}
    state_audit.to_csv(output / "state_channel_qualification.csv", index=False)
    pd.concat(list(diagnostics.values()), ignore_index=True, sort=False).to_csv(output / "all_contextual_relation_diagnostics.csv", index=False)
    summary_rows, payload = [], {"legacy_narrow_diagnostic": {"pair": LEGACY_NARROW, "automatic_promotion": False}}
    for representation in REPRESENTATIONS:
        selected = sets[representation]["relations"]
        summary_rows.append({"representation": representation, "reproducible_relation_count": len(selected),
                             "status": sets[representation]["status"], "general_candidate": len(selected) >= 3})
        payload[representation] = {"status": sets[representation]["status"], "relations": selected.to_dict("records")}
    pd.DataFrame(summary_rows).to_csv(output / "relation_set_summary.csv", index=False)
    write_json(output / "relation_sets.json", payload)
    counts = {name: len(sets[name]["relations"]) for name in REPRESENTATIONS}
    if dry_run:
        result = {**artifact, "code_readiness": "A. implementation_complete", "relation_counts": counts,
                  "state_channel_candidates": len(states), "validation_or_test_used_for_discovery": False,
                  "model_retrained": False, "fusion_computed": False, "multiple_seeds_run": False}
        write_json(output / "dry_run_summary.json", result); print(json.dumps(result, indent=2)); return result
    references = {}; floor = cfg["reproducibility"]["fisher_scale_floor"]
    for representation in REPRESENTATIONS:
        if counts[representation]:
            references[representation] = _reference_for_set(data, split, all_pairs, sets[representation]["relations"],
                                                             representation, states, floor)
    validation_manifest, validation = validation_usage_split(data.timestamps, cfg["split"]["window_size"])
    write_json(output / "validation_usage_manifest.json", validation_manifest)
    controlled, controlled_summary, coverage = controlled_validation(cfg, data, all_pairs, sets, references, states,
                                                                     validation, output, smoke)
    primary = select_primary(controlled, counts, coverage, cfg["controlled"]["minimum_primary_auroc"],
                             cfg["controlled"]["minimum_coverage"])
    primary["test_labels_read_before_freeze"] = False
    write_json(output / "frozen_primary_manifest.json", primary)
    if smoke:
        result = {**artifact, "code_readiness": "A. implementation_complete", "relation_counts": counts,
                  "state_channel_candidates": len(states), "controlled": controlled_summary, "primary": primary,
                  "formal_test_read": False, "model_retrained": False, "fusion_computed": False}
        write_json(output / "smoke_summary.json", result); print(json.dumps(result, indent=2)); return result
    real = None
    if primary["primary_frozen"]:
        representation = primary["representation"]; selected = sets[representation]["relations"]
        threshold_corr, threshold_app = correlations_for_relations(data.scaled_features, data.valid_points,
                                                                   validation["threshold_ends"], all_pairs, selected, representation, states)
        threshold_scores = score_correlations(threshold_corr, references[representation], threshold_app)["score"]
        real = formal_test_evaluation(cfg, data, all_pairs, selected, representation, references[representation], states,
                                      threshold_scores, output, primary)
    statuses = {name: sets[name]["status"] for name in REPRESENTATIONS}
    if primary["primary_frozen"]:
        scientific = ("A. lagged_general_structure_validated" if primary["representation"].startswith("lagged")
                      else "B. state_conditioned_general_structure_validated")
        if real is not None and real["metrics"]["auroc"] < .60:
            scientific = "C. controlled_validated_real_performance_weak"
    elif any(statuses[name] == "narrow" for name in ("lagged_level", "lagged_delta")):
        scientific = "D. only_narrow_lagged_relation_available"
    elif any(statuses[name] == "narrow" for name in ("conditioned_level", "conditioned_delta")):
        scientific = "E. only_narrow_conditioned_relation_available"
    else:
        scientific = "F. no_general_pairwise_structure_found"
    readiness = {"code_readiness": "A. implementation_complete", "scientific_readiness": scientific,
                 "source_hash_matches": True, "checkpoint_hash_matches": True, "relations_train_only": True,
                 "lag_selected_on_discovery_only": True, "state_qualification_train_only": True,
                 "validation_threshold_controlled_disjoint": validation_manifest["raw_timestamp_overlap"] is False,
                 "test_labels_read_only_after_primary_freeze": True, "primary_frozen": primary["primary_frozen"],
                 "model_retrained": False, "fusion_computed": False, "multiple_seeds_run": False}
    write_json(output / "batch_m2s_readiness.json", readiness); write_json(output / "resolved_config.json", cfg)
    result = {"readiness": readiness, "relation_counts": counts, "state_channel_candidates": len(states),
              "controlled_metrics": controlled.to_dict("records"), "primary": primary, "real_test": real}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str)); return result


def main() -> int:
    parser = argparse.ArgumentParser(description="SWaT Batch-M2S contextual structure analysis")
    parser.add_argument("--config", required=True); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    if args.dry_run and args.smoke:
        raise ValueError("--dry-run and --smoke are mutually exclusive")
    run(load_config(args.config), args.dry_run, args.smoke); return 0


if __name__ == "__main__":
    raise SystemExit(main())
