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

from data.swat2026_conditional_prediction import (  # noqa: E402
    ALPHAS, CROSS_LAGS, SELF_LAGS, aggregate_target_scores, circular_target_injection,
    cross_design, eligible_indices, fit_target_discovery, fixed_threshold_metrics, mechanism_status,
    refit_frozen_target, score_models, select_primary, validation_split,
)
from data.swat2026_structure_recovery import load_recovery_data  # noqa: E402
from data.swat2026_training import sha256_file  # noqa: E402
from scripts.swat2026.analyze_rec_and_structure import (  # noqa: E402
    contiguous_normal_intervals, event_diagnostics, injected_rec_scores, load_frozen_model,
    resolve, validate_m1_artifacts, write_json,
)
from scripts.swat2026.train_eval_rec_baseline import build_device, point_metrics  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.project_paths import resolve_data_path  # noqa: E402


DISCOVERY_START = pd.Timestamp("2026-03-11 09:00:00")
DISCOVERY_END = pd.Timestamp("2026-03-11 11:13:12")
CONFIRMATION_START = pd.Timestamp("2026-03-11 11:15:19")
TRAIN_END = pd.Timestamp("2026-03-11 12:10:18")
VALIDATION_START = TRAIN_END
VALIDATION_END = pd.Timestamp("2026-03-11 12:57:53")
TEST_CONTEXT_START = VALIDATION_END
TEST_START = pd.Timestamp("2026-03-11 13:00:00")
TEST_END = pd.Timestamp("2026-03-11 17:00:00")


def _bounds(timestamps: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int]:
    indices = np.flatnonzero(np.asarray((timestamps >= start) & (timestamps < end)))
    if not len(indices) or not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
        raise ValueError(f"Non-contiguous interval: {start} to {end}")
    return int(indices[0]), int(indices[-1] + 1)


def _score_available(values: np.ndarray, validity: np.ndarray, indices: np.ndarray, models) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = tuple(sorted(set(CROSS_LAGS + SELF_LAGS)))
    good = np.array([all(index - lag >= 0 and validity[index - lag] for lag in required) for index in indices])
    cross = np.full((len(indices), len(models)), np.nan); self_scores = np.full_like(cross, np.nan)
    if good.any():
        cross[good], self_scores[good] = score_models(values, indices[good], models)
    return cross, self_scores, good


def discover_targets(cfg: dict[str, Any], data, values: np.ndarray, names: list[str]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    d0, d1 = _bounds(data.timestamps, DISCOVERY_START, DISCOVERY_END)
    c0, c1 = _bounds(data.timestamps, CONFIRMATION_START, TRAIN_END)
    required = tuple(sorted(set(CROSS_LAGS + SELF_LAGS)))
    discovery = eligible_indices(data.valid_points, d0, d1, required)
    confirmation = eligible_indices(data.valid_points, c0, c1, required)
    if set(discovery) & set(confirmation):
        raise ValueError("Discovery and Confirmation overlap")
    rows, searches = [], {}
    for target_index, target in enumerate(names):
        row, detail = fit_target_discovery(values, names, target_index, discovery, confirmation, cfg)
        rows.append(row); searches[target] = detail
    diagnostics = pd.DataFrame(rows)
    manifest = {"discovery_start": DISCOVERY_START.isoformat(), "discovery_end_exclusive": DISCOVERY_END.isoformat(),
                "discovery_valid_samples": len(discovery), "guard_start": DISCOVERY_END.isoformat(),
                "guard_end_exclusive": CONFIRMATION_START.isoformat(), "guard_points": 127,
                "confirmation_start": CONFIRMATION_START.isoformat(), "confirmation_end_exclusive": TRAIN_END.isoformat(),
                "confirmation_valid_samples": len(confirmation), "raw_timestamp_overlap": False,
                "maximum_history_lag": 30, "validation_used": False, "test_used": False,
                "alpha_selection_scope": "chronological final 20 percent of Discovery"}
    return diagnostics, {"manifest": manifest, "alpha_search": searches}, {"discovery": discovery, "confirmation": confirmation}


def freeze_qualified_models(cfg, data, values, names, diagnostics):
    t0, t1 = _bounds(data.timestamps, DISCOVERY_START, TRAIN_END)
    required = tuple(sorted(set(CROSS_LAGS + SELF_LAGS)))
    train_indices = eligible_indices(data.valid_points, t0, t1, required)
    qualified = diagnostics[diagnostics.cross_predictable].copy()
    models = []
    for row in qualified.itertuples(index=False):
        models.append(refit_frozen_target(values, names, int(row.target_index), train_indices,
                                          float(row.selected_alpha_cross), float(row.selected_alpha_self),
                                          cfg["prediction"]["residual_scale_floor"]))
    return qualified, models, train_indices


def coefficient_frame(models) -> pd.DataFrame:
    rows = []
    for model in models:
        order = np.argsort(np.abs(model.cross.coef_))[::-1]
        for rank, index in enumerate(order, 1):
            label = model.predictor_names[int(index)]; channel, lag = label.rsplit("@lag", 1)
            rows.append({"target": model.target, "predictor_channel": channel, "lag": int(lag),
                         "coefficient": float(model.cross.coef_[index]),
                         "absolute_coefficient": float(abs(model.cross.coef_[index])), "coefficient_rank": rank,
                         "interpretation": "predictive contribution, not a causal claim"})
    return pd.DataFrame(rows)


def _controlled_blocks(indices: np.ndarray, window_size: int) -> list[np.ndarray]:
    blocks = []
    if not len(indices): return blocks
    starts = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1]
    ends = np.r_[starts[1:], len(indices)]
    for start, end in zip(starts, ends):
        run = indices[start:end]
        for offset in range(0, len(run) - window_size + 1, window_size):
            blocks.append(run[offset:offset + window_size])
    return blocks


def _autocorrelation(values: np.ndarray, lag: int = 1) -> float:
    x = np.asarray(values, float)
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1]) if np.std(x[:-lag]) > 0 and np.std(x[lag:]) > 0 else np.nan


def controlled_validation(cfg, data, values, models, output, smoke=False):
    v0, v1 = _bounds(data.timestamps, VALIDATION_START, VALIDATION_END)
    manifest, split = validation_split(data.valid_points, v0, v1, cfg["validation"]["history_guard_points"])
    write_json(output / "validation_usage_manifest.json", manifest)
    if not models:
        pd.DataFrame().to_csv(output / "controlled_injection_manifest.csv", index=False)
        pd.DataFrame().to_csv(output / "controlled_metrics.csv", index=False)
        summary = {"injection_count": 0, "marginal_statistics_preserved": True, "test_used": False, "coverage": 0.0}
        write_json(output / "controlled_summary.json", summary)
        return pd.DataFrame(), summary, {}, manifest
    threshold_indices = split["threshold"][:256] if smoke else split["threshold"]
    threshold_cross, threshold_self, threshold_valid = _score_available(values, data.valid_points, threshold_indices, models)
    threshold_cross_agg = aggregate_target_scores(threshold_cross)
    threshold_self_agg = aggregate_target_scores(threshold_self)
    blocks = _controlled_blocks(split["controlled"], cfg["controlled"]["window_size"])
    if smoke: blocks = blocks[:2]
    manifest_rows, normal_cross_rows, anomaly_cross_rows = [], [], []
    normal_self_rows, anomaly_self_rows, original_full, injected_full = [], [], [], []
    shifts = cfg["controlled"]["shifts"]
    full_lookup = {name: index for index, name in enumerate(data.feature_names)}
    for model_index, model in enumerate(models):
        for block_index, block in enumerate(blocks):
            original = values[block].copy()
            shift = int(shifts[(model_index + block_index) % len(shifts)])
            injected = circular_target_injection(original, model.target_index, shift)
            before, after = original[:, model.target_index], injected[:, model.target_index]
            normal_cross, normal_self = score_models(original, np.array([len(original) - 1]), models)
            anomaly_cross, anomaly_self = score_models(injected, np.array([len(injected) - 1]), models)
            normal_cross_rows.append(normal_cross[0]); anomaly_cross_rows.append(anomaly_cross[0])
            normal_self_rows.append(normal_self[0]); anomaly_self_rows.append(anomaly_self[0])
            original_rec = data.scaled_features[block].copy()
            injected_rec = original_rec.copy(); full_target = full_lookup[model.target]
            injected_rec[:, full_target] = np.roll(injected_rec[:, full_target], shift)
            original_full.append(original_rec); injected_full.append(injected_rec)
            target_normal = normal_cross[0, model_index]; target_anomaly = anomaly_cross[0, model_index]
            manifest_rows.append({"target": model.target, "window": block_index,
                                  "start": data.timestamps[block[0]].isoformat(),
                                  "end_exclusive": (data.timestamps[block[-1]] + pd.Timedelta(seconds=1)).isoformat(),
                                  "shift": shift, "seed": 1, "test_used": False, "within_window": True,
                                  "value_multiset_preserved": bool(np.array_equal(np.sort(before), np.sort(after))),
                                  "mean_difference": abs(float(before.mean() - after.mean())),
                                  "std_difference": abs(float(before.std() - after.std())),
                                  "min_difference": abs(float(before.min() - after.min())),
                                  "max_difference": abs(float(before.max() - after.max())),
                                  "autocorrelation_difference": abs(_autocorrelation(before) - _autocorrelation(after)),
                                  "cross_residual_score_before": float(target_normal),
                                  "cross_residual_score_after": float(target_anomaly),
                                  "self_history_uses_injected_target": True,
                                  "other_cross_channels_unchanged": bool(np.array_equal(np.delete(original, model.target_index, 1),
                                                                                        np.delete(injected, model.target_index, 1)))})
    injection_manifest = pd.DataFrame(manifest_rows)
    injection_manifest.to_csv(output / "controlled_injection_manifest.csv", index=False)
    if not manifest_rows:
        return pd.DataFrame(), {"injection_count": 0, "marginal_statistics_preserved": True,
                                "test_used": False, "coverage": 0.0}, {}, manifest
    normal_cross_agg = aggregate_target_scores(np.asarray(normal_cross_rows))
    anomaly_cross_agg = aggregate_target_scores(np.asarray(anomaly_cross_rows))
    normal_self_agg = aggregate_target_scores(np.asarray(normal_self_rows))
    anomaly_self_agg = aggregate_target_scores(np.asarray(anomaly_self_rows))
    device = build_device("cuda"); frozen_m1 = load_frozen_model(cfg, device)
    rec_normal = injected_rec_scores(frozen_m1, np.asarray(original_full), device)
    rec_anomaly = injected_rec_scores(frozen_m1, np.asarray(injected_full), device)
    rec_validation = pd.read_csv(resolve(cfg["data"]["m1_dir"]) / "rec_scores_validation.csv")
    rec_validation["timestamp"] = pd.to_datetime(rec_validation.timestamp)
    threshold_times = set(data.timestamps[threshold_indices])
    rec_threshold_values = rec_validation[rec_validation.timestamp.isin(threshold_times)].rec_score.to_numpy(float)
    if not len(rec_threshold_values):
        raise ValueError("M1 validation_threshold reconstruction scores are unavailable")
    preserved = bool(injection_manifest.value_multiset_preserved.all() and
                     injection_manifest[["mean_difference", "std_difference", "min_difference", "max_difference"]].to_numpy().max() < 1e-6)
    rows = []
    definitions = (
        ("cross_max", normal_cross_agg["max_target_score"], anomaly_cross_agg["max_target_score"], threshold_cross_agg["max_target_score"]),
        ("cross_top3_mean", normal_cross_agg["top3_mean_target_score"], anomaly_cross_agg["top3_mean_target_score"], threshold_cross_agg["top3_mean_target_score"]),
        ("self_max", normal_self_agg["max_target_score"], anomaly_self_agg["max_target_score"], threshold_self_agg["max_target_score"]),
        ("self_top3_mean", normal_self_agg["top3_mean_target_score"], anomaly_self_agg["top3_mean_target_score"], threshold_self_agg["top3_mean_target_score"]),
        ("rec_score", rec_normal, rec_anomaly, rec_threshold_values),
    )
    score_arrays = {}
    for name, normal, anomaly, threshold_values in definitions:
        metrics = fixed_threshold_metrics(normal, anomaly, threshold_values, cfg["validation"]["quantile"])
        coverage = 1.0 if name == "rec_score" else float(np.isfinite(normal).mean())
        rows.append({"score_name": name, **metrics, "coverage": coverage,
                     "marginal_statistics_preserved": preserved, "threshold_uses_injected_samples": False})
        score_arrays[name] = {"normal": normal, "anomaly": anomaly}
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(output / "controlled_metrics.csv", index=False)
    lookup = metrics_frame.set_index("score_name")
    cross_alarm = score_arrays["cross_max"]["anomaly"] > lookup.loc["cross_max", "threshold"]
    self_alarm = score_arrays["self_max"]["anomaly"] > lookup.loc["self_max", "threshold"]
    rec_alarm = score_arrays["rec_score"]["anomaly"] > lookup.loc["rec_score", "threshold"]
    summary = {"injection_count": len(injection_manifest), "marginal_statistics_preserved": preserved,
               "coverage": float(np.isfinite(normal_cross_agg["max_target_score"]).mean()),
               "cross_detected_self_missed_ratio": float((cross_alarm & ~self_alarm).mean()),
               "cross_detected_rec_missed_ratio": float((cross_alarm & ~rec_alarm).mean()),
               "cross_self_pearson": float(pd.Series(score_arrays["cross_max"]["anomaly"]).corr(pd.Series(score_arrays["self_max"]["anomaly"]))),
               "cross_rec_pearson": float(pd.Series(score_arrays["cross_max"]["anomaly"]).corr(pd.Series(score_arrays["rec_score"]["anomaly"]))),
               "test_used": False, "model_retrained": False, "fusion_computed": False}
    write_json(output / "controlled_summary.json", summary)
    return metrics_frame, summary, score_arrays, manifest


def _target_match(target: str, official: str) -> str:
    base = target.split(".")[0].upper(); official = str(official).upper()
    if base in official: return "exact target match"
    if len(base) >= 2 and base[:2] in official: return "same stage match"
    return "no obvious match"


def formal_test_evaluation(cfg, data, values, models, primary, controlled, output):
    if not primary["primary_frozen"]:
        raise ValueError("Test labels cannot be read before Primary freeze")
    # Test labels and official targets are opened only beyond the frozen-Primary gate.
    labels = pd.read_csv(resolve(cfg["data"]["n2_dir"]) / "point_labels_11mar.csv", encoding="utf-8-sig", dtype={"attack_id": str})
    labels["timestamp"] = pd.to_datetime(labels.timestamp)
    schedule = pd.read_csv(resolve(cfg["data"]["n2_dir"]) / "approved_attack_schedule.csv")
    rec = pd.read_csv(resolve(cfg["data"]["m1_dir"]) / "rec_scores_test.csv"); rec["timestamp"] = pd.to_datetime(rec.timestamp)
    t0, t1 = _bounds(data.timestamps, TEST_START, TEST_END)
    indices = np.arange(t0, t1, dtype=int)
    if len(indices) != cfg["evaluation"]["test_points"]:
        raise ValueError("formal Test must contain exactly 14,400 evaluation points")
    cross, self_scores, valid = _score_available(values, data.valid_points, indices, models)
    cross_agg, self_agg = aggregate_target_scores(cross), aggregate_target_scores(self_scores)
    aggregation = primary["aggregation"]
    key = "max_target_score" if aggregation == "cross_max" else "top3_mean_target_score"
    self_key = key
    threshold = float(controlled.set_index("score_name").loc[aggregation, "threshold"])
    score = cross_agg[key]
    frame = pd.DataFrame({"timestamp": data.timestamps[indices], "conditional_score": score,
                          "self_history_score": self_agg[self_key], "applicable_target_count": cross_agg["applicable_target_count"],
                          "coverage": cross_agg["coverage"]})
    for column, model in enumerate(models): frame[f"target_score__{model.target}"] = cross[:, column]
    frame = frame.merge(labels[["timestamp", "point_label", "attack_id"]], on="timestamp", how="left", validate="one_to_one")
    finite = frame.conditional_score.notna()
    metrics = point_metrics(frame.loc[finite, "point_label"].to_numpy(int), frame.loc[finite, "conditional_score"].to_numpy(float), threshold)
    frame["alarm"] = frame.conditional_score > threshold
    directory = output / "real_test"; directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "conditional_scores_test.csv", index=False)
    events = event_diagnostics(frame, schedule, "conditional_score", threshold)
    interpretation_rows = []
    for event in events.itertuples(index=False):
        current = frame[(frame.timestamp >= pd.Timestamp(event.start_time)) & (frame.timestamp < pd.Timestamp(event.end_time_exclusive))]
        if current.empty or not current.conditional_score.notna().any():
            interpretation_rows.append({"attack_id": event.attack_id, "top_residual_target": "", "top_predictor": "",
                                        "official_target_interpretation": "no applicable score", "causal_claimed": False}); continue
        point = current.loc[current.conditional_score.idxmax()]
        target_scores = np.array([point[f"target_score__{model.target}"] for model in models], float)
        target_index = int(np.nanargmax(target_scores)); model = models[target_index]
        global_index = int(np.flatnonzero(data.timestamps == point.timestamp)[0])
        x, _, _ = cross_design(values, model.target_index, np.array([global_index]))
        contribution = np.abs(x[0] * model.cross.coef_); predictor = model.predictor_names[int(np.argmax(contribution))]
        interpretation_rows.append({"attack_id": event.attack_id, "top_residual_target": model.target,
                                    "top_predictor": predictor,
                                    "official_target_interpretation": _target_match(model.target, event.official_targets),
                                    "coefficient_interpretation": "predictive contribution, not a causal chain", "causal_claimed": False})
    events = events.merge(pd.DataFrame(interpretation_rows), on="attack_id", how="left")
    events.to_csv(directory / "event_detection_summary.csv", index=False)
    false_positive = contiguous_normal_intervals(frame, schedule, "conditional_score", threshold)
    false_positive.to_csv(directory / "false_positive_intervals.csv", index=False)
    comparison = frame.merge(rec[["timestamp", "rec_score"]], on="timestamp", how="left")
    rec_threshold = float(pd.read_csv(resolve(cfg["data"]["m1_dir"]) / "rec_scores_validation.csv").rec_score.quantile(.99))
    attack = comparison.point_label.eq(1); cross_alarm = comparison.conditional_score > threshold; rec_alarm = comparison.rec_score > rec_threshold
    applicable = comparison.conditional_score.notna()
    complementarity = {"test_points": len(comparison), "applicable_points": int(applicable.sum()), "coverage": float(applicable.mean()),
        "pearson_cross_rec": float(comparison.loc[applicable, ["conditional_score", "rec_score"]].corr(method="pearson").iloc[0, 1]),
        "spearman_cross_rec": float(comparison.loc[applicable, ["conditional_score", "rec_score"]].corr(method="spearman").iloc[0, 1]),
        "cross_adds_rec_missed_attack_points": int((attack & cross_alarm & ~rec_alarm).sum()),
        "rec_adds_cross_missed_attack_points": int((attack & rec_alarm & ~cross_alarm).sum()),
        "both_hit_attack_points": int((attack & cross_alarm & rec_alarm).sum()),
        "both_miss_attack_points": int((attack & ~cross_alarm & ~rec_alarm).sum()),
        "false_positive_overlap": int((~attack & cross_alarm & rec_alarm).sum()),
        "cross_new_false_positives": int((~attack & cross_alarm & ~rec_alarm).sum()), "fusion_computed": False}
    rec_events = event_diagnostics(comparison, schedule, "rec_score", rec_threshold)
    event_comparison = events[["attack_id", "detected"]].rename(columns={"detected": "cross_detected"}).merge(
        rec_events[["attack_id", "detected"]].rename(columns={"detected": "rec_detected"}), on="attack_id")
    complementarity.update({"cross_adds_rec_missed_events": int((event_comparison.cross_detected & ~event_comparison.rec_detected).sum()),
                            "rec_adds_cross_missed_events": int((event_comparison.rec_detected & ~event_comparison.cross_detected).sum()),
                            "both_detect_events": int((event_comparison.cross_detected & event_comparison.rec_detected).sum()),
                            "both_miss_events": int((~event_comparison.cross_detected & ~event_comparison.rec_detected).sum())})
    write_json(directory / "complementarity_summary.json", complementarity)
    result = {"metrics": metrics, "events_detected": int(events.detected.sum()),
              "mean_detection_delay": float(pd.to_numeric(events.first_detection_delay_seconds, errors="coerce").mean()),
              "preexisting_alarm_count": int(events.preexisting_alarm.sum()),
              "inter_attack_false_positives": int(false_positive[false_positive.interval_type.eq("inter_attack_normal")].false_positive_points.sum()),
              "trailing_false_positives": int(false_positive[false_positive.interval_type.eq("trailing_normal")].false_positive_points.sum()),
              "coverage": float(applicable.mean()), "complementarity": complementarity}
    write_json(directory / "real_test_summary.json", result)
    return result


def _scientific_readiness(qualified_count: int, controlled: pd.DataFrame, primary: dict[str, Any], real) -> str:
    if primary["primary_frozen"]:
        return ("A. general_conditional_collaboration_validated_real_complete"
                if real is not None and real["metrics"]["auroc"] >= .60
                else "B. controlled_collaboration_validated_real_weak")
    if qualified_count == 0: return "E. no_reproducible_cross_channel_prediction"
    if qualified_count < 3: return "D. only_narrow_conditional_diagnostic"
    if controlled.empty: return "C. cross_predictable_targets_found_controlled_weak"
    rows = controlled.set_index("score_name")
    best = max(float(rows.loc["cross_max", "auroc"]), float(rows.loc["cross_top3_mean", "auroc"]))
    self_best = max(float(rows.loc["self_max", "auroc"]), float(rows.loc["self_top3_mean", "auroc"]))
    if best - self_best < .10: return "F. cross_score_not_distinct_from_self_history"
    if best - float(rows.loc["rec_score", "auroc"]) < .10: return "G. cross_score_not_distinct_from_reconstruction"
    return "C. cross_predictable_targets_found_controlled_weak"


def run(cfg: dict[str, Any], dry_run: bool = False, smoke: bool = False) -> dict[str, Any]:
    if (cfg["seed"] != 1 or tuple(cfg["prediction"]["cross_lags"]) != CROSS_LAGS or
            tuple(cfg["prediction"]["self_lags"]) != SELF_LAGS or tuple(cfg["prediction"]["alphas"]) != ALPHAS):
        raise ValueError("M2P fixed seed, lag, or alpha contract changed")
    artifact = validate_m1_artifacts(cfg)
    if sha256_file(resolve_data_path(cfg["data"]["data_path"])) != cfg["data"]["source_sha256"]:
        raise ValueError("Official source CSV hash changed")
    base = resolve(cfg["output"]["dir"]); output = base / ("dry_run" if dry_run else "smoke") if dry_run or smoke else base
    output.mkdir(parents=True, exist_ok=True)
    data = load_recovery_data(resolve_data_path(cfg["data"]["data_path"]), resolve(cfg["data"]["m1_dir"]))
    excluded = set(data.scaler_metadata["zero_variance_features"])
    active_full_indices = [index for index, name in enumerate(data.feature_names) if name not in excluded]
    names = [data.feature_names[index] for index in active_full_indices]
    values = data.scaled_features[:, active_full_indices]
    diagnostics, discovery, split = discover_targets(cfg, data, values, names)
    diagnostics.to_csv(output / "target_model_diagnostics.csv", index=False)
    write_json(output / "train_discovery_confirmation_manifest.json", discovery["manifest"])
    write_json(output / "alpha_search_diagnostics.json", discovery["alpha_search"])
    qualified_count = int(diagnostics.cross_predictable.sum()); status = mechanism_status(qualified_count)
    target_summary = {"candidate_targets": len(diagnostics), "qualified_targets": qualified_count,
                      "qualified_target_names": diagnostics.loc[diagnostics.cross_predictable, "target"].tolist(),
                      "mechanism_status": status,
                      "process_stages": sorted({target.split(".")[0][-3:-2] for target in diagnostics.loc[diagnostics.cross_predictable, "target"]}),
                      "channel_types": sorted({target[:3] for target in diagnostics.loc[diagnostics.cross_predictable, "target"]}),
                      "official_attack_targets_used_for_selection": False}
    write_json(output / "target_qualification_summary.json", target_summary)
    if dry_run:
        result = {**artifact, "code_readiness": "A. implementation_complete", **target_summary,
                  "validation_or_test_used_for_selection": False, "scaler_refitted": False,
                  "model_retrained": False, "fusion_computed": False, "multiple_seeds_run": False}
        write_json(output / "dry_run_summary.json", result); print(json.dumps(result, indent=2)); return result
    qualified, models, train_indices = freeze_qualified_models(cfg, data, values, names, diagnostics)
    coefficients = coefficient_frame(models)
    coefficients.to_csv(output / "cross_channel_coefficients.csv", index=False)
    if len(coefficients): coefficients[coefficients.coefficient_rank <= 10].to_csv(output / "top10_cross_predictors.csv", index=False)
    write_json(output / "frozen_target_model_manifest.json",
               {"targets": [{"target": model.target, "alpha_cross": model.alpha_cross, "alpha_self": model.alpha_self,
                              "train_samples": len(train_indices), "cross_reference": model.residual_reference_cross,
                              "self_reference": model.residual_reference_self} for model in models],
                "fit_scope": "complete Train only", "validation_used": False, "test_used": False,
                "scaler_source": "frozen M1 train-only StandardScaler"})
    controlled, controlled_summary, score_arrays, validation_manifest = controlled_validation(cfg, data, values, models, output, smoke)
    primary = select_primary(controlled, qualified_count, cfg["controlled"]["minimum_auroc"],
                             cfg["controlled"]["minimum_advantage"], cfg["controlled"]["minimum_coverage"])
    primary.update({"qualified_targets": qualified.target.tolist(), "test_labels_read_before_freeze": False})
    write_json(output / "frozen_primary_manifest.json", primary)
    if smoke:
        result = {**artifact, "code_readiness": "A. implementation_complete", **target_summary,
                  "controlled": controlled_summary, "primary": primary, "formal_test_read": False,
                  "model_retrained": False, "fusion_computed": False}
        write_json(output / "smoke_summary.json", result); print(json.dumps(result, indent=2)); return result
    real = None
    if primary["primary_frozen"]:
        real = formal_test_evaluation(cfg, data, values, models, primary, controlled, output)
    else:
        t0, t1 = _bounds(data.timestamps, TEST_START, TEST_END); indices = np.arange(t0, t1)
        if models:
            cross, _, valid = _score_available(values, data.valid_points, indices, models)
            coverage = float(np.isfinite(aggregate_target_scores(cross)["max_target_score"]).mean())
        else:
            coverage = 0.0
        write_json(output / "unlabeled_test_coverage.json", {"test_points": len(indices), "coverage": coverage,
                                                               "test_labels_read": False, "formal_evaluation_run": False})
    scientific = _scientific_readiness(qualified_count, controlled, primary, real)
    readiness = {"code_readiness": "A. implementation_complete", "scientific_readiness": scientific,
                 "source_hash_matches": True, "checkpoint_hash_matches": True, "m1_scaler_reused": True,
                 "targets_selected_train_only": True, "alphas_selected_discovery_only": True,
                 "validation_threshold_controlled_disjoint": validation_manifest["raw_history_overlap"] is False,
                 "test_labels_read_only_after_primary_freeze": True, "primary_frozen": primary["primary_frozen"],
                 "model_retrained": False, "fusion_computed": False, "multiple_seeds_run": False}
    write_json(output / "batch_m2p_readiness.json", readiness); write_json(output / "resolved_config.json", cfg)
    result = {"readiness": readiness, **target_summary, "controlled_metrics": controlled.to_dict("records"),
              "controlled_summary": controlled_summary, "primary": primary, "real_test": real}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str)); return result


def main() -> int:
    parser = argparse.ArgumentParser(description="SWaT Batch-M2P conditional cross-channel prediction")
    parser.add_argument("--config", required=True); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    if args.dry_run and args.smoke: raise ValueError("--dry-run and --smoke are mutually exclusive")
    run(load_config(args.config), args.dry_run, args.smoke); return 0


if __name__ == "__main__":
    raise SystemExit(main())
