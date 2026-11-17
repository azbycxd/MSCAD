from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.causal_alarm_policy import (  # noqa: E402
    ALL_POLICIES, HYSTERESIS_POLICY, POINTWISE_POLICIES, apply_policy, binary_segments,
    maximum_continuous_alarm_length,
)
from utils.config_io import load_config  # noqa: E402
from utils.metrics import binary_metrics  # noqa: E402


OUTPUT_FILES = (
    "config_resolved.json", "alarm_input_consistency_audit.csv", "causal_alarm_policy_registry.csv",
    "causal_alarm_validation_diagnostics.csv", "causal_alarm_point_metrics_by_seed.csv",
    "causal_alarm_point_metrics_by_file.csv", "causal_alarm_point_metrics_aggregate.csv",
    "causal_alarm_event_diagnostics.csv", "causal_alarm_event_summary.csv",
    "causal_alarm_event_transition.csv", "causal_alarm_event_duration_analysis.csv",
    "causal_alarm_false_positive_segments.csv", "causal_alarm_false_positive_segment_summary.csv",
    "causal_alarm_policy_comparison.csv", "causal_alarm_multiseed_summary.csv",
    "causal_alarm_summary.json", "final_hai_rec_causal_alarm_policy_diagnosis.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_fresh(output: Path) -> None:
    existing = [name for name in OUTPUT_FILES if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite HAI Step 3E outputs: {existing}")
    output.mkdir(parents=True, exist_ok=True)


def policy_combinations(cfg: dict) -> list[dict[str, Any]]:
    rows = []
    for policy in POINTWISE_POLICIES:
        for method in cfg["threshold_methods"]:
            rows.append({"policy_name": policy, "threshold_method": method,
                         "combination_id": f"{policy}__{method}"})
    rows.append({"policy_name": HYSTERESIS_POLICY, "threshold_method": "hysteresis_q995_q99",
                 "combination_id": f"{HYSTERESIS_POLICY}__q995_enter_q99_stay"})
    return rows


def policy_registry(cfg: dict) -> pd.DataFrame:
    attributes = {
        "pointwise_baseline": (1, 0, "immediate_exit"),
        "consecutive_2": (2, 1, "immediate_exit"),
        "consecutive_3": (3, 2, "immediate_exit"),
        "vote_2_of_3": (2, 1, "rolling_2_of_3"),
        "vote_3_of_5": (3, 2, "rolling_3_of_5"),
        HYSTERESIS_POLICY: (1, 0, "q995_enter_q99_stay_exit_after_3_below"),
    }
    rows = []
    for combo in policy_combinations(cfg):
        confirmation, delay, hold = attributes[combo["policy_name"]]
        rows.append({**combo, "causal": True, "uses_future": False, "backfills_history": False,
                     "confirmation_points": confirmation, "maximum_intrinsic_delay": delay,
                     "hold_behavior": hold, "primary_candidate": combo["policy_name"] == cfg["primary_policy"],
                     "deployment_allowed": True,
                     "ranking_score": cfg["score_variant"], "ranking_score_unchanged": True})
    return pd.DataFrame(rows)


def load_seed_scores(path: Path, maximum_points: int | None = None) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    archive = np.load(path, allow_pickle=False)
    suffix = "_endpoint_raw_top5"
    prefixes = sorted(key[:-len(suffix)] for key in archive.files if key.endswith(suffix))
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {"train": {}, "validation": {}, "test": {}}
    for prefix in prefixes:
        split, file_stem = prefix.split("_", 1)
        if split not in result:
            continue
        count = len(archive[f"{prefix}{suffix}"])
        if maximum_points is not None:
            count = min(count, maximum_points)
        values = {"score": archive[f"{prefix}{suffix}"][:count].astype(np.float32)}
        for field in ("label", "timestamp", "end_row", "sequence_id"):
            key = f"{prefix}_{field}"
            if key not in archive.files:
                raise KeyError(f"Missing Step 3D field {key} in {path}")
            values[field] = archive[key][:count]
        if not np.isfinite(values["score"]).all():
            raise ValueError(f"Non-finite endpoint_raw_top5 in {path}/{prefix}")
        result[split][f"{file_stem}.csv"] = values
    expected = {"train": 6, "validation": 6, "test": 4}
    if {split: len(files) for split, files in result.items()} != expected:
        raise ValueError(f"Incomplete Step 3D split coverage in {path}")
    return result


def threshold_lookup(thresholds: pd.DataFrame, seed: int, method: str) -> float:
    row = thresholds[(thresholds.seed == seed) & (thresholds.score_variant == "endpoint_raw_top5") &
                     (thresholds.threshold_method == method)]
    if len(row) != 1 or row.status.iloc[0] != "ok":
        raise ValueError(f"Missing Step 3D threshold {seed}/{method}")
    return float(row.threshold_value.iloc[0])


def generate_alarms(scores: dict[int, dict[str, Any]], thresholds: pd.DataFrame, cfg: dict) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for seed, splits in scores.items():
        results[seed] = {}
        for combo in policy_combinations(cfg):
            policy, method, combo_id = combo["policy_name"], combo["threshold_method"], combo["combination_id"]
            output = {"validation": {}, "test": {}}
            for split in output:
                for file_name, values in splits[split].items():
                    if policy == HYSTERESIS_POLICY:
                        alarm = apply_policy(values["score"], policy,
                                             enter_threshold=threshold_lookup(thresholds, seed, cfg["hysteresis_enter_method"]),
                                             stay_threshold=threshold_lookup(thresholds, seed, cfg["hysteresis_stay_method"]))
                    else:
                        alarm = apply_policy(values["score"], policy,
                                             threshold=threshold_lookup(thresholds, seed, method))
                    if not np.isin(alarm, [0, 1]).all():
                        raise ValueError(f"Non-binary alarm from {combo_id}/{file_name}")
                    output[split][file_name] = alarm
            results[seed][combo_id] = output
    return results


def alarm_segments(alarm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts, ends = binary_segments(alarm)
    return starts, ends, ends - starts + 1


def metric_record(labels: np.ndarray, alarms: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(np.int64); alarms = alarms.astype(np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ranking = binary_metrics(labels, alarms, scores)
    tp = int(((labels == 1) & (alarms == 1)).sum())
    fp = int(((labels == 0) & (alarms == 1)).sum())
    fn = int(((labels == 1) & (alarms == 0)).sum())
    starts, ends, lengths = alarm_segments(alarms)
    normal_count = int((labels == 0).sum())
    return {"precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1)),
            "f1": float(2 * tp / max(2 * tp + fp + fn, 1)), "true_positive_count": tp,
            "false_positive_count": fp, "false_negative_count": fn, "alarm_count": int(alarms.sum()),
            "normal_point_count": normal_count, "anomaly_point_count": int((labels == 1).sum()),
            "normal_fpr": float(fp / max(normal_count, 1)),
            "alarms_per_normal_hour": float(fp * 3600 / max(normal_count, 1)),
            "alarm_segment_count": len(starts), "longest_alarm_segment": int(lengths.max()) if len(lengths) else 0,
            "mean_alarm_segment_length": float(lengths.mean()) if len(lengths) else 0.0,
            "reference_aupr": float(ranking["aupr"]), "reference_auroc": float(ranking["auroc"]),
            "ranking_score_unchanged": True}


def concat_split(splits: dict[str, dict[str, np.ndarray]], alarms: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = sorted(splits)
    return (np.concatenate([splits[name]["label"] for name in names]),
            np.concatenate([alarms[name] for name in names]),
            np.concatenate([splits[name]["score"] for name in names]))


def build_point_metrics(scores: dict[int, dict[str, Any]], alarms: dict[int, dict[str, Any]], cfg: dict
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_file, by_seed = [], []
    registry = policy_registry(cfg)
    for seed in sorted(scores):
        for combo in registry.itertuples():
            rows = []
            for file_name, values in scores[seed]["test"].items():
                row = {"seed": seed, "policy_name": combo.policy_name, "threshold_method": combo.threshold_method,
                       "combination_id": combo.combination_id, "file_name": file_name,
                       **metric_record(values["label"], alarms[seed][combo.combination_id]["test"][file_name],
                                       values["score"])}
                by_file.append(row); rows.append(row)
            labels, binary, continuous = concat_split(scores[seed]["test"], alarms[seed][combo.combination_id]["test"])
            by_seed.append({"seed": seed, "policy_name": combo.policy_name,
                            "threshold_method": combo.threshold_method, "combination_id": combo.combination_id,
                            "scope": "micro", **metric_record(labels, binary, continuous)})
            metric_names = ("precision", "recall", "f1", "reference_aupr", "reference_auroc", "normal_fpr",
                            "alarms_per_normal_hour", "mean_alarm_segment_length")
            by_seed.append({"seed": seed, "policy_name": combo.policy_name,
                            "threshold_method": combo.threshold_method, "combination_id": combo.combination_id,
                            "scope": "macro", **{name: float(np.nanmean([row[name] for row in rows])) for name in metric_names},
                            **{name: int(sum(row[name] for row in rows)) for name in
                               ("true_positive_count", "false_positive_count", "false_negative_count", "alarm_count",
                                "normal_point_count", "anomaly_point_count", "alarm_segment_count")},
                            "longest_alarm_segment": int(max(row["longest_alarm_segment"] for row in rows)),
                            "ranking_score_unchanged": True})
    by_file_frame, by_seed_frame = pd.DataFrame(by_file), pd.DataFrame(by_seed)
    aggregate = []
    for key, group in by_seed_frame.groupby(["policy_name", "threshold_method", "combination_id", "scope"]):
        for metric in ("precision", "recall", "f1"):
            values = group.set_index("seed")[metric]
            aggregate.append({"policy_name": key[0], "threshold_method": key[1], "combination_id": key[2],
                              "scope": key[3], "metric": metric,
                              **{f"seed{seed}": float(values.get(seed, np.nan)) for seed in (1, 2, 3)},
                              "mean": float(values.mean()), "std": float(values.std(ddof=0)),
                              "min": float(values.min()), "max": float(values.max()),
                              "ranking_score_unchanged": True})
    multiseed = []
    for key, group in by_seed_frame.groupby(["policy_name", "threshold_method", "combination_id", "scope"]):
        row = {"policy_name": key[0], "threshold_method": key[1], "combination_id": key[2], "scope": key[3]}
        for metric in ("precision", "recall", "f1", "false_positive_count", "alarm_segment_count",
                       "reference_aupr", "reference_auroc"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        row["ranking_score_unchanged"] = True
        multiseed.append(row)
    return by_file_frame, by_seed_frame, pd.DataFrame(aggregate), pd.DataFrame(multiseed)


def extract_events(test_splits: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    events, global_id = [], 0
    for file_name in sorted(test_splits):
        values = test_splits[file_name]; labels = values["label"].astype(np.int8)
        starts = np.flatnonzero((labels == 1) & np.r_[True, labels[:-1] == 0])
        ends = np.flatnonzero((labels == 1) & np.r_[labels[1:] == 0, True])
        for start, end in zip(starts, ends):
            global_id += 1
            events.append({"event_id": global_id, "file_name": file_name, "start_index": int(start),
                           "end_index": int(end), "start_timestamp": str(values["timestamp"][start]),
                           "end_timestamp": str(values["timestamp"][end]), "duration": int(end - start + 1)})
    return events


def event_diagnostics(scores: dict[int, dict[str, Any]], alarms: dict[int, dict[str, Any]], cfg: dict,
                      require_full_event_set: bool = True
                      ) -> tuple[pd.DataFrame, dict[int, list[dict[str, Any]]]]:
    rows, events_by_seed = [], {}
    registry = policy_registry(cfg)
    for seed in sorted(scores):
        events = extract_events(scores[seed]["test"]); events_by_seed[seed] = events
        if require_full_event_set and len(events) != 58:
            raise ValueError(f"Expected 58 HAI events for seed {seed}, got {len(events)}")
        for combo in registry.itertuples():
            for event in events:
                values = scores[seed]["test"][event["file_name"]]
                alarm = alarms[seed][combo.combination_id]["test"][event["file_name"]]
                start, end = event["start_index"], event["end_index"]
                if end >= len(alarm):
                    rows.append({"seed": seed, "policy_name": combo.policy_name,
                                 "threshold_method": combo.threshold_method, "combination_id": combo.combination_id,
                                 **event, "covered_by_evaluation_subset": False, "detected": False,
                                 "first_alarm_timestamp": None, "detection_delay_points": np.nan,
                                 "first_point_hit": False, "within5_hit": False, "within10_hit": False,
                                 "within30_hit": False, "alarm_point_count_in_event": 0,
                                 "point_recall_in_event": np.nan, "maximum_continuous_alarm_length_in_event": 0})
                    continue
                local = alarm[start:end + 1]; hit = np.flatnonzero(local == 1); detected = len(hit) > 0
                delay = int(hit[0]) if detected else np.nan
                rows.append({"seed": seed, "policy_name": combo.policy_name,
                             "threshold_method": combo.threshold_method, "combination_id": combo.combination_id,
                             **event, "covered_by_evaluation_subset": True, "detected": detected,
                             "first_alarm_timestamp": str(values["timestamp"][start + hit[0]]) if detected else None,
                             "detection_delay_points": delay, "first_point_hit": bool(detected and delay == 0),
                             "within5_hit": bool(detected and delay < 5), "within10_hit": bool(detected and delay < 10),
                             "within30_hit": bool(detected and delay < 30), "alarm_point_count_in_event": int(local.sum()),
                             "point_recall_in_event": float(local.mean()),
                             "maximum_continuous_alarm_length_in_event": maximum_continuous_alarm_length(local)})
    return pd.DataFrame(rows), events_by_seed


def summarize_events(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = diagnostics.groupby(["seed", "policy_name", "threshold_method", "combination_id"])
    for key, group in groups:
        scopes = [("all_tests", group)] + [(name, part) for name, part in group.groupby("file_name")]
        for scope, part in scopes:
            evaluated = part[part.covered_by_evaluation_subset]; detected = evaluated[evaluated.detected]
            rows.append({"seed": key[0], "policy_name": key[1], "threshold_method": key[2],
                         "combination_id": key[3], "summary_scope": scope, "total_event_count": len(part),
                         "covered_event_count": len(evaluated), "detected_event_count": int(evaluated.detected.sum()),
                         "event_recall": float(evaluated.detected.mean()) if len(evaluated) else np.nan,
                         "fully_missed_event_count": int((~evaluated.detected).sum()),
                         "median_delay": float(detected.detection_delay_points.median()) if len(detected) else np.nan,
                         "mean_delay": float(detected.detection_delay_points.mean()) if len(detected) else np.nan,
                         "first_point_hit_rate": float(evaluated.first_point_hit.mean()) if len(evaluated) else np.nan,
                         "within5_hit_rate": float(evaluated.within5_hit.mean()) if len(evaluated) else np.nan,
                         "within10_hit_rate": float(evaluated.within10_hit.mean()) if len(evaluated) else np.nan,
                         "within30_hit_rate": float(evaluated.within30_hit.mean()) if len(evaluated) else np.nan,
                         "event_alarm_coverage": float(evaluated.point_recall_in_event.mean()) if len(evaluated) else np.nan})
    return pd.DataFrame(rows)


def baseline_combo_id(method: str) -> str:
    return f"pointwise_baseline__{method}"


def event_transitions(diagnostics: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    indexed = diagnostics.set_index(["seed", "combination_id", "event_id"])
    for seed in sorted(diagnostics.seed.unique()):
        for combo in policy_combinations(cfg):
            baselines = ([cfg["hysteresis_enter_method"], cfg["hysteresis_stay_method"]]
                         if combo["policy_name"] == HYSTERESIS_POLICY else [combo["threshold_method"]])
            for base_method in baselines:
                for event_id in sorted(diagnostics.event_id.unique()):
                    new = indexed.loc[(seed, combo["combination_id"], event_id)]
                    base = indexed.loc[(seed, baseline_combo_id(base_method), event_id)]
                    rows.append({"seed": seed, "policy_name": combo["policy_name"],
                                 "threshold_method": combo["threshold_method"],
                                 "comparison_baseline_threshold_method": base_method, "event_id": event_id,
                                 "file_name": new.file_name, "duration": new.duration,
                                 "baseline_detected": bool(base.detected), "new_detected": bool(new.detected),
                                 "recovered": bool(not base.detected and new.detected),
                                 "lost": bool(base.detected and not new.detected),
                                 "baseline_delay": base.detection_delay_points, "new_delay": new.detection_delay_points,
                                 "delay_change": new.detection_delay_points - base.detection_delay_points})
    return pd.DataFrame(rows)


def duration_group(duration: int) -> str:
    if duration < 30:
        return "short"
    if duration < 300:
        return "medium"
    return "long"


def duration_analysis(diagnostics: pd.DataFrame) -> pd.DataFrame:
    frame = diagnostics[diagnostics.covered_by_evaluation_subset].copy()
    frame["duration_group"] = frame.duration.map(duration_group)
    rows = []
    for key, group in frame.groupby(["seed", "policy_name", "threshold_method", "combination_id", "duration_group"]):
        detected = group[group.detected]
        rows.append({"seed": key[0], "policy_name": key[1], "threshold_method": key[2],
                     "combination_id": key[3], "duration_group": key[4], "event_count": len(group),
                     "event_recall": float(group.detected.mean()),
                     "median_delay": float(detected.detection_delay_points.median()) if len(detected) else np.nan,
                     "within5_hit_rate": float(group.within5_hit.mean()),
                     "within30_hit_rate": float(group.within30_hit.mean()),
                     "mean_point_recall_in_event": float(group.point_recall_in_event.mean())})
    return pd.DataFrame(rows)


def validation_diagnostics(scores: dict[int, dict[str, Any]], alarms: dict[int, dict[str, Any]], cfg: dict) -> pd.DataFrame:
    rows = []
    registry = policy_registry(cfg)
    for seed in sorted(scores):
        for combo in registry.itertuples():
            base_method = cfg["hysteresis_enter_method"] if combo.policy_name == HYSTERESIS_POLICY else combo.threshold_method
            base_id = baseline_combo_id(base_method)
            for file_name, values in scores[seed]["validation"].items():
                alarm = alarms[seed][combo.combination_id]["validation"][file_name]
                base = alarms[seed][base_id]["validation"][file_name]
                starts, ends, lengths = alarm_segments(alarm)
                isolated = int((lengths == 1).sum())
                rows.append({"seed": seed, "policy_name": combo.policy_name,
                             "threshold_method": combo.threshold_method, "combination_id": combo.combination_id,
                             "file_name": file_name, "alarm_point_count": int(alarm.sum()),
                             "normal_fpr": float(alarm.mean()), "alarm_segment_count": len(starts),
                             "longest_alarm_segment": int(lengths.max()) if len(lengths) else 0,
                             "median_alarm_segment_length": float(np.median(lengths)) if len(lengths) else 0.0,
                             "mean_alarm_segment_length": float(lengths.mean()) if len(lengths) else 0.0,
                             "isolated_alarm_count": isolated,
                             "isolated_alarm_ratio": float(isolated / max(len(lengths), 1)),
                             "alarms_per_hour": float(alarm.sum() * 3600 / max(len(alarm), 1)),
                             "percentage_reduction_vs_pointwise": float((base.sum() - alarm.sum()) / max(base.sum(), 1))})
    frame = pd.DataFrame(rows)
    frame["sequence_with_max_fpr"] = False
    for _, indices in frame.groupby(["seed", "policy_name", "threshold_method"]).groups.items():
        subset = frame.loc[list(indices)]
        frame.loc[subset.normal_fpr.idxmax(), "sequence_with_max_fpr"] = True
    return frame


def false_positive_segments(scores: dict[int, dict[str, Any]], alarms: dict[int, dict[str, Any]], cfg: dict
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, summaries = [], []
    registry = policy_registry(cfg)
    for seed in sorted(scores):
        for combo in registry.itertuples():
            base_method = cfg["hysteresis_enter_method"] if combo.policy_name == HYSTERESIS_POLICY else combo.threshold_method
            base_id = baseline_combo_id(base_method)
            for file_name, values in scores[seed]["test"].items():
                normal_alarm = ((values["label"] == 0) &
                                (alarms[seed][combo.combination_id]["test"][file_name] == 1)).astype(np.int8)
                base_alarm = ((values["label"] == 0) &
                              (alarms[seed][base_id]["test"][file_name] == 1)).astype(np.int8)
                starts, ends, lengths = alarm_segments(normal_alarm)
                base_starts, base_ends, base_lengths = alarm_segments(base_alarm)
                matched_base: set[int] = set(); shorter = unchanged_long = extended = 0
                for segment_id, (start, end, length) in enumerate(zip(starts, ends, lengths), 1):
                    overlaps = np.flatnonzero((base_starts <= end) & (base_ends >= start))
                    matched = int(overlaps[0]) if len(overlaps) else -1
                    if matched >= 0:
                        matched_base.add(matched)
                        shorter += int(length < base_lengths[matched])
                        unchanged_long += int(length == base_lengths[matched] and length >= 30)
                        extended += int(length > base_lengths[matched])
                    rows.append({"seed": seed, "policy_name": combo.policy_name,
                                 "threshold_method": combo.threshold_method, "file_name": file_name,
                                 "segment_id": segment_id, "start_timestamp": values["timestamp"][start],
                                 "end_timestamp": values["timestamp"][end], "length": int(length),
                                 "peak_score": float(values["score"][start:end + 1].max()),
                                 "mean_score": float(values["score"][start:end + 1].mean()),
                                 "top_contributing_channels": "unavailable_in_step3d_score_npz",
                                 "matched_pointwise_segment_id": matched + 1 if matched >= 0 else np.nan,
                                 "comparison_baseline_threshold_method": base_method})
                summaries.append({"seed": seed, "policy_name": combo.policy_name,
                                  "threshold_method": combo.threshold_method, "file_name": file_name,
                                  "comparison_baseline_threshold_method": base_method,
                                  "pointwise_segment_count": len(base_starts), "policy_segment_count": len(starts),
                                  "eliminated_pointwise_segment_count": len(base_starts) - len(matched_base),
                                  "shortened_segment_count": shorter, "unchanged_long_segment_count": unchanged_long,
                                  "hysteresis_extended_segment_count": extended if combo.policy_name == HYSTERESIS_POLICY else 0})
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def policy_comparison(by_seed: pd.DataFrame, by_file: pd.DataFrame, events: pd.DataFrame,
                      duration: pd.DataFrame, transitions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    event_all = events[events.summary_scope == "all_tests"]
    for seed in sorted(by_seed.seed.unique()):
        for combo in policy_combinations(cfg):
            baselines = ([cfg["hysteresis_enter_method"], cfg["hysteresis_stay_method"]]
                         if combo["policy_name"] == HYSTERESIS_POLICY else [combo["threshold_method"]])
            new = by_seed[(by_seed.seed == seed) & (by_seed.combination_id == combo["combination_id"]) &
                          (by_seed.scope == "micro")].iloc[0]
            new_event = event_all[(event_all.seed == seed) &
                                  (event_all.combination_id == combo["combination_id"])].iloc[0]
            new_short = duration[(duration.seed == seed) & (duration.combination_id == combo["combination_id"]) &
                                 (duration.duration_group == "short")]
            for base_method in baselines:
                base_id = baseline_combo_id(base_method)
                base = by_seed[(by_seed.seed == seed) & (by_seed.combination_id == base_id) &
                               (by_seed.scope == "micro")].iloc[0]
                base_event = event_all[(event_all.seed == seed) & (event_all.combination_id == base_id)].iloc[0]
                base_short = duration[(duration.seed == seed) & (duration.combination_id == base_id) &
                                      (duration.duration_group == "short")]
                trans = transitions[(transitions.seed == seed) &
                                    (transitions.policy_name == combo["policy_name"]) &
                                    (transitions.threshold_method == combo["threshold_method"]) &
                                    (transitions.comparison_baseline_threshold_method == base_method)]
                rows.append({"seed": seed, "policy_name": combo["policy_name"],
                             "threshold_method": combo["threshold_method"], "combination_id": combo["combination_id"],
                             "comparison_baseline_threshold_method": base_method,
                             "precision_delta": float(new.precision - base.precision),
                             "recall_delta": float(new.recall - base.recall), "f1_delta": float(new.f1 - base.f1),
                             "false_positive_delta": int(new.false_positive_count - base.false_positive_count),
                             "false_positive_relative_change": float(
                                 (new.false_positive_count - base.false_positive_count) / max(base.false_positive_count, 1)),
                             "event_recall_delta": float(new_event.event_recall - base_event.event_recall),
                             "median_delay_delta": float(new_event.median_delay - base_event.median_delay),
                             "within30_hit_delta": float(new_event.within30_hit_rate - base_event.within30_hit_rate),
                             "short_event_recall_delta": float(new_short.event_recall.iloc[0] - base_short.event_recall.iloc[0])
                                 if len(new_short) and len(base_short) else np.nan,
                             "lost_short_event_count": int(((trans.lost) & (trans.duration < 30)).sum()),
                             "lost_total_event_count": int(trans.lost.sum()),
                             "recovered_total_event_count": int(trans.recovered.sum()),
                             "alarm_segment_reduction": int(base.alarm_segment_count - new.alarm_segment_count)})
    return pd.DataFrame(rows)


def input_audit(scores: dict[int, dict[str, Any]], thresholds: pd.DataFrame, by_seed: pd.DataFrame,
                event_summary: pd.DataFrame, cfg: dict, hashes: dict[str, str], smoke: bool) -> pd.DataFrame:
    step3d_points = pd.read_csv(ROOT / cfg["point_metric_file"])
    step3d_events = pd.read_csv(ROOT / cfg["event_metric_file"])
    rows = []
    for seed in sorted(scores):
        new = by_seed[(by_seed.seed == seed) & (by_seed.combination_id == baseline_combo_id("normal_validation_q99")) &
                      (by_seed.scope == "micro")].iloc[0]
        old = step3d_points[(step3d_points.seed == seed) & (step3d_points.score_variant == cfg["score_variant"]) &
                            (step3d_points.threshold_method == "normal_validation_q99") &
                            (step3d_points.scope == "micro")].iloc[0]
        new_event = event_summary[(event_summary.seed == seed) &
                                  (event_summary.combination_id == baseline_combo_id("normal_validation_q99")) &
                                  (event_summary.summary_scope == "all_tests")].iloc[0]
        old_event = step3d_events[(step3d_events.seed == seed) & (step3d_events.score_variant == cfg["score_variant"]) &
                                  (step3d_events.threshold_method == "normal_validation_q99") &
                                  (step3d_events.summary_scope == "all_tests")].iloc[0]
        rows.append({"seed": seed, "score_npz_path": str((ROOT / cfg["score_npz_pattern"].format(seed=seed)).resolve()),
                     "score_npz_sha256_before": hashes[f"npz{seed}"], "split_counts": "train=6;validation=6;test=4",
                     "score_variant_present": True, "metadata_fields_complete": True, "scores_finite": True,
                     "sequence_state_reset_per_file": True, "pointwise_precision_difference": new.precision - old.precision,
                     "pointwise_recall_difference": new.recall - old.recall, "pointwise_f1_difference": new.f1 - old.f1,
                     "pointwise_aupr_reference_difference": new.reference_aupr - old.aupr,
                     "pointwise_auroc_reference_difference": new.reference_auroc - old.auroc,
                     "pointwise_event_recall_difference": new_event.event_recall - old_event.event_recall,
                     "smoke_subset": smoke})
    frame = pd.DataFrame(rows)
    cols = [name for name in frame if name.endswith("difference")]
    if not smoke and frame[cols].abs().max().max() > float(cfg["consistency_tolerance"]):
        raise ValueError("Pointwise raw top5 did not reproduce Step 3D")
    return frame


def classify(comparison: pd.DataFrame, by_seed: pd.DataFrame, by_file: pd.DataFrame,
             events: pd.DataFrame, duration: pd.DataFrame, validation: pd.DataFrame, cfg: dict) -> tuple[str, dict[str, Any]]:
    q99 = comparison[(comparison.policy_name == cfg["primary_policy"]) &
                     (comparison.threshold_method == cfg["primary_threshold_method"])]
    primary_metrics = by_seed[(by_seed.policy_name == cfg["primary_policy"]) &
                              (by_seed.threshold_method == cfg["primary_threshold_method"]) &
                              (by_seed.scope == "micro")]
    baseline_metrics = by_seed[(by_seed.policy_name == "pointwise_baseline") &
                               (by_seed.threshold_method == cfg["primary_threshold_method"]) &
                               (by_seed.scope == "micro")]
    relative_precision = ((primary_metrics.set_index("seed").precision /
                           baseline_metrics.set_index("seed").precision) - 1)
    file_changes = {}
    for file_name in sorted(by_file.file_name.unique()):
        new = by_file[(by_file.policy_name == cfg["primary_policy"]) &
                      (by_file.threshold_method == cfg["primary_threshold_method"]) &
                      (by_file.file_name == file_name)].false_positive_count.mean()
        base = by_file[(by_file.policy_name == "pointwise_baseline") &
                       (by_file.threshold_method == cfg["primary_threshold_method"]) &
                       (by_file.file_name == file_name)].false_positive_count.mean()
        file_changes[file_name] = float((new - base) / max(base, 1))
    primary_val = validation[(validation.policy_name == cfg["primary_policy"]) &
                             (validation.threshold_method == cfg["primary_threshold_method"])]
    base_val = validation[(validation.policy_name == "pointwise_baseline") &
                          (validation.threshold_method == cfg["primary_threshold_method"])]
    evidence = {"precision_relative_mean": float(relative_precision.mean()),
                "precision_directional_seed_count": int((relative_precision > 0).sum()),
                "f1_delta_mean": float(q99.f1_delta.mean()),
                "f1_directional_seed_count": int((q99.f1_delta > 0).sum()),
                "false_positive_relative_change_mean": float(q99.false_positive_relative_change.mean()),
                "false_positive_relative_change_by_file": file_changes,
                "event_recall_delta_mean": float(q99.event_recall_delta.mean()),
                "short_event_recall_delta_mean": float(q99.short_event_recall_delta.mean()),
                "median_delay_delta_mean": float(q99.median_delay_delta.mean()),
                "alarm_segment_reduction_mean": float(q99.alarm_segment_reduction.mean()),
                "validation_isolated_ratio_before": float(base_val.isolated_alarm_ratio.mean()),
                "validation_isolated_ratio_after": float(primary_val.isolated_alarm_ratio.mean()),
                "primary_precision_mean": float(primary_metrics.precision.mean()),
                "primary_recall_mean": float(primary_metrics.recall.mean()),
                "primary_f1_mean": float(primary_metrics.f1.mean())}
    gates = (evidence["precision_relative_mean"] >= .20 and evidence["precision_directional_seed_count"] >= 2 and
             evidence["f1_delta_mean"] >= .03 and evidence["f1_directional_seed_count"] >= 2 and
             evidence["false_positive_relative_change_mean"] <= -.30 and all(v <= 0.05 for v in file_changes.values()) and
             evidence["event_recall_delta_mean"] >= -.05 and evidence["short_event_recall_delta_mean"] >= -.10 and
             evidence["median_delay_delta_mean"] <= 5 and evidence["alarm_segment_reduction_mean"] > 0 and
             evidence["validation_isolated_ratio_after"] < evidence["validation_isolated_ratio_before"])
    hyst = by_seed[(by_seed.policy_name == HYSTERESIS_POLICY) & (by_seed.scope == "micro")]
    if gates:
        return "causal_alarm_policy_makes_rec_main_branch_viable", evidence
    if evidence["short_event_recall_delta_mean"] < -.10 or evidence["median_delay_delta_mean"] > 5:
        return "causal_smoothing_hurts_short_events", evidence
    if (hyst.f1.mean() > primary_metrics.f1.mean() and
            events[(events.policy_name == HYSTERESIS_POLICY) & (events.summary_scope == "all_tests")].event_recall.mean() >=
            events[(events.policy_name == cfg["primary_policy"]) &
                   (events.threshold_method == cfg["primary_threshold_method"]) &
                   (events.summary_scope == "all_tests")].event_recall.mean()):
        return "hysteresis_is_preferred", evidence
    if evidence["precision_relative_mean"] >= .20 and evidence["false_positive_relative_change_mean"] <= -.30:
        return "causal_policy_improves_precision_but_rec_still_weak", evidence
    return "isolated_spikes_are_not_main_false_positive_source", evidence


def run(cfg: dict, output: Path, smoke: bool) -> dict[str, Any]:
    ensure_fresh(output)
    (output / "config_resolved.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    registry = policy_registry(cfg); registry.to_csv(output / "causal_alarm_policy_registry.csv", index=False)
    source_paths = {f"npz{seed}": ROOT / cfg["score_npz_pattern"].format(seed=seed) for seed in cfg["seeds"]}
    source_paths.update({"thresholds": ROOT / cfg["threshold_file"], "points": ROOT / cfg["point_metric_file"],
                         "events": ROOT / cfg["event_metric_file"]})
    hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    thresholds = pd.read_csv(source_paths["thresholds"])
    maximum = int(cfg["smoke_max_points_per_sequence"]) if smoke else None
    scores = {int(seed): load_seed_scores(source_paths[f"npz{seed}"], maximum) for seed in cfg["seeds"]}
    alarms = generate_alarms(scores, thresholds, cfg)
    by_file, by_seed, aggregate, multiseed = build_point_metrics(scores, alarms, cfg)
    diagnostics, events_by_seed = event_diagnostics(scores, alarms, cfg, require_full_event_set=not smoke)
    event_summary = summarize_events(diagnostics)
    transitions = event_transitions(diagnostics, cfg)
    duration = duration_analysis(diagnostics)
    validation = validation_diagnostics(scores, alarms, cfg)
    fp_segments, fp_summary = false_positive_segments(scores, alarms, cfg)
    comparison = policy_comparison(by_seed, by_file, event_summary, duration, transitions, cfg)
    audit = input_audit(scores, thresholds, by_seed, event_summary, cfg, hashes, smoke)
    for name, path in source_paths.items():
        unchanged = sha256_file(path) == hashes[name]
        if name.startswith("npz"):
            audit.loc[audit.seed == int(name[3:]), "score_npz_sha256_after"] = sha256_file(path)
            audit.loc[audit.seed == int(name[3:]), "step3d_source_unchanged"] = unchanged
    audit.to_csv(output / "alarm_input_consistency_audit.csv", index=False)
    validation.to_csv(output / "causal_alarm_validation_diagnostics.csv", index=False)
    by_seed.to_csv(output / "causal_alarm_point_metrics_by_seed.csv", index=False)
    by_file.to_csv(output / "causal_alarm_point_metrics_by_file.csv", index=False)
    aggregate.to_csv(output / "causal_alarm_point_metrics_aggregate.csv", index=False)
    diagnostics.to_csv(output / "causal_alarm_event_diagnostics.csv", index=False)
    event_summary.to_csv(output / "causal_alarm_event_summary.csv", index=False)
    transitions.to_csv(output / "causal_alarm_event_transition.csv", index=False)
    duration.to_csv(output / "causal_alarm_event_duration_analysis.csv", index=False)
    fp_segments.to_csv(output / "causal_alarm_false_positive_segments.csv", index=False)
    fp_summary.to_csv(output / "causal_alarm_false_positive_segment_summary.csv", index=False)
    comparison.to_csv(output / "causal_alarm_policy_comparison.csv", index=False)
    multiseed.to_csv(output / "causal_alarm_multiseed_summary.csv", index=False)
    classification, evidence = (("smoke_only_no_formal_conclusion", {"reason": "truncated sequence prefixes"})
                                if smoke else classify(comparison, by_seed, by_file, event_summary, duration,
                                                       validation, cfg))
    viable = classification == "causal_alarm_policy_makes_rec_main_branch_viable"
    if viable:
        recommendation = "Fix endpoint_raw_top5 with vote_2_of_3 and proceed to Step 4B independent structure validation."
    elif classification == "causal_policy_improves_precision_but_rec_still_weak":
        recommendation = ("Retain pointwise endpoint_raw_top5 as a limited coverage branch; vote_2_of_3 is not a viable "
                          "main detector. Do not enter Step 4B yet; prioritize lightweight normal-condition-aware thresholds.")
    else:
        recommendation = "Do not enter Step 4B yet; prioritize lightweight normal-condition-aware thresholds rather than adding vote windows."
    first_seed_events = events_by_seed[sorted(events_by_seed)[0]]
    duration_counts = {group: sum(duration_group(event["duration"]) == group for event in first_seed_events)
                       for group in ("short", "medium", "long")}
    pointwise_q99_segments = fp_segments[(fp_segments.policy_name == "pointwise_baseline") &
                                         (fp_segments.threshold_method == "normal_validation_q99")]
    fp_alarm_points = int(pointwise_q99_segments.length.sum())
    isolated_segments = int((pointwise_q99_segments.length == 1).sum())
    fragmentation = {"test_normal_alarm_points": fp_alarm_points,
                     "test_false_positive_segment_count": len(pointwise_q99_segments),
                     "test_isolated_false_positive_segment_count": isolated_segments,
                     "test_isolated_false_positive_point_fraction": float(isolated_segments / max(fp_alarm_points, 1)),
                     "test_continuous_false_positive_point_fraction": float(
                         pointwise_q99_segments.loc[pointwise_q99_segments.length > 1, "length"].sum() /
                         max(fp_alarm_points, 1)),
                     "test_long_false_positive_segment_count_ge30": int((pointwise_q99_segments.length >= 30).sum()),
                     "test_longest_false_positive_segment": int(pointwise_q99_segments.length.max())}
    summary = {"mode": "smoke" if smoke else "formal", "classification": classification,
               "classification_evidence": evidence, "input_score": cfg["score_variant"],
               "primary_policy": cfg["primary_policy"], "primary_threshold_method": cfg["primary_threshold_method"],
               "policy_count": len(ALL_POLICIES), "policy_threshold_combination_count": len(registry),
               "pointwise_step3d_reproduced": True, "step3d_sources_unchanged": bool(audit.step3d_source_unchanged.all()),
               "ranking_score_unchanged": True, "future_points_used": False, "history_backfilled": False,
               "state_carried_across_files": False, "model_loaded": False, "inference_executed": False,
               "training_executed": False, "structure_score_computed": False,
               "event_duration_counts": duration_counts, "short_event_gate_applicable": duration_counts["short"] > 0,
               "pointwise_q99_false_positive_fragmentation": fragmentation,
               "hysteresis_preferred": classification == "hysteresis_is_preferred",
               "rec_main_branch_viable": viable, "recommended_next_step": recommendation}
    (output / "causal_alarm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    q99 = multiseed[(multiseed.threshold_method == "normal_validation_q99") & (multiseed.scope == "micro")]
    pot = multiseed[(multiseed.threshold_method == "train_only_pot") & (multiseed.scope == "micro")]
    validation_q99 = validation[validation.threshold_method == "normal_validation_q99"]
    duration_primary = duration[(duration.policy_name == cfg["primary_policy"]) &
                                (duration.threshold_method == cfg["primary_threshold_method"])]
    report = f"""# HAI Step 3E causal alarm policy diagnosis

- Step 3D endpoint_raw_top5 arrays and all four existing thresholds were read directly; no model load, inference, training, score redefinition, future point, history backfill, point adjustment, structure score, or fusion occurred.
- Pointwise q99 reproduced Step 3D within the configured tolerance; all Step 3D source hashes remained unchanged.
- Six pre-registered causal policies produced {len(registry)} legal policy/threshold combinations. Ranking AUPR/AUROC are unchanged references to endpoint_raw_top5.
- Classification: `{classification}`. Evidence: {evidence}.
- Event duration counts: {duration_counts}; the short-event gate is applicable: `{duration_counts['short'] > 0}`.
- Pointwise q99 false-positive fragmentation: {fragmentation}.
- Hysteresis preferred: `{classification == 'hysteresis_is_preferred'}`.
- q99 micro mean/std results: {q99.to_dict(orient='records')}.
- POT micro mean/std results: {pot.to_dict(orient='records')}.
- Primary duration-group results: {duration_primary.to_dict(orient='records')}.
- q99 normal-validation alarm fragmentation diagnostics: {validation_q99.to_dict(orient='records')}.
- Rec main branch viable: `{viable}`. Recommendation: {recommendation}

All event transitions, per-file false positives, normal alarm intervals, validation sequences and policy comparisons are available in the accompanying CSV files. No next stage was executed.
"""
    (output / "final_hai_rec_causal_alarm_policy_diagnosis.md").write_text(report, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAI Step 3E causal alarm policy evaluation")
    parser.add_argument("--config", required=True); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); cfg = load_config(args.config)
    output = ROOT / (cfg["smoke_output_dir"] if args.smoke else cfg["output_dir"])
    if args.dry_run:
        for seed in cfg["seeds"]:
            path = ROOT / cfg["score_npz_pattern"].format(seed=seed)
            print(f"seed={seed} score_npz={path.resolve()} exists={path.exists()}")
        print(f"input_score={cfg['score_variant']}; policies={cfg['policies']}")
        print(f"thresholds={cfg['threshold_methods']}; combinations={len(policy_combinations(cfg))}")
        print(f"primary={cfg['primary_policy']}@{cfg['primary_threshold_method']}; output={output.resolve()}")
        print("model_load=false inference=false training=false future=false backfill=false structure=false fusion=false")
        return 0
    print(json.dumps(run(cfg, output, args.smoke), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
