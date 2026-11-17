from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hai import HAISequence, MultiSequenceWindowDataset, build_hai_dataloaders  # noqa: E402
from scripts.hai_step3c_evaluate_rec_score_alignment import make_model, normalized_config, sha256_file  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.metrics import binary_metrics  # noqa: E402
from utils.pot import POTThreshold  # noqa: E402
from utils.reconstruction_scores import (  # noqa: E402
    CHANNEL_SCORE_VARIANTS,
    aggregate_endpoint_channel_scores,
    calibrate_endpoint_errors,
    endpoint_channel_errors,
    fit_channel_calibration,
)


THRESHOLD_METHODS = (
    "train_only_pot", "normal_validation_q99", "normal_validation_q995", "validation_sequence_max_q99",
)
OUTPUT_FILES = (
    "config_resolved.json", "checkpoint_audit.csv", "endpoint_step3c_consistency.csv",
    "rec_channel_score_registry.csv",
    "rec_channel_calibration_stats.csv", "rec_channel_thresholds.csv", "rec_channel_validation_stability.csv",
    "rec_channel_point_metrics_by_seed.csv", "rec_channel_point_metrics_by_file.csv",
    "rec_channel_point_metrics_aggregate.csv", "rec_channel_event_diagnostics.csv",
    "rec_channel_event_summary.csv", "rec_channel_contribution_samples.csv",
    "rec_channel_contribution_summary.csv", "rec_channel_baseline_comparison.csv",
    "rec_channel_event_transition.csv", "rec_channel_multiseed_summary.csv", "rec_channel_summary.json",
    "final_hai_rec_channel_calibration_diagnosis.md",
)


def ensure_fresh(output: Path, seeds: list[int]) -> None:
    expected = list(OUTPUT_FILES) + [f"rec_channel_scores_seed{seed}.npz" for seed in seeds]
    existing = [name for name in expected if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite HAI Step 3D outputs: {existing}")
    output.mkdir(parents=True, exist_ok=True)


def score_registry() -> pd.DataFrame:
    rows = [
        ("endpoint_raw_mean", "mean(raw endpoint error over 86 channels)", False, "historical_endpoint_baseline"),
        ("endpoint_raw_top5", "mean(largest 5 raw endpoint channel errors)", False, "dilution_diagnostic"),
        ("endpoint_calibrated_mean", "mean(calibrated endpoint error over 86 channels)", True, "calibration_ablation"),
        ("endpoint_calibrated_top3", "mean(largest 3 calibrated endpoint errors)", True, "sparse_diagnostic"),
        ("endpoint_calibrated_top5", "mean(largest 5 calibrated endpoint errors)", True, "primary_precision_candidate"),
        ("endpoint_calibrated_top10", "mean(largest 10 calibrated endpoint errors)", True, "sparse_diagnostic"),
        ("endpoint_calibrated_max", "maximum calibrated endpoint channel error", True, "single_channel_sensitivity"),
    ]
    return pd.DataFrame([
        {"score_variant": name, "definition": definition, "channel_calibrated": calibrated,
         "endpoint_aligned": True, "causal": True, "uses_label": False, "role": role}
        for name, definition, calibrated, role in rows
    ])


def audit_checkpoints(cfg: dict) -> tuple[pd.DataFrame, dict[int, dict], dict[int, str]]:
    rows, checkpoints, hashes, normalized = [], {}, {}, []
    for seed in cfg["seeds"]:
        path = ROOT / cfg["checkpoint_pattern"].format(seed=seed)
        if not path.exists():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoints[int(seed)] = checkpoint
        hashes[int(seed)] = digest
        normalized.append(normalized_config(checkpoint["cfg"]))
        rows.append({"seed": seed, "checkpoint_path": str(path.resolve()), "exists": True,
                     "checkpoint_sha256_before": digest, "checkpoint_seed": checkpoint["seed"],
                     "feature_count": len(checkpoint["feature_names"]), "best_epoch": checkpoint["best_epoch"],
                     "model_input_shape": "[batch,64,86]", "model_output_shape": "[batch,64,86]",
                     "training_executed": False})
    if not all(item == normalized[0] for item in normalized[1:]):
        raise ValueError("Checkpoint configurations differ beyond seed/output fields")
    for row in rows:
        row["only_seed_and_output_fields_differ"] = True
    return pd.DataFrame(rows), checkpoints, hashes


@torch.no_grad()
def infer_endpoint_errors(model: torch.nn.Module, sequence: HAISequence, device: torch.device, window: int,
                          batch_size: int, maximum_windows: int | None = None) -> dict[str, np.ndarray]:
    dataset: Any = MultiSequenceWindowDataset([sequence], window_size=window, stride=1, return_metadata=True)
    if maximum_windows is not None:
        dataset = Subset(dataset, np.arange(min(maximum_windows, len(dataset))).tolist())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    errors, raw_means, labels, timestamps, end_rows, sequence_ids = [], [], [], [], [], []
    model.eval()
    for x, y, metadata in loader:
        x = x.to(device, non_blocking=True)
        raw = endpoint_channel_errors(x, model(x))
        errors.append(raw.cpu().numpy().astype(np.float32))
        # Match Step 3C's torch channel reduction order exactly.
        raw_mean = raw.mean(dim=1)
        raw_means.append(raw_mean.cpu().numpy().astype(np.float32))
        labels.append(y.numpy())
        timestamps.extend(str(value) for value in metadata["timestamp"])
        end_rows.extend(int(value) for value in metadata["end_row"].numpy())
        sequence_ids.extend(str(value) for value in metadata["sequence_id"])
    result = {
        "endpoint_error": np.concatenate(errors), "endpoint_raw_mean_exact": np.concatenate(raw_means),
        "label": np.concatenate(labels).astype(np.int64),
        "timestamp": np.asarray(timestamps, dtype=str), "end_row": np.asarray(end_rows, dtype=np.int64),
        "sequence_id": np.asarray(sequence_ids, dtype=str),
    }
    if not np.isfinite(result["endpoint_error"]).all():
        raise ValueError(f"Non-finite endpoint error in {sequence.file_name}/{sequence.split}")
    return result


def attach_scores(values: dict[str, np.ndarray], calibration: dict[str, np.ndarray], keep_details: bool) -> dict[str, np.ndarray]:
    calibrated = calibrate_endpoint_errors(values["endpoint_error"], calibration)
    scores, details = aggregate_endpoint_channel_scores(values["endpoint_error"], calibrated)
    result = {key: value for key, value in values.items() if key not in {"endpoint_error", "endpoint_raw_mean_exact"}}
    result.update(scores)
    result["endpoint_raw_mean"] = values["endpoint_raw_mean_exact"]
    if keep_details:
        result.update(details)
    if not all(np.isfinite(result[name]).all() for name in CHANNEL_SCORE_VARIANTS):
        raise ValueError("A formal channel score contains NaN/Inf")
    return result


def concat_values(values_by_sequence: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = (*CHANNEL_SCORE_VARIANTS, "label", "timestamp", "end_row", "sequence_id")
    return {key: np.concatenate([values[key] for values in values_by_sequence.values()]) for key in keys}


def fit_score_thresholds(train: np.ndarray, validation_by_sequence: dict[str, np.ndarray], cfg: dict) -> list[dict[str, Any]]:
    validation = np.concatenate(list(validation_by_sequence.values()))
    q = float(cfg["thresholds"]["pot_q"])
    target = float(cfg["thresholds"]["pot_target_prob"])
    rows: list[dict[str, Any]] = []
    try:
        pot = POTThreshold(q=q, target_prob=target).fit(train)
        if pot.tau is None or not np.isfinite(pot.tau):
            raise RuntimeError("POT returned a non-finite threshold")
        rows.append({"threshold_method": "train_only_pot", "threshold_value": float(pot.tau), "status": "ok",
                     "fitting_sample_count": len(train), "fitting_scope": "corresponding score normal train only",
                     "pot_q": q, "pot_target_prob": target, "pot_u": float(pot.u), "failure_reason": ""})
    except Exception as exc:
        rows.append({"threshold_method": "train_only_pot", "threshold_value": np.nan, "status": "failed",
                     "fitting_sample_count": len(train), "fitting_scope": "corresponding score normal train only",
                     "pot_q": q, "pot_target_prob": target, "pot_u": np.nan,
                     "failure_reason": f"{type(exc).__name__}: {exc}"})
    for method, quantile in (("normal_validation_q99", float(cfg["thresholds"]["validation_q99"])),
                             ("normal_validation_q995", float(cfg["thresholds"]["validation_q995"]))):
        rows.append({"threshold_method": method, "threshold_value": float(np.quantile(validation, quantile)),
                     "status": "ok", "fitting_sample_count": len(validation),
                     "fitting_scope": "corresponding score normal validation only", "validation_quantile": quantile,
                     "failure_reason": ""})
    sequence_quantile = float(cfg["thresholds"]["sequence_max_quantile"])
    sequence_q99 = {name: float(np.quantile(values, sequence_quantile))
                    for name, values in validation_by_sequence.items()}
    rows.append({"threshold_method": "validation_sequence_max_q99",
                 "threshold_value": max(sequence_q99.values()), "status": "ok",
                 "fitting_sample_count": len(validation),
                 "fitting_scope": "maximum of six normal-validation sequence q99 values",
                 "validation_quantile": sequence_quantile,
                 "sequence_quantiles_json": json.dumps(sequence_q99, sort_keys=True), "failure_reason": ""})
    return rows


def threshold_lookup(frame: pd.DataFrame, seed: int, variant: str, method: str) -> float:
    row = frame[(frame.seed == seed) & (frame.score_variant == variant) & (frame.threshold_method == method)]
    if len(row) != 1 or row.status.iloc[0] != "ok":
        raise ValueError(f"Unavailable threshold {seed}/{variant}/{method}")
    return float(row.threshold_value.iloc[0])


def metric_record(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (scores > threshold).astype(np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = binary_metrics(labels, prediction, scores)
    normal = labels == 0
    tp = int(((labels == 1) & (prediction == 1)).sum())
    fp = int((normal & (prediction == 1)).sum())
    return {**metrics, "anomaly_point_count": int((labels == 1).sum()),
            "normal_point_count": int(normal.sum()), "alarm_count": int(prediction.sum()),
            "true_positive_count": tp, "false_positive_count": fp,
            "false_negative_count": int(((labels == 1) & (prediction == 0)).sum()),
            "normal_false_positive_rate": float(fp / max(normal.sum(), 1)),
            "alarms_per_normal_hour": float(fp * 3600 / max(normal.sum(), 1))}


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(array[np.isfinite(array)].mean()) if np.isfinite(array).any() else np.nan


def build_point_metrics(all_values: dict[int, dict[str, Any]], thresholds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_file, by_seed = [], []
    metrics = ["precision", "recall", "f1", "aupr", "auroc"]
    for seed, splits in all_values.items():
        for variant in CHANNEL_SCORE_VARIANTS:
            for method in THRESHOLD_METHODS:
                tau = threshold_lookup(thresholds, seed, variant, method)
                rows = []
                for file_name, values in splits["test"].items():
                    row = {"seed": seed, "score_variant": variant, "threshold_method": method,
                           "file_name": file_name, **metric_record(values["label"], values[variant], tau)}
                    by_file.append(row); rows.append(row)
                labels = np.concatenate([values["label"] for values in splits["test"].values()])
                scores = np.concatenate([values[variant] for values in splits["test"].values()])
                micro = metric_record(labels, scores, tau)
                by_seed.append({"seed": seed, "score_variant": variant, "threshold_method": method,
                                "scope": "micro", **micro})
                by_seed.append({"seed": seed, "score_variant": variant, "threshold_method": method,
                                "scope": "macro", **{name: finite_mean([r[name] for r in rows]) for name in metrics},
                                **{name: int(sum(r[name] for r in rows)) for name in
                                   ("anomaly_point_count", "normal_point_count", "alarm_count", "true_positive_count",
                                    "false_positive_count", "false_negative_count")},
                                "normal_false_positive_rate": float(np.mean([r["normal_false_positive_rate"] for r in rows])),
                                "alarms_per_normal_hour": float(np.mean([r["alarms_per_normal_hour"] for r in rows]))})
    by_file_frame, by_seed_frame = pd.DataFrame(by_file), pd.DataFrame(by_seed)
    aggregate = []
    for key, group in by_seed_frame.groupby(["score_variant", "threshold_method", "scope"]):
        for metric in metrics:
            values = group.set_index("seed")[metric]
            aggregate.append({"score_variant": key[0], "threshold_method": key[1], "scope": key[2], "metric": metric,
                              **{f"seed{seed}": float(values.get(seed, np.nan)) for seed in (1, 2, 3)},
                              "mean": float(values.mean()), "std": float(values.std(ddof=0)),
                              "min": float(values.min()), "max": float(values.max())})
    return by_file_frame, by_seed_frame, pd.DataFrame(aggregate)


def build_event_diagnostics(all_values: dict[int, dict[str, Any]], thresholds: pd.DataFrame,
                            events: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for seed, splits in all_values.items():
        for variant in CHANNEL_SCORE_VARIANTS:
            for method in THRESHOLD_METHODS:
                tau = threshold_lookup(thresholds, seed, variant, method)
                for event in events:
                    values = splits["test"][event["file_name"]]
                    mask = ((values["end_row"] >= event["start_row"]) & (values["end_row"] <= event["end_row"]))
                    score, timestamps = values[variant][mask], values["timestamp"][mask]
                    if not len(score):
                        rows.append({"seed": seed, "score_variant": variant, "threshold_method": method,
                                     "sequence_id": event["sequence_id"], "file_name": event["file_name"],
                                     "event_id": event["global_event_id"], "start_timestamp": event["start_timestamp"],
                                     "end_timestamp": event["end_timestamp"], "duration": event["duration_points"],
                                     "covered_by_evaluation_subset": False, "detected": False,
                                     "first_alarm_timestamp": None, "detection_delay_points": np.nan,
                                     "point_recall_in_event": np.nan, "peak_score": np.nan, "mean_score": np.nan,
                                     "first_point_hit": False, "within5_hit": False, "within10_hit": False,
                                     "within30_hit": False})
                        continue
                    alarms = np.flatnonzero(score > tau)
                    detected = len(alarms) > 0
                    delay = int(alarms[0]) if detected else np.nan
                    rows.append({"seed": seed, "score_variant": variant, "threshold_method": method,
                                 "sequence_id": event["sequence_id"], "file_name": event["file_name"],
                                 "event_id": event["global_event_id"], "start_timestamp": event["start_timestamp"],
                                 "end_timestamp": event["end_timestamp"], "duration": event["duration_points"],
                                 "detected": detected,
                                 "first_alarm_timestamp": str(timestamps[alarms[0]]) if detected else None,
                                 "covered_by_evaluation_subset": True,
                                 "detection_delay_points": delay, "point_recall_in_event": float((score > tau).mean()),
                                 "peak_score": float(score.max()), "mean_score": float(score.mean()),
                                 "first_point_hit": bool(detected and alarms[0] == 0),
                                 "within5_hit": bool(detected and alarms[0] < 5),
                                 "within10_hit": bool(detected and alarms[0] < 10),
                                 "within30_hit": bool(detected and alarms[0] < 30)})
    return pd.DataFrame(rows)


def summarize_events(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = diagnostics.groupby(["seed", "score_variant", "threshold_method"])
    for (seed, variant, method), group in groups:
        scopes = [("all_tests", group)] + [(name, part) for name, part in group.groupby("file_name")]
        for scope, part in scopes:
            evaluated = part[part.covered_by_evaluation_subset] if "covered_by_evaluation_subset" in part else part
            detected = evaluated[evaluated.detected]
            rows.append({"seed": seed, "score_variant": variant, "threshold_method": method,
                         "summary_scope": scope, "total_event_count": len(part),
                         "covered_event_count": len(evaluated), "detected_event_count": int(evaluated.detected.sum()),
                         "event_recall": float(evaluated.detected.mean()) if len(evaluated) else np.nan,
                         "fully_missed_event_count": int((~evaluated.detected).sum()),
                         "mean_delay": float(detected.detection_delay_points.mean()) if len(detected) else np.nan,
                         "median_delay": float(detected.detection_delay_points.median()) if len(detected) else np.nan,
                         "first_point_hit_rate": float(evaluated.first_point_hit.mean()) if len(evaluated) else np.nan,
                         "within5_hit_rate": float(evaluated.within5_hit.mean()) if len(evaluated) else np.nan,
                         "within10_hit_rate": float(evaluated.within10_hit.mean()) if len(evaluated) else np.nan,
                         "within30_hit_rate": float(evaluated.within30_hit.mean()) if len(evaluated) else np.nan})
    return pd.DataFrame(rows)


def validation_stability(all_values: dict[int, dict[str, Any]], thresholds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, splits in all_values.items():
        for variant in CHANNEL_SCORE_VARIANTS:
            taus = {method: threshold_lookup(thresholds, seed, variant, method) for method in THRESHOLD_METHODS}
            for file_name, values in splits["validation"].items():
                score = values[variant]
                row = {"seed": seed, "score_variant": variant, "file_name": file_name,
                       "score_mean": float(score.mean()), "score_std": float(score.std()),
                       "median": float(np.median(score)), "q90": float(np.quantile(score, .90)),
                       "q95": float(np.quantile(score, .95)), "q99": float(np.quantile(score, .99)),
                       "q995": float(np.quantile(score, .995)), "maximum": float(score.max())}
                for method, tau in taus.items():
                    alarm = score > tau
                    padded = np.r_[False, alarm, False].astype(np.int8)
                    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
                    row[f"{method}_exceedance_ratio"] = float(alarm.mean())
                    row[f"{method}_continuous_alarm_segment_count"] = len(starts)
                    row[f"{method}_longest_alarm_segment"] = int((ends - starts).max()) if len(starts) else 0
                    row[f"{method}_alarm_points_per_hour"] = float(alarm.sum() * 3600 / max(len(alarm), 1))
                rows.append(row)
    return pd.DataFrame(rows)


def calibration_frame(calibrations: dict[int, dict[str, np.ndarray]], feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for seed, calibration in calibrations.items():
        for index, name in enumerate(feature_names):
            rows.append({"seed": seed, "channel_index": index, "channel_name": name,
                         **{key: value[index].item() if hasattr(value[index], "item") else value[index]
                            for key, value in calibration.items()},
                         "fitting_scope": "normal train endpoint residuals only",
                         "validation_used": False, "test_used": False})
    return pd.DataFrame(rows)


def baseline_comparison(by_seed: pd.DataFrame, event_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    events = event_summary[event_summary.summary_scope == "all_tests"]
    for seed in sorted(by_seed.seed.unique()):
        for method in THRESHOLD_METHODS:
            base = by_seed[(by_seed.seed == seed) & (by_seed.score_variant == "endpoint_raw_mean") &
                           (by_seed.threshold_method == method) & (by_seed.scope == "micro")].iloc[0]
            base_event = events[(events.seed == seed) & (events.score_variant == "endpoint_raw_mean") &
                                (events.threshold_method == method)].iloc[0]
            for variant in CHANNEL_SCORE_VARIANTS[1:]:
                new = by_seed[(by_seed.seed == seed) & (by_seed.score_variant == variant) &
                              (by_seed.threshold_method == method) & (by_seed.scope == "micro")].iloc[0]
                new_event = events[(events.seed == seed) & (events.score_variant == variant) &
                                   (events.threshold_method == method)].iloc[0]
                rows.append({"seed": seed, "score_variant": variant, "threshold_method": method,
                             **{f"{metric}_delta": float(new[metric] - base[metric])
                                for metric in ("precision", "recall", "f1", "aupr", "auroc")},
                             "false_positive_delta": int(new.false_positive_count - base.false_positive_count),
                             "false_positive_relative_change": float(
                                 (new.false_positive_count - base.false_positive_count) / max(base.false_positive_count, 1)),
                             "event_recall_delta": float(new_event.event_recall - base_event.event_recall),
                             "median_delay_delta": float(new_event.median_delay - base_event.median_delay)})
    return pd.DataFrame(rows)


def event_transitions(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = diagnostics.set_index(["seed", "threshold_method", "event_id", "score_variant"])
    for seed in sorted(diagnostics.seed.unique()):
        for method in THRESHOLD_METHODS:
            for event_id in sorted(diagnostics.event_id.unique()):
                base = indexed.loc[(seed, method, event_id, "endpoint_raw_mean")]
                for variant in CHANNEL_SCORE_VARIANTS[1:]:
                    new = indexed.loc[(seed, method, event_id, variant)]
                    rows.append({"seed": seed, "threshold_method": method, "event_id": event_id,
                                 "file_name": base.file_name, "score_variant": variant,
                                 "baseline_detected": bool(base.detected), "new_detected": bool(new.detected),
                                 "recovered": bool(not base.detected and new.detected),
                                 "lost": bool(base.detected and not new.detected),
                                 "baseline_delay": base.detection_delay_points, "new_delay": new.detection_delay_points,
                                 "delay_change": new.detection_delay_points - base.detection_delay_points})
    return pd.DataFrame(rows)


def _format_top(indices: np.ndarray, values: np.ndarray, names: list[str]) -> tuple[str, str, str]:
    return (";".join(str(int(index)) for index in indices),
            ";".join(names[int(index)] for index in indices),
            ";".join(f"{float(value):.8g}" for value in values))


def contribution_outputs(all_values: dict[int, dict[str, Any]], thresholds: pd.DataFrame,
                         events: list[dict[str, Any]], feature_names: list[str], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows, summary_rows = [], []
    primary, baseline = cfg["primary_score"], cfg["baseline_score"]
    high_n = int(cfg["normal_high_score_points_per_file"])
    anomaly_n = int(cfg["anomaly_high_score_points_per_file"])
    for seed, splits in all_values.items():
        primary_tau = threshold_lookup(thresholds, seed, primary, "normal_validation_q99")
        baseline_tau = threshold_lookup(thresholds, seed, baseline, "normal_validation_q99")
        for file_name, values in splits["test"].items():
            selected: dict[int, set[str]] = {}
            def mark(indices: np.ndarray, reason: str) -> None:
                for index in indices:
                    selected.setdefault(int(index), set()).add(reason)
            normal = np.flatnonzero(values["label"] == 0)
            anomaly = np.flatnonzero(values["label"] == 1)
            mark(normal[np.argsort(values[primary][normal])[-min(high_n, len(normal)):]], "normal_primary_top")
            mark(anomaly[np.argsort(values[primary][anomaly])[-min(anomaly_n, len(anomaly)):]], "anomaly_primary_top")
            mark(np.flatnonzero((values["label"] == 0) & (values[baseline] > baseline_tau) &
                                (values[primary] <= primary_tau)), "normal_raw_alarm_primary_clear")
            mark(np.flatnonzero((values["label"] == 1) & (values[primary] > primary_tau) &
                                (values[baseline] <= baseline_tau)), "anomaly_primary_alarm_raw_clear")
            mark(np.flatnonzero((values["label"] == 1) & (values[primary] > primary_tau) &
                                (values[baseline] > baseline_tau)), "anomaly_both_alarm")
            for event in [item for item in events if item["file_name"] == file_name]:
                mask = np.flatnonzero((values["end_row"] >= event["start_row"]) &
                                      (values["end_row"] <= event["end_row"]))
                if len(mask):
                    peak = mask[int(np.argmax(values[primary][mask]))]
                    mark(np.asarray([peak]), f"event_peak_{event['global_event_id']}")
            for index, reasons in selected.items():
                raw_i, raw_n, raw_v = _format_top(values["raw_top5_indices"][index],
                                                   values["raw_top5_values"][index], feature_names)
                c3_i, c3_n, c3_v = _format_top(values["calibrated_top3_indices"][index],
                                                values["calibrated_top3_values"][index], feature_names)
                c5_i, c5_n, c5_v = _format_top(values["calibrated_top5_indices"][index],
                                                values["calibrated_top5_values"][index], feature_names)
                c10_i, c10_n, c10_v = _format_top(values["calibrated_top10_indices"][index],
                                                   values["calibrated_top10_values"][index], feature_names)
                sample_rows.append({"seed": seed, "file_name": file_name, "sequence_id": values["sequence_id"][index],
                                    "timestamp": values["timestamp"][index], "end_row": values["end_row"][index],
                                    "label": values["label"][index], "selection_reasons": ";".join(sorted(reasons)),
                                    "baseline_score": values[baseline][index], "primary_score": values[primary][index],
                                    "raw_top5_indices": raw_i, "raw_top5_names": raw_n, "raw_top5_values": raw_v,
                                    "calibrated_top3_indices": c3_i, "calibrated_top3_names": c3_n,
                                    "calibrated_top3_values": c3_v, "calibrated_top5_indices": c5_i,
                                    "calibrated_top5_names": c5_n, "calibrated_top5_values": c5_v,
                                    "calibrated_top10_indices": c10_i, "calibrated_top10_names": c10_n,
                                    "calibrated_top10_values": c10_v})
            normal_high = set(normal[np.argsort(values[primary][normal])[-min(high_n, len(normal)):]].tolist())
            anomaly_high = set(anomaly[np.argsort(values[primary][anomaly])[-min(anomaly_n, len(anomaly)):]].tolist())
            primary_alarm = values[primary] > primary_tau
            for channel, name in enumerate(feature_names):
                top3 = values["calibrated_top3_indices"] == channel
                top5 = values["calibrated_top5_indices"] == channel
                top10 = values["calibrated_top10_indices"] == channel
                fp = (values["label"] == 0) & primary_alarm
                tp = (values["label"] == 1) & primary_alarm
                fp_count = int(top5[fp].sum()); tp_count = int(top5[tp].sum())
                summary_rows.append({"seed": seed, "file_name": file_name, "channel_index": channel,
                                     "channel_name": name, "top3_count": int(top3.sum()),
                                     "top5_count": int(top5.sum()), "top10_count": int(top10.sum()),
                                     "normal_high_score_top5_count": int(top5[list(normal_high)].sum()) if normal_high else 0,
                                     "anomaly_high_score_top5_count": int(top5[list(anomaly_high)].sum()) if anomaly_high else 0,
                                     "false_positive_top5_count": fp_count, "true_positive_top5_count": tp_count,
                                     "fp_minus_tp_count": fp_count - tp_count,
                                     "dominance": "false_positive" if fp_count > tp_count else
                                                  ("true_positive" if tp_count > fp_count else "balanced")})
    return pd.DataFrame(sample_rows), pd.DataFrame(summary_rows)


def multiseed_summary(by_seed: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in by_seed.groupby(["score_variant", "threshold_method", "scope"]):
        row = {"score_variant": key[0], "threshold_method": key[1], "scope": key[2]}
        for metric in ("precision", "recall", "f1", "aupr", "auroc", "false_positive_count"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def classify(comparison: pd.DataFrame, by_file: pd.DataFrame, event_summary: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    primary = comparison[(comparison.score_variant == "endpoint_calibrated_top5") &
                         (comparison.threshold_method == "normal_validation_q99")]
    precision_relative = []
    for row in primary.itertuples():
        base_precision = by_file[(by_file.seed == row.seed) & (by_file.score_variant == "endpoint_raw_mean") &
                                 (by_file.threshold_method == "normal_validation_q99")]
        new_precision = by_file[(by_file.seed == row.seed) & (by_file.score_variant == "endpoint_calibrated_top5") &
                                (by_file.threshold_method == "normal_validation_q99")]
        precision_relative.append((new_precision.true_positive_count.sum() / max(new_precision.alarm_count.sum(), 1)) /
                                   max(base_precision.true_positive_count.sum() / max(base_precision.alarm_count.sum(), 1), 1e-12) - 1)
    primary = primary.copy(); primary["precision_relative_improvement"] = precision_relative
    file_compare = []
    for file_name in sorted(by_file.file_name.unique()):
        base = by_file[(by_file.score_variant == "endpoint_raw_mean") &
                       (by_file.threshold_method == "normal_validation_q99") & (by_file.file_name == file_name)]
        new = by_file[(by_file.score_variant == "endpoint_calibrated_top5") &
                      (by_file.threshold_method == "normal_validation_q99") & (by_file.file_name == file_name)]
        file_compare.append((file_name, float((new.false_positive_count.mean() - base.false_positive_count.mean()) /
                                              max(base.false_positive_count.mean(), 1))))
    calibrated_mean = comparison[(comparison.score_variant == "endpoint_calibrated_mean") &
                                 (comparison.threshold_method == "normal_validation_q99")].f1_delta.mean()
    raw_top5 = comparison[(comparison.score_variant == "endpoint_raw_top5") &
                          (comparison.threshold_method == "normal_validation_q99")].f1_delta.mean()
    evidence = {"precision_relative_mean": float(primary.precision_relative_improvement.mean()),
                "f1_delta_mean": float(primary.f1_delta.mean()),
                "directional_f1_seed_count": int((primary.f1_delta > 0).sum()),
                "aupr_delta_mean": float(primary.aupr_delta.mean()),
                "auroc_delta_mean": float(primary.auroc_delta.mean()),
                "event_recall_delta_mean": float(primary.event_recall_delta.mean()),
                "median_delay_delta_mean": float(primary.median_delay_delta.mean()),
                "normal_fp_relative_change_by_file": dict(file_compare),
                "calibrated_mean_q99_f1_delta_mean": float(calibrated_mean),
                "raw_top5_q99_f1_delta_mean": float(raw_top5)}
    substantive = (evidence["precision_relative_mean"] >= .30 and evidence["f1_delta_mean"] >= .03 and
                   evidence["directional_f1_seed_count"] >= 2 and dict(file_compare).get("test1.csv", 1) <= -.20 and
                   all(change <= .30 for name, change in file_compare if name != "test1.csv") and
                   evidence["event_recall_delta_mean"] >= -.10 and evidence["median_delay_delta_mean"] <= 30 and
                   evidence["aupr_delta_mean"] >= -.01 and evidence["auroc_delta_mean"] >= -.02)
    if substantive:
        return "channel_calibration_and_sparse_aggregation_improve_rec", evidence
    if calibrated_mean > .01:
        return "calibration_helps_but_topk_unstable", evidence
    if raw_top5 > .03:
        return "sparse_aggregation_helps_without_calibration", evidence
    conservative = comparison[(comparison.score_variant == "endpoint_calibrated_top5") &
                              (comparison.threshold_method.isin(["normal_validation_q995", "validation_sequence_max_q99"]))]
    if conservative.precision_delta.mean() > primary.precision_delta.mean() and primary.aupr_delta.mean() <= .01:
        return "precision_improves_only_with_conservative_threshold", evidence
    return "channel_dilution_is_not_main_problem", evidence


def save_scores(output: Path, seed: int, splits: dict[str, Any]) -> None:
    arrays = {}
    for split_name in ("train", "validation", "test"):
        for file_name, values in splits[split_name].items():
            prefix = f"{split_name}_{Path(file_name).stem}"
            for key in (*CHANNEL_SCORE_VARIANTS, "label", "timestamp", "end_row", "sequence_id"):
                arrays[f"{prefix}_{key}"] = values[key]
    np.savez_compressed(output / f"rec_channel_scores_seed{seed}.npz", **arrays)


def run(cfg: dict, output: Path, smoke: bool) -> dict[str, Any]:
    ensure_fresh(output, [int(seed) for seed in cfg["seeds"]])
    (output / "config_resolved.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    score_registry().to_csv(output / "rec_channel_score_registry.csv", index=False)
    audit, checkpoints, checkpoint_hashes = audit_checkpoints(cfg)
    device = torch.device("cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    bundle = build_hai_dataloaders(cfg["data_root"], window_size=int(cfg["window_size"]), stride_train=1,
                                   stride_test=1, validation_ratio=.20, batch_size=int(cfg["batch_size"]),
                                   num_workers=0, pin_memory=device.type == "cuda", return_metadata=True)
    max_windows = int(cfg["smoke_max_windows_per_sequence"]) if smoke else None
    all_values: dict[int, dict[str, Any]] = {}
    calibrations: dict[int, dict[str, np.ndarray]] = {}
    threshold_rows, consistency_rows = [], []
    for seed in cfg["seeds"]:
        seed = int(seed); model = make_model(checkpoints[seed], device)
        raw_train = {seq.file_name: infer_endpoint_errors(model, seq, device, int(cfg["window_size"]),
                                                           int(cfg["batch_size"]), max_windows)
                     for seq in bundle.train_sequences}
        calibration = fit_channel_calibration(np.concatenate([v["endpoint_error"] for v in raw_train.values()]),
                                              float(cfg["minimum_channel_scale"]))
        calibrations[seed] = calibration
        splits = {"train": {name: attach_scores(values, calibration, False) for name, values in raw_train.items()},
                  "validation": {}, "test": {}}
        del raw_train
        for split_name, sequences in (("validation", bundle.val_sequences), ("test", bundle.test_sequences)):
            for sequence in sequences:
                raw = infer_endpoint_errors(model, sequence, device, int(cfg["window_size"]),
                                            int(cfg["batch_size"]), max_windows)
                splits[split_name][sequence.file_name] = attach_scores(raw, calibration, split_name == "test")
        all_values[seed] = splits
        old = np.load(ROOT / cfg["step3c_score_pattern"].format(seed=seed), allow_pickle=False)
        for split_name, values_by_file in splits.items():
            for file_name, values in values_by_file.items():
                prefix = f"{split_name}_{Path(file_name).stem}"; count = len(values["endpoint_raw_mean"])
                historical = old[f"{prefix}_endpoint_mean"][:count]
                score_diff = np.abs(values["endpoint_raw_mean"] - historical)
                exact = {field: np.array_equal(values[field], old[f"{prefix}_{field}"][:count])
                         for field in ("sequence_id", "timestamp", "end_row", "label")}
                consistency_rows.append({"seed": seed, "split": split_name, "file_name": file_name,
                                         "score_length": count,
                                         "step3c_length": len(old[f"{prefix}_endpoint_mean"]),
                                         "reinference_score_max_absolute_difference": float(score_diff.max()),
                                         "reinference_score_mean_absolute_difference": float(score_diff.mean()),
                                         "formal_baseline_score_max_absolute_difference": 0.0,
                                         **{f"{key}_exact_match": value for key, value in exact.items()},
                                         "formal_baseline_source": str((ROOT / cfg["step3c_score_pattern"].format(seed=seed)).resolve()),
                                         "smoke": smoke})
                if not all(exact.values()):
                    raise ValueError(f"Step 3C metadata mismatch for {seed}/{prefix}")
                # Keep Step 3C's confirmed one-dimensional endpoint baseline exact. All new scores remain
                # derived from the freshly inferred per-channel residuals above.
                values["endpoint_raw_mean"] = historical.copy()
        train_concat = concat_values(splits["train"])
        for variant in CHANNEL_SCORE_VARIANTS:
            validation_by_sequence = {name: values[variant] for name, values in splits["validation"].items()}
            for row in fit_score_thresholds(train_concat[variant], validation_by_sequence, cfg):
                threshold_rows.append({"seed": seed, "score_variant": variant, **row,
                                       "test_labels_used": False, "channel_calibration_scope": "normal train only"})
        save_scores(output, seed, splits)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    consistency = pd.DataFrame(consistency_rows)
    tolerance = float(cfg["consistency_tolerance"])
    exact_cols = [f"{field}_exact_match" for field in ("sequence_id", "timestamp", "end_row", "label")]
    if consistency.reinference_score_max_absolute_difference.max() > tolerance or not consistency[exact_cols].all().all():
        raise ValueError("endpoint_raw_mean failed exact Step 3C alignment")
    consistency.to_csv(output / "endpoint_step3c_consistency.csv", index=False)
    thresholds = pd.DataFrame(threshold_rows)
    if (thresholds.status != "ok").any():
        raise RuntimeError("At least one fixed threshold failed; refusing formal comparison")
    calibration_frame(calibrations, bundle.feature_names).to_csv(output / "rec_channel_calibration_stats.csv", index=False)
    thresholds.to_csv(output / "rec_channel_thresholds.csv", index=False)
    validation_stability(all_values, thresholds).to_csv(output / "rec_channel_validation_stability.csv", index=False)
    by_file, by_seed, aggregate = build_point_metrics(all_values, thresholds)
    by_file.to_csv(output / "rec_channel_point_metrics_by_file.csv", index=False)
    by_seed.to_csv(output / "rec_channel_point_metrics_by_seed.csv", index=False)
    aggregate.to_csv(output / "rec_channel_point_metrics_aggregate.csv", index=False)
    diagnostics = build_event_diagnostics(all_values, thresholds, bundle.attack_events)
    diagnostics.to_csv(output / "rec_channel_event_diagnostics.csv", index=False)
    event_summary = summarize_events(diagnostics); event_summary.to_csv(output / "rec_channel_event_summary.csv", index=False)
    comparison = baseline_comparison(by_seed, event_summary)
    transitions = event_transitions(diagnostics)
    recovered_lost = transitions.groupby(["seed", "score_variant", "threshold_method"])[["recovered", "lost"]].sum().reset_index()
    comparison = comparison.merge(recovered_lost, on=["seed", "score_variant", "threshold_method"], how="left")
    comparison = comparison.rename(columns={"recovered": "recovered_event_count", "lost": "lost_event_count"})
    comparison.to_csv(output / "rec_channel_baseline_comparison.csv", index=False)
    transitions.to_csv(output / "rec_channel_event_transition.csv", index=False)
    samples, contribution_summary = contribution_outputs(all_values, thresholds, bundle.attack_events,
                                                         bundle.feature_names, cfg)
    samples.to_csv(output / "rec_channel_contribution_samples.csv", index=False)
    contribution_summary.to_csv(output / "rec_channel_contribution_summary.csv", index=False)
    multiseed = multiseed_summary(by_seed, comparison); multiseed.to_csv(output / "rec_channel_multiseed_summary.csv", index=False)
    classification, evidence = ("smoke_only_no_formal_conclusion", {"reason": "limited sequence prefixes"}) if smoke else \
                               classify(comparison, by_file, event_summary)
    if classification == "sparse_aggregation_helps_without_calibration":
        recommendation = ("Fix endpoint_raw_top5 as the reconstruction candidate and retain endpoint_raw_mean plus "
                          "calibrated variants as ablations. Validate a causal alarm policy before Step 4B; a model-capacity "
                          "ablation is not the next priority.")
        rec_ready = False
    else:
        recommendation = "Follow the classification-specific decision guidance in the diagnosis report."
        rec_ready = classification == "channel_calibration_and_sparse_aggregation_improve_rec"
    checkpoint_unchanged = all(sha256_file(ROOT / cfg["checkpoint_pattern"].format(seed=seed)) == digest
                               for seed, digest in checkpoint_hashes.items())
    audit["checkpoint_sha256_after"] = [sha256_file(Path(path)) for path in audit.checkpoint_path]
    audit["checkpoint_unchanged"] = audit.checkpoint_sha256_before == audit.checkpoint_sha256_after
    audit["model_eval_only"] = True; audit["optimizer_step_executed"] = False
    audit.to_csv(output / "checkpoint_audit.csv", index=False)
    primary_q99 = comparison[(comparison.score_variant == cfg["primary_score"]) &
                             (comparison.threshold_method == "normal_validation_q99")]
    summary = {"mode": "smoke" if smoke else "formal", "classification": classification,
               "classification_evidence": evidence, "baseline_score": cfg["baseline_score"],
               "primary_score": cfg["primary_score"], "score_variants": list(CHANNEL_SCORE_VARIANTS),
               "threshold_methods": list(THRESHOLD_METHODS), "checkpoint_count": 3,
               "checkpoints_unchanged": checkpoint_unchanged, "endpoint_step3c_reproduced": True,
               "maximum_step3c_score_difference": float(consistency.formal_baseline_score_max_absolute_difference.max()),
               "maximum_reinference_numeric_difference": float(consistency.reinference_score_max_absolute_difference.max()),
               "reinference_numeric_tolerance": tolerance,
               "calibration_fit_scope": "per-seed normal train endpoint residuals only",
               "test_labels_used_for_calibration_or_thresholds": False, "point_adjustment_used": False,
               "structure_score_computed": False, "training_executed": False,
               "recommended_rec_score": "endpoint_raw_top5" if classification == "sparse_aggregation_helps_without_calibration" else None,
               "rec_ready_as_main_detection_branch": rec_ready,
               "primary_q99_mean_deltas": {column: float(primary_q99[column].mean()) for column in
                                            ("precision_delta", "recall_delta", "f1_delta", "aupr_delta",
                                             "auroc_delta", "event_recall_delta", "median_delay_delta")},
               "recommended_next_step": recommendation}
    (output / "rec_channel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    top_fp = (contribution_summary.groupby(["channel_index", "channel_name"]).false_positive_top5_count.sum()
              .sort_values(ascending=False).head(15).reset_index().to_dict(orient="records"))
    primary_rows = multiseed[(multiseed.score_variant == cfg["primary_score"]) & (multiseed.scope == "micro")]
    report = f"""# HAI Step 3D endpoint channel calibration diagnosis

- Three existing rec-only checkpoints were loaded read-only; no training, model/loss change, structure score, fusion, point adjustment, future context, or test-fitted parameter was used.
- Formal endpoint raw mean is anchored exactly to Step 3C after metadata validation (maximum difference {summary['maximum_step3c_score_difference']:.3e}); fresh GPU reinference differed by at most {summary['maximum_reinference_numeric_difference']:.3e} within the recorded {summary['reinference_numeric_tolerance']:.1e} numerical tolerance.
- Fixed scores: {list(CHANNEL_SCORE_VARIANTS)}. Fixed thresholds: {list(THRESHOLD_METHODS)}. Primary was pre-registered as `{cfg['primary_score']}`.
- Channel calibration was independently fitted per seed from normal training endpoint residuals using median and 1.4826*MAD, with recorded IQR/std/unit fallback.
- Classification: `{classification}`. Evidence: {evidence}.
- Recommended reconstruction candidate: `{summary['recommended_rec_score']}`; ready as the main detection branch: `{summary['rec_ready_as_main_detection_branch']}`.
- Recommended next step: {summary['recommended_next_step']}
- Primary micro results across all four fixed thresholds: {primary_rows.to_dict(orient='records')}.
- Primary q99 deltas versus endpoint raw mean: {primary_q99.to_dict(orient='records')}.
- Channels most often contributing to primary false positives: {top_fp}.
- Full point, event, per-file false-positive, contribution, transition, calibration and validation-sequence results are stored in the accompanying CSV files.
- Checkpoints unchanged: `{checkpoint_unchanged}`. Formal scores contain no NaN/Inf; missing event delays are explicitly masked only when an event is not detected.

Decision guidance: A fixes calibrated top5 as the rec candidate; B fixes calibrated mean; C fixes raw top5; D reports only a conservative operating-point change; E pauses Step 4B for limited capacity/normal-condition ablation. No next stage was executed.
"""
    (output / "final_hai_rec_channel_calibration_diagnosis.md").write_text(report, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAI Step 3D endpoint channel calibration evaluation")
    parser.add_argument("--config", required=True); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); cfg = load_config(args.config)
    output = ROOT / (cfg["smoke_output_dir"] if args.smoke else cfg["output_dir"])
    if args.dry_run:
        audit, checkpoints, _ = audit_checkpoints(cfg)
        print(audit.to_string(index=False)); print(f"device_requested={cfg['device']}")
        print(f"fixed_scores={list(CHANNEL_SCORE_VARIANTS)}"); print(f"fixed_thresholds={list(THRESHOLD_METHODS)}")
        print(f"primary={cfg['primary_score']}; output={output.resolve()}")
        print("calibration=train-only per seed; test labels excluded; training/model/structure/fusion disabled")
        print(f"checkpoint_features={[len(checkpoints[int(seed)]['feature_names']) for seed in cfg['seeds']]}")
        return 0
    summary = run(cfg, output, args.smoke)
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
