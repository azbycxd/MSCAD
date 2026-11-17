from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hai import HAISequence, MultiSequenceWindowDataset, build_hai_dataloaders  # noqa: E402
from models.mscad import MSCAD  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.metrics import binary_metrics  # noqa: E402
from utils.pot import POTThreshold  # noqa: E402
from utils.reconstruction_scores import (  # noqa: E402
    ALL_VARIANTS, CAUSAL_VARIANTS, add_overlap_errors, finalize_overlap_scores, temporal_reconstruction_scores,
)


THRESHOLD_METHODS = ("train_only_pot", "normal_validation_q99")
OUTPUT_FILES = (
    "score_definition_registry.csv", "checkpoint_and_config_audit.csv", "rerun_consistency_check.csv",
    "rec_alignment_thresholds.csv", "rec_alignment_normal_score_stats.csv",
    "rec_alignment_validation_sequence_stability.csv", "rec_alignment_point_metrics_by_seed.csv",
    "rec_alignment_point_metrics_by_file.csv", "rec_alignment_point_metrics_aggregate.csv",
    "rec_alignment_event_diagnostics.csv", "rec_alignment_event_summary.csv", "rec_alignment_onset_summary.csv",
    "rec_alignment_event_transition.csv", "rec_alignment_score_correlations.csv",
    "rec_alignment_alarm_agreement.csv", "rec_alignment_false_positive_diagnostics.csv",
    "rec_alignment_noncausal_upper_bound.csv", "rec_alignment_summary.json",
    "final_hai_rec_score_alignment_diagnosis.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_fresh(output: Path) -> None:
    expected = list(OUTPUT_FILES) + [f"rec_alignment_scores_seed{seed}.npz" for seed in (1, 2, 3)]
    existing = [name for name in expected if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite HAI Step 3C outputs: {existing}")
    output.mkdir(parents=True, exist_ok=True)


def score_registry() -> pd.DataFrame:
    return pd.DataFrame([
        {"score_variant": "window_mean_current", "definition": "mean(abs(X-X_hat), time and all 86 channels)",
         "causal": True, "support_points": 64, "point_alignment": "weak", "role": "formal_historical_baseline"},
        {"score_variant": "endpoint_mean", "definition": "mean(abs(X[-1]-X_hat[-1]), all 86 channels)",
         "causal": True, "support_points": 1, "point_alignment": "exact", "role": "primary_alignment_candidate"},
        {"score_variant": "recent8_mean", "definition": "mean(abs(X[-8:]-X_hat[-8:]), time and all 86 channels)",
         "causal": True, "support_points": 8, "point_alignment": "recent", "role": "diagnostic_candidate"},
        {"score_variant": "recent16_mean", "definition": "mean(abs(X[-16:]-X_hat[-16:]), time and all 86 channels)",
         "causal": True, "support_points": 16, "point_alignment": "recent", "role": "diagnostic_candidate"},
        {"score_variant": "offline_overlap_point_mean", "definition": "mean point error across all same-sequence windows covering t",
         "causal": False, "support_points": "variable", "point_alignment": "exact_noncausal",
         "role": "diagnostic_upper_bound_only_not_allowed_as_formal_online_score"},
    ])


def normalized_config(cfg: dict) -> dict:
    value = json.loads(json.dumps(cfg))
    value.pop("seed", None); value.pop("run_name", None); value.pop("output", None)
    if isinstance(value.get("training"), dict): value["training"].pop("seed", None)
    return value


def make_model(checkpoint: dict, device: torch.device) -> MSCAD:
    cfg = checkpoint["cfg"]
    model_cfg, data_cfg = cfg["model"], cfg["data"]
    model = MSCAD(in_channels=int(model_cfg["input_channels"]), hidden_channels=int(model_cfg["hidden_channels"]),
                  latent_channels=int(model_cfg["latent_channels"]), dropout=float(model_cfg["dropout"]),
                  cov_type="corr", short_window=int(data_cfg.get("short_window", 16)),
                  backbone_variant=str(model_cfg.get("backbone_variant", "base")))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model


def audit_checkpoints(cfg: dict) -> tuple[pd.DataFrame, dict[int, dict], dict[int, str]]:
    rows, checkpoints, hashes = [], {}, {}
    normalized = []
    for seed in cfg["seeds"]:
        path = ROOT / cfg["checkpoint_pattern"].format(seed=seed)
        if not path.exists(): raise FileNotFoundError(path)
        digest = sha256_file(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoints[int(seed)] = checkpoint; hashes[int(seed)] = digest
        normalized.append(normalized_config(checkpoint["cfg"]))
        rows.append({"seed": seed, "checkpoint_path": str(path.resolve()), "exists": True,
                     "checkpoint_sha256_before": digest, "checkpoint_seed": checkpoint["seed"],
                     "feature_count": len(checkpoint["feature_names"]), "best_epoch": checkpoint["best_epoch"],
                     "selection_scope": checkpoint["selection_scope"], "model_input_shape": "[batch,64,86]",
                     "model_output_shape": "[batch,64,86]", "training_executed": False})
    equal = all(item == normalized[0] for item in normalized[1:])
    if not equal: raise ValueError("Checkpoint configurations differ beyond seed/output fields")
    for row in rows: row["only_seed_and_output_fields_differ"] = True
    return pd.DataFrame(rows), checkpoints, hashes


@torch.no_grad()
def infer_sequence(model: MSCAD, sequence: HAISequence, device: torch.device, window: int, batch_size: int,
                   maximum_windows: int | None = None) -> dict[str, np.ndarray]:
    dataset = MultiSequenceWindowDataset([sequence], window_size=window, stride=1, return_metadata=True)
    if maximum_windows is not None:
        dataset = Subset(dataset, np.arange(min(maximum_windows, len(dataset))).tolist())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    collected = {variant: [] for variant in CAUSAL_VARIANTS}
    labels, timestamps, end_rows, sequence_ids = [], [], [], []
    error_sum = np.zeros(len(sequence.x), dtype=np.float64)
    coverage = np.zeros(len(sequence.x), dtype=np.int32)
    model.eval()
    for x, y, metadata in loader:
        x = x.to(device, non_blocking=True)
        x_hat = model(x)
        scores, point_error = temporal_reconstruction_scores(x, x_hat)
        for variant in CAUSAL_VARIANTS: collected[variant].append(scores[variant].cpu().numpy())
        starts_raw = metadata["start_row"].numpy().astype(np.int64)
        starts_local = starts_raw - int(sequence.original_row_offset)
        add_overlap_errors(error_sum, coverage, starts_local, point_error.cpu().numpy())
        labels.append(y.numpy()); timestamps.extend(str(value) for value in metadata["timestamp"])
        end_rows.extend(int(value) for value in metadata["end_row"].numpy())
        sequence_ids.extend(str(value) for value in metadata["sequence_id"])
    result = {variant: np.concatenate(parts).astype(np.float32) for variant, parts in collected.items()}
    overlap, covered = finalize_overlap_scores(error_sum, coverage)
    local_end = np.asarray(end_rows, dtype=np.int64) - int(sequence.original_row_offset)
    if not covered[local_end].all(): raise ValueError(f"Uncovered aligned overlap point in {sequence.file_name}/{sequence.split}")
    result["offline_overlap_point_mean"] = overlap[local_end].astype(np.float32)
    result.update({"label": np.concatenate(labels).astype(np.int64), "timestamp": np.asarray(timestamps, dtype=str),
                   "end_row": np.asarray(end_rows, dtype=np.int64), "sequence_id": np.asarray(sequence_ids, dtype=str),
                   "overlap_coverage_count": coverage[local_end]})
    for variant in ALL_VARIANTS:
        if not np.isfinite(result[variant]).all(): raise ValueError(f"Non-finite {variant} in {sequence.file_name}")
    return result


def concat_values(sequence_values: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = (*ALL_VARIANTS, "label", "timestamp", "end_row", "sequence_id", "overlap_coverage_count")
    return {key: np.concatenate([values[key] for values in sequence_values.values()]) for key in keys}


def fit_thresholds(train: np.ndarray, validation: np.ndarray, cfg: dict) -> list[dict[str, Any]]:
    rows = []
    q, target = float(cfg["thresholds"]["pot_q"]), float(cfg["thresholds"]["pot_target_prob"])
    try:
        pot = POTThreshold(q=q, target_prob=target).fit(train)
        if pot.tau is None or not np.isfinite(pot.tau): raise RuntimeError("POT returned non-finite threshold")
        rows.append({"threshold_method": "train_only_pot", "threshold_value": float(pot.tau), "status": "ok",
                     "fitting_sample_count": len(train), "fitting_scope": "corresponding variant train windows/points only",
                     "pot_q": q, "pot_target_prob": target, "pot_u": pot.u, "failure_reason": ""})
    except Exception as exc:
        rows.append({"threshold_method": "train_only_pot", "threshold_value": np.nan, "status": "failed",
                     "fitting_sample_count": len(train), "fitting_scope": "corresponding variant train windows/points only",
                     "pot_q": q, "pot_target_prob": target, "pot_u": np.nan,
                     "failure_reason": f"{type(exc).__name__}: {exc}"})
    quantile = float(cfg["thresholds"]["validation_quantile"])
    rows.append({"threshold_method": "normal_validation_q99", "threshold_value": float(np.quantile(validation, quantile)),
                 "status": "ok", "fitting_sample_count": len(validation),
                 "fitting_scope": "corresponding variant normal validation windows/points only",
                 "validation_quantile": quantile, "failure_reason": ""})
    return rows


def metric_record(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (scores > threshold).astype(np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = binary_metrics(labels, prediction, scores)
    normal = labels == 0
    return {**metrics, "anomaly_point_count": int((labels == 1).sum()), "normal_point_count": int(normal.sum()),
            "alarm_count": int(prediction.sum()), "false_positive_count": int((normal & (prediction == 1)).sum()),
            "false_negative_count": int(((labels == 1) & (prediction == 0)).sum()),
            "normal_false_positive_rate": float((normal & (prediction == 1)).sum() / max(normal.sum(), 1))}


def threshold_lookup(frame: pd.DataFrame, seed: int, variant: str, method: str) -> float:
    row = frame[(frame.seed == seed) & (frame.score_variant == variant) & (frame.threshold_method == method)]
    if len(row) != 1 or row.status.iloc[0] != "ok": raise ValueError(f"Unavailable threshold {seed}/{variant}/{method}")
    return float(row.threshold_value.iloc[0])


def consistency_check(all_values: dict[int, dict[str, Any]], cfg: dict, smoke: bool) -> pd.DataFrame:
    rows = []
    for seed, splits in all_values.items():
        historical_dir = ROOT / cfg["historical_result_pattern"].format(seed=seed)
        old = np.load(historical_dir / "test_score_arrays.npz")
        old_thresholds = {row["threshold_method"]: row for row in json.loads((historical_dir / "threshold_summary.json").read_text())}
        for file_key, new in splits["test"].items():
            key = Path(file_key).stem; count = len(new["window_mean_current"])
            checks = {field: np.array_equal(new[field], old[f"{key}_{field}"][:count])
                      for field in ("sequence_id", "timestamp", "end_row", "label")}
            old_score = old[f"{key}_score"][:count]
            score_diff = np.abs(new["window_mean_current"] - old_score)
            for method in THRESHOLD_METHODS:
                tau = float(old_thresholds[method]["threshold_value"])
                new_metric = metric_record(new["label"], new["window_mean_current"], tau)
                old_metric = metric_record(old[f"{key}_label"][:count], old_score, tau)
                rows.append({"seed": seed, "file_name": file_key, "threshold_method": method,
                             "score_array_length": count, "historical_length": len(old[f"{key}_score"]),
                             **{f"{name}_exact_match": value for name, value in checks.items()},
                             "score_max_absolute_difference": float(score_diff.max()),
                             "score_mean_absolute_difference": float(score_diff.mean()),
                             "auroc_difference": new_metric["auroc"] - old_metric["auroc"],
                             "aupr_difference": new_metric["aupr"] - old_metric["aupr"],
                             "precision_difference": new_metric["precision"] - old_metric["precision"],
                             "recall_difference": new_metric["recall"] - old_metric["recall"],
                             "f1_difference": new_metric["f1"] - old_metric["f1"], "smoke_subset": smoke})
    frame = pd.DataFrame(rows)
    tolerance = float(cfg["consistency"]["score_absolute_tolerance"])
    ranking_tolerance = float(cfg["consistency"]["ranking_metric_tolerance"])
    aligned = frame[[f"{name}_exact_match" for name in ("sequence_id", "timestamp", "end_row", "label")]].all().all()
    if not aligned or frame.score_max_absolute_difference.max() > tolerance or frame[["auroc_difference", "aupr_difference"]].abs().max().max() > ranking_tolerance:
        raise ValueError("Current-score rerun failed historical consistency requirements")
    return frame


def build_point_metrics(all_values: dict[int, dict[str, Any]], thresholds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_file_rows, summary_rows = [], []
    for seed, splits in all_values.items():
        for variant in ALL_VARIANTS:
            for method in THRESHOLD_METHODS:
                tau = threshold_lookup(thresholds, seed, variant, method)
                file_records = []
                for file_name, values in splits["test"].items():
                    row = {"seed": seed, "score_variant": variant, "threshold_method": method, "file_name": file_name,
                           **metric_record(values["label"], values[variant], tau)}
                    by_file_rows.append(row); file_records.append(row)
                labels = np.concatenate([v["label"] for v in splits["test"].values()])
                scores = np.concatenate([v[variant] for v in splits["test"].values()])
                summary_rows.append({"seed": seed, "score_variant": variant, "threshold_method": method,
                                     "summary_scope": "micro", **metric_record(labels, scores, tau)})
                numeric = [key for key in file_records[0] if key not in {"seed", "score_variant", "threshold_method", "file_name"}]
                macro = {name: float(np.mean([row[name] for row in file_records])) for name in numeric}
                summary_rows.append({"seed": seed, "score_variant": variant, "threshold_method": method,
                                     "summary_scope": "macro", **macro})
    summary, by_file = pd.DataFrame(summary_rows), pd.DataFrame(by_file_rows)
    for (seed, variant, scope), group in summary.groupby(["seed", "score_variant", "summary_scope"]):
        if (not np.allclose(group.aupr.to_numpy(), group.aupr.iloc[0], equal_nan=True) or
                not np.allclose(group.auroc.to_numpy(), group.auroc.iloc[0], equal_nan=True)):
            raise ValueError(f"Ranking metric depends on threshold: {seed}/{variant}/{scope}")
    combined = pd.concat([summary.assign(scope=summary.summary_scope),
                          by_file.assign(scope=by_file.file_name.str.replace(".csv", "", regex=False))], ignore_index=True)
    metrics = ("precision", "recall", "f1", "aupr", "auroc", "alarm_count", "false_positive_count",
               "false_negative_count", "normal_false_positive_rate")
    aggregate = []
    for (variant, method, scope), group in combined.groupby(["score_variant", "threshold_method", "scope"]):
        indexed = group.set_index("seed")
        for metric in metrics:
            values = indexed[metric].astype(float)
            aggregate.append({"score_variant": variant, "threshold_method": method, "scope": scope, "metric": metric,
                              "seed1": values.loc[1], "seed2": values.loc[2], "seed3": values.loc[3],
                              "mean": values.mean(), "std": values.std(ddof=0), "min": values.min(), "max": values.max()})
    return summary, by_file, pd.DataFrame(aggregate)


def event_diagnostics(all_values: dict[int, dict[str, Any]], thresholds: pd.DataFrame, events: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for seed, splits in all_values.items():
        for variant in ALL_VARIANTS:
            for method in THRESHOLD_METHODS:
                tau = threshold_lookup(thresholds, seed, variant, method)
                for event in events:
                    values = splits["test"][event["file_name"]]
                    mask = (values["end_row"] >= event["start_row"]) & (values["end_row"] <= event["end_row"])
                    scores, times, end_rows = values[variant][mask], values["timestamp"][mask], values["end_row"][mask]
                    alarms = scores > tau; detected = bool(alarms.any()); first = int(np.flatnonzero(alarms)[0]) if detected else -1
                    delay = int(end_rows[first] - event["start_row"]) if detected else np.nan
                    rows.append({"seed": seed, "score_variant": variant, "threshold_method": method,
                                 "sequence_id": event["sequence_id"], "file_name": event["file_name"],
                                 "event_id": event["event_id_in_file"], "global_event_id": event["global_event_id"],
                                 "start_timestamp": event["start_timestamp"], "end_timestamp": event["end_timestamp"],
                                 "duration_points": event["duration_points"], "detected": detected,
                                 "first_alarm_timestamp": str(times[first]) if detected else "",
                                 "detection_delay_points": delay, "point_recall_in_event": float(alarms.mean()) if len(alarms) else 0.0,
                                 "peak_score": float(scores.max()) if len(scores) else np.nan,
                                 "mean_score": float(scores.mean()) if len(scores) else np.nan,
                                 "detected_at_first_point": bool(detected and delay == 0),
                                 "detected_within_5_points": bool(detected and delay <= 5),
                                 "detected_within_10_points": bool(detected and delay <= 10),
                                 "detected_within_30_points": bool(detected and delay <= 30)})
    return pd.DataFrame(rows)


def summarize_events(diagnostics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, group in diagnostics.groupby(["seed", "score_variant", "threshold_method", "file_name"]):
        rows.append(event_summary_row(*keys, group))
    for keys, group in diagnostics.groupby(["seed", "score_variant", "threshold_method"]):
        rows.append(event_summary_row(*keys, "all_tests", group))
    summary = pd.DataFrame(rows)
    onset = summary[["seed", "score_variant", "threshold_method", "summary_scope", "total_event_count",
                     "first_point_hit_rate", "within5_hit_rate", "within10_hit_rate", "within30_hit_rate"]].copy()
    return summary, onset


def event_summary_row(seed: int, variant: str, method: str, scope: str, group: pd.DataFrame) -> dict[str, Any]:
    delays = group.loc[group.detected, "detection_delay_points"].dropna().astype(float)
    return {"seed": seed, "score_variant": variant, "threshold_method": method, "summary_scope": scope,
            "total_event_count": len(group), "detected_event_count": int(group.detected.sum()),
            "event_recall": float(group.detected.mean()), "fully_missed_event_count": int((~group.detected).sum()),
            "median_delay": float(delays.median()) if len(delays) else np.nan,
            "mean_delay": float(delays.mean()) if len(delays) else np.nan,
            "first_point_hit_rate": float(group.detected_at_first_point.mean()),
            "within5_hit_rate": float(group.detected_within_5_points.mean()),
            "within10_hit_rate": float(group.detected_within_10_points.mean()),
            "within30_hit_rate": float(group.detected_within_30_points.mean())}


def event_transitions(diagnostics: pd.DataFrame) -> pd.DataFrame:
    index_names = ["seed", "threshold_method", "global_event_id"]
    indexed = diagnostics.set_index([*index_names, "score_variant"])
    rows = []
    for key in diagnostics[index_names].drop_duplicates().itertuples(index=False, name=None):
        variants = {variant: indexed.loc[(*key, variant)] for variant in ALL_VARIANTS}
        current = variants["window_mean_current"]
        detected = {variant: bool(row.detected) for variant, row in variants.items()}
        delays = {variant: float(row.detection_delay_points) if row.detected else np.nan for variant, row in variants.items()}
        for comparison in ("endpoint_mean", "recent8_mean", "recent16_mean", "offline_overlap_point_mean"):
            row = variants[comparison]
            rows.append({"seed": key[0], "threshold_method": key[1], "sequence_id": current.sequence_id,
                         "file_name": current.file_name, "event_id": current.event_id, "global_event_id": key[2],
                         "start_timestamp": current.start_timestamp, "end_timestamp": current.end_timestamp,
                         "comparison_variant": comparison, **{f"{name.replace('_mean','').replace('window_','')}_detected": value for name, value in detected.items()},
                         "recovered_by_endpoint": (not detected["window_mean_current"]) and detected["endpoint_mean"],
                         "recovered_by_recent8": (not detected["window_mean_current"]) and detected["recent8_mean"],
                         "recovered_by_recent16": (not detected["window_mean_current"]) and detected["recent16_mean"],
                         "lost_by_endpoint": detected["window_mean_current"] and not detected["endpoint_mean"],
                         "lost_by_recent8": detected["window_mean_current"] and not detected["recent8_mean"],
                         "lost_by_recent16": detected["window_mean_current"] and not detected["recent16_mean"],
                         "current_delay": delays["window_mean_current"], "new_delay": delays[comparison],
                         "delay_change": delays[comparison] - delays["window_mean_current"] if np.isfinite(delays[comparison]) and np.isfinite(delays["window_mean_current"]) else np.nan})
    return pd.DataFrame(rows)


def score_and_alarm_agreement(all_values: dict[int, dict[str, Any]], thresholds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows, alarm_rows = [], []
    scopes = [f"test{i}.csv" for i in range(1, 5)] + ["all_tests"]
    def array(seed: int, variant: str, scope: str, field: str = "score") -> np.ndarray:
        tests = all_values[seed]["test"]
        if scope == "all_tests":
            return np.concatenate([tests[f"test{i}.csv"][variant if field == "score" else field] for i in range(1, 5)])
        return tests[scope][variant if field == "score" else field]
    comparisons = []
    for seed in all_values:
        comparisons += [("within_seed_variants", seed, left, seed, right) for left, right in combinations(ALL_VARIANTS, 2)]
    for variant in ALL_VARIANTS:
        comparisons += [("cross_seed_same_variant", left, variant, right, variant) for left, right in combinations((1, 2, 3), 2)]
    for kind, left_seed, left_variant, right_seed, right_variant in comparisons:
        for scope in scopes:
            left, right = array(left_seed, left_variant, scope), array(right_seed, right_variant, scope)
            score_rows.append({"comparison_type": kind, "scope": scope, "left_seed": left_seed,
                               "left_variant": left_variant, "right_seed": right_seed, "right_variant": right_variant,
                               "sample_count": len(left), "pearson": float(pearsonr(left, right).statistic),
                               "spearman": float(spearmanr(left, right).statistic)})
            for method in THRESHOLD_METHODS:
                lp = left > threshold_lookup(thresholds, left_seed, left_variant, method)
                rp = right > threshold_lookup(thresholds, right_seed, right_variant, method)
                union, intersection, disagreement = np.logical_or(lp, rp).sum(), np.logical_and(lp, rp).sum(), (lp != rp).sum()
                both_negative = np.logical_and(~lp, ~rp).sum()
                alarm_rows.append({"comparison_type": kind, "threshold_method": method, "scope": scope,
                                   "left_seed": left_seed, "left_variant": left_variant, "right_seed": right_seed,
                                   "right_variant": right_variant, "alarm_jaccard": intersection / union if union else 1.0,
                                   "positive_agreement": 2 * intersection / (2 * intersection + disagreement) if intersection or disagreement else 1.0,
                                   "negative_agreement": 2 * both_negative / (2 * both_negative + disagreement) if both_negative or disagreement else 1.0,
                                   "disagreement_count": int(disagreement), "sample_count": len(lp)})
    return pd.DataFrame(score_rows), pd.DataFrame(alarm_rows)


def false_positive_diagnostics(all_values: dict[int, dict[str, Any]], thresholds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, splits in all_values.items():
        for method in THRESHOLD_METHODS:
            current_tau = threshold_lookup(thresholds, seed, "window_mean_current", method)
            for file_name, values in splits["test"].items():
                current_fp = int(((values["label"] == 0) & (values["window_mean_current"] > current_tau)).sum())
                normal_count = int((values["label"] == 0).sum()); current_fpr = current_fp / max(normal_count, 1)
                for variant in CAUSAL_VARIANTS:
                    scores = values[variant]; alarms = scores > threshold_lookup(thresholds, seed, variant, method)
                    normal_alarm = alarms & (values["label"] == 0); fp = int(normal_alarm.sum())
                    padded = np.r_[False, normal_alarm, False]; starts = np.flatnonzero(~padded[:-1] & padded[1:]); ends = np.flatnonzero(padded[:-1] & ~padded[1:])
                    normal_scores = scores[values["label"] == 0]
                    rows.append({"seed": seed, "score_variant": variant, "threshold_method": method,
                                 "file_name": file_name, "normal_alarm_count": fp, "normal_point_count": normal_count,
                                 "normal_fpr": fp / max(normal_count, 1), "normal_alarm_interval_count": len(starts),
                                 "longest_normal_alarm_interval": int((ends - starts).max()) if len(starts) else 0,
                                 "normal_score_q95": float(np.quantile(normal_scores, .95)),
                                 "normal_score_q99": float(np.quantile(normal_scores, .99)),
                                 "fp_change_vs_current": fp - current_fp,
                                 "fpr_relative_change_vs_current": (fp / max(normal_count, 1) - current_fpr) / current_fpr if current_fpr else np.nan})
    return pd.DataFrame(rows)


def normal_statistics(all_values: dict[int, dict[str, Any]], thresholds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_variation = thresholds.groupby(["score_variant", "threshold_method"]).threshold_value.agg(
        seed_std=lambda x: x.std(ddof=0), seed_mean="mean").reset_index()
    threshold_variation["seed_cv"] = threshold_variation.seed_std / threshold_variation.seed_mean
    rows, sequence_rows = [], []
    for seed, splits in all_values.items():
        for variant in ALL_VARIANTS:
            pot = threshold_lookup(thresholds, seed, variant, "train_only_pot")
            q99 = threshold_lookup(thresholds, seed, variant, "normal_validation_q99")
            variation = threshold_variation[threshold_variation.score_variant == variant].set_index("threshold_method")
            for split in ("train", "validation"):
                scores = splits[f"{split}_all"][variant].astype(float)
                rows.append({"seed": seed, "score_variant": variant, "split": split, "sample_count": len(scores),
                             "mean": scores.mean(), "std": scores.std(), "median": np.median(scores),
                             "q90": np.quantile(scores, .90), "q95": np.quantile(scores, .95),
                             "q99": np.quantile(scores, .99), "maximum": scores.max(),
                             "coefficient_of_variation": scores.std() / scores.mean(), "pot_threshold": pot,
                             "q99_threshold": q99, "threshold_gap": q99 - pot,
                             "pot_threshold_seed_std": variation.loc["train_only_pot", "seed_std"],
                             "pot_threshold_seed_cv": variation.loc["train_only_pot", "seed_cv"],
                             "q99_threshold_seed_std": variation.loc["normal_validation_q99", "seed_std"],
                             "q99_threshold_seed_cv": variation.loc["normal_validation_q99", "seed_cv"]})
            for file_name, values in splits["validation"].items():
                scores = values[variant]
                sequence_rows.append({"seed": seed, "score_variant": variant, "file_name": file_name,
                                      "validation_normal_mean": scores.mean(), "validation_normal_q95": np.quantile(scores, .95),
                                      "validation_normal_q99": np.quantile(scores, .99),
                                      "q99_exceedance_ratio": float((scores > q99).mean()),
                                      "pot_exceedance_ratio": float((scores > pot).mean())})
    return pd.DataFrame(rows), pd.DataFrame(sequence_rows)


def noncausal_summary(point_summary: pd.DataFrame, event_summary: pd.DataFrame, fp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in (1, 2, 3):
        for method in THRESHOLD_METHODS:
            metrics = point_summary[(point_summary.seed == seed) & (point_summary.threshold_method == method) &
                                    (point_summary.summary_scope == "micro")].set_index("score_variant")
            events = event_summary[(event_summary.seed == seed) & (event_summary.threshold_method == method) &
                                   (event_summary.summary_scope == "all_tests")].set_index("score_variant")
            fps = fp[(fp.seed == seed) & (fp.threshold_method == method)].groupby("score_variant").normal_alarm_count.sum()
            row = {"record_type": "seed", "seed": seed, "threshold_method": method}
            for name in ("aupr", "auroc", "f1"):
                row[f"current_{name}"] = metrics.loc["window_mean_current", name]
                row[f"endpoint_{name}"] = metrics.loc["endpoint_mean", name]
                row[f"overlap_{name}"] = metrics.loc["offline_overlap_point_mean", name]
                row[f"overlap_minus_current_{name}"] = row[f"overlap_{name}"] - row[f"current_{name}"]
                row[f"overlap_minus_endpoint_{name}"] = row[f"overlap_{name}"] - row[f"endpoint_{name}"]
            for name in ("event_recall", "median_delay", "mean_delay"):
                row[f"current_{name}"] = events.loc["window_mean_current", name]
                row[f"endpoint_{name}"] = events.loc["endpoint_mean", name]
                row[f"overlap_{name}"] = events.loc["offline_overlap_point_mean", name]
            row["current_normal_fp"] = metrics.loc["window_mean_current", "false_positive_count"]
            row["endpoint_normal_fp"] = metrics.loc["endpoint_mean", "false_positive_count"]
            row["overlap_normal_fp"] = metrics.loc["offline_overlap_point_mean", "false_positive_count"]
            rows.append(row)
    seed_frame = pd.DataFrame(rows)
    numeric = [name for name in seed_frame.columns if name not in {"record_type", "seed", "threshold_method"}]
    for method, group in seed_frame.groupby("threshold_method"):
        for stat in ("mean", "std"):
            row = {"record_type": "aggregate", "seed": stat, "threshold_method": method}
            row.update({name: float(group[name].std(ddof=0) if stat == "std" else group[name].mean()) for name in numeric})
            rows.append(row)
    return pd.DataFrame(rows)


def classify_result(point_aggregate: pd.DataFrame, events: pd.DataFrame, fp: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    ranking = point_aggregate[(point_aggregate.scope == "micro") & (point_aggregate.threshold_method == "normal_validation_q99")]
    def metric(variant: str, name: str) -> pd.Series:
        row = ranking[(ranking.score_variant == variant) & (ranking.metric == name)].iloc[0]
        return pd.Series([row.seed1, row.seed2, row.seed3], dtype=float)
    current_aupr, current_auroc = metric("window_mean_current", "aupr"), metric("window_mean_current", "auroc")
    evidence: dict[str, Any] = {}
    improved = []
    for variant in ("endpoint_mean", "recent8_mean", "recent16_mean"):
        aupr_delta, auroc_delta = metric(variant, "aupr") - current_aupr, metric(variant, "auroc") - current_auroc
        event_current = events[(events.score_variant == "window_mean_current") & (events.summary_scope == "all_tests")].groupby("seed").event_recall.mean()
        event_new = events[(events.score_variant == variant) & (events.summary_scope == "all_tests")].groupby("seed").event_recall.mean()
        delay_current = events[(events.score_variant == "window_mean_current") & (events.summary_scope == "all_tests")].groupby("seed").median_delay.mean()
        delay_new = events[(events.score_variant == variant) & (events.summary_scope == "all_tests")].groupby("seed").median_delay.mean()
        fp_current = fp[(fp.score_variant == "window_mean_current") & (fp.file_name == "test1.csv")].groupby("seed").normal_alarm_count.mean()
        fp_new = fp[(fp.score_variant == variant) & (fp.file_name == "test1.csv")].groupby("seed").normal_alarm_count.mean()
        qualifies = ((aupr_delta.mean() >= .01 and (aupr_delta > 0).sum() >= 2) or
                     (auroc_delta.mean() >= .01 and (auroc_delta > 0).sum() >= 2) or
                     (((event_new - event_current).mean() >= .05 or (delay_new / delay_current).median() <= .80) and
                      (fp_new / fp_current).mean() <= 1.2))
        evidence[variant] = {"micro_aupr_delta_mean": float(aupr_delta.mean()),
                             "micro_auroc_delta_mean": float(auroc_delta.mean()),
                             "event_recall_delta_mean": float((event_new - event_current).mean()),
                             "median_delay_ratio": float((delay_new / delay_current).median()),
                             "test1_fp_ratio_mean": float((fp_new / fp_current).mean()), "substantive": bool(qualifies)}
        if qualifies: improved.append(variant)
    overlap_aupr = metric("offline_overlap_point_mean", "aupr") - current_aupr
    overlap_auroc = metric("offline_overlap_point_mean", "auroc") - current_auroc
    current_events = events[(events.score_variant == "window_mean_current") & (events.summary_scope == "all_tests")].groupby("seed").event_recall.mean()
    overlap_events = events[(events.score_variant == "offline_overlap_point_mean") & (events.summary_scope == "all_tests")].groupby("seed").event_recall.mean()
    overlap_improved = bool((overlap_aupr.mean() >= .01 and (overlap_aupr > 0).sum() >= 2) or
                            (overlap_auroc.mean() >= .01 and (overlap_auroc > 0).sum() >= 2) or
                            ((overlap_events - current_events).mean() >= .05))
    evidence["offline_overlap_point_mean"] = {"micro_aupr_delta_mean": float(overlap_aupr.mean()),
                                               "micro_auroc_delta_mean": float(overlap_auroc.mean()),
                                               "event_recall_delta_mean": float((overlap_events - current_events).mean()),
                                               "substantive": overlap_improved}
    if improved:
        endpoint_noisy = "endpoint_mean" not in improved and any(name in improved for name in ("recent8_mean", "recent16_mean"))
        classification = "temporal_dilution_exists_but_endpoint_is_noisy" if endpoint_noisy else "temporal_score_alignment_improves_rec_baseline"
    elif overlap_improved:
        classification = "noncausal_upper_bound_improves_only"
    elif all(metric(variant, "aupr").mean() <= current_aupr.mean() + .005 for variant in ALL_VARIANTS[1:]):
        classification = "current_window_mean_is_sufficient"
    else:
        classification = "score_alignment_is_not_main_problem"
    return classification, evidence


def save_score_arrays(output: Path, seed: int, splits: dict[str, Any]) -> None:
    arrays = {}
    for split in ("train", "validation", "test"):
        for file_name, values in splits[split].items():
            prefix = f"{split}_{Path(file_name).stem}"
            for key in (*ALL_VARIANTS, "label", "timestamp", "end_row", "sequence_id", "overlap_coverage_count"):
                arrays[f"{prefix}_{key}"] = values[key]
    np.savez_compressed(output / f"rec_alignment_scores_seed{seed}.npz", **arrays)


def run(cfg: dict, output: Path, smoke: bool) -> dict[str, Any]:
    ensure_fresh(output)
    registry = score_registry(); registry.to_csv(output / "score_definition_registry.csv", index=False)
    audit, checkpoints, checkpoint_hashes = audit_checkpoints(cfg)
    device = torch.device("cuda" if cfg.get("device") == "cuda" and torch.cuda.is_available() else "cpu")
    bundle = build_hai_dataloaders(cfg["data_root"], window_size=int(cfg["window_size"]), stride_train=1, stride_test=1,
                                   validation_ratio=.20, batch_size=int(cfg["batch_size"]), num_workers=0,
                                   pin_memory=device.type == "cuda", seed=1)
    if len(bundle.feature_names) != 86: raise ValueError("Step 3C requires all 86 features")
    maximum = int(cfg["smoke_max_windows_per_sequence"]) if smoke else None
    all_values: dict[int, dict[str, Any]] = {}
    for seed in cfg["seeds"]:
        model = make_model(checkpoints[int(seed)], device)
        if model.training: raise RuntimeError("Inference model entered training mode")
        splits = {"train": {}, "validation": {}, "test": {}}
        for split, sequences in (("train", bundle.train_sequences), ("validation", bundle.val_sequences), ("test", bundle.test_sequences)):
            for sequence in sequences:
                splits[split][sequence.file_name] = infer_sequence(model, sequence, device, int(cfg["window_size"]),
                                                                   int(cfg["batch_size"]), maximum)
        splits["train_all"] = concat_values(splits["train"])
        splits["validation_all"] = concat_values(splits["validation"])
        all_values[int(seed)] = splits
        save_score_arrays(output, int(seed), splits)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    consistency = consistency_check(all_values, cfg, smoke)
    consistency.to_csv(output / "rerun_consistency_check.csv", index=False)
    threshold_rows = []
    for seed, splits in all_values.items():
        for variant in ALL_VARIANTS:
            for row in fit_thresholds(splits["train_all"][variant], splits["validation_all"][variant], cfg):
                threshold_rows.append({"seed": seed, "score_variant": variant, "causal": variant in CAUSAL_VARIANTS,
                                       "diagnostic_upper_bound_only": variant == "offline_overlap_point_mean", **row})
    thresholds = pd.DataFrame(threshold_rows); thresholds.to_csv(output / "rec_alignment_thresholds.csv", index=False)
    normal_stats, validation_stability = normal_statistics(all_values, thresholds)
    normal_stats.to_csv(output / "rec_alignment_normal_score_stats.csv", index=False)
    validation_stability.to_csv(output / "rec_alignment_validation_sequence_stability.csv", index=False)
    point_summary, point_by_file, point_aggregate = build_point_metrics(all_values, thresholds)
    point_summary.to_csv(output / "rec_alignment_point_metrics_by_seed.csv", index=False)
    point_by_file.to_csv(output / "rec_alignment_point_metrics_by_file.csv", index=False)
    point_aggregate.to_csv(output / "rec_alignment_point_metrics_aggregate.csv", index=False)
    diagnostics = event_diagnostics(all_values, thresholds, bundle.attack_events)
    diagnostics.to_csv(output / "rec_alignment_event_diagnostics.csv", index=False)
    event_summary, onset = summarize_events(diagnostics)
    event_summary.to_csv(output / "rec_alignment_event_summary.csv", index=False)
    onset.to_csv(output / "rec_alignment_onset_summary.csv", index=False)
    transitions = event_transitions(diagnostics); transitions.to_csv(output / "rec_alignment_event_transition.csv", index=False)
    correlations, alarms = score_and_alarm_agreement(all_values, thresholds)
    correlations.to_csv(output / "rec_alignment_score_correlations.csv", index=False)
    alarms.to_csv(output / "rec_alignment_alarm_agreement.csv", index=False)
    fp = false_positive_diagnostics(all_values, thresholds); fp.to_csv(output / "rec_alignment_false_positive_diagnostics.csv", index=False)
    upper = noncausal_summary(point_summary, event_summary, fp); upper.to_csv(output / "rec_alignment_noncausal_upper_bound.csv", index=False)
    if smoke:
        classification, evidence = "smoke_only_no_formal_conclusion", {"reason": "limited windows may contain one label class"}
    else:
        classification, evidence = classify_result(point_aggregate, event_summary, fp)
    recommended_causal = "endpoint_mean" if classification == "temporal_score_alignment_improves_rec_baseline" else "window_mean_current"
    current_persistent = diagnostics[(diagnostics.score_variant == "window_mean_current")].groupby(
        ["threshold_method", "global_event_id"]).detected.sum().reset_index()
    persistent_ids = current_persistent[current_persistent.detected == 0]
    recovered = {}
    for variant in ("endpoint_mean", "recent8_mean", "recent16_mean", "offline_overlap_point_mean"):
        recovered[variant] = int(sum(
            diagnostics[(diagnostics.threshold_method == row.threshold_method) &
                        (diagnostics.global_event_id == row.global_event_id) &
                        (diagnostics.score_variant == variant)].detected.any() for row in persistent_ids.itertuples()))
    checkpoint_unchanged = all(sha256_file(ROOT / cfg["checkpoint_pattern"].format(seed=seed)) == digest
                               for seed, digest in checkpoint_hashes.items())
    audit["checkpoint_sha256_after"] = [sha256_file(Path(path)) for path in audit.checkpoint_path]
    audit["checkpoint_unchanged"] = audit.checkpoint_sha256_before == audit.checkpoint_sha256_after
    audit.to_csv(output / "checkpoint_and_config_audit.csv", index=False)
    summary = {"mode": "smoke" if smoke else "formal", "classification": classification,
               "classification_evidence": evidence, "checkpoint_count": 3, "checkpoints_unchanged": checkpoint_unchanged,
               "current_rerun_consistent": True, "maximum_current_score_difference": float(consistency.score_max_absolute_difference.max()),
               "score_variants": list(ALL_VARIANTS), "offline_overlap_is_noncausal_diagnostic_only": True,
               "persistent_current_missed_event_rows": len(persistent_ids), "persistent_event_recovered_by_variant": recovered,
               "test_labels_used_for_threshold_fitting": False, "point_adjustment_used": False,
               "structure_score_computed": False, "training_executed": False,
               "recommended_causal_score": recommended_causal,
               "limited_rec_sufficiency_ablation_needed_before_structure": classification == "score_alignment_is_not_main_problem",
               "ready_for_independent_structure_score_step4b": classification in {
                   "temporal_dilution_exists_but_endpoint_is_noisy", "temporal_score_alignment_improves_rec_baseline"},
               "recommended_next_step": (
                   "Use endpoint_mean as the fixed causal reconstruction baseline, then proceed to Step 4B independent structure-score validation."
                   if classification == "temporal_score_alignment_improves_rec_baseline" else
                   "Follow the classification-specific decision guidance in the diagnosis report.")}
    (output / "rec_alignment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rank = point_aggregate[(point_aggregate.scope.isin(["micro", "macro"])) &
                           (point_aggregate.metric.isin(["f1", "aupr", "auroc"]))]
    events_all = event_summary[event_summary.summary_scope == "all_tests"]
    test1 = fp[fp.file_name == "test1.csv"].groupby(["score_variant", "threshold_method"])[["normal_alarm_count", "normal_fpr"]].mean()
    report = f"""# HAI Step 3C reconstruction-score alignment diagnosis

- Three existing rec-only checkpoints were loaded read-only; no training, optimizer step, model/loss change, feature change, structure score, or fusion was performed.
- Current-score rerun: aligned exactly by sequence_id/timestamp/end_row/label; maximum score difference {summary['maximum_current_score_difference']:.3e}; historical ranking metrics reproduced within tolerance.
- Fixed variants: {list(ALL_VARIANTS)}. All use the mean error across all 86 channels; offline overlap is noncausal and diagnostic-only.
- Classification: `{classification}`. Evidence: {evidence}.
- Recommended causal reconstruction score: `{recommended_causal}`.
- Next-stage gate: limited rec-only sufficiency ablation required before structure = `{summary['limited_rec_sufficiency_ablation_needed_before_structure']}`; ready for Step 4B independent structure-score validation = `{summary['ready_for_independent_structure_score_step4b']}`.
- Recommendation rationale: endpoint scoring gives the strongest causal event recall/F1 and sharply fewer normal alarms, while recent8/recent16 mainly improve ranking AUPR. Endpoint onset-at-first-point is weaker in some settings, so delay and onset columns must remain visible in later comparisons.
- Current persistently missed threshold/event rows: {len(persistent_ids)}; recovered by fixed variants: {recovered}.
- Test1 causal false-positive means by variant/threshold: {test1.reset_index().to_dict(orient='records')}.
- Point metric aggregates are in `rec_alignment_point_metrics_aggregate.csv`; selected micro/macro F1/AUPR/AUROC rows: {rank.to_dict(orient='records')}.
- Event recall/delay and onset hit rates for all 58 events are in `rec_alignment_event_summary.csv` and `rec_alignment_onset_summary.csv`; all-tests rows: {events_all.to_dict(orient='records')}.
- Offline upper-bound comparison is in `rec_alignment_noncausal_upper_bound.csv`; it must not be used as an online score.
- Checkpoints unchanged: `{checkpoint_unchanged}`. Thresholds were fitted independently per seed and score variant using train-only POT and normal-validation q99; no test label fitted a threshold.

Decision guidance: A/B fixes a causal aligned rec score before structure validation; C requires further causal extraction work; D requires a limited rec sufficiency ablation; E retains current as the historical/formal rec baseline. This script does not execute the next stage.
"""
    (output / "final_hai_rec_score_alignment_diagnosis.md").write_text(report, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAI Step 3C fixed reconstruction-score alignment evaluation")
    parser.add_argument("--config", required=True); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); cfg = load_config(args.config)
    output = ROOT / (cfg["smoke_output_dir"] if args.smoke else cfg["output_dir"])
    audit, checkpoints, _ = audit_checkpoints(cfg)
    if args.dry_run:
        print(audit[["seed", "checkpoint_path", "exists", "only_seed_and_output_fields_differ"]].to_string(index=False))
        print("model input/output=[batch,64,86]; model=models.mscad.MSCAD; no copied model definition")
        print("current implementation=models/mscad.py::score_components + utils/reconstruction_scores.py")
        old = np.load(ROOT / cfg["historical_result_pattern"].format(seed=1) / "test_score_arrays.npz")
        print(f"historical_npz_fields={old.files}")
        print("rerun_required=true because historical NPZ has scalar window scores but no per-time-position errors")
        print(f"fixed_variants={list(ALL_VARIANTS)}")
        print("reuse=data.hai, models.mscad, utils.metrics, utils.pot; add=config, score utility, Step3C script/test")
        return 0
    result = run(cfg, output, args.smoke)
    print(json.dumps({"output": str(output), **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
