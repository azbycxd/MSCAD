from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat2026_structure import (  # noqa: E402
    RelationReference,
    aggregate_deviation,
    circular_relation_injection,
    fit_robust_reference,
    relation_deviation,
    rolling_pair_correlations,
    screen_train_relations,
    select_relation_sets,
    window_batch_pair_correlations,
)
from data.swat2026_training import (  # noqa: E402
    SWaTWindowDataset,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    prepare_swat_training_data,
    sha256_file,
)
from scripts.swat2026.train_eval_rec_baseline import (  # noqa: E402
    build_device,
    make_model,
    point_metrics,
    reconstruction_score,
    threshold_from_validation,
)
from utils.config_io import load_config  # noqa: E402
from utils.project_paths import (  # noqa: E402
    resolve_checkpoint_path,
    resolve_data_path,
    resolve_output_path,
    resolve_repo_path,
)


REQUIRED_M1 = (
    "batch_m1_readiness.json", "rec_scores_train.csv", "rec_scores_validation.csv", "rec_scores_test.csv",
    "scaler_metadata.json", "prepared_data_manifest.json", "baseline_metrics.json", "checkpoints/best_seed1.pt",
)


def resolve(value: str | Path) -> Path:
    return resolve_repo_path(value)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def validate_m1_artifacts(cfg: dict[str, Any]) -> dict[str, Any]:
    m1 = resolve_output_path(cfg["data"]["m1_dir"])
    missing = [name for name in REQUIRED_M1 if not (m1 / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Required M1 artifacts are missing: {missing}")
    readiness = read_json(m1 / "batch_m1_readiness.json")
    checkpoint = resolve_checkpoint_path(cfg["data"]["checkpoint"])
    actual_hash = sha256_file(checkpoint)
    if readiness["classification"] != "A. single_seed_rec_baseline_complete":
        raise ValueError("M1 readiness is not A. single_seed_rec_baseline_complete")
    if actual_hash != cfg["data"]["checkpoint_sha256"]:
        raise ValueError("M1 checkpoint SHA-256 mismatch; rerun M1 instead of M2")
    return {"m1_readiness": readiness["classification"], "checkpoint_sha256": actual_hash,
            "required_artifacts_present": True, "model_retrained": False}


def load_frozen_model(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(resolve_checkpoint_path(cfg["data"]["checkpoint"]), map_location=device, weights_only=False)
    if checkpoint.get("old_checkpoint_loaded") is not False or checkpoint.get("test_used") is not False:
        raise ValueError("M1 checkpoint provenance is invalid")
    m1_cfg = checkpoint["config"]
    model = make_model(m1_cfg, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _rows(prepared, split: str) -> pd.DataFrame:
    return prepared.valid_window_rows(split)


@torch.no_grad()
def frozen_rec_errors(model: torch.nn.Module, prepared, split: str, device: torch.device,
                      limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    dataset = SWaTWindowDataset(prepared, split)
    count = len(dataset) if limit is None else min(limit, len(dataset))
    scores, channels = [], []
    for start in range(0, count, 128):
        windows = torch.stack([dataset[index][0] for index in range(start, min(start + 128, count))]).to(device)
        estimate = model(windows)
        error = (estimate - windows).abs()
        scores.append(error.mean(dim=(1, 2)).cpu().numpy())
        channels.append(error.mean(dim=1).cpu().numpy())
    return np.concatenate(scores), np.concatenate(channels)


@torch.no_grad()
def injected_rec_scores(model: torch.nn.Module, windows: np.ndarray, device: torch.device) -> np.ndarray:
    values = []
    for start in range(0, len(windows), 128):
        batch = torch.from_numpy(windows[start : start + 128].astype(np.float32)).to(device)
        values.append(reconstruction_score(batch, model(batch)).cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def distribution_row(segment: str, values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "segment": segment, "count": len(x), "mean": float(np.mean(x)), "median": float(np.median(x)),
        "std": float(np.std(x)), "q90": float(np.quantile(x, .90)), "q95": float(np.quantile(x, .95)),
        "q99": float(np.quantile(x, .99)), "min": float(np.min(x)), "max": float(np.max(x)),
    }


def contiguous_normal_intervals(test: pd.DataFrame, schedule: pd.DataFrame, score_column: str,
                                threshold: float) -> pd.DataFrame:
    normal = test[test.point_label.eq(0)].copy()
    groups = normal.timestamp.diff().dt.total_seconds().ne(1).cumsum()
    first = pd.Timestamp(f"{schedule.iloc[0].date} {schedule.iloc[0].start_time}")
    last = pd.Timestamp(f"{schedule.iloc[-1].date} {schedule.iloc[-1].end_time}")
    rows = []
    for _, frame in normal.groupby(groups):
        start, end = frame.timestamp.iloc[0], frame.timestamp.iloc[-1] + pd.Timedelta(seconds=1)
        kind = "pre_attack_normal" if end <= first else ("trailing_normal" if start >= last else "inter_attack_normal")
        alarm = frame[score_column].to_numpy() > threshold
        rows.append({"interval_type": kind, "start_time": start.isoformat(), "end_time_exclusive": end.isoformat(),
                     "normal_points": len(frame), "false_positive_points": int(alarm.sum()),
                     "false_positive_rate": float(alarm.mean()), "mean_score": float(frame[score_column].mean())})
    return pd.DataFrame(rows)


def event_diagnostics(test: pd.DataFrame, schedule: pd.DataFrame, score_column: str, threshold: float) -> pd.DataFrame:
    rows = []
    test = test.sort_values("timestamp").reset_index(drop=True)
    for _, attack in schedule.sort_values("attack_id").iterrows():
        start = pd.Timestamp(f"{attack.date} {attack.start_time}")
        end = pd.Timestamp(f"{attack.date} {attack.end_time}")
        current = test[(test.timestamp >= start) & (test.timestamp < end)]
        alarms = current[score_column].to_numpy() > threshold
        detected = bool(alarms.any())
        first_alarm = current.timestamp.iloc[int(np.flatnonzero(alarms)[0])] if detected else pd.NaT
        previous = test[test.timestamp < start].tail(1)
        preexisting = bool(len(previous) and float(previous.iloc[0][score_column]) > threshold)
        rows.append({
            "attack_id": int(attack.attack_id), "attack_name": attack.attack_name, "official_targets": attack.target,
            "start_time": start.isoformat(), "end_time_exclusive": end.isoformat(), "detected": detected,
            "first_detection_timestamp": first_alarm.isoformat() if detected else "",
            "first_detection_delay_seconds": float((first_alarm - start).total_seconds()) if detected else "",
            "preexisting_alarm": preexisting, "alarm_points": int(alarms.sum()), "attack_points": len(current),
        })
    return pd.DataFrame(rows)


def rec_diagnostics(cfg: dict[str, Any], prepared, model: torch.nn.Module, device: torch.device,
                    output: Path, smoke: bool) -> dict[str, Any]:
    directory = output / "rec_diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    m1 = resolve(cfg["data"]["m1_dir"])
    train = pd.read_csv(m1 / "rec_scores_train.csv")
    validation = pd.read_csv(m1 / "rec_scores_validation.csv")
    test = pd.read_csv(m1 / "rec_scores_test.csv", dtype={"attack_id": str})
    for frame in (train, validation, test):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    metrics = read_json(m1 / "baseline_metrics.json")
    threshold = float(metrics["threshold"])
    schedule = pd.read_csv(resolve(cfg["data"]["n2_dir"]) / "approved_attack_schedule.csv", encoding="utf-8-sig")
    rows = [distribution_row("train", train.rec_score), distribution_row("validation", validation.rec_score),
            distribution_row("test_all", test.rec_score),
            distribution_row("test_normal", test.loc[test.point_label.eq(0), "rec_score"]),
            distribution_row("test_attack", test.loc[test.point_label.eq(1), "rec_score"]),
            distribution_row("trailing_1200_seconds", test.tail(1200).rec_score)]
    for _, attack in schedule.iterrows():
        start, end = pd.Timestamp(f"{attack.date} {attack.start_time}"), pd.Timestamp(f"{attack.date} {attack.end_time}")
        rows.append(distribution_row(f"attack_{int(attack.attack_id)}", test.loc[(test.timestamp >= start) & (test.timestamp < end), "rec_score"]))
    pd.DataFrame(rows).to_csv(directory / "rec_score_distribution_by_segment.csv", index=False)
    binned = test.set_index("timestamp").resample("15min").agg(
        count=("rec_score", "size"), mean=("rec_score", "mean"), median=("rec_score", "median"),
        std=("rec_score", "std"), q90=("rec_score", lambda x: x.quantile(.90)),
        q95=("rec_score", lambda x: x.quantile(.95)), q99=("rec_score", lambda x: x.quantile(.99)),
        attack_points=("point_label", "sum"),
    ).reset_index()
    binned.to_csv(directory / "rec_score_time_bins.csv", index=False)
    event = event_diagnostics(test, schedule, "rec_score", threshold)
    event.to_csv(directory / "rec_event_diagnostics.csv", index=False)
    fp = contiguous_normal_intervals(test, schedule, "rec_score", threshold)
    fp.to_csv(directory / "rec_false_positive_diagnostics.csv", index=False)

    limit = 256 if smoke else None
    channel_rows = []
    channel_cache = {}
    for split in ("train", "validation", "test"):
        _, channel_error = frozen_rec_errors(model, prepared, split, device, limit)
        channel_cache[split] = channel_error
        labels = _rows(prepared, split).iloc[: len(channel_error)]
        segments = {split: np.ones(len(channel_error), dtype=bool)}
        if split == "test":
            segments = {"test_normal": labels.point_label.astype(int).to_numpy() == 0,
                        "test_attack": labels.point_label.astype(int).to_numpy() == 1,
                        "trailing_1200_seconds": np.arange(len(labels)) >= max(0, len(labels) - 1200)}
        for segment, mask in segments.items():
            if not np.asarray(mask, dtype=bool).any():
                continue
            for index, feature in enumerate(prepared.feature_names):
                values = channel_error[mask, index]
                channel_rows.append({"segment": segment, "feature": feature, "count": len(values),
                                     "mean_abs_error": float(values.mean()), "median_abs_error": float(np.median(values)),
                                     "q99_abs_error": float(np.quantile(values, .99)),
                                     "contribution_share": float(values.mean() / max(channel_error[mask].mean(axis=0).sum(), 1e-12))})
    pd.DataFrame(channel_rows).to_csv(directory / "rec_channel_attribution.csv", index=False)
    train_mask = ((prepared.timestamps >= TRAIN_START) & (prepared.timestamps < TRAIN_END)) & prepared.valid_points
    test_mask = (prepared.timestamps >= TEST_START) & (prepared.timestamps < TEST_END)
    shift_rows = []
    for index, feature in enumerate(prepared.feature_names):
        train_values = prepared.raw_features[train_mask, index]
        test_values = prepared.raw_features[test_mask, index]
        q01, q99 = np.quantile(train_values, [.01, .99])
        lower, upper = np.min(train_values), np.max(train_values)
        shift_rows.append({"feature": feature, "train_q01": q01, "train_q99": q99,
                           "test_outside_train_q01_q99_ratio": float(((test_values < q01) | (test_values > q99)).mean()),
                           "test_outside_train_min_max_ratio": float(((test_values < lower) | (test_values > upper)).mean()),
                           "test_mean_shift_in_train_std": float((test_values.mean() - train_values.mean()) / max(train_values.std(), 1e-12))})
    shift = pd.DataFrame(shift_rows).sort_values("test_outside_train_q01_q99_ratio", ascending=False)
    shift.to_csv(directory / "feature_distribution_shift.csv", index=False)
    rho = float(pd.Series(test.rec_score).corr(pd.Series(np.arange(len(test))), method="spearman"))
    from utils.metrics import binary_metrics
    labels = test.point_label.to_numpy(dtype=int)
    scores = test.rec_score.to_numpy(float)
    negated = binary_metrics(labels, (scores < -threshold).astype(int), -scores)
    trailing = test.tail(1200)
    trailing_fp = int((trailing.rec_score > threshold).sum())
    normal_median, attack_median = test.loc[test.point_label.eq(0), "rec_score"].median(), test.loc[test.point_label.eq(1), "rec_score"].median()
    labels_diag = []
    if abs(rho) >= .30 or trailing_fp == 1200:
        labels_diag.append("A. global_operating_regime_shift")
    if float(shift.test_outside_train_q01_q99_ratio.mean()) >= .10:
        labels_diag.append("B. scaler_extrapolation")
    if float(shift.head(5).test_outside_train_q01_q99_ratio.mean()) > 2 * max(float(shift.tail(5).test_outside_train_q01_q99_ratio.mean()), 1e-12):
        labels_diag.append("C. channel_specific_drift")
    if float((test.loc[test.point_label.eq(0), "rec_score"] > threshold).mean()) > .10:
        labels_diag.append("D. threshold_miscalibration")
    if not labels_diag:
        labels_diag = ["F. mixed_or_unclear"]
    summary = {
        "primary_diagnoses": labels_diag, "threshold": threshold, "test_time_spearman": rho,
        "test_normal_median": float(normal_median), "test_attack_median": float(attack_median),
        "attack_normal_ranking_reversed": bool(attack_median < normal_median),
        "negated_score_auroc_diagnostic_only": negated["auroc"],
        "trailing_1200_alarm_points": trailing_fp,
        "trailing_explanation": "late-session operating-regime/channel drift relative to train-only scaling and validation threshold" if trailing_fp == 1200 else "partial late-session threshold exceedance",
        "preexisting_alarm_event_count": int(event.preexisting_alarm.sum()),
        "model_retrained": False, "primary_score_changed": False,
    }
    write_json(directory / "rec_diagnosis_summary.json", summary)
    return summary


def relation_set_payload(sets: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> dict[str, Any]:
    payload = {"selection_scope": "train_only", "test_labels_used": False, "old_relation_sets_used": False,
               "primary": "strict_set + max + window128", "expanded_is_diagnostic_only": True}
    for name, frame in sets.items():
        payload[name] = frame[["pair_name", "channel_i", "channel_j", "channel_i_index", "channel_j_index"]].to_dict("records")
    payload["thresholds"] = cfg["relations"]
    payload["insufficient_strict_relations"] = len(sets["strict_set"]) < 3
    return payload


def score_relation_set(prepared, rows: pd.DataFrame, relations: pd.DataFrame, reference: RelationReference,
                       window_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ends = rows.end_index.to_numpy(dtype=np.int64)
    correlations, _ = rolling_pair_correlations(prepared.scaled_features, ends, relations, window_size, prepared.valid_points)
    deviation = relation_deviation(correlations, reference)
    maximum, top3 = aggregate_deviation(deviation)
    return maximum, top3, deviation


def fit_set_reference(prepared, relations: pd.DataFrame, window_size: int, floor: float) -> tuple[RelationReference, np.ndarray]:
    train_rows = _rows(prepared, "train")
    ends = train_rows.end_index.to_numpy(dtype=np.int64)
    correlations, _ = rolling_pair_correlations(prepared.scaled_features, ends, relations, window_size, prepared.valid_points)
    return fit_robust_reference(correlations, floor), correlations


def metrics_for_control(normal: np.ndarray, anomaly: np.ndarray, quantile: float) -> dict[str, Any]:
    labels = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(anomaly), dtype=int)])
    scores = np.concatenate([normal, anomaly])
    threshold = threshold_from_validation(normal, quantile)
    return {"threshold": threshold, **point_metrics(labels, scores, threshold)}


def controlled_analysis(cfg: dict[str, Any], prepared, model, device, sets, references,
                        validation_scores: dict[str, np.ndarray], output: Path, smoke: bool) -> dict[str, Any]:
    directory = output / "controlled"; directory.mkdir(parents=True, exist_ok=True)
    rows = _rows(prepared, "validation")
    if smoke:
        rows = rows.head(min(128, len(rows))).copy()
    windows = np.stack([prepared.scaled_features[int(row.start_index): int(row.end_index) + 1]
                        for row in rows.itertuples(index=False)]).astype(np.float32)
    injected = windows.copy()
    strict = sets["strict_set"]
    shifts = list(cfg["controlled"]["shifts"])
    manifest = []
    for index in range(len(injected)):
        relation_index = index % len(strict)
        relation = strict.iloc[relation_index]
        shift = shifts[index % len(shifts)]
        channel = int(relation.channel_j_index)
        original = injected[index, :, channel].copy()
        injected[index] = circular_relation_injection(injected[index], channel, shift)
        changed = injected[index, :, channel]
        manifest.append({
            "injection_id": index, "source_validation_window_id": int(rows.iloc[index].window_id),
            "score_timestamp": rows.iloc[index].score_timestamp, "relation": relation.pair_name,
            "shifted_channel": relation.channel_j, "shift_seconds": shift, "boundary_handling": "circular_within_window",
            "seed": int(cfg["controlled"]["seed"]), "value_multiset_preserved": bool(np.array_equal(np.sort(original), np.sort(changed))),
            "mean_abs_change": abs(float(original.mean() - changed.mean())),
            "std_abs_change": abs(float(original.std() - changed.std())),
            "min_abs_change": abs(float(original.min() - changed.min())),
            "max_abs_change": abs(float(original.max() - changed.max())), "test_used": False,
        })
    pd.DataFrame(manifest).to_csv(directory / "controlled_injection_manifest.csv", index=False)
    original_rec = validation_scores["rec_score"][:len(rows)]
    injected_rec = injected_rec_scores(model, injected, device)
    score_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {"rec_score": (original_rec, injected_rec)}
    for set_name, output_name in (("strict_set", "strict_max_128"), ("diverse_set", "diverse_max_128"),
                                  ("expanded_set", "expanded_max_128")):
        relation = sets[set_name]
        reference = references[(set_name, 128)]
        normal_corr = window_batch_pair_correlations(windows, relation)
        injected_corr = window_batch_pair_correlations(injected, relation)
        normal_max, _ = aggregate_deviation(relation_deviation(normal_corr, reference))
        injected_max, _ = aggregate_deviation(relation_deviation(injected_corr, reference))
        score_arrays[output_name] = (normal_max, injected_max)
    relation = sets["strict_set"]
    reference64 = references[("strict_set", 64)]
    normal64 = window_batch_pair_correlations(windows[:, -64:, :], relation)
    injected64 = window_batch_pair_correlations(injected[:, -64:, :], relation)
    score_arrays["strict_max_64"] = (
        aggregate_deviation(relation_deviation(normal64, reference64))[0],
        aggregate_deviation(relation_deviation(injected64, reference64))[0],
    )
    metric_rows = []
    metric_lookup = {}
    for name, (normal, anomaly) in score_arrays.items():
        metric = metrics_for_control(normal, anomaly, float(cfg["controlled"]["validation_quantile"]))
        metric_lookup[name] = metric
        metric_rows.append({"score_name": name, **{key: value for key, value in metric.items() if key != "confusion_matrix"},
                            **{f"cm_{key}": value for key, value in metric["confusion_matrix"].items()}})
    pd.DataFrame(metric_rows).to_csv(directory / "controlled_metrics.csv", index=False)
    rec_normal, rec_anomaly = score_arrays["rec_score"]
    struct_normal, struct_anomaly = score_arrays["strict_max_128"]
    rec_threshold, struct_threshold = metric_lookup["rec_score"]["threshold"], metric_lookup["strict_max_128"]["threshold"]
    rec_alarm, struct_alarm = rec_anomaly > rec_threshold, struct_anomaly > struct_threshold
    comp = {
        "injected_count": len(rows), "structure_detected_rec_missed_count": int((struct_alarm & ~rec_alarm).sum()),
        "structure_detected_rec_missed_ratio": float((struct_alarm & ~rec_alarm).mean()),
        "rec_detected_structure_missed_count": int((rec_alarm & ~struct_alarm).sum()),
        "rec_detected_structure_missed_ratio": float((rec_alarm & ~struct_alarm).mean()),
        "both_detected_count": int((rec_alarm & struct_alarm).sum()), "both_missed_count": int((~rec_alarm & ~struct_alarm).sum()),
        "injected_score_pearson": float(pd.Series(rec_anomaly).corr(pd.Series(struct_anomaly), method="pearson")),
        "injected_score_spearman": float(pd.Series(rec_anomaly).corr(pd.Series(struct_anomaly), method="spearman")),
    }
    pd.DataFrame([comp]).to_csv(directory / "controlled_complementarity.csv", index=False)
    quality = pd.DataFrame(manifest)
    preserved = bool(quality.value_multiset_preserved.all() and
                     quality[["mean_abs_change", "std_abs_change", "min_abs_change", "max_abs_change"]].to_numpy().max() < 1e-6)
    auc = float(metric_lookup["strict_max_128"]["auroc"])
    rec_auc = float(metric_lookup["rec_score"]["auroc"])
    if not preserved:
        status = "D. invalid_injection"
    elif auc >= float(cfg["controlled"]["validated_auroc_min"]) and auc > rec_auc:
        status = "A. independent_structure_mechanism_validated"
    elif auc >= float(cfg["controlled"]["weak_auroc_min"]):
        status = "B. weak_but_observable"
    else:
        status = "C. no_structure_sensitivity"
    summary = {"mechanism_status": status, "injection_count": len(rows), "marginal_statistics_preserved": preserved,
               "strict_max_128_auroc": auc, "rec_score_auroc": rec_auc, "test_used": False,
               "relation_selection_changed": False, **comp}
    write_json(directory / "controlled_summary.json", summary)
    return summary


def structure_event_evaluation(test: pd.DataFrame, schedule: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return event_diagnostics(test, schedule, "strict_max_128", threshold)


def target_alignment(events: pd.DataFrame, attribution: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, attack in schedule.sort_values("attack_id").iterrows():
        event = events[events.attack_id.eq(int(attack.attack_id))].iloc[0]
        start, end = pd.Timestamp(event.start_time), pd.Timestamp(event.end_time_exclusive)
        subset = attribution[(attribution.timestamp >= start) & (attribution.timestamp < end)]
        top = subset.sort_values("top_relation_score", ascending=False).iloc[0]
        channels = [top.channel_i, top.channel_j]
        targets = [value.strip() for value in re.split(r"[,–-]", str(attack.target)) if value.strip()]
        bases = [str(value).replace(".Pv", "") for value in channels]
        exact = any(target in bases for target in targets)
        target_stages = {match.group(1) for target in targets if (match := re.search(r"(\d)", target))}
        channel_stages = {match.group(1) for channel in bases if (match := re.search(r"(\d)", channel))}
        related = bool(not exact and target_stages & channel_stages)
        rows.append({"attack_id": int(attack.attack_id), "official_targets": attack.target,
                     "top_relation": top.top_relation, "relation_channels": f"{top.channel_i},{top.channel_j}",
                     "exact_target_match": exact, "same_stage_or_related_match": related,
                     "no_obvious_match": not exact and not related,
                     "interpretation_note": "diagnostic alignment only; no causal claim and not used for relation selection"})
    return pd.DataFrame(rows)


def run(cfg: dict[str, Any], dry_run: bool = False, smoke: bool = False) -> dict[str, Any]:
    if int(cfg["seed"]) != 1:
        raise ValueError("Batch-M2 authorizes seed=1 only")
    artifact_audit = validate_m1_artifacts(cfg)
    base_output = resolve_output_path(cfg["output"]["dir"])
    output = base_output / ("dry_run" if dry_run else "smoke") if (dry_run or smoke) else base_output
    output.mkdir(parents=True, exist_ok=True)
    prepared = prepare_swat_training_data(resolve_data_path(cfg["data"]["data_path"]), resolve_output_path(cfg["data"]["n2_dir"]), None, 128)
    scaler_m1 = read_json(resolve(cfg["data"]["m1_dir"]) / "scaler_metadata.json")
    if scaler_m1["feature_order"] != prepared.feature_names or scaler_m1["source_csv_sha256"] != prepared.source_sha256:
        raise ValueError("M1/N2 feature or source contract changed")
    relations_cfg = cfg["relations"]
    train_mask = (prepared.timestamps >= TRAIN_START) & (prepared.timestamps < TRAIN_END)
    train_values = prepared.scaled_features[train_mask]
    train_validity = prepared.valid_points[train_mask]
    diagnostics, channels, _ = screen_train_relations(
        train_values, train_validity, prepared.feature_names, scaler_m1["zero_variance_features"], 128, relations_cfg
    )
    sets = select_relation_sets(diagnostics, int(relations_cfg["strict_max_pairs"]),
                                int(relations_cfg["diverse_max_pairs"]), int(relations_cfg["diverse_max_degree"]))
    screening = output / "relation_screening"; screening.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(screening / "all_pair_diagnostics.csv", index=False)
    channels.to_csv(screening / "channel_diagnostics.csv", index=False)
    write_json(screening / "relation_sets.json", relation_set_payload(sets, cfg))
    summary_rows = []
    for name, frame in sets.items():
        degree = Counter(frame.channel_i.tolist() + frame.channel_j.tolist())
        summary_rows.append({"relation_set": name, "pair_count": len(frame),
                             "involved_channel_count": len(degree), "maximum_channel_degree": max(degree.values(), default=0),
                             "primary": name == "strict_set", "diagnostic_only": name == "expanded_set"})
    pd.DataFrame(summary_rows).to_csv(screening / "relation_set_summary.csv", index=False)
    strict_count = len(sets["strict_set"])
    if dry_run:
        result = {**artifact_audit, "dry_run": True, "legal_candidate_channels": int((~channels.zero_variance_train).sum()),
                  "all_pairs": len(diagnostics), "strict_count": strict_count,
                  "diverse_count": len(sets["diverse_set"]), "expanded_count": len(sets["expanded_set"]),
                  "would_retrain_model": False, "would_fuse_scores": False}
        write_json(output / "dry_run_summary.json", result); print(json.dumps(result, indent=2)); return result
    device = build_device("cuda")
    model = load_frozen_model(cfg, device)
    rec_summary = rec_diagnostics(cfg, prepared, model, device, output, smoke)
    if strict_count < 3:
        readiness = {"classification": "C. insufficient_strict_relations", "strict_relation_count": strict_count,
                     "reconstruction_diagnosis_complete": True, "rec_diagnosis": rec_summary,
                     "controlled_validation_complete": False, "real_structure_evaluation_complete": False,
                     "thresholds_relaxed_after_test": False, "model_retrained": False, "fusion_computed": False}
        write_json(output / "batch_m2_readiness.json", readiness)
        print(json.dumps(readiness, indent=2)); return readiness

    floor = float(relations_cfg["fisher_scale_floor"])
    references: dict[tuple[str, int], RelationReference] = {}
    for set_name in sets:
        references[(set_name, 128)], _ = fit_set_reference(prepared, sets[set_name], 128, floor)
    references[("strict_set", 64)], _ = fit_set_reference(prepared, sets["strict_set"], 64, floor)
    score_frames: dict[str, pd.DataFrame] = {}
    score_arrays: dict[str, dict[str, np.ndarray]] = {}
    attribution_test = None
    for split in ("train", "validation", "test"):
        rows = _rows(prepared, split)
        current: dict[str, np.ndarray] = {}
        strict_max, strict_top3, strict_deviation = score_relation_set(prepared, rows, sets["strict_set"], references[("strict_set", 128)], 128)
        current["strict_max_128"], current["strict_top3_mean"] = strict_max, strict_top3
        current["diverse_max_128"] = score_relation_set(prepared, rows, sets["diverse_set"], references[("diverse_set", 128)], 128)[0]
        current["expanded_max_128"] = score_relation_set(prepared, rows, sets["expanded_set"], references[("expanded_set", 128)], 128)[0]
        current["strict_max_window64"] = score_relation_set(prepared, rows, sets["strict_set"], references[("strict_set", 64)], 64)[0]
        frame = pd.DataFrame({"timestamp": pd.to_datetime(rows.score_timestamp), **current})
        if split == "test":
            frame["point_label"] = rows.point_label.astype(int).to_numpy()
            frame["attack_id"] = rows.attack_id.to_numpy()
            frame["phase"] = rows.phase.to_numpy(); frame["evaluation_eligible"] = rows.evaluation_eligible.astype(bool).to_numpy()
            top_index = np.argmax(strict_deviation, axis=1)
            strict = sets["strict_set"].reset_index(drop=True)
            attribution_test = pd.DataFrame({
                "timestamp": frame.timestamp, "top_relation": strict.iloc[top_index].pair_name.to_numpy(),
                "channel_i": strict.iloc[top_index].channel_i.to_numpy(), "channel_j": strict.iloc[top_index].channel_j.to_numpy(),
                "top_relation_score": strict_deviation[np.arange(len(top_index)), top_index],
            })
        frame.to_csv(output / f"structure_scores_{split}.csv", index=False)
        score_frames[split] = frame; score_arrays[split] = current
    assert attribution_test is not None
    attribution_test.to_csv(output / "relation_attribution_test.csv", index=False)
    m1 = resolve(cfg["data"]["m1_dir"])
    rec_validation = pd.read_csv(m1 / "rec_scores_validation.csv").rec_score.to_numpy(float)
    controlled = controlled_analysis(cfg, prepared, model, device, sets, references,
                                     {"rec_score": rec_validation}, output, smoke)
    if smoke:
        result = {**artifact_audit, "smoke": True, "strict_count": strict_count,
                  "controlled": controlled, "model_retrained": False, "formal_test_evaluation": False}
        write_json(output / "smoke_summary.json", result); print(json.dumps(result, indent=2)); return result

    real_dir = output / "real_test"; real_dir.mkdir(parents=True, exist_ok=True)
    validation_threshold = threshold_from_validation(score_arrays["validation"]["strict_max_128"],
                                                     float(cfg["evaluation"]["validation_quantile"]))
    test = score_frames["test"]
    labels, scores = test.point_label.to_numpy(int), test.strict_max_128.to_numpy(float)
    metrics = point_metrics(labels, scores, validation_threshold)
    metrics.update({"threshold": validation_threshold, "threshold_source": "validation_normal_q99",
                    "primary_score": "strict_max_128", "point_adjustment": False,
                    "test_relation_selection": False, "test_threshold_tuning": False})
    write_json(real_dir / "structure_metrics.json", metrics)
    schedule = pd.read_csv(resolve(cfg["data"]["n2_dir"]) / "approved_attack_schedule.csv", encoding="utf-8-sig")
    events = structure_event_evaluation(test, schedule, validation_threshold)
    events.to_csv(real_dir / "structure_event_summary.csv", index=False)
    fp = contiguous_normal_intervals(test, schedule, "strict_max_128", validation_threshold)
    fp.to_csv(real_dir / "structure_false_positive_summary.csv", index=False)
    top_events = []
    for _, event in events.iterrows():
        subset = attribution_test[(attribution_test.timestamp >= pd.Timestamp(event.start_time)) &
                                  (attribution_test.timestamp < pd.Timestamp(event.end_time_exclusive))]
        top = subset.sort_values("top_relation_score", ascending=False).iloc[0]
        top_events.append({"attack_id": event.attack_id, "top_relation": top.top_relation,
                           "channel_i": top.channel_i, "channel_j": top.channel_j, "peak_score": top.top_relation_score})
    pd.DataFrame(top_events).to_csv(real_dir / "top_relation_by_event.csv", index=False)
    alignment = target_alignment(events, attribution_test, schedule)
    alignment.to_csv(real_dir / "relation_target_alignment.csv", index=False)

    comp_dir = output / "complementarity"; comp_dir.mkdir(parents=True, exist_ok=True)
    rec_test = pd.read_csv(m1 / "rec_scores_test.csv"); rec_test["timestamp"] = pd.to_datetime(rec_test.timestamp)
    if not rec_test.timestamp.equals(test.timestamp):
        raise ValueError("M1 reconstruction and M2 structure timestamps do not align exactly")
    correlation_rows = []
    for name in ("strict_max_128", "strict_top3_mean", "diverse_max_128"):
        correlation_rows.append({"structure_score": name,
                                 "pearson": float(rec_test.rec_score.corr(test[name], method="pearson")),
                                 "spearman": float(rec_test.rec_score.corr(test[name], method="spearman"))})
    pd.DataFrame(correlation_rows).to_csv(comp_dir / "score_correlations.csv", index=False)
    rec_threshold = float(read_json(m1 / "baseline_metrics.json")["threshold"])
    rec_alarm = rec_test.rec_score.to_numpy(float) > rec_threshold
    struct_alarm = scores > validation_threshold
    attack = labels == 1; normal = ~attack
    point = {
        "attack_points": int(attack.sum()), "rec_hit_structure_miss": int((attack & rec_alarm & ~struct_alarm).sum()),
        "structure_hit_rec_miss": int((attack & struct_alarm & ~rec_alarm).sum()),
        "both_hit": int((attack & rec_alarm & struct_alarm).sum()), "both_miss": int((attack & ~rec_alarm & ~struct_alarm).sum()),
        "structure_supplement_of_rec_missed_attack_points": float((attack & struct_alarm & ~rec_alarm).sum() / max((attack & ~rec_alarm).sum(), 1)),
        "rec_false_positives": int((normal & rec_alarm).sum()), "structure_false_positives": int((normal & struct_alarm).sum()),
        "false_positive_overlap": int((normal & rec_alarm & struct_alarm).sum()),
    }
    pd.DataFrame([point]).to_csv(comp_dir / "point_level_complementarity.csv", index=False)
    rec_events = event_diagnostics(rec_test, schedule, "rec_score", rec_threshold)
    event_comp = events[["attack_id", "detected"]].rename(columns={"detected": "structure_detected"}).merge(
        rec_events[["attack_id", "detected"]].rename(columns={"detected": "rec_detected"}), on="attack_id")
    event_comp["structure_adds_event"] = event_comp.structure_detected & ~event_comp.rec_detected
    event_comp["rec_adds_event"] = event_comp.rec_detected & ~event_comp.structure_detected
    event_comp.to_csv(comp_dir / "event_level_complementarity.csv", index=False)
    comp_summary = {
        **point, "structure_added_events": int(event_comp.structure_adds_event.sum()),
        "rec_added_events": int(event_comp.rec_adds_event.sum()),
        "both_detect_events": int((event_comp.rec_detected & event_comp.structure_detected).sum()),
        "both_miss_events": int((~event_comp.rec_detected & ~event_comp.structure_detected).sum()),
        "controlled_relation_complementarity": controlled,
        "low_correlation_alone_claims_complementarity": False,
        "false_positive_effect": "reduces" if point["structure_false_positives"] < point["rec_false_positives"] else
                                 ("increases" if point["structure_false_positives"] > point["rec_false_positives"] else "transfers"),
        "fusion_computed": False,
    }
    write_json(comp_dir / "complementarity_summary.json", comp_summary)
    mechanism = controlled["mechanism_status"]
    classification = ("A. structure_mechanism_and_real_evaluation_complete"
                      if mechanism == "A. independent_structure_mechanism_validated"
                      else "B. real_evaluation_complete_mechanism_weak")
    readiness = {
        "classification": classification, "m1_artifacts_valid": True, "checkpoint_hash_matches": True,
        "relations_selected_train_only": True, "validation_threshold_fixed": True,
        "test_used_final_evaluation_only": True, "controlled_validation_complete": True,
        "real_test_complete": True, "complementarity_complete": True, "model_retrained": False,
        "fusion_computed": False, "multiple_seeds_run": False, "test_driven_relation_changes": False,
        "mechanism_status": mechanism, "strict_relation_count": strict_count,
    }
    write_json(output / "batch_m2_readiness.json", readiness)
    write_json(output / "resolved_config.json", cfg)
    result = {"readiness": classification, "rec_diagnosis": rec_summary, "controlled": controlled,
              "real_metrics": metrics, "real_events_detected": int(events.detected.sum()),
              "strict_count": strict_count, "diverse_count": len(sets["diverse_set"]),
              "expanded_count": len(sets["expanded_set"]), "model_retrained": False, "fusion_computed": False}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str)); return result


def main() -> int:
    parser = argparse.ArgumentParser(description="SWaT Batch-M2 independent structure analysis")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.smoke:
        raise ValueError("--dry-run and --smoke are mutually exclusive")
    return 0 if run(load_config(args.config), args.dry_run, args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
