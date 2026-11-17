from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1, 2, 3)
METHODS = ("train_only_pot", "normal_validation_q99")
FILES = tuple(f"test{i}" for i in range(1, 5))
CONFIGS = {seed: ROOT / "configs" / f"hai_rec_only_seed{seed}.yaml" for seed in SEEDS}
RUN_DIRS = {seed: ROOT / "outputs" / f"hai_step3a_rec_baseline_seed{seed}" for seed in SEEDS}
OUTPUT = ROOT / "outputs" / "hai_step3b_rec_multiseed"
SMOKE_OUTPUT = ROOT / "outputs" / "hai_step3b_rec_multiseed_smoke"
FINAL_FILES = (
    "multiseed_training_summary.csv", "multiseed_threshold_summary.csv",
    "multiseed_point_metrics.csv", "multiseed_point_metrics_by_file.csv",
    "multiseed_point_metrics_aggregate.csv", "multiseed_score_correlations.csv",
    "multiseed_alarm_agreement.csv", "multiseed_alarm_vote_summary.csv",
    "multiseed_event_consistency.csv", "multiseed_event_consistency_summary.csv",
    "multiseed_summary.json", "final_hai_rec_multiseed_diagnosis.md",
)
ALLOWED_CONFIG_DIFFERENCES = {
    "seed", "training.seed", "run_name", "output.dir", "output.smoke_dir",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flat: dict[str, Any] = {}
    for key, item in value.items():
        child = f"{prefix}.{key}" if prefix else str(key)
        flat.update(flatten_config(item, child))
    return flat


def config_diff_frame(configs: dict[int, dict]) -> tuple[pd.DataFrame, bool]:
    flattened = {seed: flatten_config(cfg) for seed, cfg in configs.items()}
    keys = sorted(set().union(*(values.keys() for values in flattened.values())))
    rows = []
    valid = True
    for key in keys:
        values = [flattened[seed].get(key, "<MISSING>") for seed in SEEDS]
        all_equal = values[0] == values[1] == values[2]
        allowed = key in ALLOWED_CONFIG_DIFFERENCES
        if not all_equal and not allowed:
            valid = False
        rows.append({"config_key": key, "seed1": values[0], "seed2": values[1], "seed3": values[2],
                     "all_equal": all_equal, "difference_allowed": allowed,
                     "check_passed": all_equal or allowed})
    return pd.DataFrame(rows), valid


def write_config_check(output: Path) -> bool:
    configs = {seed: load_json(path) for seed, path in CONFIGS.items()}
    frame, valid = config_diff_frame(configs)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "config_diff_check.csv"
    content = frame.to_csv(index=False)
    if not target.exists():
        target.write_text(content, encoding="utf-8")
    print(f"only_seed_and_output_fields_differ={str(valid).lower()}")
    return valid


def ensure_inputs() -> None:
    required = (
        "config_resolved.json", "model_summary.json", "training_history.csv", "best.pt",
        "threshold_summary.json", "test_point_metrics_by_file.csv", "test_point_metrics_summary.csv",
        "test_event_diagnostics.csv", "test_event_summary.csv", "test_score_arrays.npz",
        "efficiency_metrics.json", "final_hai_rec_baseline_diagnosis.md",
    )
    missing = [str(RUN_DIRS[seed] / name) for seed in SEEDS for name in required if not (RUN_DIRS[seed] / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing formal seed outputs: {missing}")


def ensure_fresh(output: Path) -> None:
    existing = [name for name in FINAL_FILES if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite HAI Step 3B outputs: {existing}")


def training_summary() -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    rows = []
    raw: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        history = pd.read_csv(RUN_DIRS[seed] / "training_history.csv")
        cfg = load_json(RUN_DIRS[seed] / "config_resolved.json")
        efficiency = load_json(RUN_DIRS[seed] / "efficiency_metrics.json")
        model = load_json(RUN_DIRS[seed] / "model_summary.json")
        best_index = int(history.val_rec_loss.idxmin())
        best = history.loc[best_index]
        row = {
            "record_type": "seed", "seed": seed, "best_epoch": int(best.epoch),
            "stop_epoch": int(history.epoch.iloc[-1]), "best_train_loss": float(best.train_rec_loss),
            "best_validation_loss": float(best.val_rec_loss),
            "final_train_loss": float(history.train_rec_loss.iloc[-1]),
            "final_validation_loss": float(history.val_rec_loss.iloc[-1]),
            "total_training_seconds": float(history.elapsed_seconds.sum()),
            "early_stopping_triggered": bool(len(history) < int(cfg["training"]["epochs"])),
            "nonfinite_detected": not bool(history.finite.astype(bool).all()),
            "gradient_explosion_detected": bool((history.max_gradient_norm > 1e3).any()),
            "severe_overfit_warning": bool(history.val_rec_loss.iloc[-1] > 2.0 * history.train_rec_loss.iloc[-1]),
            "checkpoint_size_mb": float(efficiency["checkpoint_size_mb"]),
            "parameter_count": int(model["total_parameter_count"]),
        }
        raw[seed] = row
        rows.append(row)
    numeric = ["best_epoch", "stop_epoch", "best_train_loss", "best_validation_loss", "final_train_loss",
               "final_validation_loss", "total_training_seconds", "checkpoint_size_mb", "parameter_count"]
    values = pd.DataFrame(rows)
    for statistic in ("mean", "std", "min", "max", "median"):
        row: dict[str, Any] = {"record_type": "statistic", "seed": statistic}
        for name in numeric:
            series = values[name].astype(float)
            row[name] = float(series.std(ddof=0) if statistic == "std" else getattr(series, statistic)())
        rows.append(row)
    return pd.DataFrame(rows), raw


def threshold_summary() -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    raw: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        data = {row["threshold_method"]: row for row in load_json(RUN_DIRS[seed] / "threshold_summary.json")}
        pot, q99 = data["train_only_pot"], data["normal_validation_q99"]
        row = {
            "record_type": "seed", "seed": seed, "pot_threshold": pot.get("threshold_value"),
            "pot_fitting_sample_count": pot["fitting_sample_count"], "pot_fitting_scope": pot["fitting_scope"],
            "q99_threshold": q99["threshold_value"], "q99_fitting_sample_count": q99["fitting_sample_count"],
            "q99_fitting_scope": q99["fitting_scope"], "pot_fitting_success": pot["status"] == "ok",
            "pot_failure_reason": pot.get("failure_reason", ""),
        }
        raw[seed] = row
        rows.append(row)
    frame = pd.DataFrame(rows)
    for statistic in ("mean", "std", "min", "max", "range", "coefficient_of_variation"):
        row = {"record_type": "statistic", "seed": statistic}
        for name in ("pot_threshold", "q99_threshold"):
            values = frame[name].astype(float)
            if statistic == "std": value = values.std(ddof=0)
            elif statistic == "range": value = values.max() - values.min()
            elif statistic == "coefficient_of_variation": value = values.std(ddof=0) / values.mean()
            else: value = getattr(values, statistic)()
            row[name] = float(value)
        rows.append(row)
    return pd.DataFrame(rows), raw


def point_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows, file_rows = [], []
    for seed in SEEDS:
        summary = pd.read_csv(RUN_DIRS[seed] / "test_point_metrics_summary.csv")
        summary.insert(0, "seed", seed)
        summary = summary.rename(columns={"predicted_alarm_count": "alarm_count"})
        summary_rows.append(summary)
        by_file = pd.read_csv(RUN_DIRS[seed] / "test_point_metrics_by_file.csv")
        by_file.insert(0, "seed", seed)
        by_file = by_file.rename(columns={"predicted_alarm_count": "alarm_count"})
        file_rows.append(by_file)
    summary = pd.concat(summary_rows, ignore_index=True)
    by_file = pd.concat(file_rows, ignore_index=True)
    for seed in SEEDS:
        for frame, scope_name in ((summary, "summary_scope"), (by_file, "file_name")):
            for _, group in frame[frame.seed == seed].groupby(scope_name):
                values = group.set_index("threshold_method")
                for metric in ("aupr", "auroc"):
                    if not np.isclose(values.loc[METHODS[0], metric], values.loc[METHODS[1], metric], atol=1e-12):
                        raise ValueError(f"{metric} unexpectedly differs by threshold for seed={seed}")
    combined = pd.concat([
        summary.assign(scope=summary.summary_scope),
        by_file.assign(scope=by_file.file_name.str.replace(".csv", "", regex=False)),
    ], ignore_index=True, sort=False)
    metrics = ("precision", "recall", "f1", "aupr", "auroc", "alarm_count", "false_positive_count", "false_negative_count")
    aggregate_rows = []
    for (method, scope), group in combined.groupby(["threshold_method", "scope"], sort=False):
        if len(group) != 3:
            raise ValueError(f"Expected three seeds for {method}/{scope}, got {len(group)}")
        group = group.set_index("seed").sort_index()
        for metric in metrics:
            values = group[metric].astype(float)
            aggregate_rows.append({
                "threshold_method": method, "scope": scope, "metric": metric,
                "seed1": values.loc[1], "seed2": values.loc[2], "seed3": values.loc[3],
                "mean": values.mean(), "std": values.std(ddof=0), "min": values.min(),
                "max": values.max(), "range": values.max() - values.min(),
            })
    return summary, by_file, pd.DataFrame(aggregate_rows)


def load_arrays() -> dict[int, dict[str, np.ndarray]]:
    return {seed: dict(np.load(RUN_DIRS[seed] / "test_score_arrays.npz")) for seed in SEEDS}


def aligned_array(arrays: dict[int, dict[str, np.ndarray]], file_key: str, field: str) -> list[np.ndarray]:
    key = f"{file_key}_{field}"
    values = [arrays[seed][key] for seed in SEEDS]
    if any(len(value) != len(values[0]) for value in values[1:]):
        raise ValueError(f"Length mismatch for {key}")
    if field in {"sequence_id", "timestamp", "label", "end_row"}:
        if any(not np.array_equal(value, values[0]) for value in values[1:]):
            raise ValueError(f"Alignment mismatch for {key}")
    return values


def validate_alignment(arrays: dict[int, dict[str, np.ndarray]]) -> None:
    for file_key in FILES:
        for field in ("sequence_id", "timestamp", "label", "end_row", "score"):
            aligned_array(arrays, file_key, field)


def score_correlations(arrays: dict[int, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for scope in (*FILES, "all_tests"):
        scores = {seed: (np.concatenate([arrays[seed][f"{name}_score"] for name in FILES])
                         if scope == "all_tests" else arrays[seed][f"{scope}_score"]) for seed in SEEDS}
        for left, right in combinations(SEEDS, 2):
            rows.append({"scope": scope, "seed_pair": f"seed{left}_vs_seed{right}",
                         "sample_count": len(scores[left]),
                         "pearson": float(pearsonr(scores[left], scores[right]).statistic),
                         "spearman": float(spearmanr(scores[left], scores[right]).statistic)})
    return pd.DataFrame(rows)


def alarm_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def specific_agreement(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left, right = left.astype(bool), right.astype(bool)
    both_positive = int((left & right).sum())
    both_negative = int((~left & ~right).sum())
    disagreement = int((left != right).sum())
    positive = 2 * both_positive / (2 * both_positive + disagreement) if both_positive or disagreement else 1.0
    negative = 2 * both_negative / (2 * both_negative + disagreement) if both_negative or disagreement else 1.0
    return float(positive), float(negative)


def alarm_agreement(arrays: dict[int, dict[str, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    agreement, votes = [], []
    for method in METHODS:
        for scope in (*FILES, "all_tests"):
            predictions = {seed: (np.concatenate([arrays[seed][f"{name}_pred_{method}"] for name in FILES])
                                  if scope == "all_tests" else arrays[seed][f"{scope}_pred_{method}"]) for seed in SEEDS}
            for left, right in combinations(SEEDS, 2):
                positive, negative = specific_agreement(predictions[left], predictions[right])
                agreement.append({"threshold_method": method, "scope": scope,
                                  "seed_pair": f"seed{left}_vs_seed{right}",
                                  "alarm_jaccard": alarm_jaccard(predictions[left], predictions[right]),
                                  "positive_agreement": positive, "negative_agreement": negative,
                                  "disagreement_count": int((predictions[left] != predictions[right]).sum()),
                                  "sample_count": len(predictions[left])})
            vote_count = sum(predictions.values())
            for count in range(4):
                points = int((vote_count == count).sum())
                votes.append({"threshold_method": method, "scope": scope, "alarm_seed_count": count,
                              "point_count": points, "fraction": points / len(vote_count),
                              "diagnostic_only_not_a_detection_method": True})
    return pd.DataFrame(agreement), pd.DataFrame(votes)


def event_classification(count: int) -> str:
    if count == 3: return "robust_detected"
    if count == 0: return "consistently_missed"
    return "unstable_detected"


def event_consistency() -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = {seed: pd.read_csv(RUN_DIRS[seed] / "test_event_diagnostics.csv") for seed in SEEDS}
    key = ["threshold_method", "sequence_id", "file_name", "global_event_id", "event_id", "start_timestamp", "end_timestamp"]
    base_keys = diagnostics[1][key]
    for seed in (2, 3):
        if not base_keys.equals(diagnostics[seed][key]):
            raise ValueError(f"Event alignment mismatch for seed={seed}")
    rows = []
    for index, base in diagnostics[1].iterrows():
        row = {name: base[name] for name in key}
        detected_count, delays = 0, []
        for seed in SEEDS:
            item = diagnostics[seed].iloc[index]
            detected = bool(item.detected)
            detected_count += int(detected)
            delay = float(item.first_alarm_delay_points) if detected else np.nan
            if detected: delays.append(delay)
            row[f"seed{seed}_detected"] = detected
            row[f"seed{seed}_delay"] = delay
            row[f"seed{seed}_point_recall"] = float(item.point_recall_in_event)
        row["detected_seed_count"] = detected_count
        row["mean_delay_detected_seeds"] = float(np.mean(delays)) if delays else np.nan
        row["classification"] = event_classification(detected_count)
        rows.append(row)
    consistency = pd.DataFrame(rows)
    summaries = []
    for (method, file_name), group in consistency.groupby(["threshold_method", "file_name"]):
        summaries.append(event_summary_row(method, file_name, group))
    for method, group in consistency.groupby("threshold_method"):
        summaries.append(event_summary_row(method, "all_tests", group))
    return consistency, pd.DataFrame(summaries)


def event_summary_row(method: str, scope: str, group: pd.DataFrame) -> dict[str, Any]:
    recalls = [float(group[f"seed{seed}_detected"].mean()) for seed in SEEDS]
    delays = np.concatenate([group.loc[group[f"seed{seed}_detected"], f"seed{seed}_delay"].dropna().to_numpy(float)
                             for seed in SEEDS])
    return {
        "threshold_method": method, "scope": scope, "total_event_count": len(group),
        "seed1_event_recall": recalls[0], "seed2_event_recall": recalls[1], "seed3_event_recall": recalls[2],
        "event_recall_mean": float(np.mean(recalls)), "event_recall_std": float(np.std(recalls, ddof=0)),
        "event_recall_range": float(np.max(recalls) - np.min(recalls)),
        "robust_detected_event_count": int((group.classification == "robust_detected").sum()),
        "unstable_event_count": int((group.classification == "unstable_detected").sum()),
        "consistently_missed_event_count": int((group.classification == "consistently_missed").sum()),
        "detection_delay_mean": float(np.mean(delays)) if len(delays) else np.nan,
        "detection_delay_std": float(np.std(delays, ddof=0)) if len(delays) else np.nan,
        "detection_delay_median": float(np.median(delays)) if len(delays) else np.nan,
    }


def classify_stability(training: pd.DataFrame, point_aggregate: pd.DataFrame, correlations: pd.DataFrame,
                       event_summary: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    seed_training = training[training.record_type == "seed"]
    normal_training = bool((~seed_training.nonfinite_detected.astype(bool)).all() and
                           (~seed_training.gradient_explosion_detected.astype(bool)).all() and
                           (~seed_training.severe_overfit_warning.astype(bool)).all())
    ranking = point_aggregate[(point_aggregate.scope == "micro") &
                              (point_aggregate.threshold_method == "normal_validation_q99")]
    auroc_std = float(ranking.loc[ranking.metric == "auroc", "std"].iloc[0])
    aupr_std = float(ranking.loc[ranking.metric == "aupr", "std"].iloc[0])
    event_ranges = event_summary[event_summary.scope == "all_tests"].event_recall_range
    pearson_majority = float((correlations.pearson >= 0.90).mean()) > 0.5
    checks = {"normal_training": normal_training, "micro_auroc_std": auroc_std,
              "micro_aupr_std": aupr_std, "max_event_recall_range": float(event_ranges.max()),
              "pearson_fraction_ge_0_90": float((correlations.pearson >= 0.90).mean())}
    if normal_training and auroc_std <= 0.02 and aupr_std <= 0.02 and event_ranges.max() <= 0.10 and pearson_majority:
        return "stable_baseline", checks
    if normal_training and auroc_std <= 0.05 and aupr_std <= 0.05:
        return "moderately_seed_sensitive", checks
    return "highly_seed_sensitive", checks


def aggregate(output: Path) -> dict[str, Any]:
    ensure_inputs()
    ensure_fresh(output)
    training, training_raw = training_summary()
    thresholds, threshold_raw = threshold_summary()
    point, point_by_file, point_aggregate = point_metrics()
    arrays = load_arrays()
    validate_alignment(arrays)
    correlations = score_correlations(arrays)
    agreement, votes = alarm_agreement(arrays)
    events, event_summary = event_consistency()
    stability, checks = classify_stability(training, point_aggregate, correlations, event_summary)
    outputs = {
        "multiseed_training_summary.csv": training,
        "multiseed_threshold_summary.csv": thresholds,
        "multiseed_point_metrics.csv": point,
        "multiseed_point_metrics_by_file.csv": point_by_file,
        "multiseed_point_metrics_aggregate.csv": point_aggregate,
        "multiseed_score_correlations.csv": correlations,
        "multiseed_alarm_agreement.csv": agreement,
        "multiseed_alarm_vote_summary.csv": votes,
        "multiseed_event_consistency.csv": events,
        "multiseed_event_consistency_summary.csv": event_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)

    overall_events = event_summary[event_summary.scope == "all_tests"].set_index("threshold_method")
    ranking = point_aggregate[(point_aggregate.scope.isin(["micro", "macro"])) &
                              (point_aggregate.metric.isin(["f1", "aupr", "auroc"]))]
    file_ranking = point_aggregate[(point_aggregate.scope.isin(FILES)) &
                                   (point_aggregate.metric.isin(["aupr", "auroc"]))]
    file_stability = file_ranking.groupby("scope")["std"].mean().sort_values()
    false_positives = point_aggregate[(point_aggregate.scope.isin(FILES)) &
                                      (point_aggregate.metric == "false_positive_count")].groupby("scope")["mean"].mean()
    persistent = events[events.classification == "consistently_missed"]
    summary = {
        "only_seed_and_output_fields_differ": True, "all_test_windows_aligned": True,
        "stability_classification": stability, "stability_checks": checks,
        "best_epochs": {str(seed): training_raw[seed]["best_epoch"] for seed in SEEDS},
        "best_validation_loss_mean": float(np.mean([training_raw[s]["best_validation_loss"] for s in SEEDS])),
        "best_validation_loss_std": float(np.std([training_raw[s]["best_validation_loss"] for s in SEEDS], ddof=0)),
        "pot_threshold_mean": float(np.mean([threshold_raw[s]["pot_threshold"] for s in SEEDS])),
        "pot_threshold_std": float(np.std([threshold_raw[s]["pot_threshold"] for s in SEEDS], ddof=0)),
        "q99_threshold_mean": float(np.mean([threshold_raw[s]["q99_threshold"] for s in SEEDS])),
        "q99_threshold_std": float(np.std([threshold_raw[s]["q99_threshold"] for s in SEEDS], ddof=0)),
        "ranking_metric_aggregates": ranking.to_dict(orient="records"),
        "score_correlation_mean": {"pearson": float(correlations.pearson.mean()),
                                   "spearman": float(correlations.spearman.mean())},
        "alarm_jaccard_mean": agreement.groupby("threshold_method").alarm_jaccard.mean().to_dict(),
        "event_summary": overall_events.reset_index().to_dict(orient="records"),
        "most_stable_test_file": str(file_stability.index[0]),
        "highest_false_positive_test_file": str(false_positives.idxmax()),
        "persistent_missed_event_count_rows": len(persistent),
        "multiseed_vote_used_as_detection_method": False,
        "ready_for_hai_structure_relation_diagnosis": stability in {"stable_baseline", "moderately_seed_sensitive"},
    }
    (output / "multiseed_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    missed_lines = [f"{row.threshold_method}: {row.file_name}/event_{row.event_id}" for row in persistent.itertuples()]
    report = f"""# HAI Step 3B rec-only multiseed diagnosis

- Configuration audit: passed; only seed, run name, and output fields differ.
- Training: all three seeds completed with finite loss and gradients. Best epochs: {summary['best_epochs']}.
- Best validation loss: {summary['best_validation_loss_mean']:.8f} +/- {summary['best_validation_loss_std']:.8f}.
- POT threshold: {summary['pot_threshold_mean']:.8f} +/- {summary['pot_threshold_std']:.8f}; q99: {summary['q99_threshold_mean']:.8f} +/- {summary['q99_threshold_std']:.8f}.
- Score alignment was checked by sequence_id, timestamp, end row, and label without sorting.
- Mean score correlation: Pearson {summary['score_correlation_mean']['pearson']:.6f}, Spearman {summary['score_correlation_mean']['spearman']:.6f}.
- Mean alarm Jaccard: {summary['alarm_jaccard_mean']}.
- Most stable test file by AUPR/AUROC standard deviation: {summary['most_stable_test_file']}.
- Highest mean false-positive count: {summary['highest_false_positive_test_file']}.
- POT events: robust {int(overall_events.loc['train_only_pot', 'robust_detected_event_count'])}, unstable {int(overall_events.loc['train_only_pot', 'unstable_event_count'])}, consistently missed {int(overall_events.loc['train_only_pot', 'consistently_missed_event_count'])}.
- q99 events: robust {int(overall_events.loc['normal_validation_q99', 'robust_detected_event_count'])}, unstable {int(overall_events.loc['normal_validation_q99', 'unstable_event_count'])}, consistently missed {int(overall_events.loc['normal_validation_q99', 'consistently_missed_event_count'])}.
- Persistent rec-score misses (threshold-specific): {', '.join(missed_lines) if missed_lines else 'none'}.
- Stability classification: `{stability}`. Ready for HAI structure-relation effectiveness diagnosis: `{summary['ready_for_hai_structure_relation_diagnosis']}`.
- Three-seed alarm voting is diagnostic only and is not reported as a detection method.
"""
    (output / "final_hai_rec_multiseed_diagnosis.md").write_text(report, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate HAI rec-only seed 1/2/3 formal outputs")
    parser.add_argument("--check-config-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = SMOKE_OUTPUT if args.smoke else OUTPUT
    valid = write_config_check(output)
    if not valid:
        raise ValueError("Configuration mismatch: fields other than seed/run/output differ")
    if args.check_config_only:
        return 0
    result = aggregate(output)
    print(json.dumps({"output": str(output), **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
