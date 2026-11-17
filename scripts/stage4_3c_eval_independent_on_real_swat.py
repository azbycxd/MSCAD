from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat import _load_swat_csv  # noqa: E402
from data.transforms import StandardScaler  # noqa: E402
from scripts.analyze_structure_score_effectiveness import build_device, fit_train_threshold, load_model  # noqa: E402
from scripts.stage4_2b_diagnose_relation_validity import rolling_pair_diagnostics  # noqa: E402
from scripts.stage4_3b_compare_structure_scores import (  # noqa: E402
    checkpoint_for_seed,
    checkpoint_structure_pairs,
    fit_independent_models_64,
)
from utils.config_io import load_config  # noqa: E402
from utils.controlled_relation_injection import pair_score  # noqa: E402
from utils.metrics import binary_metrics  # noqa: E402


WINDOW_64 = 64
WINDOW_128 = 128
SCORE_TYPES = ("rec", "existing_short", "existing_long", "independent_64_max", "independent_128_max")
THRESHOLD_METHODS = ("pot_train_only", "normal_q99")


def load_fixed_strict_set(source_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    candidate_sets = json.loads((source_dir / "candidate_relation_sets.json").read_text(encoding="utf-8"))
    strict = candidate_sets["strict_stable"]["relations"]
    if len(strict) != 6:
        raise ValueError(f"Expected fixed strict_stable to contain 6 relations, got {len(strict)}")
    pair_ids = {str(row["pair_id"]) for row in strict}
    model_frame = pd.read_csv(source_dir / "normal_relation_models.csv")
    models = model_frame[model_frame.pair_id.astype(str).isin(pair_ids)].to_dict("records")
    if len(models) != 6:
        raise ValueError("Normal relation models do not cover all six fixed strict relations")
    floors = {float(row["scale_floor"]) for row in models}
    if len(floors) != 1:
        raise ValueError("Inconsistent 128 scale floors")
    return strict, models, floors.pop()


def common_alignment_indices(point_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if point_count < WINDOW_128:
        raise ValueError("Not enough points for common alignment")
    test_indices = np.arange(WINDOW_128 - 1, point_count, dtype=np.int64)
    score64_indices = test_indices - (WINDOW_64 - 1)
    score128_indices = test_indices - (WINDOW_128 - 1)
    return test_indices, score64_indices, score128_indices


def align_scores(
    scores_64: np.ndarray,
    scores_128: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    test_indices, index64, index128 = common_alignment_indices(len(labels))
    return scores_64[index64], scores_128[index128], labels[test_indices], test_indices


@torch.no_grad()
def infer_existing_series(
    model: torch.nn.Module,
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    count = len(values) - WINDOW_64 + 1
    rec = np.empty(count, dtype=np.float64)
    short = np.empty(count, dtype=np.float64)
    long = np.empty(count, dtype=np.float64)
    for offset in range(0, count, batch_size):
        starts = range(offset, min(count, offset + batch_size))
        windows = np.stack([values[start : start + WINDOW_64] for start in starts])
        xb = torch.as_tensor(windows, dtype=torch.float32, device=device)
        x_hat = model(xb)
        components = model.score_components(xb, x_hat)
        length = len(windows)
        rec[offset : offset + length] = components["rec"].detach().cpu().numpy()
        short[offset : offset + length] = components["short"].detach().cpu().numpy()
        long[offset : offset + length] = components["long"].detach().cpu().numpy()
    return {"rec": rec, "existing_short": short, "existing_long": long}


def independent_series_scores(
    values: np.ndarray,
    models: list[dict[str, Any]],
    scale_floor: float,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_scores = np.empty((len(values) - window_size + 1, len(models)), dtype=np.float64)
    correlations = np.empty_like(pair_scores)
    for index, model in enumerate(models):
        diagnostics = rolling_pair_diagnostics(
            values[:, int(model["channel_i_index"])],
            values[:, int(model["channel_j_index"])],
            window_size,
        )
        corr = diagnostics["correlations"]
        correlations[:, index] = corr
        pair_scores[:, index] = [
            pair_score(value, float(model["normal_median_fisher"]),
                       float(model["normal_robust_scale"]), scale_floor)
            for value in corr
        ]
    finite = np.isfinite(pair_scores)
    aggregate = np.max(np.where(finite, pair_scores, -np.inf), axis=1)
    aggregate[~finite.any(axis=1)] = 0.0
    return aggregate, pair_scores, correlations


def fit_thresholds(normal_scores: np.ndarray, eval_cfg: dict[str, Any]) -> dict[str, float]:
    pot, method = fit_train_threshold(normal_scores, {**eval_cfg, "threshold_mode": "pot"})
    if not method.startswith("pot_train_only"):
        raise ValueError(f"Unexpected POT method: {method}")
    return {"pot_train_only": float(pot), "normal_q99": float(np.quantile(normal_scores, 0.99))}


def point_complementarity(
    labels: np.ndarray,
    rec_scores: np.ndarray,
    independent_scores: np.ndarray,
    rec_threshold: float,
    independent_threshold: float,
) -> dict[str, int | float]:
    labels = np.asarray(labels, dtype=int)
    anomaly = labels == 1
    normal = ~anomaly
    rec_alarm = np.asarray(rec_scores) > rec_threshold
    independent_alarm = np.asarray(independent_scores) > independent_threshold
    rec_detected = anomaly & rec_alarm
    independent_detected = anomaly & independent_alarm
    rec_missed = anomaly & ~rec_alarm
    independent_missed = anomaly & ~independent_alarm
    added_true = independent_detected & ~rec_alarm
    added_false = normal & independent_alarm & ~rec_alarm
    both = anomaly & rec_alarm & independent_alarm
    neither = anomaly & ~rec_alarm & ~independent_alarm
    incremental_precision = int(added_true.sum()) / max(1, int(added_true.sum() + added_false.sum()))
    return {
        "total_anomaly_points": int(anomaly.sum()),
        "rec_detected": int(rec_detected.sum()), "rec_missed": int(rec_missed.sum()),
        "independent128_detected": int(independent_detected.sum()),
        "independent128_missed": int(independent_missed.sum()),
        "independent128_detected_rec_missed": int(added_true.sum()),
        "independent128_complement_rate": int(added_true.sum()) / max(1, int(anomaly.sum())),
        "rec_detected_independent128_missed": int((rec_detected & ~independent_alarm).sum()),
        "both_detected": int(both.sum()), "neither_detected": int(neither.sum()),
        "rec_normal_alarm_count": int((normal & rec_alarm).sum()),
        "independent128_normal_alarm_count": int((normal & independent_alarm).sum()),
        "independent128_new_false_alarm_count": int(added_false.sum()),
        "incremental_precision": float(incremental_precision),
    }


def contiguous_events(labels: np.ndarray, test_indices: np.ndarray) -> list[dict[str, int]]:
    labels = np.asarray(labels, dtype=int)
    padded = np.concatenate([[0], labels, [0]])
    transitions = np.flatnonzero(np.diff(padded))
    events = []
    for event_id, (start, end_exclusive) in enumerate(zip(transitions[::2], transitions[1::2]), start=1):
        end = int(end_exclusive - 1)
        events.append({
            "event_id": event_id, "aligned_start": int(start), "aligned_end": end,
            "start_index": int(test_indices[start]), "end_index": int(test_indices[end]),
            "duration": int(end_exclusive - start),
        })
    return events


def top_relation_attribution(
    pair_scores: np.ndarray,
    pair_rows: list[dict[str, Any]],
    position: int,
) -> dict[str, Any]:
    scores = pair_scores[position]
    order = np.argsort(np.where(np.isfinite(scores), -scores, np.inf), kind="stable")
    top = order[: min(3, len(order))]
    return {
        "top1_relation": pair_rows[int(top[0])]["pair_id"],
        "top1_relation_peak_score": float(scores[int(top[0])]),
        "top3_relations": "|".join(str(pair_rows[int(index)]["pair_id"]) for index in top),
    }


def event_diagnostics(
    labels: np.ndarray,
    test_indices: np.ndarray,
    predictions: dict[str, np.ndarray],
    pair_scores: np.ndarray,
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for event in contiguous_events(labels, test_indices):
        start, end = event["aligned_start"], event["aligned_end"]
        section = slice(start, end + 1)
        row: dict[str, Any] = {**event}
        for name in ("rec", "existing_long", "independent_64_max", "independent_128_max"):
            alarms = np.flatnonzero(predictions[name][section])
            row[f"{name}_detected"] = bool(len(alarms))
            row[f"{name}_point_recall_in_event"] = float(predictions[name][section].mean())
            row[f"{name}_first_alarm_delay"] = int(alarms[0]) if len(alarms) else -1
        row["independent128_recovers_rec_missed_event"] = bool(
            row["independent_128_max_detected"] and not row["rec_detected"]
        )
        row["independent64_detected"] = row["independent_64_max_detected"]
        row["independent128_detected"] = row["independent_128_max_detected"]
        row["independent128_first_alarm_delay"] = row["independent_128_max_first_alarm_delay"]
        row["independent128_point_recall_in_event"] = row["independent_128_max_point_recall_in_event"]
        peak = start + int(np.nanargmax(np.max(np.where(np.isfinite(pair_scores[section]), pair_scores[section], -np.inf), axis=1)))
        row.update(top_relation_attribution(pair_scores, pair_rows, peak))
        output.append(row)
    return output


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0, 0.0
    return float(pearsonr(left, right).statistic), float(spearmanr(left, right).statistic)


def validate_finite(frame: pd.DataFrame, columns: list[str]) -> None:
    if np.any(~np.isfinite(frame[columns].to_numpy(dtype=float))):
        raise ValueError("Non-finite output value")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False, encoding="utf-8-sig")


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_data(project_root: Path, cfg: dict[str, Any], smoke: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    data_cfg = cfg["data"]
    normal_raw, normal_labels, feature_names = _load_swat_csv(
        str(project_root / data_cfg["train_csv"]), data_cfg["label_col"], data_cfg["timestamp_col"]
    )
    test_raw, test_labels, test_features = _load_swat_csv(
        str(project_root / data_cfg["test_csv"]), data_cfg["label_col"], data_cfg["timestamp_col"]
    )
    if feature_names != test_features:
        raise ValueError("Train/test feature order differs")
    n_val = int(max(WINDOW_64 * 2, round(len(normal_raw) * float(data_cfg.get("val_ratio", 0.2)))))
    n_val = min(max(WINDOW_64 * 2, n_val), len(normal_raw) - WINDOW_64 - 1)
    split = len(normal_raw) - n_val
    train_raw = normal_raw[:split]
    if np.any(normal_labels[:split] != 0):
        raise ValueError("Normal training split contains anomaly labels")
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_raw).astype(np.float64, copy=False)
    test_norm = scaler.transform(test_raw).astype(np.float64, copy=False)
    original_count = len(test_norm)
    original_positive = int(test_labels.sum())
    if smoke:
        train_norm = train_norm[: min(len(train_norm), 16384)]
        first_attack = int(np.flatnonzero(test_labels == 1)[0])
        start = max(0, first_attack - 8192)
        end = min(len(test_norm), first_attack + 8192)
        test_norm = test_norm[start:end]
        test_labels = test_labels[start:end]
        original_indices = np.arange(start, end, dtype=np.int64)
    else:
        original_indices = np.arange(len(test_norm), dtype=np.int64)
    return train_norm, test_norm, test_labels, feature_names, {
        "normal_csv": data_cfg["train_csv"], "test_csv": data_cfg["test_csv"],
        "original_test_point_count": original_count, "original_test_positive_count": original_positive,
        "normal_threshold_training_point_count": len(train_norm),
        "analyzed_test_point_count": len(test_norm), "smoke": smoke,
        "test_data_read": True, "attack_labels_read": True,
        "scaler": "StandardScaler fitted on the configured normal training split only",
        "original_test_indices": original_indices,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    source_dir = project_root / "outputs/stage4_3a_candidate_relation_sensitivity"
    source_4b = project_root / "outputs/stage4_3b_controlled_score_comparison"
    output_dir = project_root / (
        "outputs/stage4_3c_real_swat_single_score_smoke" if args.smoke else args.output_dir
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    required = [
        source_dir / "candidate_relation_sets.json", source_dir / "normal_relation_models.csv",
        source_4b / "controlled_score_comparison_summary.json",
        project_root / "configs/swat_improve_v4_struct_finetune.yaml",
        project_root / "Swat/normal.csv", project_root / "Swat/merged.csv",
    ] + [checkpoint_for_seed(project_root, seed) for seed in seeds]
    missing = [str(path) for path in required if not path.exists()]
    print(f"Step 4.3C strict_stable only, seeds={seeds}, common t>=127")
    print("Scores: rec, existing_short, existing_long, independent_64_max, independent_128_max")
    print("Thresholds: train-only POT and normal q99; no fusion or test-label tuning")
    print(f"Output={output_dir}")
    for seed in seeds:
        path = checkpoint_for_seed(project_root, seed)
        if path.exists():
            print(f"seed={seed} checkpoint={path} pairs={len(checkpoint_structure_pairs(path))}")
    if missing:
        for path in missing:
            print(f"missing path: {path}")
        if args.dry_run:
            return {"dry_run": True, "missing": missing}
        raise FileNotFoundError(missing[0])
    if args.dry_run:
        return {"dry_run": True, "missing": []}

    cfg = load_config(str(project_root / "configs/swat_improve_v4_struct_finetune.yaml"))
    strict_rows, models_128, floor_128 = load_fixed_strict_set(source_dir)
    train_values, test_values, test_labels, feature_names, data_info = _load_data(project_root, cfg, args.smoke)
    reference = train_values[: int(len(train_values) * 0.80)]
    models_64, floor_64 = fit_independent_models_64(reference, strict_rows)

    train_ind64, _, _ = independent_series_scores(train_values, models_64, floor_64, WINDOW_64)
    train_ind128, _, _ = independent_series_scores(train_values, models_128, floor_128, WINDOW_128)
    test_ind64, _, _ = independent_series_scores(test_values, models_64, floor_64, WINDOW_64)
    test_ind128, pair_scores_128, correlations_128 = independent_series_scores(
        test_values, models_128, floor_128, WINDOW_128
    )
    test_indices_local, score64_indices, score128_indices = common_alignment_indices(len(test_values))
    labels_aligned = test_labels[test_indices_local]
    original_indices = data_info.pop("original_test_indices")[test_indices_local]
    independent_aligned = {
        "independent_64_max": test_ind64[score64_indices],
        "independent_128_max": test_ind128[score128_indices],
    }
    pair_scores_aligned = pair_scores_128[score128_indices]
    correlations_aligned = correlations_128[score128_indices]

    threshold_train: dict[tuple[int, str], dict[str, float]] = {}
    aligned_by_seed: dict[int, dict[str, np.ndarray]] = {}
    train_by_seed: dict[int, dict[str, np.ndarray]] = {}
    device = build_device(str(cfg.get("device", "cuda")))
    for seed in seeds:
        checkpoint = checkpoint_for_seed(project_root, seed)
        if len(checkpoint_structure_pairs(checkpoint)) != 40:
            raise ValueError(f"Checkpoint seed {seed} does not contain 40 original pairs")
        model = load_model(cfg, checkpoint, len(feature_names), device)
        train_existing = infer_existing_series(model, train_values, device, args.batch_size)
        test_existing = infer_existing_series(model, test_values, device, args.batch_size)
        aligned = {
            name: test_existing[name][score64_indices]
            for name in ("rec", "existing_short", "existing_long")
        } | independent_aligned
        training = train_existing | {
            "independent_64_max": train_ind64,
            "independent_128_max": train_ind128,
        }
        aligned_by_seed[seed] = aligned
        train_by_seed[seed] = training
        for score_name in SCORE_TYPES:
            threshold_train[(seed, score_name)] = fit_thresholds(training[score_name], cfg["eval"])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metric_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[int, str, str], np.ndarray] = {}
    positive_ratio = float(labels_aligned.mean())
    for seed in seeds:
        for score_name in SCORE_TYPES:
            score = aligned_by_seed[seed][score_name]
            for method in THRESHOLD_METHODS:
                threshold = threshold_train[(seed, score_name)][method]
                pred = score > threshold
                predictions[(seed, score_name, method)] = pred
                metrics = binary_metrics(labels_aligned, pred.astype(int), score)
                metric_rows.append({
                    "seed": seed, "score_type": score_name, "threshold_method": method,
                    **metrics, "threshold": threshold,
                    "threshold_fit_sample_count": len(train_by_seed[seed][score_name]),
                    "aligned_test_count": len(labels_aligned), "test_positive_ratio": positive_ratio,
                    "normal_test_false_positive_count": int(((labels_aligned == 0) & pred).sum()),
                    "anomaly_detected_count": int(((labels_aligned == 1) & pred).sum()),
                })
    metric_frame = pd.DataFrame(metric_rows)
    summary_rows: list[dict[str, Any]] = []
    for (score_name, method), group in metric_frame.groupby(["score_type", "threshold_method"], sort=True):
        row: dict[str, Any] = {"score_type": score_name, "threshold_method": method, "seed_count": len(group)}
        for field in ("precision", "recall", "f1", "aupr", "auroc", "threshold",
                      "normal_test_false_positive_count", "anomaly_detected_count"):
            row[field] = float(group[field].mean())
            row[f"{field}_std"] = float(group[field].std(ddof=1))
        row["test_positive_ratio"] = positive_ratio
        summary_rows.append(row)

    correlation_rows: list[dict[str, Any]] = []
    correlation_pairs = (
        ("rec", "existing_short"), ("rec", "existing_long"),
        ("rec", "independent_64_max"), ("rec", "independent_128_max"),
        ("existing_long", "independent_128_max"),
        ("independent_64_max", "independent_128_max"),
    )
    for seed in seeds:
        scores = aligned_by_seed[seed]
        for sample_group, mask in (
            ("normal", labels_aligned == 0), ("anomaly", labels_aligned == 1),
            ("all", np.ones(len(labels_aligned), dtype=bool)),
        ):
            for left, right in correlation_pairs:
                pearson, spearman = _safe_correlation(scores[left][mask], scores[right][mask])
                correlation_rows.append({
                    "seed": seed, "sample_group": sample_group, "score_pair": f"{left}_vs_{right}",
                    "pearson": pearson, "spearman": spearman, "sample_count": int(mask.sum()),
                })

    complement_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    event_summary_rows: list[dict[str, Any]] = []
    events = contiguous_events(labels_aligned, original_indices)
    for seed in seeds:
        for method in THRESHOLD_METHODS:
            complement_rows.append({
                "seed": seed, "threshold_method": method,
                **point_complementarity(
                    labels_aligned, aligned_by_seed[seed]["rec"], independent_aligned["independent_128_max"],
                    threshold_train[(seed, "rec")][method],
                    threshold_train[(seed, "independent_128_max")][method],
                ),
            })
            event_predictions = {
                name: predictions[(seed, name, method)]
                for name in ("rec", "existing_long", "independent_64_max", "independent_128_max")
            }
            diagnostics = event_diagnostics(
                labels_aligned, original_indices, event_predictions, pair_scores_aligned, models_128
            )
            for row in diagnostics:
                event_rows.append({"seed": seed, "threshold_method": method, **row})
            event_summary_rows.append({
                "seed": seed, "threshold_method": method, "total_event_count": len(diagnostics),
                "rec_detected_event_count": sum(row["rec_detected"] for row in diagnostics),
                "rec_fully_missed_event_count": sum(not row["rec_detected"] for row in diagnostics),
                "independent128_detected_event_count": sum(row["independent_128_max_detected"] for row in diagnostics),
                "independent128_recovered_rec_missed_event_count": sum(row["independent128_recovers_rec_missed_event"] for row in diagnostics),
                "both_missed_event_count": sum(not row["rec_detected"] and not row["independent_128_max_detected"] for row in diagnostics),
                "independent128_earlier_than_rec_event_count": sum(
                    row["independent_128_max_detected"] and (
                        not row["rec_detected"] or row["independent_128_max_first_alarm_delay"] < row["rec_first_alarm_delay"]
                    ) for row in diagnostics
                ),
            })

    attribution_rows: list[dict[str, Any]] = []
    top_global = np.argsort(-independent_aligned["independent_128_max"], kind="stable")[:100]
    event_peaks = []
    for event in events:
        section = slice(event["aligned_start"], event["aligned_end"] + 1)
        event_peaks.append(event["aligned_start"] + int(np.argmax(independent_aligned["independent_128_max"][section])))
    selected_positions = sorted(set(int(value) for value in np.concatenate([top_global, np.asarray(event_peaks, dtype=int)])))
    event_id_by_position = np.full(len(labels_aligned), -1, dtype=int)
    for event in events:
        event_id_by_position[event["aligned_start"] : event["aligned_end"] + 1] = event["event_id"]
    for seed in seeds:
        for position in selected_positions:
            scores = pair_scores_aligned[position]
            order = np.argsort(np.where(np.isfinite(scores), -scores, np.inf), kind="stable")[:3]
            top = int(order[0])
            attribution_rows.append({
                "seed": seed, "test_index": int(original_indices[position]),
                "event_id": int(event_id_by_position[position]), "y_true": int(labels_aligned[position]),
                "rec_score": float(aligned_by_seed[seed]["rec"][position]),
                "independent128_score": float(independent_aligned["independent_128_max"][position]),
                "top1_pair": models_128[top]["pair_id"], "top1_pair_score": float(scores[top]),
                "top2_pair": models_128[int(order[1])]["pair_id"], "top2_pair_score": float(scores[int(order[1])]),
                "top3_pair": models_128[int(order[2])]["pair_id"], "top3_pair_score": float(scores[int(order[2])]),
                "current_corr_top1": float(correlations_aligned[position, top]),
                "normal_median_corr_top1": float(models_128[top]["median_corr"]),
                "normalized_deviation_top1": float(scores[top]),
            })

    frequency_rows: list[dict[str, Any]] = []
    finite_scores = np.where(np.isfinite(pair_scores_aligned), pair_scores_aligned, -np.inf)
    top_order = np.argsort(-finite_scores, axis=1, kind="stable")[:, :3]
    for method in THRESHOLD_METHODS:
        independent_alarm = predictions[(seeds[0], "independent_128_max", method)]
        for pair_index, model in enumerate(models_128):
            event_top1 = 0
            event_top3 = 0
            for position in event_peaks:
                event_top1 += int(top_order[position, 0] == pair_index)
                event_top3 += int(pair_index in top_order[position])
            frequency_rows.append({
                "threshold_method": method, "pair_id": model["pair_id"],
                "top1_event_count": event_top1, "top3_event_count": event_top3,
                "normal_false_alarm_top1_count": int(((labels_aligned == 0) & independent_alarm & (top_order[:, 0] == pair_index)).sum()),
                "true_anomaly_top1_count": int(((labels_aligned == 1) & (top_order[:, 0] == pair_index)).sum()),
            })

    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "single_score_metrics.csv", metric_rows)
    write_csv(output_dir / "single_score_seed_summary.csv", summary_rows)
    write_csv(output_dir / "score_correlations.csv", correlation_rows)
    write_csv(output_dir / "point_level_complementarity.csv", complement_rows)
    write_csv(output_dir / "event_level_diagnostics.csv", event_rows)
    write_csv(output_dir / "event_level_summary.csv", event_summary_rows)
    write_csv(output_dir / "independent_relation_attribution.csv", attribution_rows)
    write_csv(output_dir / "top_relation_frequency_by_event.csv", frequency_rows)
    np.savez_compressed(
        output_dir / "aligned_score_arrays.npz",
        test_index=original_indices, label=labels_aligned.astype(np.int8),
        independent_64_max=independent_aligned["independent_64_max"].astype(np.float32),
        independent_128_max=independent_aligned["independent_128_max"].astype(np.float32),
        **{f"seed_{seed}_{name}": aligned_by_seed[seed][name].astype(np.float32)
           for seed in seeds for name in ("rec", "existing_short", "existing_long")},
    )

    metric_lookup = {
        (int(row["seed"]), row["score_type"], row["threshold_method"]): row for row in metric_rows
    }
    comp_frame = pd.DataFrame(complement_rows)
    event_summary_frame = pd.DataFrame(event_summary_rows)
    independent_better_seeds = sum(
        metric_lookup[(seed, "independent_128_max", "pot_train_only")]["aupr"]
        > metric_lookup[(seed, "rec", "pot_train_only")]["aupr"] for seed in seeds
    )
    recovered_points = float(comp_frame[comp_frame.threshold_method == "pot_train_only"]["independent128_detected_rec_missed"].mean())
    new_false = float(comp_frame[comp_frame.threshold_method == "pot_train_only"]["independent128_new_false_alarm_count"].mean())
    incremental_precision = float(comp_frame[comp_frame.threshold_method == "pot_train_only"]["incremental_precision"].mean())
    recovered_events = float(event_summary_frame[event_summary_frame.threshold_method == "pot_train_only"]["independent128_recovered_rec_missed_event_count"].mean())
    supports_fusion = bool(independent_better_seeds >= 2 and recovered_points > 0 and incremental_precision > positive_ratio)
    summary = {
        "scope": {
            **data_info, "common_aligned_point_count": len(labels_aligned),
            "warmup_excluded_point_count": WINDOW_128 - 1,
            "aligned_positive_count": int(labels_aligned.sum()),
            "aligned_positive_ratio": positive_ratio,
            "strict_relation_count": len(strict_rows), "strict_channel_count": len({int(row[key]) for row in strict_rows for key in ("channel_i_index", "channel_j_index")}),
            "seeds": seeds, "model_or_training_modified": False, "checkpoint_modified": False,
            "structure_pairs_modified": False, "fusion_performed": False,
        },
        "threshold_config": {"pot_q": cfg["eval"]["pot_q"], "pot_target_prob": cfg["eval"]["pot_target_prob"], "q99": 0.99},
        "single_score_seed_summary": summary_rows,
        "pot_point_complementarity_mean": {
            "independent128_detected_rec_missed": recovered_points,
            "independent128_new_false_alarm_count": new_false,
            "incremental_precision": incremental_precision,
        },
        "pot_recovered_rec_missed_events_mean": recovered_events,
        "independent128_aupr_better_than_rec_seed_count": independent_better_seeds,
        "supports_followup_fusion_experiment": supports_fusion,
        "event_label_limitation": f"Configured merged.csv contains {len(events)} contiguous positive-label event(s) after common alignment.",
    }
    with (output_dir / "real_swat_independent_score_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(_safe_json(summary), handle, ensure_ascii=False, indent=2, allow_nan=False)

    diagnosis = f"""# Step 4.3C Final Diagnosis

This stage evaluates the fixed six-relation strict set on the configured real SWaT test file. It does not perform score fusion.

1. The controlled relation-only result from Step 4.3B and the real SWaT result here are distinct; see `single_score_seed_summary.csv` for the real-data metrics.
2. Real SWaT does not provide a cooperative-anomaly type label, so independent detections cannot all be interpreted as pure cooperative anomalies.
3. POT independent-128 recovers a mean of {recovered_points:.1f} points missed by rec, with {new_false:.1f} new normal alarms and incremental precision {incremental_precision:.4f}.
4. It recovers a mean of {recovered_events:.1f} fully rec-missed contiguous event(s) under the configured labels.
5. The configured `merged.csv` yields only {len(events)} contiguous positive-label event(s) after alignment; attack-subtype event claims are therefore not possible from this file.
6. The fixed six relations cover only five channels and cannot represent all cooperative behavior in the SWaT system.
7. Follow-up fusion experimentation is {'supported' if supports_fusion else 'not supported'} by the predefined complementarity check, but no fusion or weight search was run here.
"""
    (output_dir / "final_diagnosis.md").write_text(diagnosis, encoding="utf-8")
    print(json.dumps(_safe_json({
        "original_test_points": data_info["original_test_point_count"],
        "aligned_points": len(labels_aligned), "positive_points": int(labels_aligned.sum()),
        "event_count": len(events), "independent_better_seed_count": independent_better_seeds,
        "pot_recovered_points": recovered_points, "pot_new_false_alarms": new_false,
        "pot_incremental_precision": incremental_precision,
        "supports_fusion": supports_fusion,
    }), ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 4.3C real SWaT independent-score evaluation")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--output-dir", default="outputs/stage4_3c_real_swat_single_score")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
