"""Audit persisted scores and compute exact literature-compatible diagnostics.

This module never imports a model, opens a checkpoint, or performs inference.
It reads only frozen metrics/manifests and already-persisted score arrays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
METHODS = ("LSTM-AE", "USAD", "GDN", "Ours-REC", "Ours-DUAL", "Ours-TAIL")
DATASETS = ("SWaT", "HAI21")
TOLERANCE = 1e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def confusion_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    tp = int(((labels == 1) & (prediction == 1)).sum())
    fp = int(((labels == 0) & (prediction == 1)).sum())
    fn = int(((labels == 1) & (prediction == 0)).sum())
    tn = int(((labels == 0) & (prediction == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    return confusion_metrics(labels, np.asarray(scores) > threshold)


def exact_oracle(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    """Exact F1 sweep for the frozen `score > threshold` convention.

    All unique score thresholds plus the all-positive boundary are evaluated.
    Among exactly tied best F1 values, the largest numeric threshold wins.
    """
    started = time.perf_counter()
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    unique = np.unique(scores)
    candidates = np.concatenate(([np.nextafter(unique[0], -np.inf)], unique))
    normal = np.sort(scores[labels == 0])
    anomaly = np.sort(scores[labels == 1])
    fp = len(normal) - np.searchsorted(normal, candidates, side="right")
    tp = len(anomaly) - np.searchsorted(anomaly, candidates, side="right")
    fn = len(anomaly) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    best_f1 = float(f1.max())
    tied = np.flatnonzero(f1 == best_f1)
    index = int(tied[np.argmax(candidates[tied])])
    threshold = float(candidates[index])
    metrics = threshold_metrics(labels, scores, threshold)
    metrics.update({
        "threshold": threshold,
        "n_unique_thresholds": int(len(candidates)),
        "runtime_seconds": time.perf_counter() - started,
        "tie_rule": "largest numeric threshold among exact best-F1 ties",
        "comparator": ">",
    })
    return metrics


def anomaly_intervals(labels: np.ndarray, segment_lengths: list[int]) -> list[tuple[int, int, int]]:
    """Return [start,end) label-one intervals, never crossing a segment boundary."""
    labels = np.asarray(labels, dtype=np.int8)
    if sum(segment_lengths) != len(labels):
        raise ValueError("segment lengths do not cover the aligned label timeline")
    intervals: list[tuple[int, int, int]] = []
    offset = 0
    for segment_id, length in enumerate(segment_lengths):
        segment = labels[offset : offset + length]
        padded = np.pad(segment, (1, 1))
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        intervals.extend((offset + int(a), offset + int(b), segment_id) for a, b in zip(starts, ends))
        offset += length
    return intervals


def point_adjustment(prediction: np.ndarray, labels: np.ndarray, intervals: list[tuple[int, int, int]]) -> np.ndarray:
    adjusted = np.asarray(prediction, dtype=np.int8).copy()
    original = adjusted.copy()
    for start, end, _ in intervals:
        if original[start:end].any():
            adjusted[start:end] = 1
    if np.any((original == 1) & (adjusted == 0)):
        raise AssertionError("point adjustment removed a positive prediction")
    if np.any((labels == 0) & (adjusted != original)):
        raise AssertionError("point adjustment changed a false-positive/normal position")
    return adjusted


def pa_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float, intervals: list[tuple[int, int, int]]) -> dict[str, float | int]:
    prediction = np.asarray(scores) > threshold
    adjusted = point_adjustment(prediction, labels, intervals)
    result = confusion_metrics(labels, adjusted)
    result["threshold"] = float(threshold)
    result["hit_intervals"] = int(sum(bool(prediction[a:b].any()) for a, b, _ in intervals))
    return result


def exact_pa_oracle(labels: np.ndarray, scores: np.ndarray, intervals: list[tuple[int, int, int]]) -> dict[str, float | int]:
    """Exact PA-F1 sweep using interval maxima; no threshold grid approximation."""
    started = time.perf_counter()
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    unique = np.unique(scores)
    candidates = np.concatenate(([np.nextafter(unique[0], -np.inf)], unique))
    normal = np.sort(scores[labels == 0])
    fp = len(normal) - np.searchsorted(normal, candidates, side="right")

    interval_max = np.asarray([scores[a:b].max() for a, b, _ in intervals], dtype=np.float64)
    interval_length = np.asarray([b - a for a, b, _ in intervals], dtype=np.int64)
    order = np.argsort(interval_max, kind="stable")
    sorted_max = interval_max[order]
    prefix_length = np.concatenate(([0], np.cumsum(interval_length[order], dtype=np.int64)))
    cut = np.searchsorted(sorted_max, candidates, side="right")
    tp = int(interval_length.sum()) - prefix_length[cut]
    fn = int(interval_length.sum()) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    best_f1 = float(f1.max())
    tied = np.flatnonzero(f1 == best_f1)
    index = int(tied[np.argmax(candidates[tied])])
    threshold = float(candidates[index])
    result = pa_at_threshold(labels, scores, threshold, intervals)
    result.update({
        "n_unique_thresholds": int(len(candidates)),
        "runtime_seconds": time.perf_counter() - started,
        "tie_rule": "largest numeric threshold among exact best-F1 ties",
        "comparator": ">",
    })
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def strict_rows() -> list[dict]:
    rows: list[dict] = []
    for directory, method in (("lstm_ae", "LSTM-AE"), ("usad", "USAD"), ("gdn", "GDN")):
        for dataset, prefix in (("SWaT", "swat"), ("HAI21", "hai")):
            path = ROOT / "outputs" / "thesis_baselines" / directory / f"{prefix}_metrics.json"
            value = read_json(path)
            rows.append({
                "dataset": dataset,
                "method": method,
                "auroc": float(value["AUROC"]),
                "aupr": float(value["AUPR"]),
                "precision_q99": float(value["Precision"]),
                "recall_q99": float(value["Recall"]),
                "f1_q99": float(value["F1"]),
                "fpr_q99": float(value["FPR"]),
                "threshold_q99": float(value["threshold"]),
                "n_points": int(value["endpoint_count"]),
                "strict_source": str(path.relative_to(ROOT)).replace("\\", "/"),
            })

    path = ROOT / "outputs" / "thesis_innovation3b_tail_fusion_final" / "cross_dataset_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    method_map = {"REC": "Ours-REC", "DUAL": "Ours-DUAL", "TAIL_FUSION": "Ours-TAIL"}
    for item in raw:
        if item["dataset"] not in DATASETS or item["method"] not in method_map:
            continue
        rows.append({
            "dataset": item["dataset"],
            "method": method_map[item["method"]],
            "auroc": float(item["auroc"]),
            "aupr": float(item["aupr"]),
            "precision_q99": float(item["precision"]),
            "recall_q99": float(item["recall"]),
            "f1_q99": float(item["f1"]),
            "fpr_q99": float(item["fpr"]),
            "threshold_q99": float(item["threshold"]),
            "n_points": int(item["tn"]) + int(item["fp"]) + int(item["fn"]) + int(item["tp"]),
            "strict_source": str(path.relative_to(ROOT)).replace("\\", "/"),
        })
    order = {(dataset, method): i for i, (dataset, method) in enumerate((d, m) for d in DATASETS for m in METHODS)}
    return sorted(rows, key=lambda row: order[(row["dataset"], row["method"])])


def availability(strict: list[dict]) -> list[dict]:
    by_key = {(r["dataset"], r["method"]): r for r in strict}
    rows = []
    baseline_dir = {"LSTM-AE": "lstm_ae", "USAD": "usad", "GDN": "gdn"}
    for dataset in DATASETS:
        for method in METHODS:
            strict_row = by_key[(dataset, method)]
            if (dataset, method) == ("SWaT", "Ours-REC"):
                score_path = "outputs/thesis_midterm_s1/a0_rec_baseline/test_scores.npz"
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "score_path": score_path + "#endpoint_mae",
                    "label_path": score_path + "#label",
                    "mask_path": score_path + "#endpoint_timestamp + outputs/thesis_midterm_s1/a0_rec_baseline/metrics.json#segments",
                    "q99_threshold": strict_row["threshold_q99"],
                    "n_scores": strict_row["n_points"],
                    "n_labels": strict_row["n_points"],
                    "available": True,
                    "provenance": "S1 A0 frozen archive; Innovation-3B formal REC row",
                    "notes": "higher=more anomalous; two independently windowed hard-gap segments; no score transformation",
                })
            else:
                if method in baseline_dir:
                    prefix = "swat" if dataset == "SWaT" else "hai"
                    provenance = f"outputs/thesis_baselines/{baseline_dir[method]}/{prefix}_metrics.json + manifest.json"
                else:
                    provenance = "outputs/thesis_innovation3b_tail_fusion_final/cross_dataset_metrics.csv + manifest.json"
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "score_path": "",
                    "label_path": "",
                    "mask_path": "",
                    "q99_threshold": strict_row["threshold_q99"],
                    "n_scores": None,
                    "n_labels": None,
                    "available": False,
                    "provenance": provenance,
                    "notes": "SCORE_NOT_PERSISTED; strict aggregate metrics exist, but raw score/label/mask vectors do not; no reinference permitted",
                })
    return rows


def run() -> dict:
    strict = strict_rows()
    if len(strict) != 12:
        raise RuntimeError(f"expected 12 strict rows, found {len(strict)}")
    avail = availability(strict)
    score_file = ROOT / "outputs/thesis_midterm_s1/a0_rec_baseline/test_scores.npz"
    metrics_file = ROOT / "outputs/thesis_midterm_s1/a0_rec_baseline/metrics.json"
    a0_metrics = read_json(metrics_file)
    with np.load(score_file, allow_pickle=False) as payload:
        scores = np.asarray(payload["endpoint_mae"], dtype=np.float64)
        labels = np.asarray(payload["label"], dtype=np.int8)
        timestamps = np.asarray(payload["endpoint_timestamp"], dtype=np.int64)
    if len(scores) != len(labels) or len(scores) != len(timestamps) or not np.isfinite(scores).all():
        raise RuntimeError("invalid frozen Ours-REC arrays")
    segment_lengths = [int(x["window_count"]) for x in a0_metrics["segments"]]
    if sum(segment_lengths) != len(scores):
        raise RuntimeError("frozen segment counts do not cover scores")
    offset = 0
    segment_audit = []
    for item, length in zip(a0_metrics["segments"], segment_lengths):
        part = timestamps[offset : offset + length]
        if len(part) > 1 and not np.all(np.diff(part) == 1):
            raise RuntimeError("timestamp discontinuity inside frozen segment")
        segment_audit.append({
            "segment_id": item["segment_id"],
            "length": length,
            "first_timestamp": int(part[0]),
            "last_timestamp": int(part[-1]),
        })
        offset += length

    strict_ref = next(x for x in strict if x["dataset"] == "SWaT" and x["method"] == "Ours-REC")
    strict_calc = threshold_metrics(labels, scores, strict_ref["threshold_q99"])
    strict_calc["auroc"] = float(roc_auc_score(labels, scores))
    strict_calc["aupr"] = float(average_precision_score(labels, scores))
    checks = {
        "auroc": abs(strict_calc["auroc"] - strict_ref["auroc"]),
        "aupr": abs(strict_calc["aupr"] - strict_ref["aupr"]),
        "precision": abs(strict_calc["precision"] - strict_ref["precision_q99"]),
        "recall": abs(strict_calc["recall"] - strict_ref["recall_q99"]),
        "f1": abs(strict_calc["f1"] - strict_ref["f1_q99"]),
        "fpr": abs(strict_calc["fpr"] - strict_ref["fpr_q99"]),
    }
    if max(checks.values()) > TOLERANCE:
        raise RuntimeError(f"STRICT_RESULT_MISMATCH: {checks}")

    intervals = anomaly_intervals(labels, segment_lengths)
    oracle = exact_oracle(labels, scores)
    pa_q99 = pa_at_threshold(labels, scores, strict_ref["threshold_q99"], intervals)
    pa_oracle = exact_pa_oracle(labels, scores, intervals)
    old_oracle_f1 = 0.768959
    old_oracle_check = abs(float(oracle["f1"]) - old_oracle_f1)
    if old_oracle_check > 1e-6:
        raise RuntimeError(f"Ours-REC oracle cross-check failed: {oracle['f1']} vs {old_oracle_f1}")

    literature = {
        "SWaT|Ours-REC": {
            "f1_oracle": oracle["f1"],
            "threshold_oracle": oracle["threshold"],
            "precision_oracle": oracle["precision"],
            "recall_oracle": oracle["recall"],
            "fpr_oracle": oracle["fpr"],
            "pa_f1_q99": pa_q99["f1"],
            "pa_precision_q99": pa_q99["precision"],
            "pa_recall_q99": pa_q99["recall"],
            "pa_fpr_q99": pa_q99["fpr"],
            "pa_f1_oracle": pa_oracle["f1"],
            "pa_threshold_oracle": pa_oracle["threshold"],
            "pa_precision_oracle": pa_oracle["precision"],
            "pa_recall_oracle": pa_oracle["recall"],
            "pa_fpr_oracle": pa_oracle["fpr"],
            "n_unique_thresholds": oracle["n_unique_thresholds"],
            "oracle_runtime_seconds": oracle["runtime_seconds"],
            "pa_oracle_runtime_seconds": pa_oracle["runtime_seconds"],
            "strict_reproduction": "PASS",
        }
    }

    metric_rows = []
    for row in strict:
        key = f"{row['dataset']}|{row['method']}"
        result = literature.get(key)
        full = dict(row)
        full["availability_status"] = "AVAILABLE" if result else "SCORE_NOT_PERSISTED"
        if result:
            full.update(result)
            full["delta_oracle"] = result["f1_oracle"] - row["f1_q99"]
            full["delta_pa_q99"] = result["pa_f1_q99"] - row["f1_q99"]
            full["delta_pa_oracle"] = result["pa_f1_oracle"] - row["f1_q99"]
            full["notes"] = "retrospective diagnostics only; Oracle and PA are not deployable strict results"
        else:
            for name in (
                "f1_oracle", "threshold_oracle", "precision_oracle", "recall_oracle", "fpr_oracle",
                "pa_f1_q99", "pa_precision_q99", "pa_recall_q99", "pa_fpr_q99",
                "pa_f1_oracle", "pa_threshold_oracle", "pa_precision_oracle", "pa_recall_oracle", "pa_fpr_oracle",
                "delta_oracle", "delta_pa_q99", "delta_pa_oracle", "n_unique_thresholds",
                "oracle_runtime_seconds", "pa_oracle_runtime_seconds", "strict_reproduction",
            ):
                full[name] = None
            full["notes"] = "SCORE_NOT_PERSISTED; no retraining or reinference performed"
        metric_rows.append(full)

    interval_rows = []
    for index, (start, end, segment_id) in enumerate(intervals, start=1):
        interval_rows.append({
            "interval_index": index,
            "segment_id": segment_audit[segment_id]["segment_id"],
            "start_aligned_index": start,
            "end_exclusive_aligned_index": end,
            "length": end - start,
            "q99_hit": bool((scores[start:end] > strict_ref["threshold_q99"]).any()),
            "pa_oracle_hit": bool((scores[start:end] > pa_oracle["threshold"]).any()),
        })

    return {
        "availability_rows": avail,
        "metric_rows": metric_rows,
        "strict_rows": strict,
        "literature": literature,
        "threshold_details": {
            "SWaT": {
                "Ours-REC": {"q99": strict_ref["threshold_q99"], "oracle": oracle, "pa_oracle": pa_oracle},
                "unavailable_methods": [m for m in METHODS if m != "Ours-REC"],
            },
            "HAI21": {"unavailable_methods": list(METHODS)},
        },
        "event_pa_audit": {
            "SWaT": {
                "segments": segment_audit,
                "anomaly_interval_count": len(intervals),
                "q99_hit_intervals": pa_q99["hit_intervals"],
                "pa_oracle_hit_intervals": pa_oracle["hit_intervals"],
                "intervals": interval_rows,
                "cross_segment_adjustment": False,
                "false_positives_removed": 0,
            },
            "HAI21": {
                "status": "SCORE_NOT_PERSISTED",
                "cross_file_adjustment": False,
                "notes": "No PA was computed because no frozen raw score vector is available for any in-scope method.",
            },
        },
        "checks": {
            "strict_absolute_differences": checks,
            "strict_tolerance": TOLERANCE,
            "strict_reproduction": "PASS",
            "old_oracle_f1_reference": old_oracle_f1,
            "old_oracle_absolute_difference": old_oracle_check,
            "old_oracle_cross_check": "PASS",
            "common_mask_note": "Frozen Ours-REC strict timeline has 449793 endpoints; completed baseline metrics record 449791. Existing strict metrics were not redefined.",
            "no_training": True,
            "no_checkpoint_read": True,
            "no_score_modification": True,
            "no_score_transformation": True,
        },
        "source_hashes": {
            "swat_ours_rec_scores_sha256": sha256(score_file),
            "strict_ours_table_sha256": sha256(ROOT / "outputs/thesis_innovation3b_tail_fusion_final/cross_dataset_metrics.csv"),
        },
        "status": "LITERATURE_COMPATIBLE_EVAL_PARTIAL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    result = run()
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "available": sum(int(x["available"]) for x in result["availability_rows"]),
        "total": len(result["availability_rows"]),
        "swat_ours_rec": result["literature"]["SWaT|Ours-REC"],
        "checks": result["checks"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
