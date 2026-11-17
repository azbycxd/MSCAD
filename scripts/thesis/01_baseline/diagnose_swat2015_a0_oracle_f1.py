"""Compute a Test-label oracle F1 diagnostic for the unchanged S1 A0 score."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat2015_official import OfficialSWaTError, resolve_repo_path  # noqa: E402
from utils.metrics import binary_metrics  # noqa: E402


TOLERANCE = 1e-6
TIE_TOLERANCE = 1e-12


def evaluate(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (score >= threshold).astype(np.int8)
    result = binary_metrics(y_true, prediction, score)
    tn = int(((y_true == 0) & (prediction == 0)).sum())
    fp = int(((y_true == 0) & (prediction == 1)).sum())
    fn = int(((y_true == 1) & (prediction == 0)).sum())
    tp = int(((y_true == 1) & (prediction == 1)).sum())
    result.update({"fpr": fp / max(tn + fp, 1), "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return result


def oracle_from_test_labels(y_true: np.ndarray, score: np.ndarray) -> tuple[float, dict[str, Any]]:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator, out=np.zeros_like(denominator), where=denominator > 0)
    best = float(np.max(f1))
    candidates = np.flatnonzero(np.abs(f1 - best) <= TIE_TOLERANCE)
    evaluated: list[tuple[float, float, dict[str, Any]]] = []
    for index in candidates:
        threshold = float(thresholds[index])
        metrics = evaluate(y_true, score, threshold)
        evaluated.append((metrics["fpr"], -threshold, metrics))
    _, neg_threshold, metrics = min(evaluated, key=lambda item: (item[0], item[1]))
    return -neg_threshold, metrics


def verify_two_frozen_segments(timestamp: np.ndarray, manifest: dict[str, Any]) -> None:
    offset = 0
    for segment in manifest["test_segments"]:
        count = int(segment["row_count"]) - 63
        expected_start = int(np.datetime64(segment["start"], "s").astype(np.int64)) + 63
        expected_end = int(np.datetime64(segment["end_inclusive"], "s").astype(np.int64))
        part = timestamp[offset : offset + count]
        if len(part) != count or int(part[0]) != expected_start or int(part[-1]) != expected_end:
            raise OfficialSWaTError("Oracle endpoints differ from frozen A0 segments")
        if len(part) > 1 and not np.all(np.diff(part) == 1):
            raise OfficialSWaTError("Oracle endpoint window crosses a hard session boundary")
        offset += count
    if offset != len(timestamp):
        raise OfficialSWaTError("Oracle endpoint count differs from frozen A0")


def run(a0_output: Path, segment_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing S4 output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    a0_metrics = json.loads((a0_output / "metrics.json").read_text(encoding="utf-8"))
    with np.load(a0_output / "test_scores.npz", allow_pickle=False) as payload:
        score = np.asarray(payload["endpoint_mae"], dtype=np.float32)
        labels = np.asarray(payload["label"], dtype=np.int8)
        timestamp = np.asarray(payload["endpoint_timestamp"], dtype=np.int64)
    expected_count = int(a0_metrics["test_evaluation_point_count"])
    if len(score) != expected_count or len(labels) != expected_count or not np.isfinite(score).all():
        raise OfficialSWaTError("Oracle did not read the complete unchanged A0 score/label arrays")
    manifest = json.loads(segment_manifest_path.read_text(encoding="utf-8"))
    verify_two_frozen_segments(timestamp, manifest)
    oracle_threshold, oracle = oracle_from_test_labels(labels, score)
    formal = a0_metrics["primary_endpoint_mae"]
    if abs(oracle["auroc"] - formal["auroc"]) > TOLERANCE or abs(oracle["aupr"] - formal["aupr"]) > TOLERANCE:
        raise OfficialSWaTError("implementation mismatch: unchanged-score ranking is not invariant")
    if oracle["f1"] + TIE_TOLERANCE < formal["f1"]:
        raise OfficialSWaTError("oracle F1 is unexpectedly below formal q99 F1")
    if oracle["f1"] >= 0.65:
        diagnosis = "THRESHOLD_TRANSFER_DOMINATES_F1_FAILURE"
    elif oracle["f1"] >= 0.45:
        diagnosis = "BOTH_SCORE_AND_THRESHOLD_LIMITED"
    else:
        diagnosis = "RECONSTRUCTION_SCORE_ITSELF_LIMITED"
    formal_threshold = float(a0_metrics["thresholds"]["endpoint_mae"])
    result = {
        "experiment_id": "S15-A0-ORACLE-F1",
        "scientific_status": "ORACLE_DIAGNOSTIC_ONLY / INVALID_FOR_FORMAL_BENCHMARK",
        "formal_result": False,
        "oracle_only": True,
        "threshold_selected_from": "Test point labels; diagnostic only",
        "formal_q99_threshold": formal_threshold,
        "formal_q99_f1": formal["f1"],
        "oracle_threshold": oracle_threshold,
        "oracle_threshold_comparator": ">= (sklearn precision_recall_curve convention)",
        "oracle_precision": oracle["precision"],
        "oracle_recall": oracle["recall"],
        "oracle_f1": oracle["f1"],
        "oracle_fpr": oracle["fpr"],
        "tn": oracle["tn"],
        "fp": oracle["fp"],
        "fn": oracle["fn"],
        "tp": oracle["tp"],
        "oracle_f1_gain": oracle["f1"] - formal["f1"],
        "threshold_difference": oracle_threshold - formal_threshold,
        "threshold_ratio": oracle_threshold / formal_threshold,
        "auroc": oracle["auroc"],
        "aupr": oracle["aupr"],
        "ranking_invariant": "PASS",
        "diagnosis": diagnosis,
        "test_endpoint_count": len(score),
        "test_segment_count": len(manifest["test_segments"]),
        "score_modification": False,
        "retraining": False,
        "formal_threshold_change": False,
    }
    output_path = output_dir / "oracle_f1_diagnostic.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0-output", default="outputs/thesis_midterm_s1/a0_rec_baseline")
    parser.add_argument("--segment-manifest", default="outputs/thesis_t25_swat2015_protocol/segment_manifest.json")
    parser.add_argument("--output", default="outputs/thesis_midterm_s4_oracle")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(
        resolve_repo_path(ROOT, args.a0_output),
        resolve_repo_path(ROOT, args.segment_manifest),
        resolve_repo_path(ROOT, args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
