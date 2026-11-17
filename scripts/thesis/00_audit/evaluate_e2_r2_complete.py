"""E2-R2 complete literature-compatible evaluation from frozen raw scores only.

No model, checkpoint, scaler, coefficient, original dataset, or inference code is
opened here.  Historical Strict rows are read as immutable references; all new
diagnostics operate on the E2-R1D raw-score archives and frozen common masks.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/thesis_evaluation/literature_compatible"
RAW = BASE / "raw_scores"
MASKS = BASE / "common_masks"
OUT = BASE / "final"
RECOVERY = BASE / "recovery/recovery_summary.json"
METHODS = ("LSTM-AE", "USAD", "GDN", "Ours-REC", "Ours-DUAL", "Ours-TAIL")
SLUG = {x: x.lower().replace("-", "_") for x in METHODS}
DATASETS = ("SWaT", "HAI21")
DS_SLUG = {"SWaT": "swat", "HAI21": "hai"}
COMMON_COUNT = {"SWaT": 449791, "HAI21": 401685}
TOLERANCE = 1e-6


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E2 = load_module("e2r2_metric_core", ROOT / "scripts/thesis/00_audit/evaluate_literature_compatible.py")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def preconditions() -> dict[str, Any]:
    value = json.loads(RECOVERY.read_text(encoding="utf-8"))
    checks = {
        "final_status": value["status"] == "E2_RAW_SCORES_COMPLETED",
        "twelve_of_twelve": value["passed"] == value["total"] == 12,
        "all_raw_score_reproduced": all(x["status"] == "RAW_SCORE_REPRODUCED" for x in value["results"]),
        "strict_tolerance": max(x["max_diff"] for x in value["results"]) <= TOLERANCE,
        "common_counts": value["common_masks_unchanged"] == COMMON_COUNT,
    }
    if not all(checks.values()):
        raise RuntimeError(f"E2_R2_PRECONDITION_FAILED: {checks}")
    return {"status": "PASS", "checks": checks, "max_strict_difference": max(x["max_diff"] for x in value["results"])}


def segment_lengths(ids: np.ndarray) -> tuple[list[int], list[str]]:
    ids = np.asarray(ids).astype(str)
    if not len(ids):
        raise RuntimeError("empty common mask")
    cut = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    lengths = np.diff(cut).astype(int).tolist()
    names = [str(ids[i]) for i in cut[:-1]]
    if len(names) != len(set(names)):
        raise RuntimeError("segment/file id is non-contiguous")
    return lengths, names


def load_common(dataset: str) -> dict[str, np.ndarray]:
    path = MASKS / f"{DS_SLUG[dataset]}_common_mask.npz"
    with np.load(path, allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}
    if len(data["local_index"]) != COMMON_COUNT[dataset] or not np.all(data["valid_endpoint"]):
        raise RuntimeError(f"{dataset} common mask changed")
    return data


def load_and_map(dataset: str, method: str, common: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    path = RAW / DS_SLUG[dataset] / SLUG[method] / "test_raw_scores.npz"
    with np.load(path, allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}
    if not np.isfinite(data["score"]).all() or not np.isin(data["label"], [0, 1]).all():
        raise RuntimeError(f"invalid score/label: {dataset}/{method}")
    keep = np.asarray(data["local_index"] >= 64)
    mapped = {key: value[keep] for key, value in data.items() if value.ndim == 1 and len(value) == len(keep)}
    if len(mapped["score"]) != COMMON_COUNT[dataset]:
        raise RuntimeError(f"common count mismatch: {dataset}/{method}")
    common_id = np.asarray(common["file_id"] if dataset == "HAI21" else common["segment_id"]).astype(str)
    raw_id = np.asarray(mapped["file_id"]).astype(str)
    if not np.array_equal(raw_id, common_id):
        raise RuntimeError(f"identity mismatch (segment/file): {dataset}/{method}")
    for key in ("local_index", "global_index"):
        if not np.array_equal(mapped[key], common[key]):
            raise RuntimeError(f"identity mismatch ({key}): {dataset}/{method}")
    if dataset == "SWaT":
        if not np.array_equal(mapped["timestamp"], common["timestamp"]):
            raise RuntimeError(f"identity mismatch (timestamp): {dataset}/{method}")
        if not np.array_equal(mapped["label"], common["label"]):
            raise RuntimeError(f"identity mismatch (label): {dataset}/{method}")
    return {"path": np.asarray(str(path)), **mapped}


def q99_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    value = E2.threshold_metrics(labels, scores, threshold)
    value["auroc"] = float(roc_auc_score(labels, scores))
    value["aupr"] = float(average_precision_score(labels, scores))
    value["threshold"] = float(threshold)
    return value


def evaluate_one(dataset: str, method: str, strict: dict[str, Any], common: dict[str, np.ndarray],
                 shared_labels: np.ndarray | None) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    mapped = load_and_map(dataset, method, common)
    labels = np.asarray(mapped["label"], dtype=np.int8)
    scores = np.asarray(mapped["score"], dtype=np.float64)
    if shared_labels is not None and not np.array_equal(labels, shared_labels):
        raise RuntimeError(f"common labels disagree: {dataset}/{method}")
    lengths, names = segment_lengths(mapped["file_id"])
    intervals = E2.anomaly_intervals(labels, lengths)
    q99 = q99_metrics(labels, scores, float(strict["threshold_q99"]))
    oracle = E2.exact_oracle(labels, scores)
    pa_q99 = E2.pa_at_threshold(labels, scores, float(strict["threshold_q99"]), intervals)
    pa_oracle = E2.exact_pa_oracle(labels, scores, intervals)
    row = {
        "dataset": dataset, "method": method,
        "historical_auroc": strict["auroc"], "historical_aupr": strict["aupr"],
        "historical_f1_q99": strict["f1_q99"], "historical_fpr_q99": strict["fpr_q99"],
        "q99_threshold": strict["threshold_q99"],
        "common_auroc": q99["auroc"], "common_aupr": q99["aupr"],
        "common_precision_q99": q99["precision"], "common_recall_q99": q99["recall"],
        "common_f1_q99": q99["f1"], "common_fpr_q99": q99["fpr"],
        "oracle_threshold": oracle["threshold"], "oracle_precision": oracle["precision"],
        "oracle_recall": oracle["recall"], "oracle_f1": oracle["f1"], "oracle_fpr": oracle["fpr"],
        "pa_q99_precision": pa_q99["precision"], "pa_q99_recall": pa_q99["recall"],
        "pa_q99_f1": pa_q99["f1"], "pa_q99_fpr": pa_q99["fpr"],
        "pa_oracle_threshold": pa_oracle["threshold"], "pa_oracle_precision": pa_oracle["precision"],
        "pa_oracle_recall": pa_oracle["recall"], "pa_oracle_f1": pa_oracle["f1"], "pa_oracle_fpr": pa_oracle["fpr"],
        "n_common_points": len(labels), "n_anomaly_points": int(labels.sum()),
        "n_normal_points": int((labels == 0).sum()), "n_anomaly_intervals": len(intervals),
        "delta_oracle": oracle["f1"] - q99["f1"], "delta_pa_q99": pa_q99["f1"] - q99["f1"],
        "delta_pa_oracle": pa_oracle["f1"] - q99["f1"],
        "threshold_ratio": oracle["threshold"] / strict["threshold_q99"] if strict["threshold_q99"] > 0 else None,
    }
    per_segment = []
    offset = 0
    for name, length in zip(names, lengths):
        count = sum(1 for a, b, sid in intervals if sid == len(per_segment))
        per_segment.append({"segment_or_file": name, "point_count": length, "anomaly_interval_count": count})
        offset += length
    audit = {"dataset": dataset, "method": method, "raw_archive": str(mapped["path"]),
             "common_mapping": "EXACT_IDENTITY_MATCH", "segment_or_file_count": len(lengths),
             "point_anomaly_interval_count": len(intervals), "per_segment_or_file": per_segment,
             "pa_q99_hit_intervals": pa_q99["hit_intervals"], "pa_oracle_hit_intervals": pa_oracle["hit_intervals"],
             "cross_boundary_adjustment": False, "normal_false_positives_removed": 0,
             "oracle_unique_thresholds": oracle["n_unique_thresholds"], "pa_oracle_unique_thresholds": pa_oracle["n_unique_thresholds"]}
    return row, audit, labels


def descending_ranks(rows: list[dict[str, Any]]) -> None:
    metrics = {
        "common_auroc": "rank_common_auroc", "common_aupr": "rank_common_aupr",
        "oracle_f1": "rank_oracle_f1", "pa_q99_f1": "rank_pa_q99_f1", "pa_oracle_f1": "rank_pa_oracle_f1",
    }
    for dataset in DATASETS:
        subset = [x for x in rows if x["dataset"] == dataset]
        for metric, field in metrics.items():
            ordered = sorted({float(x[metric]) for x in subset}, reverse=True)
            rank = {value: i + 1 for i, value in enumerate(ordered)}
            for row in subset:
                row[field] = rank[float(row[metric])]


def historical_ours_rec_regression(strict_map: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    path = RAW / "swat/ours_rec/test_raw_scores.npz"
    with np.load(path, allow_pickle=False) as z:
        score = np.asarray(z["score"], dtype=np.float64); label = np.asarray(z["label"], dtype=np.int8)
        ids = np.asarray(z["file_id"]).astype(str)
    lengths, _ = segment_lengths(ids); intervals = E2.anomaly_intervals(label, lengths)
    q = float(strict_map[("SWaT", "Ours-REC")]["threshold_q99"])
    oracle = E2.exact_oracle(label, score); pa_q99 = E2.pa_at_threshold(label, score, q, intervals); pa_oracle = E2.exact_pa_oracle(label, score, intervals)
    expected = {"oracle_f1": 0.768959, "pa_q99_f1": 0.272041, "pa_oracle_f1": 0.887275}
    actual = {"oracle_f1": oracle["f1"], "pa_q99_f1": pa_q99["f1"], "pa_oracle_f1": pa_oracle["f1"]}
    delta = {key: abs(actual[key] - expected[key]) for key in expected}
    return {"status": "PASS" if max(delta.values()) <= TOLERANCE else "FAIL", "expected_rounded": expected, "actual": actual,
            "absolute_difference": delta, "max_absolute_difference": max(delta.values()), "historical_endpoint_count": len(label)}


def table_payloads(rows: list[dict[str, Any]], strict_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    table_a = [{"Dataset": x["dataset"], "Method": x["method"], "AUROC": x["auroc"], "AUPR": x["aupr"],
                "F1-q99": x["f1_q99"], "FPR-q99": x["fpr_q99"], "Protocol": "HISTORICAL_STRICT",
                "Test_tuning": False, "PA": False} for x in strict_rows]
    table_b = [{"Dataset": x["dataset"], "Method": x["method"], "AUROC": x["common_auroc"], "AUPR": x["common_aupr"],
                "F1-q99": x["common_f1_q99"], "FPR-q99": x["common_fpr_q99"], "Protocol": "COMMON_MASK_Q99",
                "Same_endpoint_set": True, "Threshold_source": "Frozen Normal Validation q99"} for x in rows]
    table_c = [{"Dataset": x["dataset"], "Method": x["method"], "F1-q99-common": x["common_f1_q99"],
                "F1-Oracle": x["oracle_f1"], "PA-F1@q99": x["pa_q99_f1"], "PA-F1-Oracle": x["pa_oracle_f1"],
                "Oracle_note": "Test-best threshold", "PA_note": "Classic Point Adjustment",
                "PA_Oracle_note": "Test-best threshold + PA"} for x in rows]
    detailed = []
    fields = ["dataset", "method", "common_auroc", "common_aupr", "common_f1_q99", "oracle_f1", "pa_q99_f1", "pa_oracle_f1",
              "common_fpr_q99", "oracle_fpr", "delta_oracle", "delta_pa_q99", "delta_pa_oracle",
              "rank_common_auroc", "rank_common_aupr", "rank_oracle_f1", "rank_pa_q99_f1", "rank_pa_oracle_f1"]
    for row in rows:
        detailed.append({key: row[key] for key in fields})
    transfer = [{"dataset":x["dataset"],"method":x["method"],"q99_threshold":x["q99_threshold"],"oracle_threshold":x["oracle_threshold"],
                 "F1_q99":x["common_f1_q99"],"F1_oracle":x["oracle_f1"],"delta_oracle":x["delta_oracle"],
                 "FPR_q99":x["common_fpr_q99"],"FPR_oracle":x["oracle_fpr"],"PA_F1_q99":x["pa_q99_f1"],
                 "PA_F1_oracle":x["pa_oracle_f1"],"delta_pa_q99":x["delta_pa_q99"],"delta_pa_oracle":x["delta_pa_oracle"],
                 "threshold_ratio":x["threshold_ratio"]} for x in rows]
    return {"table_a_historical_strict":table_a,"table_b_common_mask_strict":table_b,"table_c_literature_compatible":table_c,
            "swat_literature_compatible":[x for x in detailed if x["dataset"]=="SWaT"],
            "hai_literature_compatible":[x for x in detailed if x["dataset"]=="HAI21"],"threshold_transfer_analysis":transfer}


def literature_schema(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result=[]; primary={"AUROC":"historical_auroc","AUPR":"historical_aupr","F1-q99":"historical_f1_q99","FPR-q99":"historical_fpr_q99"}
    for row in rows:
        if not row["method"].startswith("Ours-"): continue
        for metric,field in primary.items():
            result.append({"method":row["method"],"year":2026,"dataset":row["dataset"],"metric_name":metric,
                "reported_value":row[field],"protocol":"HISTORICAL_STRICT; frozen Normal Validation q99 for operating-point metrics",
                "PA":False,"Test_best":False,"source_type":"LOCAL_REPRODUCED_RESULT",
                "source":"outputs/thesis_evaluation/literature_compatible/final/table_a_historical_strict.csv",
                "direct_comparability":"LOCAL_PRIMARY_ONLY","notes":"No Test tuning; no PA; E2-R1D raw score reproduction passed"})
    for method in ("Park-Shim 2026","LatentFlow 2026","MAD-TCN 2026","CCM-TAD 2026"):
        result.append({"method":method,"year":2026,"dataset":"","metric_name":"","reported_value":"",
            "protocol":"PENDING_SEPARATE_LITERATURE_AUDIT","PA":"UNKNOWN","Test_best":"UNKNOWN",
            "source_type":"EXTERNAL_PLACEHOLDER","source":"","direct_comparability":False,
            "notes":"Schema placeholder only; external paper value intentionally blank"})
    return result


def main() -> int:
    started=time.perf_counter(); gate=preconditions(); OUT.mkdir(parents=True,exist_ok=True)
    strict_rows=E2.strict_rows(); strict_map={(x["dataset"],x["method"]):x for x in strict_rows}
    if len(strict_rows)!=12: raise RuntimeError("expected 12 historical Strict rows")
    rows=[]; audits=[]; hashes={}; common_labels={}
    for dataset in DATASETS:
        common=load_common(dataset); shared=None
        hashes[f"{dataset}_common_mask_sha256"]=sha256(MASKS/f"{DS_SLUG[dataset]}_common_mask.npz")
        for method in METHODS:
            row,audit,labels=evaluate_one(dataset,method,strict_map[(dataset,method)],common,shared)
            if shared is None: shared=labels
            rows.append(row); audits.append(audit)
            hashes[f"{dataset}_{method}_raw_score_sha256"]=sha256(RAW/DS_SLUG[dataset]/SLUG[method]/"test_raw_scores.npz")
        common_labels[dataset]=shared
    descending_ranks(rows)
    regression=historical_ours_rec_regression(strict_map)
    if regression["status"]!="PASS": raise RuntimeError(f"historical Ours-REC regression failed: {regression}")
    payloads=table_payloads(rows,strict_rows); payloads["literature_comparison_schema"]=literature_schema(rows)
    for name,value in payloads.items(): dump_json(OUT/f"{name}_rows.json",value)
    oracle_details={"protocol":{"comparator":">","sweep":"all unique scores plus all-positive boundary","tie_rule":"largest numeric threshold among exact best-F1 ties"},
                    "rows":[{k:v for k,v in x.items() if k.startswith("oracle_") or k in {"dataset","method","q99_threshold","threshold_ratio","delta_oracle"}} for x in rows],
                    "swat_historical_ours_rec_regression":regression}
    pa_audit={"protocol":{"classic_point_adjustment":True,"normal_predictions_unchanged":True,"boundary_source":"common-mask segment/file identity",
                           "pa_oracle":"exact interval-max and normal-score sweep"},"rows":audits}
    dump_json(OUT/"oracle_details.json",oracle_details); dump_json(OUT/"pa_audit.json",pa_audit)
    manifest={"phase":"Thesis Evaluation-E2-R2","status":"LITERATURE_COMPATIBLE_EVAL_COMPLETED","preconditions":gate,
              "datasets":list(DATASETS),"methods":list(METHODS),"combination_count":12,"common_mask_counts":COMMON_COUNT,
              "strict_primary_unchanged":True,"common_mask_protocol":"exact identity intersection; no length trimming",
              "metrics":{"historical_strict":"Primary; unchanged","common_q99":"frozen Validation q99 on common endpoints",
                         "oracle":"exact Test-best threshold diagnostic","pa_q99":"classic PA after frozen q99","pa_oracle":"exact Test-best threshold plus PA diagnostic"},
              "historical_ours_rec_regression":regression,"source_hashes":hashes,"elapsed_seconds":time.perf_counter()-started,
              "restrictions":{"training":False,"inference":False,"model_change":False,"score_change":False,"scaler_recalculation":False,
                              "ridge_recalculation":False,"tail_recalibration":False,"checkpoint_read":False,"hp_search":False,
                              "new_fusion":False,"test_driven_model_change":False,"strict_result_overwrite":False,"commit":False,"push":False}}
    dump_json(OUT/"manifest.json",manifest)
    print(json.dumps({"status":manifest["status"],"elapsed_seconds":manifest["elapsed_seconds"],"rows":rows,
                      "historical_ours_rec_regression":regression},ensure_ascii=False,allow_nan=False),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
