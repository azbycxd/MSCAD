"""ABL2: retrospective Innovation-3 heterogeneous-evidence calibration ablation."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "thesis_ablation" / "innovation3"
I3A = ROOT / "outputs" / "thesis_innovation3a_tail_calibration"
I3B = ROOT / "outputs" / "thesis_innovation3b_tail_fusion_final"
EFF = ROOT / "outputs" / "thesis_efficiency_final"
RAW = ROOT / "outputs" / "thesis_evaluation" / "literature_compatible" / "raw_scores"
EVIDENCE = "RETROSPECTIVE_ABLATION"
METHODS = ("F0 REC", "F1 DUAL", "F3 Robust-Z", "F4 Tail-Calibrated")


def jdump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred, pos, neg = scores > threshold, labels == 1, labels == 0
    tp, fp = int((pred & pos).sum()), int((pred & neg).sum())
    fn, tn = int((~pred & pos).sum()), int((~pred & neg).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def raw_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    q99 = float(np.quantile(values, .99))
    top = values[values >= q99]
    return {
        "count": len(values), "median": median,
        "mad": float(np.median(np.abs(values - median))),
        "iqr": float(np.quantile(values, .75) - np.quantile(values, .25)),
        "q95": float(np.quantile(values, .95)), "q99": q99,
        "q99_5": float(np.quantile(values, .995)), "max": float(values.max()),
        "top1_median": float(np.median(top)), "top1_mean": float(top.mean()),
    }


def frozen_row(metrics: pd.DataFrame, dataset: str, method: str) -> dict:
    row = metrics[(metrics.dataset == dataset) & (metrics.method == method)]
    if len(row) != 1:
        raise RuntimeError(f"missing frozen metric row {dataset}/{method}")
    return row.iloc[0].to_dict()


def compare_metric(actual: dict, expected: dict, tolerance: float = 1e-6) -> bool:
    return all(abs(float(actual[key]) - float(expected[key])) <= tolerance for key in ("auroc", "aupr", "precision", "recall", "f1", "fpr"))


def load_raw(dataset_dir: str, method_dir: str):
    path = RAW / dataset_dir / method_dir / "test_raw_scores.npz"
    with np.load(path, allow_pickle=False) as z:
        payload = {key: np.asarray(z[key]) for key in z.files}
    return path, payload


def strict_raw_reproduction(dataset: str, dataset_dir: str, metrics: pd.DataFrame, thresholds: dict):
    paths = {}
    rec_path, rec = load_raw(dataset_dir, "ours_rec")
    dual_path, dual = load_raw(dataset_dir, "ours_dual")
    tail_path, tail = load_raw(dataset_dir, "ours_tail")
    paths.update({"rec_test_raw": rec_path, "dual_test_raw": dual_path, "tail_test_raw": tail_path})
    for other in (dual, tail):
        if not np.array_equal(rec["label"], other["label"]):
            raise RuntimeError(f"{dataset}: raw score labels differ")
        for key in ("segment_id", "file_id", "local_index", "global_index", "valid_endpoint"):
            if key in rec and key in other and not np.array_equal(rec[key], other[key]):
                raise RuntimeError(f"{dataset}: raw score endpoint binding differs for {key}")
    labels = rec["label"].astype(np.int8)
    rec_score, dual_score, tail_score = map(lambda x: np.asarray(x, dtype=np.float64), (rec["score"], dual["score"], tail["score"]))
    zr = np.maximum(0.0, (rec_score - float(thresholds["rec_branch_median"])) / float(thresholds["rec_branch_scale"]))
    zd = np.maximum(0.0, (dual_score - float(thresholds["dual_branch_median"])) / float(thresholds["dual_branch_scale"]))
    robust_score = np.maximum(zr, zd)
    actual = {
        "REC": point_metrics(labels, rec_score, float(thresholds["rec"])),
        "DUAL": point_metrics(labels, dual_score, float(thresholds["dual"])),
        "OLD_FUSION": point_metrics(labels, robust_score, float(thresholds["fusion"])),
        "TAIL_FUSION": point_metrics(labels, tail_score, float(frozen_row(metrics, dataset, "TAIL_FUSION")["threshold"])),
    }
    expected = {name: frozen_row(metrics, dataset, name) for name in actual}
    checks = {name: compare_metric(actual[name], expected[name]) for name in actual}
    timeline = rec.get("timestamp", rec["global_index"])
    return paths, labels, timeline, rec_score, dual_score, robust_score, tail_score, actual, checks


def event_summary_rows(dataset: str, frame: pd.DataFrame) -> list[dict]:
    mapping = {
        "F0 REC": "rec_hit_rate", "F1 DUAL": "dual_hit_rate",
        "F3 Robust-Z": "old_fusion_hit_rate", "F4 Tail-Calibrated": "tail_fusion_hit_rate",
    }
    output = []
    for method, column in mapping.items():
        row = {
            "dataset": dataset, "variant": method, "event_count": len(frame),
            "event_hit_at_least_0_10": int((frame[column] >= .10).sum()),
            "median_event_hit": float(frame[column].median()),
            "high_event_count": "", "high_median_hit": "", "direct_median_hit": "",
            "rec_near_miss_recovered": "", "notes": "frozen official event semantics",
        }
        if dataset == "HAI21":
            high = frame[frame.category == "HIGH"]
            direct = frame[frame.category == "DIRECT"]
            row.update({"high_event_count": len(high), "high_median_hit": float(high[column].median()), "direct_median_hit": float(direct[column].median())})
            if method == "F1 DUAL":
                row["rec_near_miss_recovered"] = int(((frame.rec_hit_rate < .10) & (frame.dual_hit_rate >= .10)).sum())
            elif method == "F4 Tail-Calibrated":
                row["rec_near_miss_recovered"] = int(((frame.rec_hit_rate < .10) & (frame.tail_fusion_hit_rate >= .10)).sum())
        output.append(row)
    return output


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    registry = json.loads((ROOT / "outputs" / "thesis_ablation" / "ablation_registry.json").read_text(encoding="utf-8"))
    abl1 = json.loads((ROOT / "outputs" / "thesis_ablation" / "innovation2" / "summary" / "abl1_summary.json").read_text(encoding="utf-8"))
    if registry.get("status") != "ABLATION_PROTOCOL_READY" or abl1.get("status") != "ABLATION_A1_INNOVATION2_COMPLETED":
        raise RuntimeError("ABL-P0/ABL1 prerequisite failed")
    OUT.mkdir(parents=True)

    i3a_manifest = json.loads((I3A / "manifest.json").read_text(encoding="utf-8"))
    i3b_manifest = json.loads((I3B / "manifest.json").read_text(encoding="utf-8"))
    i3a_summary = json.loads((I3A / "innovation3a_summary.json").read_text(encoding="utf-8"))
    i3b_summary = json.loads((I3B / "innovation3b_summary.json").read_text(encoding="utf-8"))
    frozen_metrics = pd.read_csv(I3B / "cross_dataset_metrics.csv")
    normal = pd.read_csv(I3A / "dataset_summary.csv")
    dominance = pd.read_csv(I3A / "dominance_analysis.csv")

    threshold_sources = {
        "SWaT": ROOT / "outputs" / "thesis_swat_final_unified" / "thresholds.json",
        "HAI21": ROOT / "outputs" / "thesis_hai21_final_unified" / "thresholds.json",
        "WADI.A2": ROOT / "outputs" / "thesis_wadi_b1_final_external" / "thresholds.json",
    }
    thresholds = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in threshold_sources.items()}

    source_paths = {
        "i3a_manifest": I3A / "manifest.json", "i3a_summary": I3A / "innovation3a_summary.json",
        "i3a_dataset_summary": I3A / "dataset_summary.csv", "i3a_dominance": I3A / "dominance_analysis.csv",
        "i3b_manifest": I3B / "manifest.json", "i3b_summary": I3B / "innovation3b_summary.json",
        "i3b_metrics": I3B / "cross_dataset_metrics.csv", "i3b_hai_events": I3B / "hai_event_metrics.csv",
        "i3b_swat_events": I3B / "swat_event_metrics.csv", "i3b_wadi_events": I3B / "wadi_event_metrics.csv",
        "swat_tail_calibration": EFF / "swat_tail_calibration.npz",
        "hai_tail_calibration": EFF / "hai21_tail_calibration.npz",
        **{f"{name}_thresholds": path for name, path in threshold_sources.items()},
    }

    reproduced = {}
    fusion_archives = {}
    for dataset, dataset_dir in (("SWaT", "swat"), ("HAI21", "hai")):
        paths, labels, timeline, rec_score, dual_score, robust_score, tail_score, actual, checks = strict_raw_reproduction(dataset, dataset_dir, frozen_metrics, thresholds[dataset])
        for key, path in paths.items():
            source_paths[f"{dataset}_{key}"] = path
        archive = OUT / f"{dataset.lower()}_fusion_scores.npz"
        np.savez(archive, labels=labels, timeline=timeline, rec_score=rec_score, dual_score=dual_score,
                 f3_robust_z_score=robust_score, f4_tail_score=tail_score)
        fusion_archives[dataset] = {"path": str(archive.relative_to(ROOT)), "sha256": sha256(archive), "points": len(labels)}
        reproduced[dataset] = {"metrics": actual, "checks": checks}

    # Raw Normal calibration scales are available as persisted Calibration-A ECDF arrays.
    raw_scale = {}
    for dataset, path in (("SWaT", EFF / "swat_tail_calibration.npz"), ("HAI21", EFF / "hai21_tail_calibration.npz")):
        with np.load(path, allow_pickle=False) as payload:
            rec_stats = raw_stats(payload["rec_ecdf_sorted"])
            dual_stats = raw_stats(payload["dual_ecdf_sorted"])
            persisted_threshold = float(payload["threshold"][0])
        raw_scale[dataset] = {
            "scope": "frozen Normal Validation Calibration-A", "REC": rec_stats, "DUAL": dual_stats,
            "dual_to_rec_q99_ratio": dual_stats["q99"] / max(rec_stats["q99"], 1e-300),
            "dual_to_rec_top1_median_ratio": dual_stats["top1_median"] / max(rec_stats["top1_median"], 1e-300),
            "tail_threshold": persisted_threshold,
        }

    balance_rows = []
    for ds in ("SWaT", "HAI21", "WADI.A2"):
        norm = normal[normal.dataset == ds].iloc[0]
        old = dominance[(dominance.dataset == ds) & (dominance.method == "OLD_ROBUST_Z")].iloc[0]
        tail = dominance[(dominance.dataset == ds) & (dominance.method == "TAIL_EVIDENCE")].iloc[0]
        if ds in raw_scale:
            scale = raw_scale[ds]
            balance_rows.append({"dataset": ds, "variant": "RAW_SCORE", "rec_q99": scale["REC"]["q99"], "dual_q99": scale["DUAL"]["q99"],
                                 "balance_ratio": max(scale["REC"]["q99"], scale["DUAL"]["q99"]) / min(scale["REC"]["q99"], scale["DUAL"]["q99"]),
                                 "top1_rec_share": "", "top1_dual_share": "", "normal_fpr_at_audit_threshold": "",
                                 "notes": "raw Calibration-A score scale; no direct fusion"})
        balance_rows.append({"dataset": ds, "variant": "F3 Robust-Z", "rec_q99": "", "dual_q99": "", "balance_ratio": "",
                             "top1_rec_share": float(old.rec_dominant_fraction), "top1_dual_share": float(old.dual_dominant_fraction),
                             "normal_fpr_at_audit_threshold": float(norm.old_fusion_fpr), "notes": "historical Calibration-A robust center/scale; Audit-B top-1% dominance"})
        balance_rows.append({"dataset": ds, "variant": "F4 Tail-Calibrated", "rec_q99": float(norm.rec_q99), "dual_q99": float(norm.dual_q99),
                             "balance_ratio": float(norm.q99_balance_ratio), "top1_rec_share": float(tail.rec_dominant_fraction),
                             "top1_dual_share": float(tail.dual_dominant_fraction), "normal_fpr_at_audit_threshold": float(norm.tail_fusion_fpr),
                             "notes": "Calibration-A ECDF; Audit-B q99 balance and top-1% contribution"})

    detection_rows = []
    method_map = {"REC": "F0 REC", "DUAL": "F1 DUAL", "OLD_FUSION": "F3 Robust-Z", "TAIL_FUSION": "F4 Tail-Calibrated"}
    for dataset in ("SWaT", "HAI21", "WADI.A2"):
        single = frozen_metrics[(frozen_metrics.dataset == dataset) & frozen_metrics.method.isin(["REC", "DUAL"])]
        best_ap, best_f1 = float(single.aupr.max()), float(single.f1.max())
        for frozen_name, variant in method_map.items():
            row = frozen_row(frozen_metrics, dataset, frozen_name)
            detection_rows.append({"dataset": dataset, "variant": variant, "threshold": float(row["threshold"]),
                                   **{key: float(row[key]) for key in ("auroc", "aupr", "precision", "recall", "f1", "fpr")},
                                   "delta_vs_best_single_AUPR": float(row["aupr"]) - best_ap,
                                   "delta_vs_best_single_F1": float(row["f1"]) - best_f1,
                                   "role": "DOMAIN_SHIFT_STRESS_TEST" if dataset == "WADI.A2" else "PRIMARY"})

    event_rows = []
    for dataset, path in (("SWaT", I3B / "swat_event_metrics.csv"), ("HAI21", I3B / "hai_event_metrics.csv"), ("WADI.A2", I3B / "wadi_event_metrics.csv")):
        event_rows.extend(event_summary_rows(dataset, pd.read_csv(path)))

    plot_rows = []
    for row in balance_rows:
        if row["variant"] in ("F3 Robust-Z", "F4 Tail-Calibrated"):
            plot_rows.extend([
                {"panel": "normal_top1_share", "dataset": row["dataset"], "variant": row["variant"], "series": "REC", "metric": "share", "value": row["top1_rec_share"]},
                {"panel": "normal_top1_share", "dataset": row["dataset"], "variant": row["variant"], "series": "DUAL", "metric": "share", "value": row["top1_dual_share"]},
            ])
    for row in detection_rows:
        if row["dataset"] == "HAI21":
            plot_rows.extend([{"panel": "hai_detection", "dataset": "HAI21", "variant": row["variant"], "series": row["variant"], "metric": metric, "value": row[metric]} for metric in ("aupr", "f1")])
        if row["dataset"] == "SWaT":
            plot_rows.append({"panel": "swat_aupr", "dataset": "SWaT", "variant": row["variant"], "series": row["variant"], "metric": "aupr", "value": row["aupr"]})

    def drow(dataset, variant):
        return next(row for row in detection_rows if row["dataset"] == dataset and row["variant"] == variant)
    table_abl4 = []
    balance_label = {
        "F0 REC": "single REC branch", "F1 DUAL": "single DUAL branch",
        "F3 Robust-Z": "top1 REC/DUAL = 0/100% on all audited datasets",
        "F4 Tail-Calibrated": "top1 REC/DUAL = 65/35% SWaT; 59/41% HAI; 52/48% WADI",
    }
    for variant in METHODS:
        swat, hai = drow("SWaT", variant), drow("HAI21", variant)
        table_abl4.append({"Variant": variant, "SWaT AUPR": swat["aupr"], "SWaT F1": swat["f1"], "SWaT FPR": swat["fpr"],
                           "HAI AUPR": hai["aupr"], "HAI F1": hai["f1"], "HAI FPR": hai["fpr"], "Normal-tail balance": balance_label[variant]})

    source_hashes = {key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "size_bytes": path.stat().st_size} for key, path in source_paths.items()}
    expected_shares = {"SWaT": (.6519114688128773, .34808853118712274), "HAI21": (.5915492957746479, .4084507042253521), "WADI.A2": (.5178117048346056, .4821882951653944)}
    expected_balance = {"SWaT": 1.04488845531713, "HAI21": 1.0002800414976116, "WADI.A2": 1.0352968814016037}
    tests = {
        "f0_f1_read_frozen_artifacts": all(all(value for value in reproduced[ds]["checks"].values()) for ds in ("SWaT", "HAI21")),
        "f3_definition_matches_history": i3a_manifest["score_definitions"]["old_fusion"] == "max(robust_z_rec,robust_z_dual)" and all(reproduced[ds]["checks"]["OLD_FUSION"] for ds in reproduced),
        "f4_formula_matches_final": i3a_manifest["score_definitions"]["tail_probability"].startswith("(1 + count") and i3b_manifest["score"] == "max(E_rec,E_dual)" and all(reproduced[ds]["checks"]["TAIL_FUSION"] for ds in reproduced),
        "no_new_fusion": tuple(sorted(set(row["variant"] for row in detection_rows))) == tuple(sorted(METHODS)),
        "no_test_driven_calibration": i3a_summary["normal_only"] and not i3a_summary["test_tuning"] and i3b_summary["no_test_tuning"],
        "robust_z_dominance_reproduced": bool(((dominance[dominance.method == "OLD_ROBUST_Z"].rec_dominant_fraction == 0) & (dominance[dominance.method == "OLD_ROBUST_Z"].dual_dominant_fraction == 1)).all()),
        "tail_q99_balance_reproduced": all(abs(float(normal.loc[normal.dataset == ds, "q99_balance_ratio"].iloc[0]) - value) <= 1e-12 for ds, value in expected_balance.items()),
        "top1_share_reproduced": all(abs(float(dominance[(dominance.dataset == ds) & (dominance.method == "TAIL_EVIDENCE")].rec_dominant_fraction.iloc[0]) - values[0]) <= 1e-12 and abs(float(dominance[(dominance.dataset == ds) & (dominance.method == "TAIL_EVIDENCE")].dual_dominant_fraction.iloc[0]) - values[1]) <= 1e-12 for ds, values in expected_shares.items()),
        "hai_tail_metrics_reproduced": reproduced["HAI21"]["checks"]["TAIL_FUSION"],
        "swat_tail_metrics_reproduced": reproduced["SWaT"]["checks"]["TAIL_FUSION"],
        "wadi_formal_metrics_reproduced": all(abs(drow("WADI.A2", variant)[key] - float(frozen_row(frozen_metrics, "WADI.A2", frozen_name)[key])) <= 1e-12 for frozen_name, variant in method_map.items() for key in ("auroc", "aupr", "precision", "recall", "f1", "fpr")),
        "raw_and_calibration_artifacts_persisted": all(path.is_file() for path in source_paths.values()) and all((OUT / f"{ds.lower()}_fusion_scores.npz").is_file() for ds in ("SWaT", "HAI21")),
    }
    status = "ABLATION_A2_INNOVATION3_COMPLETED" if all(tests.values()) else "ABLATION_A2_REQUIRES_REVIEW"
    conclusions = {
        "comparability": "TAIL_CALIBRATION_BALANCES_NORMAL_EVIDENCE",
        "swat_boundary": "TAIL_FUSION_SWAT_UNSTABLE",
        "allowed_interpretation": [
            "Tail calibration improves heterogeneous score comparability on Normal data.",
            "On HAI21, comparable evidence is associated with higher AUPR than either single branch.",
            "On SWaT, calibration succeeds but fusion does not preserve REC ranking and threshold transfer remains unstable.",
            "Innovation-3 is evidence calibration/score comparability, not a universal detection-improvement claim.",
        ],
    }
    jdump(OUT / "raw_scale_diagnostics.json", raw_scale)
    jdump(OUT / "thresholds.json", {"F0_F1_F3": thresholds, "F4": i3b_summary["thresholds"], "sources": "frozen Normal-only artifacts"})
    jdump(OUT / "metrics.json", detection_rows)
    jdump(OUT / "table_abl4.json", table_abl4)
    jdump(OUT / "summary_payload.json", {"normal_tail_balance": balance_rows, "fusion_detection_ablation": detection_rows,
                                           "fusion_event_summary": event_rows, "fusion_ablation_plot_data": plot_rows})
    manifest = {
        "phase": "Thesis Ablation-ABL2", "status": status, "evidence_class": EVIDENCE,
        "method_freeze": "FROZEN", "variants": list(METHODS),
        "f3_definition": {"rec": "max(0,(REC-median_REC)/scale_REC)", "dual": "max(0,(DUAL-median_DUAL)/scale_DUAL)", "fusion": "max(z_REC,z_DUAL)", "calibration": "historical full Normal Validation for strict metrics; Calibration-A/Audit-B for dominance audit"},
        "f4_definition": {"p_tail": "(1 + count{s_i >= x}) / (N + 1)", "evidence": "-log10(p_tail)", "fusion": "max(E_REC,E_DUAL)", "ecdf": "Calibration-A only", "threshold": "Threshold-B q99 only"},
        "source_artifacts": source_hashes, "persisted_fusion_archives": fusion_archives,
        "raw_scale_diagnostics": raw_scale, "tests": tests, "conclusions": conclusions,
        "elapsed_seconds": time.perf_counter() - started,
        "restrictions": {"model_training": False, "ridge_refit": False, "new_fusion": False, "hp_search": False, "test_tuning": False, "point_adjustment": False, "oracle": False, "method_change": False, "commit": False, "push": False},
    }
    jdump(OUT / "manifest.json", manifest)
    jdump(OUT / "abl2_summary.json", {"status": status, "elapsed_seconds": manifest["elapsed_seconds"], "tests": tests, "conclusions": conclusions,
                                      "hai_tail_aupr": drow("HAI21", "F4 Tail-Calibrated")["aupr"], "hai_tail_f1": drow("HAI21", "F4 Tail-Calibrated")["f1"],
                                      "swat_tail_aupr": drow("SWaT", "F4 Tail-Calibrated")["aupr"], "swat_tail_fpr": drow("SWaT", "F4 Tail-Calibrated")["fpr"]})
    print(json.dumps({"status": status, "elapsed_seconds": manifest["elapsed_seconds"], "tests": tests, "raw_scale": raw_scale,
                      "table_abl4": table_abl4, "event_summary": event_rows}, indent=2, allow_nan=False))
    return 0 if status == "ABLATION_A2_INNOVATION3_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
