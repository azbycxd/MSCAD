"""Audit score/label alignment for three Historical CoMuT cases; no model training."""

from __future__ import annotations

import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
D1_PATH = ROOT / "scripts" / "thesis" / "00_audit" / "run_comut_d1_relation_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("comut_d1_frozen_alignment", D1_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot import frozen D1 implementation")
D1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D1)

DATA = ROOT / "dataset" / "CoMuT_historical_v1"
OUT = ROOT / "outputs" / "thesis_comut_d1a_alignment"
CASES = [
    "rmj-short-mode-correlation-on-2",
    "rmj-medium-mode-correlation-on-2",
    "rmj-large-mode-correlation-on-2",
]


def intervals(labels: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[0, labels.astype(np.int8), 0])
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite D1A output: {OUT}")
    assert D1.WINDOW == 64 and D1.STRIDE == 1
    interval_rows: list[dict] = []
    region_rows: list[dict] = []
    metric_rows: list[dict] = []
    case_summary: dict = {}
    for case in CASES:
        train = pd.read_csv(DATA / case / "train_no_anomaly.csv")
        test = pd.read_csv(DATA / case / "test.csv")
        columns = D1.feature_columns(train)
        train_values = train[columns].to_numpy(float)[:8000]
        test_values = test[columns].to_numpy(float)
        labels_full = test["is_anomaly"].to_numpy(np.int8)
        recovered = intervals(labels_full)
        assert len(recovered) == 1
        anomaly_start, anomaly_end = recovered[0]
        expected_length = {"short": 40, "medium": 80, "large": 160}[
            next(name for name in ("short", "medium", "large") if name in case)
        ]
        assert anomaly_end - anomaly_start == expected_length

        train_rho, pairs, train_valid = D1.pair_correlations(train_values)
        test_rho, test_pairs, test_valid = D1.pair_correlations(test_values)
        assert pairs == [(0, 1)] == test_pairs
        assert train_valid.all() and test_valid.all()
        rho_ref = float(np.median(train_rho[:, 0]))
        raw_score, v2_score, reference = D1.relation_scores(train_values, test_values)
        endpoints = np.arange(63, len(test), dtype=np.int64)
        endpoint_labels = labels_full[endpoints]
        overlap_labels = np.asarray(
            [labels_full[endpoint - 63 : endpoint + 1].any() for endpoint in endpoints], dtype=np.int8
        )
        assert len(raw_score) == len(v2_score) == len(endpoints)

        head_end = min(anomaly_start + 63, anomaly_end)
        core_start = anomaly_start + 63
        tail_end = min(anomaly_end + 63, len(test))
        region_definitions = {
            "anomaly_head": (anomaly_start, head_end),
            "anomaly_core": (core_start, anomaly_end) if core_start < anomaly_end else (core_start, core_start),
            "post_anomaly_tail": (anomaly_end, tail_end),
        }
        interval_rows.append({
            "case_name": case,
            "anomaly_start": anomaly_start,
            "anomaly_end_exclusive": anomaly_end,
            "anomaly_length": anomaly_end - anomaly_start,
            "anomaly_head_start": anomaly_start,
            "anomaly_head_end_exclusive": head_end,
            "anomaly_head_count": head_end - anomaly_start,
            "anomaly_core_start": core_start,
            "anomaly_core_end_exclusive": anomaly_end,
            "anomaly_core_count": max(anomaly_end - core_start, 0),
            "post_tail_start": anomaly_end,
            "post_tail_end_exclusive": tail_end,
            "post_tail_count": tail_end - anomaly_end,
        })

        # Endpoint-indexed masks; far normal means the full window has no anomaly overlap.
        masks = {
            name: (endpoints >= left) & (endpoints < right)
            for name, (left, right) in region_definitions.items()
        }
        masks["far_normal"] = overlap_labels == 0
        endpoint_to_score = {int(endpoint): index for index, endpoint in enumerate(endpoints)}
        core_rhos = [test_rho[endpoint_to_score[t], 0] for t in range(core_start, anomaly_end)] if core_start < anomaly_end else []
        for region in ["far_normal", "anomaly_head", "anomaly_core", "post_anomaly_tail"]:
            values = raw_score[masks[region]]
            region_rows.append({
                "case_name": case,
                "region": region,
                "endpoint_count": len(values),
                "score_median": float(np.median(values)) if len(values) else None,
                "score_q95": float(np.quantile(values, 0.95)) if len(values) else None,
                "score_q99": float(np.quantile(values, 0.99)) if len(values) else None,
                "rho_ref": rho_ref,
                "anomaly_core_rho_median": float(np.median(core_rhos)) if core_rhos else None,
            })
        for method, score in [("RawCorr", raw_score), ("V2", v2_score)]:
            metric_rows.append({
                "case_name": case,
                "method": method,
                "endpoint_label_auroc": float(roc_auc_score(endpoint_labels, score)),
                "window_overlap_diagnostic_auroc": float(roc_auc_score(overlap_labels, score)),
                "overlap_metric_status": "DIAGNOSTIC_ONLY",
                "score_flipped": False,
            })
        far = raw_score[masks["far_normal"]]
        core = raw_score[masks["anomaly_core"]]
        core_high = bool(len(core) and np.median(core) > np.quantile(far, 0.99))
        raw_endpoint_auc = metric_rows[-2]["endpoint_label_auroc"]
        raw_overlap_auc = metric_rows[-2]["window_overlap_diagnostic_auroc"]
        case_summary[case] = {
            "rho_ref": rho_ref,
            "v2_mu": reference["v2_mu_median_fisher_z"][0],
            "v2_sigma": reference["v2_sigma_robust"][0],
            "core_count": len(core),
            "core_score_gt_far_normal_q99": core_high,
            "rawcorr_endpoint_auroc": raw_endpoint_auc,
            "rawcorr_window_overlap_diagnostic_auroc": raw_overlap_auc,
        }

    large = case_summary["rmj-large-mode-correlation-on-2"]
    if not large["core_score_gt_far_normal_q99"]:
        status = "RELATION_SCORE_IMPLEMENTATION_PROBLEM"
    elif large["rawcorr_endpoint_auroc"] >= 0.80:
        status = "RELATION_SIGNAL_AND_ALIGNMENT_OK"
    elif large["rawcorr_window_overlap_diagnostic_auroc"] > large["rawcorr_endpoint_auroc"]:
        status = "RELATION_SIGNAL_PRESENT_ALIGNMENT_MISMATCH"
    else:
        status = "RELATION_SIGNAL_PRESENT_ENDPOINT_METRIC_WEAK"
    summary = {
        "status": status,
        "elapsed_seconds": time.perf_counter() - started,
        "dataset": str(DATA.resolve()),
        "window": 64,
        "stride": 1,
        "score_semantics": "one score summarizes the complete causal window [t-63,t], not endpoint t alone",
        "reverse_windowing_note": "TimeEval ReverseWindowing distributes window scores over covered points; D1 assigns each complete-window score only to its endpoint. They are semantically different.",
        "case_summary": case_summary,
        "window_overlap_metric": "DIAGNOSTIC_ONLY",
        "no_model_training": True,
        "no_parameter_search": True,
        "no_score_flip": True,
    }
    OUT.mkdir(parents=True)
    write_csv(OUT / "interval_regions.csv", interval_rows)
    write_csv(OUT / "score_region_summary.csv", region_rows)
    write_csv(OUT / "alignment_metrics.csv", metric_rows)
    (OUT / "d1a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
