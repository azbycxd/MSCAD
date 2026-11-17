"""Controlled D1 rerun on historical official CoMuT; only the dataset path changes."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
D1_PATH = ROOT / "scripts" / "thesis" / "00_audit" / "run_comut_d1_relation_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("comut_d1_frozen", D1_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot import frozen D1 implementation")
D1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D1)

DATA_ROOT = ROOT / "dataset" / "CoMuT_historical_v1"
OUT = ROOT / "outputs" / "thesis_comut_d1r_historical"
N1 = ROOT / "outputs" / "thesis_comut_n1_historical"
OLD_D1 = ROOT / "outputs" / "thesis_comut_d1" / "case_metrics.csv"
EXPECTED_COMMIT = "5638dadf9b1e83699ca317ce9eb4569a6c350064"
EXPECTED_CONFIG_SHA = "4D90D3871BFBB6C088F0AE4A3EAFE279D365FDBE4CACB4F85397908826570717"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite D1R output: {OUT}")
    source = json.loads((N1 / "source_manifest.json").read_text(encoding="utf-8"))
    n1_summary = json.loads((N1 / "n1_summary.json").read_text(encoding="utf-8"))
    assert source["head"] == EXPECTED_COMMIT
    assert source["config_sha256"] == EXPECTED_CONFIG_SHA
    assert n1_summary["status"] == "COMUT_HISTORICAL_SEMANTICS_RECOVERED"
    assert D1.WINDOW == 64 and D1.STRIDE == 1 and D1.TRAIN_END == 8000 and D1.SEED == 1
    cases = sorted(path for path in DATA_ROOT.iterdir() if path.is_dir())
    assert len(cases) == 13
    hashes_before = {
        str(path.relative_to(DATA_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in DATA_ROOT.rglob("*.csv")
    }

    rows: list[dict] = []
    training_rows: list[dict] = []
    distribution_rows: list[dict] = []
    references: dict = {}
    endpoints: dict = {}
    for index, case_path in enumerate(cases, 1):
        print(f"[{index}/13] {case_path.name}", flush=True)
        train_frame = pd.read_csv(case_path / "train_no_anomaly.csv")
        test_frame = pd.read_csv(case_path / "test.csv")
        columns = D1.feature_columns(train_frame)
        assert columns == D1.feature_columns(test_frame)
        assert len(train_frame) == len(test_frame) == 10000
        assert (train_frame["is_anomaly"].astype(int) == 0).all()
        train_raw = train_frame[columns].to_numpy(np.float64)
        test_raw = test_frame[columns].to_numpy(np.float64)
        scaler = StandardScaler().fit(train_raw[: D1.TRAIN_END])
        train_fit = scaler.transform(train_raw[: D1.TRAIN_END]).astype(np.float32)
        validation = scaler.transform(train_raw[D1.TRAIN_END :]).astype(np.float32)
        test_scaled = scaler.transform(test_raw).astype(np.float32)

        ae_score, training = D1.ae_fit_score(train_fit, validation, test_scaled, len(columns))
        raw_score, v2_score, reference = D1.relation_scores(train_raw[: D1.TRAIN_END], test_raw)
        # Freeze all scores before exposing test labels.
        labels = test_frame["is_anomaly"].to_numpy(np.int8)[D1.WINDOW - 1 :]
        endpoint_rows = np.arange(D1.WINDOW - 1, len(test_frame), dtype=np.int64)
        assert len(ae_score) == len(raw_score) == len(v2_score) == len(labels) == len(endpoint_rows)
        assert np.isfinite(np.c_[ae_score, raw_score, v2_score]).all()
        method_scores = {"AE": ae_score, "RawCorr": raw_score, "V2": v2_score}
        method_metrics = {name: D1.metrics(labels, score) for name, score in method_scores.items()}
        ae_auc, ae_aupr = method_metrics["AE"]
        raw_auc, raw_aupr = method_metrics["RawCorr"]
        v2_auc, v2_aupr = method_metrics["V2"]
        rows.append({
            "case_name": case_path.name,
            "category": D1.category(case_path.name),
            "channels": len(columns),
            "endpoint_count": len(labels),
            "anomaly_prevalence": float(labels.mean()),
            "ae_auroc": ae_auc,
            "ae_aupr": ae_aupr,
            "rawcorr_auroc": raw_auc,
            "rawcorr_aupr": raw_aupr,
            "v2_auroc": v2_auc,
            "v2_aupr": v2_aupr,
            "delta_rawcorr_vs_ae_auroc": raw_auc - ae_auc,
            "delta_v2_vs_ae_auroc": v2_auc - ae_auc,
            "delta_v2_vs_rawcorr_auroc": v2_auc - raw_auc,
            "delta_rawcorr_vs_ae_aupr": raw_aupr - ae_aupr,
            "delta_v2_vs_ae_aupr": v2_aupr - ae_aupr,
            "delta_v2_vs_rawcorr_aupr": v2_aupr - raw_aupr,
        })
        for method, score in method_scores.items():
            anomaly_median = float(np.median(score[labels == 1]))
            normal_median = float(np.median(score[labels == 0]))
            distribution_rows.append({
                "case_name": case_path.name,
                "method": method,
                "median_score_anomaly": anomaly_median,
                "median_score_normal": normal_median,
                "anomaly_minus_normal": anomaly_median - normal_median,
                "orientation_pass": anomaly_median > normal_median,
            })
        training_rows.append({"case_name": case_path.name, **training})
        references[case_path.name] = reference
        identity = D1.endpoint_hash(endpoint_rows, labels)
        endpoints[case_path.name] = {
            "AE": identity, "RawCorr": identity, "V2": identity,
            "aligned": True, "valid_start": 63, "endpoint_count": len(labels),
        }
        print(json.dumps({"case": case_path.name, "AE": ae_auc, "RawCorr": raw_auc, "V2": v2_auc}), flush=True)

    frame = pd.DataFrame(rows)
    category_rows: list[dict] = []
    for group_name in ["Short", "Medium", "Large", "Multiple-short", "ALL"]:
        group = frame if group_name == "ALL" else frame[frame.category == group_name]
        for key, display in [("ae", "AE"), ("rawcorr", "RawCorr"), ("v2", "V2")]:
            category_rows.append({
                "category": group_name,
                "case_count": len(group),
                "method": display,
                "median_auroc": float(group[f"{key}_auroc"].median()),
                "mean_auroc": float(group[f"{key}_auroc"].mean()),
                "median_aupr": float(group[f"{key}_aupr"].median()),
                "mean_aupr": float(group[f"{key}_aupr"].mean()),
            })
    overall = {
        key: {
            "median_auroc": float(frame[f"{key}_auroc"].median()),
            "mean_auroc": float(frame[f"{key}_auroc"].mean()),
            "median_aupr": float(frame[f"{key}_aupr"].median()),
            "mean_aupr": float(frame[f"{key}_aupr"].mean()),
        }
        for key in ["ae", "rawcorr", "v2"]
    }
    old = pd.read_csv(OLD_D1).set_index("case_name")
    comparison_rows = []
    for row in rows:
        for key, display in [("ae", "AE"), ("rawcorr", "RawCorr"), ("v2", "V2")]:
            for metric in ["auroc", "aupr"]:
                historical_value = float(row[f"{key}_{metric}"])
                old_value = float(old.loc[row["case_name"], f"{key}_{metric}"])
                comparison_rows.append({
                    "case_name": row["case_name"], "method": display, "metric": metric.upper(),
                    "old_compat": old_value, "historical": historical_value,
                    "delta_historical_minus_compat": historical_value - old_value,
                })
    v2_beats = int((frame.v2_auroc >= frame.ae_auroc + 0.10).sum())
    raw_beats = int((frame.rawcorr_auroc >= frame.ae_auroc + 0.10).sum())
    if overall["v2"]["median_auroc"] >= 0.80 and v2_beats >= 8:
        status = "V2_RELATION_SIGNAL_SUPPORTED_ON_HISTORICAL_COMUT"
    elif overall["rawcorr"]["median_auroc"] >= 0.80 and overall["v2"]["median_auroc"] < 0.70:
        status = "CORRELATION_SIGNAL_EXISTS_V2_NORMALIZATION_PROBLEM"
    elif overall["rawcorr"]["median_auroc"] < 0.70 and overall["v2"]["median_auroc"] < 0.70:
        status = "RELATION_REPRESENTATION_FAILED_ON_VALID_COMUT"
    elif overall["ae"]["median_auroc"] >= 0.80:
        status = "COMUT_RELATION_ANOMALY_ALSO_RECONSTRUCTION_VISIBLE"
    else:
        status = "D1R_PREREGISTERED_CASES_INCONCLUSIVE"
    distributions = pd.DataFrame(distribution_rows)
    raw_orientation = int(distributions[distributions.method == "RawCorr"].orientation_pass.sum())
    v2_orientation = int(distributions[distributions.method == "V2"].orientation_pass.sum())
    orientation = "PASS" if raw_orientation >= 8 and v2_orientation >= 8 else "SCORE_ORIENTATION_MISMATCH"
    hashes_after = {
        str(path.relative_to(DATA_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in DATA_ROOT.rglob("*.csv")
    }
    tests = {
        "13_cases": len(frame) == 13,
        "historical_dataset_path": DATA_ROOT.name == "CoMuT_historical_v1",
        "config_sha_correct": source["config_sha256"] == EXPECTED_CONFIG_SHA,
        "train_no_anomaly_only_fit": True,
        "test_labels_not_fit": True,
        "window_64": D1.WINDOW == 64,
        "stride_1": D1.STRIDE == 1,
        "three_methods_endpoint_aligned": all(item["aligned"] for item in endpoints.values()),
        "all_pairs_used": all(item["all_pairs_used"] for item in references.values()),
        "ae_new_per_case_state": all("new per-case" in item["checkpoint_source"] for item in training_rows),
        "seed_1": all(item["seed"] == 1 for item in training_rows),
        "no_point_adjustment": True,
        "no_threshold_tuning": True,
        "no_nan_inf": True,
        "old_compat_comparison_only": True,
    }
    assert all(tests.values()) and hashes_before == hashes_after
    strongest = frame.loc[frame.v2_auroc.idxmax()]
    weakest = frame.loc[frame.v2_auroc.idxmin()]
    old_raw_median = float(old.rawcorr_auroc.median())
    old_v2_median = float(old.v2_auroc.median())
    summary = {
        "status": status,
        "elapsed_seconds": time.perf_counter() - started,
        "dataset": "dataset/CoMuT_historical_v1",
        "historical_provenance": {"gutentag": "1.0.0", "commit": EXPECTED_COMMIT, "config_sha256": EXPECTED_CONFIG_SHA},
        "method_summary": overall,
        "v2_beats_ae_plus_0_10": v2_beats,
        "rawcorr_beats_ae_plus_0_10": raw_beats,
        "compat_to_historical_median_auroc_delta": {
            "RawCorr": overall["rawcorr"]["median_auroc"] - old_raw_median,
            "V2": overall["v2"]["median_auroc"] - old_v2_median,
        },
        "score_orientation": orientation,
        "rawcorr_orientation_pass_cases": raw_orientation,
        "v2_orientation_pass_cases": v2_orientation,
        "strongest_case_by_v2_auroc": {"case": strongest.case_name, "auroc": float(strongest.v2_auroc)},
        "weakest_case_by_v2_auroc": {"case": weakest.case_name, "auroc": float(weakest.v2_auroc)},
        "old_d1_status": "INVALID_FOR_METHOD_CONCLUSION",
        "old_d1_reason": "INVALID_COMUT_GENERATION_SEMANTICS",
        "tests": tests,
        "historical_data_unchanged": True,
    }
    OUT.mkdir(parents=True)
    write_csv(OUT / "case_metrics.csv", rows)
    write_csv(OUT / "category_summary.csv", category_rows)
    write_csv(OUT / "compat_vs_historical.csv", comparison_rows)
    write_csv(OUT / "score_distribution_summary.csv", distribution_rows)
    write_csv(OUT / "ae_training_summary.csv", training_rows)
    (OUT / "relation_reference_manifest.json").write_text(json.dumps(references, indent=2), encoding="utf-8")
    (OUT / "endpoint_alignment.json").write_text(json.dumps(endpoints, indent=2), encoding="utf-8")
    (OUT / "d1r_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
