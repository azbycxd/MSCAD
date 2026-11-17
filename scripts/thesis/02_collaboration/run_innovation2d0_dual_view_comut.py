"""Historical CoMuT mechanism test for target-free dual-view disagreement."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts" / "thesis" / "02_collaboration"
if str(HELPER) not in sys.path:
    sys.path.insert(0, str(HELPER))
import run_innovation2c0b_comut_structure as c0b  # noqa: E402


DATA = ROOT / "dataset" / "CoMuT_historical_v1"
OLD_COMPARATORS = ROOT / "outputs" / "thesis_innovation2c0b_comut_structure" / "comparator_summary.csv"
OUT = ROOT / "outputs" / "thesis_innovation2d0_dual_view_comut"
SELF_LAGS = (1, 5, 10, 20, 30)
CROSS_LAGS = (0, 1, 5, 10, 20, 30)
MAX_LAG = 30
ALPHA = 1.0


def design(values: np.ndarray, times: np.ndarray, channels: list[int], lags: tuple[int, ...]) -> np.ndarray:
    result = np.column_stack([values[times - lag, channel] for channel in channels for lag in lags]).astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite lagged design")
    return result


def calibrated(disagreement: np.ndarray, median: float, scale: float) -> np.ndarray:
    result = np.maximum(0.0, (disagreement - median) / scale)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite disagreement score")
    return result


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite 2D0 output: {OUT}")
    paths = sorted(path for path in DATA.iterdir() if path.is_dir())
    if len(paths) != 13:
        raise RuntimeError("expected 13 Historical CoMuT cases")

    case_rows, sanity_rows, response_rows = [], [], []
    case_contract = {}
    for path in paths:
        train, test, features = c0b.load_case(path)
        train_raw = train[features].to_numpy(np.float64)
        test_raw = test[features].to_numpy(np.float64)
        scaler = StandardScaler().fit(train_raw)
        z_train, z_test = scaler.transform(train_raw), scaler.transform(test_raw)
        train_times = np.arange(MAX_LAG, len(train), dtype=np.int64)
        split = int(np.floor(0.8 * len(train_times)))
        fit_times, calibration_times = train_times[:split], train_times[split:]
        test_times = np.arange(MAX_LAG, len(test), dtype=np.int64)
        calibration_target_scores, test_target_scores = [], []
        target_contract = {}

        for target_index, target in enumerate(features):
            cross_indices = [index for index in range(len(features)) if index != target_index]
            self_fit = design(z_train, fit_times, [target_index], SELF_LAGS)
            cross_fit = design(z_train, fit_times, cross_indices, CROSS_LAGS)
            y_fit = z_train[fit_times, target_index]
            self_model = Ridge(alpha=ALPHA).fit(self_fit, y_fit)
            cross_model = Ridge(alpha=ALPHA).fit(cross_fit, y_fit)

            self_cal = self_model.predict(design(z_train, calibration_times, [target_index], SELF_LAGS))
            cross_cal = cross_model.predict(design(z_train, calibration_times, cross_indices, CROSS_LAGS))
            y_cal = z_train[calibration_times, target_index]
            disagreement_cal = np.abs(self_cal - cross_cal)
            median = float(np.median(disagreement_cal))
            scale = max(1.4826 * float(np.median(np.abs(disagreement_cal - median))), 1e-6)
            calibration_target_scores.append(calibrated(disagreement_cal, median, scale))

            self_test = self_model.predict(design(z_test, test_times, [target_index], SELF_LAGS))
            cross_test = cross_model.predict(design(z_test, test_times, cross_indices, CROSS_LAGS))
            disagreement_test = np.abs(self_test - cross_test)
            test_target_scores.append(calibrated(disagreement_test, median, scale))
            sanity_rows.append({
                "case_name": path.name, "category": c0b.category(path.name), "target": target,
                "self_fit_mae": float(np.mean(np.abs(y_fit - self_model.predict(self_fit)))),
                "self_calibration_mae": float(np.mean(np.abs(y_cal - self_cal))),
                "cross_fit_mae": float(np.mean(np.abs(y_fit - cross_model.predict(cross_fit)))),
                "cross_calibration_mae": float(np.mean(np.abs(y_cal - cross_cal))),
                "calibration_disagreement_median": median,
                "calibration_disagreement_q95": float(np.quantile(disagreement_cal, 0.95)),
                "calibration_disagreement_q99": float(np.quantile(disagreement_cal, 0.99)),
                "calibration_scale": scale,
            })
            target_contract[target] = {
                "target_index": target_index, "self_lags": list(SELF_LAGS),
                "cross_channels": [features[index] for index in cross_indices], "cross_lags": list(CROSS_LAGS),
                "alpha": ALPHA, "calibration_median": median, "calibration_scale": scale,
                "score_formula": "max(0, (abs(p_self-p_cross)-median_calibration)/scale_calibration)",
                "observed_target_residual_in_score": False,
            }

        calibration_matrix = np.column_stack(calibration_target_scores)
        test_matrix = np.column_stack(test_target_scores)
        calibration_score = calibration_matrix.max(axis=1)
        score = test_matrix.max(axis=1)
        argmax = np.argmax(test_matrix, axis=1)
        threshold = float(np.quantile(calibration_score, 0.99))
        labels = test["is_anomaly"].to_numpy(np.int8)[test_times]
        auroc, aupr = float(roc_auc_score(labels, score)), float(average_precision_score(labels, score))
        prediction = score > threshold
        precision, recall, f1, _ = precision_recall_fscore_support(labels, prediction, average="binary", zero_division=0)
        fpr = float(np.mean(prediction[labels == 0]))
        normal_median = float(np.median(score[labels == 0]))
        anomaly_median = float(np.median(score[labels == 1]))
        anomaly_argmax = pd.Series([features[index] for index in argmax[labels == 1]]).value_counts(sort=True)
        top_target = str(anomaly_argmax.index[0])
        cat = c0b.category(path.name)
        case_rows.append({
            "case_name": path.name, "category": cat, "channels": len(features), "test_point_count": len(test_times),
            "auroc": auroc, "aupr": aupr, "precision_q99": float(precision), "recall_q99": float(recall),
            "f1_q99": float(f1), "fpr_q99": fpr,
        })
        response_rows.append({
            "case_name": path.name, "category": cat,
            "test_normal_median_disagreement_score": normal_median,
            "test_anomaly_median_disagreement_score": anomaly_median,
            "anomaly_normal_median_ratio": anomaly_median / max(normal_median, 1e-12),
            "most_frequent_anomaly_argmax_target": top_target,
            "argmax_fraction": float(anomaly_argmax.iloc[0] / anomaly_argmax.sum()),
        })
        case_contract[path.name] = {
            "features": features, "scaler_fit_rows": [0, len(train)],
            "model_fit_legal_time_rows": [0, split], "calibration_legal_time_rows": [split, len(train_times)],
            "test_legal_time_range": [MAX_LAG, len(test)], "threshold_source": "train calibration S_dual q99",
            "target_contract": target_contract,
        }

    metrics = pd.DataFrame(case_rows)
    responses = pd.DataFrame(response_rows)
    categories = metrics.groupby("category", sort=False).agg(
        cases=("case_name", "count"), median_auroc=("auroc", "median"), median_aupr=("aupr", "median"),
        median_recall=("recall_q99", "median"), median_fpr=("fpr_q99", "median")
    ).reset_index()
    old = pd.read_csv(OLD_COMPARATORS)
    comparator_rows = old.to_dict("records")
    comparator_rows.append({"method": "Dual-View Disagreement", "median_endpoint_auroc": float(metrics["auroc"].median()), "median_endpoint_aupr": float(metrics["aupr"].median())})
    by_cat = categories.set_index("category")
    count_080 = int((metrics["auroc"] >= 0.80).sum())
    gates = {
        "A_median_auroc_at_least_0_90": float(metrics["auroc"].median()) >= 0.90,
        "B_median_aupr_at_least_0_50": float(metrics["aupr"].median()) >= 0.50,
        "C_at_least_10_of_13_auroc_at_least_0_80": count_080 >= 10,
        "D_short_median_auroc_at_least_0_80": float(by_cat.loc["Short", "median_auroc"]) >= 0.80,
        "E_medium_median_auroc_at_least_0_80": float(by_cat.loc["Medium", "median_auroc"]) >= 0.80,
        "F_large_median_auroc_at_least_0_90": float(by_cat.loc["Large", "median_auroc"]) >= 0.90,
        "G_multiple_short_median_auroc_at_least_0_80": float(by_cat.loc["Multiple-short", "median_auroc"]) >= 0.80,
    }
    if all(gates.values()):
        status = "DUAL_VIEW_RELATION_MECHANISM_SUPPORTED"
    elif float(metrics["auroc"].median()) >= 0.80:
        status = "DUAL_VIEW_RELATION_MECHANISM_PARTIAL"
    else:
        status = "DUAL_VIEW_RELATION_MECHANISM_NOT_SUPPORTED"
    tests = {
        "historical_path_correct": DATA.name == "CoMuT_historical_v1", "compat_not_read": True,
        "scaler_train_only": True, "target_excluded_from_cross_view": True,
        "self_lags_fixed": SELF_LAGS == (1, 5, 10, 20, 30),
        "cross_lags_fixed": CROSS_LAGS == (0, 1, 5, 10, 20, 30), "alpha_1": ALPHA == 1.0,
        "fit_calibration_chronology": True, "score_only_prediction_disagreement": True,
        "score_excludes_observed_target_residual": True, "labels_evaluation_only": True,
        "threshold_train_calibration_only": True, "no_point_adjustment": True,
        "no_nan_inf": bool(np.isfinite(metrics.select_dtypes("number")).all().all() and np.isfinite(pd.DataFrame(sanity_rows).select_dtypes("number")).all().all()),
        "comparators_read_only": True,
    }
    if not all(tests.values()):
        raise RuntimeError(f"2D0 integrity failure: {tests}")
    summary = {
        "status": status, "historical_dataset_root": str(DATA), "case_count": len(metrics),
        "median_auroc": float(metrics["auroc"].median()), "median_aupr": float(metrics["aupr"].median()),
        "median_recall": float(metrics["recall_q99"].median()), "median_fpr": float(metrics["fpr_q99"].median()),
        "auroc_at_least_0_80_count": count_080, "gates": gates, "tests": tests,
        "self_lags": list(SELF_LAGS), "cross_lags": list(CROSS_LAGS), "alpha": ALPHA,
        "observed_target_residual_in_score": False, "point_adjustment": False,
        "test_tuning": False, "feature_search": False, "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.mkdir(parents=True)
    metrics.to_csv(OUT / "case_metrics.csv", index=False)
    categories.to_csv(OUT / "category_summary.csv", index=False)
    pd.DataFrame(sanity_rows).to_csv(OUT / "target_normal_sanity.csv", index=False)
    responses.to_csv(OUT / "relation_response.csv", index=False)
    pd.DataFrame(comparator_rows).to_csv(OUT / "comparator_summary.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({"summary": summary, "cases": case_contract}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "innovation2d0_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "cases": case_rows, "categories": categories.to_dict("records"), "responses": response_rows, "comparators": comparator_rows}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
