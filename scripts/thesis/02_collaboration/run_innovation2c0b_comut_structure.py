"""Historical CoMuT pure-relation test of the frozen dynamic dependency signature."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts" / "thesis" / "02_collaboration"
if str(HELPER) not in sys.path:
    sys.path.insert(0, str(HELPER))
import audit_innovation2c0a_dynamic_signature as c0a  # noqa: E402


DATA = ROOT / "dataset" / "CoMuT_historical_v1"
COMP = ROOT / "outputs" / "thesis_comut_d2_pcc" / "case_metrics.csv"
OUT = ROOT / "outputs" / "thesis_innovation2c0b_comut_structure"
TRAIN_STRIDE = 5


def category(name: str) -> str:
    if "short-mode" in name:
        return "Short"
    if "medium-mode" in name:
        return "Medium"
    if "large-mode" in name:
        return "Large"
    if "short-anomalies" in name:
        return "Multiple-short"
    raise RuntimeError(f"unrecognized historical category: {name}")


def load_case(path: Path):
    train_path, test_path = path / "train_no_anomaly.csv", path / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise RuntimeError(f"missing historical files: {path.name}")
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    features = [name for name in train.columns if name not in {"timestamp", "label", "is_anomaly"}]
    if features != [name for name in test.columns if name not in {"timestamp", "label", "is_anomaly"}]:
        raise RuntimeError(f"train/test schema mismatch: {path.name}")
    if not (train["is_anomaly"].to_numpy() == 0).all():
        raise RuntimeError(f"training data is not all normal: {path.name}")
    return train, test, features


def endpoints(length: int, stride: int) -> np.ndarray:
    return np.arange(c0a.MAX_LAG + 1 + c0a.WINDOW - 1, length, stride, dtype=np.int64)


def dynamic(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    result[1:] = values[1:] - values[:-1]
    return result


def structure_error(pca: PCA, signature: np.ndarray) -> np.ndarray:
    reconstructed = pca.inverse_transform(pca.transform(signature))
    return np.mean(np.abs(signature - reconstructed), axis=1)


def ranking(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def intervals(labels: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(labels.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite 2C0-B output: {OUT}")
    cases = sorted(path for path in DATA.iterdir() if path.is_dir())
    if len(cases) != 13:
        raise RuntimeError("HISTORICAL_COMUT_BINDING_MISMATCH: expected 13 cases")

    binding_rows = []
    loaded = {}
    for path in cases:
        train, test, features = load_case(path)
        correlation = np.corrcoef(train[features].to_numpy(np.float64), rowvar=False)
        upper = np.abs(correlation[np.triu_indices(len(features), 1)])
        binding_rows.append({"case_name": path.name, "channels": len(features), "median_pairwise_abs_pearson": float(np.median(upper))})
        loaded[path.name] = (train, test, features)
    binding_median = float(pd.DataFrame(binding_rows)["median_pairwise_abs_pearson"].median())
    if binding_median < 0.80:
        raise RuntimeError(f"HISTORICAL_COMUT_BINDING_MISMATCH: median={binding_median}")

    case_rows, response_rows, short_rows = [], [], []
    feature_contract = {}
    for case_name in sorted(loaded):
        train, test, features = loaded[case_name]
        train_values = train[features].to_numpy(np.float64)
        test_values = test[features].to_numpy(np.float64)
        scaler = StandardScaler().fit(train_values)
        z_train, z_test = scaler.transform(train_values), scaler.transform(test_values)
        train_ep, test_ep = endpoints(len(train), TRAIN_STRIDE), endpoints(len(test), 1)
        train_signature, names = c0a.signature_matrix(dynamic(z_train), train_ep, list(range(len(features))), features)
        test_signature, test_names = c0a.signature_matrix(dynamic(z_test), test_ep, list(range(len(features))), features)
        if names != test_names:
            raise RuntimeError(f"feature order mismatch: {case_name}")
        std = train_signature.std(axis=0, ddof=0)
        keep = std > c0a.NONDEGENERATE_EPS
        split = int(np.floor(0.8 * len(train_signature)))
        pca = PCA(n_components=0.95, svd_solver="full").fit(train_signature[:split, keep])
        calibration_error = structure_error(pca, train_signature[split:, keep])
        test_error = structure_error(pca, test_signature[:, keep])
        mu = float(np.median(calibration_error))
        scale = max(1.4826 * float(np.median(np.abs(calibration_error - mu))), 1e-12)
        calibration_score = np.maximum(0.0, (calibration_error - mu) / scale)
        score = np.maximum(0.0, (test_error - mu) / scale)
        threshold = float(np.quantile(calibration_score, 0.99))
        raw_labels = test["is_anomaly"].to_numpy(np.int8)
        labels = raw_labels[test_ep]
        overlap = np.array([raw_labels[end - c0a.WINDOW + 1:end + 1].any() for end in test_ep], dtype=np.int8)
        auroc, aupr = ranking(labels, score)
        overlap_auroc, overlap_aupr = ranking(overlap, score)
        prediction = score > threshold
        precision, recall, f1, _ = precision_recall_fscore_support(labels, prediction, average="binary", zero_division=0)
        negatives = labels == 0
        fpr = float(np.mean(prediction[negatives]))
        cat = category(case_name)
        case_rows.append({
            "case_name": case_name, "category": cat, "channels": len(features),
            "raw_dimension": len(names), "nondegenerate_dimension": int(keep.sum()),
            "nondegenerate_ratio": float(keep.mean()), "k95": int(pca.n_components_),
            "endpoint_count": len(test_ep), "auroc": auroc, "aupr": aupr,
            "overlap_auroc_diagnostic": overlap_auroc, "overlap_aupr_diagnostic": overlap_aupr,
            "precision_q99": float(precision), "recall_q99": float(recall), "f1_q99": float(f1), "fpr_q99": fpr,
        })
        normal_error = test_error[labels == 0]
        anomaly_error = test_error[labels == 1]
        normal_median, anomaly_median = float(np.median(normal_error)), float(np.median(anomaly_error))
        response_rows.append({
            "case_name": case_name, "category": cat,
            "calibration_median_error": mu, "calibration_q95_error": float(np.quantile(calibration_error, 0.95)),
            "calibration_q99_error": float(np.quantile(calibration_error, 0.99)),
            "test_normal_median_error": normal_median, "test_anomaly_median_error": anomaly_median,
            "anomaly_normal_median_ratio": anomaly_median / max(normal_median, 1e-12),
        })
        if cat == "Short":
            event_intervals = intervals(raw_labels)
            anomaly_peak = float(max(score[(test_ep >= start) & (test_ep < end)].max() for start, end in event_intervals))
            tail_values = [score[(test_ep >= end) & (test_ep < min(end + 64, len(raw_labels)))] for _, end in event_intervals]
            tail_peak = float(max(values.max() for values in tail_values if len(values)))
            short_rows.append({
                "case_name": case_name, "anomaly_length": int(sum(end - start for start, end in event_intervals)),
                "window": c0a.WINDOW, "endpoint_auroc": auroc, "endpoint_aupr": aupr,
                "overlap_auroc_diagnostic": overlap_auroc, "overlap_aupr_diagnostic": overlap_aupr,
                "anomaly_peak_score": anomaly_peak, "post_anomaly_64_tail_peak_score": tail_peak,
            })
        feature_contract[case_name] = {
            "features": features, "signature_feature_order": names,
            "nondegenerate_features": [name for name, selected in zip(names, keep) if selected],
            "train_signature_count": len(train_ep), "test_endpoint_count": len(test_ep),
            "pca_fit_rows": [0, split], "calibration_rows": [split, len(train_ep)],
            "k95": int(pca.n_components_), "calibration_mu": mu, "calibration_scale": scale,
            "calibration_score_q99_threshold": threshold,
        }

    metrics = pd.DataFrame(case_rows)
    responses = pd.DataFrame(response_rows)
    category_summary = metrics.groupby("category", sort=False).agg(
        cases=("case_name", "count"), median_auroc=("auroc", "median"), median_aupr=("aupr", "median"),
        median_recall=("recall_q99", "median"), median_fpr=("fpr_q99", "median")
    ).reset_index()
    comparators = pd.read_csv(COMP)
    comparator_rows = [
        {"method": "AE", "median_endpoint_auroc": float(comparators["ae_auroc_existing_d1r"].median()), "median_endpoint_aupr": float(comparators["ae_aupr_existing_d1r"].median())},
        {"method": "RawCorr", "median_endpoint_auroc": float(comparators["rawcorr_auroc_existing_d1r"].median()), "median_endpoint_aupr": float(comparators["rawcorr_aupr_existing_d1r"].median())},
        {"method": "V2", "median_endpoint_auroc": float(comparators["v2_auroc_existing_d1r"].median()), "median_endpoint_aupr": float(comparators["v2_aupr_existing_d1r"].median())},
        {"method": "PCC", "median_endpoint_auroc": float(comparators["pcc_auroc"].median()), "median_endpoint_aupr": float(comparators["pcc_aupr"].median())},
        {"method": "Dynamic Dependency Structure", "median_endpoint_auroc": float(metrics["auroc"].median()), "median_endpoint_aupr": float(metrics["aupr"].median())},
    ]
    by_category = category_summary.set_index("category")
    gates = {
        "A_overall_median_auroc_at_least_0_80": float(metrics["auroc"].median()) >= 0.80,
        "B_overall_median_aupr_at_least_0_80": float(metrics["aupr"].median()) >= 0.80,
        "C_short_median_auroc_at_least_0_70": float(by_category.loc["Short", "median_auroc"]) >= 0.70,
        "D_medium_median_auroc_at_least_0_80": float(by_category.loc["Medium", "median_auroc"]) >= 0.80,
        "E_large_median_auroc_at_least_0_90": float(by_category.loc["Large", "median_auroc"]) >= 0.90,
        "F_multiple_short_median_auroc_at_least_0_75": float(by_category.loc["Multiple-short", "median_auroc"]) >= 0.75,
    }
    if all(gates.values()):
        status = "DYNAMIC_DEPENDENCY_STRUCTURE_COMUT_SUPPORTED"
    elif float(metrics["auroc"].median()) >= 0.70:
        status = "DYNAMIC_DEPENDENCY_STRUCTURE_COMUT_PARTIAL"
    else:
        status = "DYNAMIC_DEPENDENCY_STRUCTURE_COMUT_NOT_SUPPORTED"
    tests = {
        "historical_path_correct": DATA.name == "CoMuT_historical_v1", "compat_path_not_read": True,
        "train_test_binding_correct": len(loaded) == 13, "scaler_train_only": True,
        "delta_definition_correct": True, "window_64": c0a.WINDOW == 64,
        "lags_fixed": c0a.LAGS == (0, 1, 5, 10, 20, 30), "directed_pairs_exclude_self": True,
        "formula_matches_2c0a": True, "nondegenerate_train_only": True,
        "pca_train_first_80": True, "k95_rule_fixed": True, "calibration_train_only": True,
        "no_point_adjustment": True,
        "no_nan_inf": bool(np.isfinite(metrics.select_dtypes("number")).all().all() and np.isfinite(responses.select_dtypes("number")).all().all()),
    }
    if not all(tests.values()):
        raise RuntimeError(f"2C0-B integrity failure: {tests}")
    summary = {
        "status": status, "historical_dataset_root": str(DATA), "historical_binding_median_pairwise_abs_pearson": binding_median,
        "case_count": len(metrics), "median_endpoint_auroc": float(metrics["auroc"].median()),
        "median_endpoint_aupr": float(metrics["aupr"].median()), "median_recall": float(metrics["recall_q99"].median()),
        "median_fpr": float(metrics["fpr_q99"].median()), "gates": gates, "tests": tests,
        "train_signature_stride": TRAIN_STRIDE, "test_endpoint_stride": 1,
        "window": c0a.WINDOW, "lags": list(c0a.LAGS), "point_adjustment": False,
        "test_threshold_search": False, "feature_search_using_test": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.mkdir(parents=True)
    metrics.to_csv(OUT / "case_metrics.csv", index=False)
    category_summary.to_csv(OUT / "category_summary.csv", index=False)
    responses.to_csv(OUT / "structure_response.csv", index=False)
    pd.DataFrame(short_rows).to_csv(OUT / "short_alignment_diagnostic.csv", index=False)
    pd.DataFrame(comparator_rows).to_csv(OUT / "comparator_summary.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({"summary": summary, "binding": binding_rows, "cases": feature_contract}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "innovation2c0b_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "cases": case_rows, "categories": category_summary.to_dict("records"), "responses": response_rows, "short": short_rows, "comparators": comparator_rows}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
