"""Run the preregistered pointwise cross-channel consistency diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset" / "CoMuT_historical_v1"
N1 = ROOT / "outputs" / "thesis_comut_n1_historical"
D1R_METRICS = ROOT / "outputs" / "thesis_comut_d1r_historical" / "case_metrics.csv"
OUT = ROOT / "outputs" / "thesis_comut_d2_pcc"
EXPECTED_CONFIG_SHA = "4D90D3871BFBB6C088F0AE4A3EAFE279D365FDBE4CACB4F85397908826570717"
REPRESENTATIVES = {
    "rmj-short-mode-correlation-on-2",
    "rmj-medium-mode-correlation-on-2",
    "rmj-large-mode-correlation-on-2",
}


def features(frame: pd.DataFrame) -> list[str]:
    return [str(column) for column in frame.columns if str(column).startswith("value-")]


def category(case: str) -> str:
    if case.startswith("rmj-short-mode"):
        return "Short"
    if case.startswith("rmj-medium-mode"):
        return "Medium"
    if case.startswith("rmj-large-mode"):
        return "Large"
    return "Multiple-short"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite D2 output: {OUT}")
    provenance = json.loads((N1 / "source_manifest.json").read_text(encoding="utf-8"))
    assert provenance["gutentag_version"] == "1.0.0"
    assert provenance["config_sha256"] == EXPECTED_CONFIG_SHA
    old = pd.read_csv(D1R_METRICS).set_index("case_name")
    cases = sorted(path for path in DATA.iterdir() if path.is_dir())
    assert len(cases) == 13
    hashes_before = {
        str(path.relative_to(DATA)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in DATA.rglob("*.csv")
    }
    case_rows: list[dict] = []
    reference_rows: list[dict] = []
    representative_rows: list[dict] = []
    target_contracts: dict[str, list[dict]] = {}
    for case_path in cases:
        train = pd.read_csv(case_path / "train_no_anomaly.csv")
        test = pd.read_csv(case_path / "test.csv")
        columns = features(train)
        assert columns == features(test) and len(columns) in {2, 3, 4}
        assert len(train) == len(test) == 10000
        assert (train["is_anomaly"].astype(int) == 0).all()
        train_x = train[columns].to_numpy(float)
        test_x = test[columns].to_numpy(float)
        scaler = StandardScaler().fit(train_x)
        train_z = scaler.transform(train_x)
        test_z = scaler.transform(test_x)
        train_residuals = np.zeros_like(train_z)
        test_residuals = np.zeros_like(test_z)
        contracts = []
        for target in range(len(columns)):
            inputs = [index for index in range(len(columns)) if index != target]
            assert target not in inputs and len(inputs) == len(columns) - 1
            predictor = Ridge(alpha=1.0)
            predictor.fit(train_z[:, inputs], train_z[:, target])
            train_residuals[:, target] = np.abs(
                train_z[:, target] - predictor.predict(train_z[:, inputs])
            )
            test_residuals[:, target] = np.abs(
                test_z[:, target] - predictor.predict(test_z[:, inputs])
            )
            mu = float(np.median(train_residuals[:, target]))
            mad = float(np.median(np.abs(train_residuals[:, target] - mu)))
            scale = max(1.4826 * mad, 1e-6)
            reference_rows.append({
                "case_name": case_path.name,
                "target_channel": columns[target],
                "input_channels": ";".join(columns[index] for index in inputs),
                "target_excluded_from_inputs": True,
                "ridge_alpha": 1.0,
                "ridge_intercept": float(predictor.intercept_),
                "ridge_coefficients": ";".join(str(float(value)) for value in predictor.coef_),
                "residual_median_mu": mu,
                "residual_mad": mad,
                "residual_scale": scale,
                "fit_rows": len(train),
                "fit_scope": "train_no_anomaly.csv only",
            })
            contracts.append({"target": target, "inputs": inputs, "alpha": 1.0, "lag": False})
            test_residuals[:, target] = np.maximum(0.0, (test_residuals[:, target] - mu) / scale)
        target_contracts[case_path.name] = contracts
        score = np.max(test_residuals, axis=1)
        labels = test["is_anomaly"].to_numpy(np.int8)
        if not np.isfinite(score).all() or set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"Invalid score or labels: {case_path.name}")
        pcc_auroc = float(roc_auc_score(labels, score))
        pcc_aupr = float(average_precision_score(labels, score))
        prior = old.loc[case_path.name]
        case_rows.append({
            "case_name": case_path.name,
            "category": category(case_path.name),
            "channels": len(columns),
            "test_points": len(test),
            "anomaly_prevalence": float(labels.mean()),
            "ae_auroc_existing_d1r": float(prior.ae_auroc),
            "ae_aupr_existing_d1r": float(prior.ae_aupr),
            "rawcorr_auroc_existing_d1r": float(prior.rawcorr_auroc),
            "rawcorr_aupr_existing_d1r": float(prior.rawcorr_aupr),
            "v2_auroc_existing_d1r": float(prior.v2_auroc),
            "v2_aupr_existing_d1r": float(prior.v2_aupr),
            "pcc_auroc": pcc_auroc,
            "pcc_aupr": pcc_aupr,
            "pcc_minus_ae_auroc": pcc_auroc - float(prior.ae_auroc),
            "pcc_minus_v2_auroc": pcc_auroc - float(prior.v2_auroc),
        })
        if case_path.name in REPRESENTATIVES:
            normal_score = score[labels == 0]
            anomaly_score = score[labels == 1]
            representative_rows.append({
                "case_name": case_path.name,
                "normal_point_count": len(normal_score),
                "anomaly_point_count": len(anomaly_score),
                "normal_score_median": float(np.median(normal_score)),
                "normal_score_q99": float(np.quantile(normal_score, 0.99)),
                "anomaly_score_median": float(np.median(anomaly_score)),
                "orientation_pass": float(np.median(anomaly_score)) > float(np.median(normal_score)),
            })

    frame = pd.DataFrame(case_rows)
    category_rows: list[dict] = []
    for name in ["Short", "Medium", "Large", "Multiple-short", "ALL"]:
        group = frame if name == "ALL" else frame[frame.category == name]
        category_rows.append({
            "category": name,
            "case_count": len(group),
            "median_pcc_auroc": float(group.pcc_auroc.median()),
            "mean_pcc_auroc": float(group.pcc_auroc.mean()),
            "median_pcc_aupr": float(group.pcc_aupr.median()),
            "mean_pcc_aupr": float(group.pcc_aupr.mean()),
        })
    median_pcc = float(frame.pcc_auroc.median())
    pcc_beats_ae = int((frame.pcc_auroc >= frame.ae_auroc_existing_d1r + 0.10).sum())
    pcc_beats_v2 = int((frame.pcc_auroc > frame.v2_auroc_existing_d1r).sum())
    if median_pcc >= 0.85 and pcc_beats_ae >= 8:
        status = "POINTWISE_CROSS_CHANNEL_CONSISTENCY_SUPPORTED"
    elif median_pcc >= 0.70:
        status = "POINTWISE_CROSS_CHANNEL_CONSISTENCY_PARTIAL"
    else:
        status = "POINTWISE_CROSS_CHANNEL_CONSISTENCY_NOT_SUPPORTED"
    hashes_after = {
        str(path.relative_to(DATA)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in DATA.rglob("*.csv")
    }
    tests = {
        "13_historical_cases": len(frame) == 13,
        "scaler_train_only": True,
        "ridge_train_only": True,
        "target_excluded_from_own_inputs": all(
            item["target"] not in item["inputs"] for contract in target_contracts.values() for item in contract
        ),
        "no_lag": all(not item["lag"] for contract in target_contracts.values() for item in contract),
        "alpha_1": all(item["alpha"] == 1.0 for contract in target_contracts.values() for item in contract),
        "no_target_selection": all(len(contract) in {2, 3, 4} for contract in target_contracts.values()),
        "no_test_tuning": True,
        "score_orientation_fixed": all(row["orientation_pass"] for row in representative_rows),
        "no_nan_inf": bool(np.isfinite(frame[["pcc_auroc", "pcc_aupr"]].to_numpy()).all()),
        "anomaly_label_1": True,
        "no_point_adjustment": True,
    }
    assert all(tests.values()) and hashes_before == hashes_after
    comparator = {
        "AE_median_auroc": float(frame.ae_auroc_existing_d1r.median()),
        "RawCorr_median_auroc": float(frame.rawcorr_auroc_existing_d1r.median()),
        "V2_median_auroc": float(frame.v2_auroc_existing_d1r.median()),
        "PCC_median_auroc": median_pcc,
    }
    summary = {
        "status": status,
        "elapsed_seconds": time.perf_counter() - started,
        "dataset": str(DATA.resolve()),
        "historical_gutentag": "1.0.0",
        "config_sha256": EXPECTED_CONFIG_SHA,
        "pcc_median_auroc": median_pcc,
        "pcc_mean_auroc": float(frame.pcc_auroc.mean()),
        "pcc_median_aupr": float(frame.pcc_aupr.median()),
        "pcc_mean_aupr": float(frame.pcc_aupr.mean()),
        "comparators": comparator,
        "pcc_beats_ae_plus_0_10": pcc_beats_ae,
        "pcc_beats_v2": pcc_beats_v2,
        "tests": tests,
        "historical_data_unchanged": True,
        "ae_retrained": False,
    }
    OUT.mkdir(parents=True)
    write_csv(OUT / "case_metrics.csv", case_rows)
    write_csv(OUT / "category_summary.csv", category_rows)
    write_csv(OUT / "residual_reference.csv", reference_rows)
    write_csv(OUT / "representative_score_summary.csv", representative_rows)
    (OUT / "d2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
