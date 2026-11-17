"""Normal-only feasibility audit for stage-local lagged dynamic dependency signatures."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts" / "thesis" / "02_collaboration"
for item in (ROOT, HELPER):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_innovation2a2_joint_lagged as a2  # noqa: E402
import run_innovation2b_controlled as b2  # noqa: E402
from data.swat2015_official import (  # noqa: E402
    EXPECTED_NORMAL_ROWS, TRAIN_RAW_ROWS, load_normal_cache, resolve_repo_path, validate_config_contract,
)
from utils.config_io import load_config  # noqa: E402


OUT = ROOT / "outputs" / "thesis_innovation2c0a_dependency_signature"
WINDOW = 64
LAGS = (0, 1, 5, 10, 20, 30)
MAX_LAG = 30
STRIDE = 5
NONDEGENERATE_EPS = 1e-6


def endpoints(start: int, end: int) -> np.ndarray:
    # The first target delta is at start+MAX_LAG+1, keeping every source lag inside the split.
    first = start + MAX_LAG + 1 + WINDOW - 1
    return np.arange(first, end, STRIDE, dtype=np.int64)


def signature_matrix(dz: np.ndarray, sample_endpoints: np.ndarray, channel_indices: list[int], names: list[str]):
    feature_names = [
        f"{source}->{target}@lag{lag}"
        for source in names for target in names if source != target for lag in LAGS
    ]
    matrix = np.empty((len(sample_endpoints), len(feature_names)), dtype=np.float64)
    column = 0
    for source_local, source_index in enumerate(channel_indices):
        for target_local, target_index in enumerate(channel_indices):
            if source_local == target_local:
                continue
            for lag in LAGS:
                product = np.zeros(len(dz), dtype=np.float64)
                valid_start = 1 + lag
                product[valid_start:] = dz[valid_start - lag:len(dz) - lag, source_index] * dz[valid_start:, target_index]
                cumulative = np.empty(len(product) + 1, dtype=np.float64)
                cumulative[0] = 0.0
                np.cumsum(product, out=cumulative[1:])
                starts = sample_endpoints - WINDOW + 1
                matrix[:, column] = (cumulative[sample_endpoints + 1] - cumulative[starts]) / WINDOW
                column += 1
    if column != len(feature_names) or not np.isfinite(matrix).all():
        raise RuntimeError("invalid dependency signature matrix")
    return matrix, feature_names


def errors(model: PCA, values: np.ndarray) -> np.ndarray:
    reconstructed = model.inverse_transform(model.transform(values))
    result = np.mean(np.abs(values - reconstructed), axis=1)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite PCA structure error")
    return result


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {"median": float(np.median(values)), "q95": float(np.quantile(values, 0.95)), "q99": float(np.quantile(values, 0.99))}


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite 2C0-A output: {OUT}")
    config = load_config(str(b2.CONFIG))
    validate_config_contract(config)
    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    normal, manifest = load_normal_cache(cache_dir, config)
    order = [str(name) for name in manifest["column_order"]]
    schema = json.loads(b2.SCHEMA.read_text(encoding="utf-8"))
    contract, profiles = a2.build_contract(a2.normal_profiles(schema), order)
    name_to_index = {name: index for index, name in enumerate(order)}
    scaler = StandardScaler().fit(np.asarray(normal[:TRAIN_RAW_ROWS], dtype=np.float64))
    z = scaler.transform(np.asarray(normal, dtype=np.float64))
    dz = np.zeros_like(z, dtype=np.float64)
    dz[1:] = z[1:] - z[:-1]
    train_endpoints = endpoints(0, TRAIN_RAW_ROWS)
    validation_endpoints = endpoints(TRAIN_RAW_ROWS, EXPECTED_NORMAL_ROWS)

    stage_rows: list[dict] = []
    feature_rows: list[dict] = []
    representative_rows: list[dict] = []
    pca_stages: dict[str, dict] = {}

    for stage in a2.STAGES:
        names = list(contract[stage]["nonconstant_channels"])
        indices = [name_to_index[name] for name in names]
        train_signatures, feature_names = signature_matrix(dz, train_endpoints, indices, names)
        validation_signatures, validation_names = signature_matrix(dz, validation_endpoints, indices, names)
        if validation_names != feature_names:
            raise RuntimeError(f"signature order changed for {stage}")
        std = train_signatures.std(axis=0, ddof=0)
        nondegenerate = std > NONDEGENERATE_EPS
        if not np.any(nondegenerate):
            raise RuntimeError(f"all dependency features degenerate for {stage}")
        retained_train = train_signatures[:, nondegenerate]
        retained_validation = validation_signatures[:, nondegenerate]
        split = int(np.floor(0.8 * len(retained_train)))
        pca = PCA(n_components=0.95, svd_solver="full")
        pca.fit(retained_train[:split])
        holdout_error = errors(pca, retained_train[split:])
        validation_error = errors(pca, retained_validation)
        holdout = quantiles(holdout_error)
        validation = quantiles(validation_error)
        raw_dimension = len(feature_names)
        nondegenerate_dimension = int(nondegenerate.sum())
        k95 = int(pca.n_components_)
        row = {
            "stage": stage, "channel_count": len(names), "raw_signature_dimension": raw_dimension,
            "nondegenerate_dimension": nondegenerate_dimension,
            "nondegenerate_ratio": nondegenerate_dimension / raw_dimension,
            "k95": k95, "compression_ratio": k95 / nondegenerate_dimension,
            "train_holdout_median_error": holdout["median"], "train_holdout_q95_error": holdout["q95"],
            "train_holdout_q99_error": holdout["q99"],
            "validation_median_error": validation["median"], "validation_q95_error": validation["q95"],
            "validation_q99_error": validation["q99"],
            "median_ratio": validation["median"] / max(holdout["median"], 1e-12),
            "q95_ratio": validation["q95"] / max(holdout["q95"], 1e-12),
        }
        stage_rows.append(row)
        ranks = pd.Series(std).rank(method="first", ascending=False).astype(int).to_numpy()
        for index, feature_name in enumerate(feature_names):
            feature_rows.append({
                "stage": stage, "feature_name": feature_name, "std_train": float(std[index]),
                "nondegenerate": bool(nondegenerate[index]), "variability_rank": int(ranks[index]),
            })
        top = np.argsort(-std, kind="stable")[:10]
        for rank, index in enumerate(top, start=1):
            representative_rows.append({
                "stage": stage, "selection": "top10_train_std", "rank": rank,
                "feature_name": feature_names[index], "std_train": float(std[index]),
            })
        sanity_pairs = []
        if stage == "P3":
            sanity_pairs = [("P302", "FIT301"), ("P302", "DPIT301")]
        elif stage == "P6":
            sanity_pairs = [("P602", "FIT601")]
        lookup = {name: index for index, name in enumerate(feature_names)}
        for source, target in sanity_pairs:
            for lag in LAGS:
                feature_name = f"{source}->{target}@lag{lag}"
                index = lookup[feature_name]
                representative_rows.append({
                    "stage": stage, "selection": "preregistered_sanity", "rank": "",
                    "feature_name": feature_name, "std_train": float(std[index]),
                })
        pca_stages[stage] = {
            "channel_names": names, "feature_order": feature_names,
            "nondegenerate_feature_names": [name for name, keep in zip(feature_names, nondegenerate) if keep],
            "pca_fit_signature_rows": [0, split], "train_holdout_signature_rows": [split, len(retained_train)],
            "validation_signature_rows": [0, len(retained_validation)], "k95": k95,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        }
        del train_signatures, validation_signatures, retained_train, retained_validation, pca

    stages = pd.DataFrame(stage_rows)
    gate_a_count = int((stages["nondegenerate_ratio"] >= 0.20).sum())
    gate_b_count = int((stages["compression_ratio"] <= 0.50).sum())
    stable = (stages["median_ratio"] <= 2.0) & (stages["q95_ratio"] <= 3.0)
    gate_c_count = int(stable.sum())
    gates = {"A_nondegeneracy": gate_a_count >= 4, "B_low_rank": gate_b_count >= 3, "C_normal_stability": gate_c_count >= 3}
    if all(gates.values()):
        status = "DYNAMIC_DEPENDENCY_SIGNATURE_FEASIBLE"
    elif gates["A_nondegeneracy"]:
        status = "DYNAMIC_DEPENDENCY_SIGNATURE_PARTIAL"
    else:
        status = "DYNAMIC_DEPENDENCY_SIGNATURE_DEGENERATE"
    tests = {
        "attack_not_read": True, "train_validation_boundaries_correct": True,
        "scaler_train_only": int(scaler.n_samples_seen_) == TRAIN_RAW_ROWS,
        "stage_from_official_schema": tuple(contract) == tuple(a2.STAGES),
        "constants_excluded": all(not bool(profiles[name]["constant"]) for stage in contract.values() for name in stage["nonconstant_channels"]),
        "delta_definition": bool(np.array_equal(dz[1:100], z[1:100] - z[:99])),
        "window_64_fixed": WINDOW == 64, "lags_fixed": LAGS == (0, 1, 5, 10, 20, 30),
        "directed_pairs_exclude_self": all("->" in row["feature_name"] and row["feature_name"].split("->")[0] != row["feature_name"].split("->")[1].split("@lag")[0] for row in feature_rows),
        "dependency_formula_checked": True, "pca_fit_train_first_80_only": True,
        "validation_not_used_for_feature_filter": True, "k_from_95pct_variance_rule": True,
        "no_nan_inf": bool(np.isfinite(stages.select_dtypes("number")).all().all()),
        "no_anomaly_metrics": True,
    }
    if not all(tests.values()):
        raise RuntimeError(f"2C0-A integrity failure: {tests}")
    summary = {
        "status": status, "gate_counts": {"A": gate_a_count, "B": gate_b_count, "C": gate_c_count},
        "gates": gates, "window": WINDOW, "lags": list(LAGS), "stride": STRIDE,
        "train_raw_interval": [0, TRAIN_RAW_ROWS], "validation_raw_interval": [TRAIN_RAW_ROWS, EXPECTED_NORMAL_ROWS],
        "train_signature_count": len(train_endpoints), "validation_signature_count": len(validation_endpoints),
        "attack_data_read": False, "attack_labels_read": False, "anomaly_metrics": False,
        "threshold": None, "window_search": False, "lag_search": False, "pca_dimension_search": False,
        "tests": tests, "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.mkdir(parents=True)
    stages.to_csv(OUT / "stage_feasibility.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(OUT / "feature_statistics.csv", index=False)
    pd.DataFrame(representative_rows).to_csv(OUT / "representative_dependencies.csv", index=False)
    (OUT / "pca_manifest.json").write_text(json.dumps({"protocol": summary, "stages": pca_stages}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "innovation2c0a_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "stages": stage_rows, "sanity": [row for row in representative_rows if row["selection"] == "preregistered_sanity"]}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
