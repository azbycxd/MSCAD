"""RETROSPECTIVE_VALIDITY_AUDIT: frozen 2D1 controlled replay orchestration.

Only the published runner's design, Ridge, calibration and metric functions are
called. Its main is never called. This module writes no files and reads no Attack
data. The caller must freeze all normal statistics before evaluate_controls.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TAG = "RETROSPECTIVE_VALIDITY_AUDIT"
TARGETS = ("AIT202", "FIT201", "DPIT301", "FIT301", "FIT401", "FIT601")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values):
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {
        "median": median,
        "mad": mad,
        "scaled_mad": 1.4826 * mad,
        "scale": max(1.4826 * mad, 1e-6),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "q99_median_ratio": float(np.quantile(values, 0.99)) / median if median != 0 else None,
    }


def prepare(root):
    """Fit six frozen predictors and obtain only Normal calibration statistics."""
    root = Path(root).resolve()
    for path in (root, root / "scripts/thesis/02_collaboration"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    d1 = importlib.import_module("run_innovation2d1_swat_controlled")
    if d1.ROOT.resolve() != root:
        raise RuntimeError("CONTROLLED_SOURCE_ROOT_MISMATCH")
    b2, a2, d0 = d1.b2, d1.a2, d1.d0
    required = [b2.CONFIG, b2.SCHEMA, b2.A2_STABLE, d1.B2 / "donor_manifest.csv",
                d1.OUT / "target_metrics.csv", d1.OUT / "manifest.json", d1.OUT / "normal_sanity.csv"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"STOP_A missing controlled artifacts: {missing}")
    config = d1.load_config(str(b2.CONFIG))
    d1.validate_config_contract(config)
    cache_dir = d1.resolve_repo_path(root, config["paths"]["cache_dir"])
    normal, cache_manifest = d1.load_normal_cache(cache_dir, config)
    order = [str(x) for x in cache_manifest["column_order"]]
    mapping = {name: index for index, name in enumerate(order)}
    schema = json.loads(b2.SCHEMA.read_text(encoding="utf-8"))
    contract, _ = a2.build_contract(a2.normal_profiles(schema), order)
    stable = pd.read_csv(b2.A2_STABLE)
    if tuple(stable["target"].astype(str)) != TARGETS or b2.EXPECTED_TARGETS != TARGETS:
        raise RuntimeError("CONTROLLED_FROZEN_TARGET_MISMATCH")
    official_manifest = json.loads((d1.OUT / "manifest.json").read_text(encoding="utf-8"))
    official_metrics = pd.read_csv(d1.OUT / "target_metrics.csv").set_index("target")
    if tuple(official_metrics.index.astype(str)) != TARGETS:
        raise RuntimeError("CONTROLLED_METRIC_TARGET_MISMATCH")
    donors = pd.read_csv(d1.B2 / "donor_manifest.csv")
    if len(donors) != 6 * 2000 or set(donors.target.astype(str)) != set(TARGETS):
        raise RuntimeError("CONTROLLED_DONOR_TARGET_OR_COUNT_MISMATCH")
    scaler = d1.StandardScaler().fit(np.asarray(normal[:d1.TRAIN_RAW_ROWS], dtype=np.float64))
    z = scaler.transform(np.asarray(normal, dtype=np.float64))
    train_times = np.arange(d0.MAX_LAG, d1.TRAIN_RAW_ROWS, dtype=np.int64)
    cal_times = np.arange(b2.CALIBRATION_START + d0.MAX_LAG, b2.CALIBRATION_END, dtype=np.int64)
    cases, normal_stats, audit_cases = {}, [], []
    for row in stable.itertuples(index=False):
        target, stage = str(row.target), str(row.stage)
        j = mapping[target]
        source_names = [name for name in contract[stage]["nonconstant_channels"] if name != target]
        source_indices = [mapping[name] for name in source_names]
        frozen = official_manifest["targets"][target]
        if source_names != frozen["cross_sources"]:
            raise RuntimeError(f"CONTROLLED_SOURCE_LIST_MISMATCH: {target}")
        if frozen["calibration_normal_interval"] != [b2.CALIBRATION_START, b2.CALIBRATION_END]:
            raise RuntimeError(f"CONTROLLED_CALIBRATION_MISMATCH: {target}")
        paired = donors[donors.target == target].sort_values("sample_index")
        if not np.array_equal(paired.sample_index.to_numpy(np.int64), np.arange(2000)):
            raise RuntimeError(f"CONTROLLED_SAMPLE_INDEX_MISMATCH: {target}")
        clean_times = paired.clean_endpoint.to_numpy(np.int64)
        donor_times = paired.donor_endpoint.to_numpy(np.int64)
        if min(clean_times.min(), donor_times.min()) < b2.EVALUATION_START + d0.MAX_LAG or max(clean_times.max(), donor_times.max()) >= b2.EVALUATION_END:
            raise RuntimeError(f"CONTROLLED_DONOR_RANGE_MISMATCH: {target}")
        if not np.array_equal(np.abs(clean_times - donor_times), paired.donor_separation.to_numpy(np.int64)):
            raise RuntimeError(f"CONTROLLED_DONOR_SEPARATION_MISMATCH: {target}")
        # Exact classes and parameters exposed by the official controlled runner.
        self_model = d1.Ridge(alpha=d0.ALPHA).fit(
            d0.design(z, train_times, [j], d0.SELF_LAGS), z[train_times, j])
        cross_model = d1.Ridge(alpha=d0.ALPHA).fit(
            d0.design(z, train_times, source_indices, d0.CROSS_LAGS), z[train_times, j])
        self_cal = self_model.predict(d0.design(z, cal_times, [j], d0.SELF_LAGS))
        cross_cal = cross_model.predict(d0.design(z, cal_times, source_indices, d0.CROSS_LAGS))
        y_cal = z[cal_times, j]
        residual_self, residual_cross = y_cal - self_cal, y_cal - cross_cal
        covariance = float(np.mean((residual_cross - residual_cross.mean()) * (residual_self - residual_self.mean())))
        kappa = covariance - 0.5 * float(np.var(residual_self))
        calibration_raw = {"S1": np.abs(residual_self), "S2": np.abs(residual_cross), "S3": np.abs(self_cal - cross_cal)}
        references = {}
        for score_name, values in calibration_raw.items():
            stats = _stats(values)
            normalized = d0.calibrated(values, stats["median"], stats["scale"])
            stats["threshold"] = float(np.quantile(normalized, 0.99))
            references[score_name] = stats
            normal_stats.append({"audit_label": TAG, "dataset": "SWaT controlled", "target": target, "stage": stage,
                                 "score": score_name, "scope": "CONTROLLED_HALF_VALIDATION", "n": len(cal_times),
                                 "var_self": float(np.var(residual_self)), "var_cross": float(np.var(residual_cross)),
                                 "cov_cross_self": covariance, "kappa": kappa, **stats})
        cases[target] = {"target": target, "stage": stage, "target_index": j, "source_names": source_names,
                         "source_indices": source_indices, "self_model": self_model, "cross_model": cross_model,
                         "clean_times": clean_times, "donor_times": donor_times, "references": references,
                         "calibration_raw": calibration_raw, "frozen_metrics": official_metrics.loc[target].to_dict()}
        audit_cases.append({"target": target, "stage": stage, "sample_count": len(paired),
                            "cross_sources": source_names, "target_excluded": target not in source_names,
                            "pair_hash": hashlib.sha256(paired[["sample_index", "clean_endpoint", "donor_endpoint"]].to_numpy(np.int64).tobytes()).hexdigest(),
                            "self_and_target_from_clean": True, "cross_only_from_donor": True,
                            "clean_range": [int(clean_times.min()), int(clean_times.max())],
                            "donor_range": [int(donor_times.min()), int(donor_times.max())]})
    audit = {"audit_label": TAG, "status": "PASS", "targets": list(TARGETS), "total_pairs": len(donors),
             "calibration_interval": [b2.CALIBRATION_START, b2.CALIBRATION_END],
             "calibration_endpoint_start": int(cal_times[0]), "calibration_endpoint_end_exclusive": int(cal_times[-1] + 1),
             "no_cross_target_aggregation": True, "no_new_corruption": True, "attack_data_read": False,
             "donor_manifest_sha256": _sha(d1.B2 / "donor_manifest.csv"), "cases": audit_cases}
    return {"d1": d1, "z": z, "cases": cases, "normal_stats": normal_stats, "donor_audit": audit,
            "input_paths": [str(p) for p in required], "cache_dir": str(cache_dir), "replay": None}


def _predict_case(state, case):
    d0 = state["d1"].d0
    z, j = state["z"], case["target_index"]
    clean, donor = case["clean_times"], case["donor_times"]
    p_self = case["self_model"].predict(d0.design(z, clean, [j], d0.SELF_LAGS))
    p_cross = case["cross_model"].predict(d0.design(z, clean, case["source_indices"], d0.CROSS_LAGS))
    p_donor = case["cross_model"].predict(d0.design(z, donor, case["source_indices"], d0.CROSS_LAGS))
    return z[clean, j], p_self, p_cross, p_donor


def replay_s3(state):
    """Reproduce only the formally frozen S3 controlled metrics before new metrics."""
    b2, d0 = state["d1"].b2, state["d1"].d0
    rows = []
    for target, case in state["cases"].items():
        y, self_pred, cross_pred, donor_pred = _predict_case(state, case)
        case["predictions"] = (y, self_pred, cross_pred, donor_pred)
        ref = case["references"]["S3"]
        clean = d0.calibrated(np.abs(self_pred - cross_pred), ref["median"], ref["scale"])
        corrupt = d0.calibrated(np.abs(self_pred - donor_pred), ref["median"], ref["scale"])
        pair = b2.balanced_metrics(clean, corrupt)
        observed = {"dual_auroc": pair["auroc"], "dual_aupr": pair["aupr"],
                    "corruption_recall": float(np.mean(corrupt > ref["threshold"])),
                    "clean_fpr": float(np.mean(clean > ref["threshold"])),
                    "calibration_q99_threshold": ref["threshold"]}
        differences = {key: value - float(case["frozen_metrics"][key]) for key, value in observed.items()}
        matches = {key: round(observed[key], 6) == round(float(case["frozen_metrics"][key]), 6)
                   for key in observed if key != "calibration_q99_threshold"}
        rows.append({"target": target, "passed": all(matches.values()), "observed": observed,
                     "expected": {key: float(case["frozen_metrics"][key]) for key in observed}, "differences": differences, "checks": matches})
    result = {"audit_label": TAG, "passed": all(row["passed"] for row in rows), "metric_decimals": 6,
              "controlled_threshold_difference_diagnostic_only": True, "cases": rows,
              "median_auroc": float(np.median([r["observed"]["dual_auroc"] for r in rows])),
              "median_aupr": float(np.median([r["observed"]["dual_aupr"] for r in rows]))}
    state["replay"] = result
    return result


def evaluate_controls(state, normal_freeze_sha):
    """Compute new controlled S1/S2 metrics only after root freezes normal stats."""
    if not state["replay"] or not state["replay"]["passed"]:
        raise RuntimeError("STOP_B controlled S3 reproduction did not pass")
    if not isinstance(normal_freeze_sha, str) or len(normal_freeze_sha) != 64:
        raise RuntimeError("NORMAL_STATISTICS_NOT_HASH_FROZEN")
    b2, d0 = state["d1"].b2, state["d1"].d0
    rows, checks = [], []
    for target, case in state["cases"].items():
        y, self_pred, cross_pred, donor_pred = case["predictions"]
        raw = {"S1": (np.abs(y - self_pred), np.abs(y - self_pred)),
               "S2": (np.abs(y - cross_pred), np.abs(y - donor_pred)),
               "S3": (np.abs(self_pred - cross_pred), np.abs(self_pred - donor_pred))}
        equal_delta = float(np.max(np.abs(raw["S1"][0] - raw["S1"][1])))
        if equal_delta != 0:
            raise RuntimeError(f"S1_CONSISTENCY_FLAG: {target}")
        signed_delta_s2 = (y - donor_pred) - (y - cross_pred)
        signed_delta_s3 = (self_pred - donor_pred) - (self_pred - cross_pred)
        signed_delta_difference = float(np.max(np.abs(signed_delta_s2 - signed_delta_s3)))
        checks.append({"target": target, "s1_pair_max_abs_difference": equal_delta,
                       "s2_s3_signed_change_max_abs_difference": signed_delta_difference})
        for score_name, (clean_raw, corrupt_raw) in raw.items():
            ref = case["references"][score_name]
            clean = d0.calibrated(clean_raw, ref["median"], ref["scale"])
            corrupt = d0.calibrated(corrupt_raw, ref["median"], ref["scale"])
            metrics = b2.balanced_metrics(clean, corrupt)
            if score_name == "S1" and metrics["auroc"] != 0.5:
                raise RuntimeError(f"S1_CONSISTENCY_FLAG: AUROC {target}")
            rows.append({"audit_label": TAG, "target": target, "stage": case["stage"], "score": score_name,
                         "auroc": metrics["auroc"], "aupr": metrics["aupr"],
                         "recall_q99": float(np.mean(corrupt > ref["threshold"])),
                         "fpr_q99": float(np.mean(clean > ref["threshold"])), "threshold": ref["threshold"],
                         "cross_sources": ";".join(case["source_names"]), "sample_pairs": len(clean),
                         "normal_median": ref["median"], "normal_mad": ref["mad"],
                         "normal_scaled_mad": ref["scaled_mad"], "normal_scale": ref["scale"],
                         "normal_q95": ref["q95"], "normal_q99": ref["q99"],
                         "s1_pair_max_abs_difference": equal_delta, "normal_stats_sha256": normal_freeze_sha})
    return rows, {"audit_label": TAG, "s1_consistency": "PASS", "checks": checks}
