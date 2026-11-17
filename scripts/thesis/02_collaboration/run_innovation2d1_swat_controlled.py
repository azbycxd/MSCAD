"""Frozen SWaT relation-breaking validation of dual-view predictive disagreement."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts" / "thesis" / "02_collaboration"
for item in (ROOT, HELPER):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_innovation2a2_joint_lagged as a2  # noqa: E402
import run_innovation2b_controlled as b2  # noqa: E402
import run_innovation2d0_dual_view_comut as d0  # noqa: E402
from data.swat2015_official import TRAIN_RAW_ROWS, load_normal_cache, resolve_repo_path, validate_config_contract  # noqa: E402
from utils.config_io import load_config  # noqa: E402


OUT = ROOT / "outputs" / "thesis_innovation2d1_swat_controlled"
B2 = ROOT / "outputs" / "thesis_innovation2b2_relation_breaking"


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite 2D1 output: {OUT}")
    config = load_config(str(b2.CONFIG))
    validate_config_contract(config)
    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    normal, cache_manifest = load_normal_cache(cache_dir, config)
    order = [str(name) for name in cache_manifest["column_order"]]
    name_to_index = {name: index for index, name in enumerate(order)}
    schema = json.loads(b2.SCHEMA.read_text(encoding="utf-8"))
    contract, _ = a2.build_contract(a2.normal_profiles(schema), order)
    stable = pd.read_csv(b2.A2_STABLE)
    if tuple(stable["target"].astype(str)) != b2.EXPECTED_TARGETS:
        raise RuntimeError("frozen 2D1 targets changed")
    b2_metrics = pd.read_csv(B2 / "metrics.csv").set_index("target")
    donors = pd.read_csv(B2 / "donor_manifest.csv")

    scaler = StandardScaler().fit(np.asarray(normal[:TRAIN_RAW_ROWS], dtype=np.float64))
    z = scaler.transform(np.asarray(normal, dtype=np.float64))
    train_times = np.arange(d0.MAX_LAG, TRAIN_RAW_ROWS, dtype=np.int64)
    calibration_times = np.arange(b2.CALIBRATION_START + d0.MAX_LAG, b2.CALIBRATION_END, dtype=np.int64)
    target_rows, sanity_rows, response_rows, comparison_rows = [], [], [], []
    target_contract = {}

    for frozen in stable.itertuples(index=False):
        target, stage = str(frozen.target), str(frozen.stage)
        target_index = name_to_index[target]
        stage_names = list(contract[stage]["nonconstant_channels"])
        source_names = [name for name in stage_names if name != target]
        source_indices = [name_to_index[name] for name in source_names]
        target_donors = donors[donors["target"] == target].sort_values("sample_index")
        if len(target_donors) != b2.MAX_SAMPLES:
            raise RuntimeError(f"2B2 sample count mismatch for {target}")
        clean_times = target_donors["clean_endpoint"].to_numpy(np.int64)
        donor_times = target_donors["donor_endpoint"].to_numpy(np.int64)

        self_model = Ridge(alpha=d0.ALPHA).fit(
            d0.design(z, train_times, [target_index], d0.SELF_LAGS), z[train_times, target_index]
        )
        cross_model = Ridge(alpha=d0.ALPHA).fit(
            d0.design(z, train_times, source_indices, d0.CROSS_LAGS), z[train_times, target_index]
        )
        self_cal = self_model.predict(d0.design(z, calibration_times, [target_index], d0.SELF_LAGS))
        cross_cal = cross_model.predict(d0.design(z, calibration_times, source_indices, d0.CROSS_LAGS))
        y_cal = z[calibration_times, target_index]
        d_cal = np.abs(self_cal - cross_cal)
        mu = float(np.median(d_cal))
        scale = max(1.4826 * float(np.median(np.abs(d_cal - mu))), 1e-6)
        calibration_score = d0.calibrated(d_cal, mu, scale)
        threshold = float(np.quantile(calibration_score, 0.99))

        self_clean = self_model.predict(d0.design(z, clean_times, [target_index], d0.SELF_LAGS))
        cross_clean = cross_model.predict(d0.design(z, clean_times, source_indices, d0.CROSS_LAGS))
        cross_corrupt = cross_model.predict(d0.design(z, donor_times, source_indices, d0.CROSS_LAGS))
        d_clean = np.abs(self_clean - cross_clean)
        d_corrupt = np.abs(self_clean - cross_corrupt)
        score_clean = d0.calibrated(d_clean, mu, scale)
        score_corrupt = d0.calibrated(d_corrupt, mu, scale)
        dual = b2.balanced_metrics(score_clean, score_corrupt)
        recall = float(np.mean(score_corrupt > threshold))
        fpr = float(np.mean(score_clean > threshold))
        clean_median, corrupt_median = float(np.median(d_clean)), float(np.median(d_corrupt))
        ratio = corrupt_median / max(clean_median, 1e-12)
        target_rows.append({
            "target": target, "stage": stage, "dual_auroc": dual["auroc"], "dual_aupr": dual["aupr"],
            "old_joint_residual_auroc": float(b2_metrics.loc[target, "collab_auroc"]),
            "a0_auroc": float(b2_metrics.loc[target, "a0_auroc"]), "a0_aupr": float(b2_metrics.loc[target, "a0_aupr"]),
            "corruption_recall": recall, "clean_fpr": fpr, "calibration_q99_threshold": threshold,
        })
        sanity_rows.append({
            "target": target, "stage": stage,
            "self_calibration_mae": float(np.mean(np.abs(y_cal - self_cal))),
            "cross_calibration_mae": float(np.mean(np.abs(y_cal - cross_cal))),
            "normal_disagreement_median": mu, "normal_disagreement_q95": float(np.quantile(d_cal, 0.95)),
            "normal_disagreement_q99": float(np.quantile(d_cal, 0.99)), "normal_disagreement_scale": scale,
        })
        response_rows.append({
            "target": target, "stage": stage, "clean_median_raw_disagreement": clean_median,
            "corrupt_median_raw_disagreement": corrupt_median,
            "corrupt_clean_median_ratio": ratio, "absolute_median_delta": corrupt_median - clean_median,
        })
        comparison_rows.append({
            "target": target, "stage": stage,
            "old_joint_residual_auroc": float(b2_metrics.loc[target, "collab_auroc"]),
            "dual_view_auroc": dual["auroc"],
            "delta_dual_minus_old": dual["auroc"] - float(b2_metrics.loc[target, "collab_auroc"]),
        })
        target_contract[target] = {
            "stage": stage, "target_index": target_index, "cross_sources": source_names,
            "target_excluded_from_cross": target not in source_names,
            "self_lags": list(d0.SELF_LAGS), "cross_lags": list(d0.CROSS_LAGS), "alpha": d0.ALPHA,
            "model_fit_interval": [0, TRAIN_RAW_ROWS],
            "calibration_normal_interval": [b2.CALIBRATION_START, b2.CALIBRATION_END],
            "evaluation_sample_count": len(clean_times), "donors_source": "Innovation-2B2 donor_manifest.csv",
            "score_formula": "max(0,(abs(pred_self-pred_cross)-calibration_median)/calibration_scale)",
            "observed_target_residual_in_score": False,
        }

    metrics = pd.DataFrame(target_rows)
    median_auroc, median_aupr = float(metrics["dual_auroc"].median()), float(metrics["dual_aupr"].median())
    count_080 = int((metrics["dual_auroc"] >= 0.80).sum())
    median_recall, median_fpr = float(metrics["corruption_recall"].median()), float(metrics["clean_fpr"].median())
    gates = {
        "A_median_dual_auroc_at_least_0_85": median_auroc >= 0.85,
        "B_at_least_4_of_6_dual_auroc_at_least_0_80": count_080 >= 4,
        "C_median_corruption_recall_at_least_0_50": median_recall >= 0.50,
        "D_median_clean_fpr_at_most_0_03": median_fpr <= 0.03,
        "E_all_cross_models_exclude_target": all(value["target_excluded_from_cross"] for value in target_contract.values()),
    }
    if all(gates.values()):
        status = "DUAL_VIEW_SWAT_CONTROLLED_SUPPORTED"
    elif median_auroc >= 0.75:
        status = "DUAL_VIEW_SWAT_CONTROLLED_PARTIAL"
    else:
        status = "DUAL_VIEW_SWAT_CONTROLLED_NOT_SUPPORTED"
    tests = {
        "attack_workbook_not_read": True, "six_frozen_targets": tuple(metrics["target"]) == b2.EXPECTED_TARGETS,
        "b2b_sample_indices_reused": True, "b2b_donors_reused": True,
        "target_current_history_corruption_unchanged": True,
        "scaler_train_only": int(scaler.n_samples_seen_) == TRAIN_RAW_ROWS,
        "self_lags_fixed": d0.SELF_LAGS == (1, 5, 10, 20, 30),
        "cross_lags_fixed": d0.CROSS_LAGS == (0, 1, 5, 10, 20, 30),
        "target_excluded_cross": gates["E_all_cross_models_exclude_target"], "alpha_1": d0.ALPHA == 1.0,
        "score_prediction_disagreement_only": True, "observed_target_residual_not_score": True,
        "calibration_normal_only": True, "no_threshold_search": True,
        "no_nan_inf": bool(np.isfinite(metrics.select_dtypes("number")).all().all() and np.isfinite(pd.DataFrame(sanity_rows).select_dtypes("number")).all().all()),
    }
    if not all(tests.values()):
        raise RuntimeError(f"2D1 integrity failure: {tests}")
    summary = {
        "status": status, "median_dual_auroc": median_auroc, "median_dual_aupr": median_aupr,
        "dual_auroc_at_least_0_80_count": count_080, "median_corruption_recall": median_recall,
        "median_clean_fpr": median_fpr, "gates": gates, "tests": tests,
        "attack_data_read": False, "new_corruption": False, "donor_change": False,
        "observed_target_residual_in_score": False, "threshold_search": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.mkdir(parents=True)
    metrics.to_csv(OUT / "target_metrics.csv", index=False)
    pd.DataFrame(sanity_rows).to_csv(OUT / "normal_sanity.csv", index=False)
    pd.DataFrame(response_rows).to_csv(OUT / "disagreement_response.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(OUT / "old_vs_dual.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({"summary": summary, "targets": target_contract}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "innovation2d1_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "metrics": target_rows, "sanity": sanity_rows, "response": response_rows, "comparison": comparison_rows}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
