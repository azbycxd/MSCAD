"""Exact deterministic Innovation-2B refit and effective-corruption audit."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts" / "thesis" / "02_collaboration"
for item in (ROOT, HELPER):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_innovation2a2_joint_lagged as a2  # noqa: E402
import run_innovation2b_controlled as b2  # noqa: E402
from data.swat2015_official import (  # noqa: E402
    TRAIN_RAW_ROWS,
    load_normal_cache,
    resolve_repo_path,
    validate_config_contract,
)
from utils.config_io import load_config  # noqa: E402


OUT = ROOT / "outputs" / "thesis_innovation2b_diagnostic_refit"
ORIGINAL = ROOT / "outputs" / "thesis_innovation2b_controlled" / "controlled_metrics.csv"


def q(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability))


def spearman(left: pd.Series, right: pd.Series) -> float:
    return float(np.corrcoef(left.rank(method="average"), right.rank(method="average"))[0, 1])


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {OUT}")

    config = load_config(str(b2.CONFIG))
    validate_config_contract(config)
    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    normal, cache_manifest = load_normal_cache(cache_dir, config)
    order = [str(name) for name in cache_manifest["column_order"]]
    schema = json.loads(b2.SCHEMA.read_text(encoding="utf-8"))
    contract, profiles = a2.build_contract(a2.normal_profiles(schema), order)
    stable = pd.read_csv(b2.A2_STABLE)
    if tuple(stable["target"].astype(str)) != b2.EXPECTED_TARGETS:
        raise b2.ControlledError("frozen target order changed")

    scaler = StandardScaler().fit(np.asarray(normal[:TRAIN_RAW_ROWS], dtype=np.float64))
    z = scaler.transform(np.asarray(normal, dtype=np.float64))
    name_to_index = {name: index for index, name in enumerate(order)}
    train_times = np.arange(b2.MAX_LAG, TRAIN_RAW_ROWS, dtype=np.int64)
    calibration_times = np.arange(b2.CALIBRATION_START + b2.MAX_LAG, b2.CALIBRATION_END, dtype=np.int64)
    evaluation_endpoints = np.arange(b2.EVALUATION_START + b2.WINDOW - 1, b2.EVALUATION_END, dtype=np.int64)
    selected_positions, donor_positions = b2.deterministic_donors(len(evaluation_endpoints))
    current_endpoints = evaluation_endpoints[selected_positions]
    donor_endpoints = evaluation_endpoints[donor_positions]

    original = pd.read_csv(ORIGINAL).set_index("target")
    reproduction_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    sample_rows: list[dict] = []
    strength_rows: list[dict] = []
    discrete_rows: list[dict] = []
    continuous_rows: list[dict] = []

    for frozen in stable.itertuples(index=False):
        target = str(frozen.target)
        stage = str(frozen.stage)
        stage_names = list(contract[stage]["nonconstant_channels"])
        stage_indices = [name_to_index[name] for name in stage_names]
        target_local = stage_names.index(target)
        target_index = name_to_index[target]
        self_columns = np.arange(target_local * len(b2.LAGS), (target_local + 1) * len(b2.LAGS))
        all_columns = np.arange(len(stage_names) * len(b2.LAGS))
        cross_columns = np.setdiff1d(all_columns, self_columns, assume_unique=True)
        feature_names = [f"{name}@lag{lag}" for name in stage_names for lag in b2.LAGS]

        x_train = a2.lagged_design(z, train_times, stage_indices)
        y_train = z[train_times, target_index]
        self_model = b2.fit_ridge(x_train[:, self_columns], y_train)
        joint_model = b2.fit_ridge(x_train, y_train)

        x_cal = a2.lagged_design(z, calibration_times, stage_indices)
        y_cal = z[calibration_times, target_index]
        e_self_cal = np.abs(y_cal - self_model.predict(x_cal[:, self_columns]))
        e_joint_cal = np.abs(y_cal - joint_model.predict(x_cal))
        c_cal = e_joint_cal - e_self_cal
        reference_median, reference_scale = b2.robust_reference(c_cal)
        threshold = q(b2.score(c_cal, reference_median, reference_scale), 0.99)

        x_clean = a2.lagged_design(z, current_endpoints, stage_indices)
        x_corrupt = x_clean.copy()
        source_names = [name for name in stage_names if name != target]
        for source in source_names:
            source_local = stage_names.index(source)
            source_global = name_to_index[source]
            for lag_offset, lag in enumerate(b2.LAGS):
                x_corrupt[:, source_local * len(b2.LAGS) + lag_offset] = z[donor_endpoints - lag, source_global]
        if not np.array_equal(x_clean[:, self_columns], x_corrupt[:, self_columns]):
            raise b2.ControlledError(f"self feature mismatch for {target}")

        y = z[current_endpoints, target_index]
        self_clean_pred = self_model.predict(x_clean[:, self_columns])
        self_corrupt_pred = self_model.predict(x_corrupt[:, self_columns])
        joint_clean_pred = joint_model.predict(x_clean)
        joint_corrupt_pred = joint_model.predict(x_corrupt)
        e_self_clean = np.abs(y - self_clean_pred)
        e_self_corrupt = np.abs(y - self_corrupt_pred)
        e_joint_clean = np.abs(y - joint_clean_pred)
        e_joint_corrupt = np.abs(y - joint_corrupt_pred)
        clean_score = b2.score(e_joint_clean - e_self_clean, reference_median, reference_scale)
        corrupt_score = b2.score(e_joint_corrupt - e_self_corrupt, reference_median, reference_scale)
        metrics = b2.balanced_metrics(clean_score, corrupt_score)
        reproduced = {
            "collab_auroc": metrics["auroc"],
            "collab_aupr": metrics["aupr"],
            "corruption_recall": float(np.mean(corrupt_score > threshold)),
            "normal_fpr": float(np.mean(clean_score > threshold)),
        }
        reproduction_row = {"target": target}
        for metric, value in reproduced.items():
            expected = float(original.loc[target, metric])
            reproduction_row[f"refit_{metric}"] = value
            reproduction_row[f"original_{metric}"] = expected
            reproduction_row[f"abs_diff_{metric}"] = abs(value - expected)
        reproduction_rows.append(reproduction_row)

        delta_x = x_corrupt[:, cross_columns] - x_clean[:, cross_columns]
        distance = np.linalg.norm(delta_x, axis=1)
        delta_prediction = np.abs(delta_x @ np.asarray(joint_model.coef_)[cross_columns])
        if not np.isfinite(distance).all() or not np.isfinite(delta_prediction).all():
            raise b2.ControlledError(f"non-finite diagnostic for {target}")

        median_normal_joint = float(np.median(e_joint_cal))
        protected_denominator = max(median_normal_joint, b2.ROBUST_FLOOR)
        median_shift = float(np.median(delta_prediction))
        p90_shift = q(delta_prediction, 0.90)
        for index, (feature, coefficient) in enumerate(zip(feature_names, np.asarray(joint_model.coef_))):
            coefficient_rows.append({
                "target": target, "stage": stage, "feature_index": index,
                "feature_name": feature, "coefficient": float(coefficient),
                "intercept": float(joint_model.intercept_),
                "is_self": bool(index in self_columns), "is_cross": bool(index in cross_columns),
            })
        for index in range(len(current_endpoints)):
            sample_rows.append({
                "target": target, "sample_index": index,
                "selected_position": int(selected_positions[index]), "donor_position": int(donor_positions[index]),
                "clean_endpoint": int(current_endpoints[index]), "donor_endpoint": int(donor_endpoints[index]),
                "cross_feature_distance": float(distance[index]),
                "delta_cross_pred": float(delta_prediction[index]),
                "clean_joint_prediction": float(joint_clean_pred[index]),
                "corrupt_joint_prediction": float(joint_corrupt_pred[index]),
                "clean_self_prediction": float(self_clean_pred[index]),
                "corrupt_self_prediction": float(self_corrupt_pred[index]),
            })

        continuous_changes: list[np.ndarray] = []
        for source in source_names:
            source_local = stage_names.index(source)
            kind = str(profiles[source]["observed_value_kind"])
            for lag_offset, lag in enumerate(b2.LAGS):
                column = source_local * len(b2.LAGS) + lag_offset
                changes = np.abs(x_corrupt[:, column] - x_clean[:, column])
                if kind == "continuous_numeric":
                    continuous_changes.append(changes)
                    continuous_rows.append({
                        "target": target, "stage": stage, "source": source, "lag": lag,
                        "median_abs_standardized_change": float(np.median(changes)),
                        "p90_abs_standardized_change": q(changes, 0.90),
                    })
                else:
                    discrete_rows.append({
                        "target": target, "stage": stage, "source": source, "lag": lag,
                        "state_change_fraction": float(np.mean(changes != 0.0)),
                    })
        continuous_flat = np.concatenate(continuous_changes) if continuous_changes else np.array([], dtype=float)
        strength_rows.append({
            "target": target, "stage": stage, "validation_gain_2a2": float(frozen.joint_validation_gain),
            "cross_feature_distance_median": float(np.median(distance)),
            "cross_feature_distance_p90": q(distance, 0.90), "cross_feature_distance_p99": q(distance, 0.99),
            "cross_feature_distance_zero_fraction": float(np.mean(distance == 0.0)),
            "delta_cross_pred_median": median_shift, "delta_cross_pred_p90": p90_shift,
            "delta_cross_pred_p99": q(delta_prediction, 0.99),
            "delta_cross_pred_zero_fraction": float(np.mean(delta_prediction == 0.0)),
            "median_normal_e_joint": median_normal_joint, "q90_normal_e_joint": q(e_joint_cal, 0.90),
            "q99_normal_e_joint": q(e_joint_cal, 0.99),
            "ratio_denominator": protected_denominator,
            "denominator_was_protected": median_normal_joint < b2.ROBUST_FLOOR,
            "median_shift_ratio": median_shift / protected_denominator,
            "p90_shift_ratio": p90_shift / protected_denominator,
            "continuous_median_abs_standardized_change": float(np.median(continuous_flat)) if len(continuous_flat) else None,
            "continuous_p90_abs_standardized_change": q(continuous_flat, 0.90) if len(continuous_flat) else None,
            "collab_auroc": metrics["auroc"], "a0_auroc": float(original.loc[target, "a0_auroc"]),
        })

    reproduction = pd.DataFrame(reproduction_rows)
    difference_columns = [name for name in reproduction if name.startswith("abs_diff_")]
    max_difference = float(reproduction[difference_columns].to_numpy().max())
    OUT.mkdir(parents=True)
    reproduction.to_csv(OUT / "refit_reproduction.csv", index=False)
    if max_difference > 1e-8:
        (OUT / "diagnostic_summary.json").write_text(json.dumps({
            "status": "DETERMINISTIC_REFIT_REPRODUCTION_MISMATCH",
            "max_metric_absolute_difference": max_difference,
            "elapsed_seconds": time.perf_counter() - started,
        }, indent=2) + "\n", encoding="utf-8")
        raise b2.ControlledError(f"deterministic reproduction mismatch: {max_difference}")

    strength = pd.DataFrame(strength_rows)
    pd.DataFrame(coefficient_rows).to_csv(OUT / "fitted_coefficients.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(OUT / "sample_prediction_shift.csv", index=False)
    strength.to_csv(OUT / "target_corruption_strength.csv", index=False)
    pd.DataFrame(discrete_rows).to_csv(OUT / "discrete_state_change.csv", index=False)
    pd.DataFrame(continuous_rows).to_csv(OUT / "continuous_feature_change.csv", index=False)

    correlations = {
        "A_gain_vs_median_prediction_shift": spearman(strength["validation_gain_2a2"], strength["delta_cross_pred_median"]),
        "B_median_prediction_shift_vs_collab_auroc": spearman(strength["delta_cross_pred_median"], strength["collab_auroc"]),
        "C_median_shift_ratio_vs_collab_auroc": spearman(strength["median_shift_ratio"], strength["collab_auroc"]),
    }
    weak_count = int((strength["delta_cross_pred_median"] < 0.25 * strength["median_normal_e_joint"]).sum())
    effective_count = int((strength["delta_cross_pred_median"] > strength["median_normal_e_joint"]).sum())
    if weak_count >= 3:
        diagnosis = "CORRUPTION_EFFECTIVELY_WEAK"
    elif effective_count >= 4 and float(original["collab_auroc"].median()) < 0.70:
        diagnosis = "SCORE_NOT_SENSITIVE_DESPITE_EFFECTIVE_CORRUPTION"
    else:
        diagnosis = "MIXED_DIAGNOSIS"
    summary = {
        "refit_status": "DETERMINISTIC_REFIT_REPRODUCED",
        "max_metric_absolute_difference": max_difference,
        "weak_target_count": weak_count,
        "effective_target_count": effective_count,
        "original_median_collab_auroc": float(original["collab_auroc"].median()),
        "spearman": correlations,
        "final_diagnosis": diagnosis,
        "attack_data_read": False, "model_selection": False, "feature_search": False,
        "lag_search": False, "alpha_search": False, "new_corruption": False,
        "score_change": False, "test_tuning": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "targets": strength_rows, "discrete": discrete_rows}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
