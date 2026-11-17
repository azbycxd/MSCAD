from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


CROSS_LAGS = (0, 5, 10, 20, 30)
SELF_LAGS = (1, 5, 10, 20, 30)
ALPHAS = (.01, .1, 1.0, 10.0, 100.0)


@dataclass
class TargetModels:
    target: str
    target_index: int
    predictor_indices: np.ndarray
    predictor_names: list[str]
    alpha_cross: float
    alpha_self: float
    cross: Ridge
    self_history: Ridge
    residual_reference_cross: dict[str, float]
    residual_reference_self: dict[str, float]


def eligible_indices(validity: np.ndarray, start: int, end: int, required_lags: tuple[int, ...]) -> np.ndarray:
    maximum = max(required_lags)
    candidates = np.arange(start + maximum, end, dtype=np.int64)
    usable = np.ones(len(candidates), dtype=bool)
    for lag in required_lags:
        usable &= np.asarray(validity)[candidates - lag]
    return candidates[usable]


def cross_design(values: np.ndarray, target_index: int, indices: np.ndarray,
                 lags: tuple[int, ...] = CROSS_LAGS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Other-channel current/history only; target current/history never enter X."""
    channel_indices = np.array([index for index in range(values.shape[1]) if index != target_index], dtype=int)
    blocks = [values[np.asarray(indices) - lag][:, channel_indices] for lag in lags]
    return np.concatenate(blocks, axis=1), values[np.asarray(indices), target_index], channel_indices


def self_history_design(values: np.ndarray, target_index: int, indices: np.ndarray,
                        lags: tuple[int, ...] = SELF_LAGS) -> tuple[np.ndarray, np.ndarray]:
    x = np.column_stack([values[np.asarray(indices) - lag, target_index] for lag in lags])
    return x, values[np.asarray(indices), target_index]


def predictor_labels(feature_names: list[str], target_index: int,
                     lags: tuple[int, ...] = CROSS_LAGS) -> list[tuple[str, int]]:
    other = [name for index, name in enumerate(feature_names) if index != target_index]
    return [(name, lag) for lag in lags for name in other]


def chronological_discovery_split(indices: np.ndarray, fit_ratio: float = .80) -> tuple[np.ndarray, np.ndarray]:
    boundary = int(np.floor(len(indices) * fit_ratio))
    if boundary <= 0 or boundary >= len(indices):
        raise ValueError("Discovery cannot be split chronologically")
    return np.asarray(indices[:boundary]), np.asarray(indices[boundary:])


def select_alpha(values: np.ndarray, target_index: int, fit_indices: np.ndarray, selection_indices: np.ndarray,
                 kind: str, alphas: tuple[float, ...] = ALPHAS) -> tuple[float, list[dict[str, float]]]:
    if kind == "cross":
        x_fit, y_fit, _ = cross_design(values, target_index, fit_indices)
        x_select, y_select, _ = cross_design(values, target_index, selection_indices)
    elif kind == "self":
        x_fit, y_fit = self_history_design(values, target_index, fit_indices)
        x_select, y_select = self_history_design(values, target_index, selection_indices)
    else:
        raise ValueError("kind must be cross or self")
    rows = []
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver="lsqr").fit(x_fit, y_fit)
        rows.append({"alpha": float(alpha), "selection_mae": float(mean_absolute_error(y_select, model.predict(x_select)))})
    selected = sorted(rows, key=lambda row: (row["selection_mae"], row["alpha"]))[0]
    return selected["alpha"], rows


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    finite = np.isfinite(actual) & np.isfinite(prediction)
    if not finite.all() or not len(actual):
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "correlation": np.nan}
    correlation = float(np.corrcoef(actual, prediction)[0, 1]) if np.std(actual) > 0 and np.std(prediction) > 0 else np.nan
    return {"mae": float(mean_absolute_error(actual, prediction)),
            "rmse": float(mean_squared_error(actual, prediction) ** .5),
            "r2": float(r2_score(actual, prediction)), "correlation": correlation}


def fit_target_discovery(values: np.ndarray, feature_names: list[str], target_index: int,
                         discovery_indices: np.ndarray, confirmation_indices: np.ndarray,
                         cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    fit_indices, selection_indices = chronological_discovery_split(discovery_indices, cfg["prediction"]["discovery_fit_ratio"])
    alpha_cross, cross_search = select_alpha(values, target_index, fit_indices, selection_indices, "cross")
    alpha_self, self_search = select_alpha(values, target_index, fit_indices, selection_indices, "self")
    x_discovery, y_discovery, predictor_indices = cross_design(values, target_index, discovery_indices)
    x_confirmation, y_confirmation, _ = cross_design(values, target_index, confirmation_indices)
    cross = Ridge(alpha=alpha_cross, solver="lsqr").fit(x_discovery, y_discovery)
    selection_mae = next(row["selection_mae"] for row in cross_search if row["alpha"] == alpha_cross)
    cross_metrics = regression_metrics(y_confirmation, cross.predict(x_confirmation))
    sx_discovery, sy_discovery = self_history_design(values, target_index, discovery_indices)
    sx_confirmation, sy_confirmation = self_history_design(values, target_index, confirmation_indices)
    self_model = Ridge(alpha=alpha_self, solver="lsqr").fit(sx_discovery, sy_discovery)
    self_metrics = regression_metrics(sy_confirmation, self_model.predict(sx_confirmation))
    median = float(np.median(y_discovery))
    median_metrics = regression_metrics(y_confirmation, np.full(len(y_confirmation), median))
    improvement = 1.0 - cross_metrics["mae"] / max(median_metrics["mae"], 1e-12)
    scale = max(float(np.quantile(y_discovery, .75) - np.quantile(y_discovery, .25)), 1e-12)
    q = cfg["qualification"]; reasons = []
    if len(confirmation_indices) < q["minimum_confirmation_samples"]: reasons.append("insufficient_confirmation_samples")
    if not np.isfinite(list(cross_metrics.values())).all(): reasons.append("non_finite_cross_metrics")
    if cross_metrics["r2"] < q["minimum_cross_r2"]: reasons.append("cross_r2_below_minimum")
    if improvement < q["minimum_median_improvement"]: reasons.append("median_improvement_below_minimum")
    if cross_metrics["correlation"] < q["minimum_prediction_correlation"]: reasons.append("prediction_correlation_below_minimum")
    if cross_metrics["mae"] > q["maximum_confirmation_selection_mae_ratio"] * selection_mae:
        reasons.append("confirmation_mae_collapse")
    row = {"target": feature_names[target_index], "target_index": target_index,
           "selected_alpha_cross": alpha_cross, "selected_alpha_self": alpha_self,
           "predictor_count": len(predictor_indices) * len(CROSS_LAGS),
           "discovery_fit_samples": len(fit_indices), "discovery_selection_samples": len(selection_indices),
           "confirmation_samples": len(confirmation_indices), "discovery_mae": selection_mae,
           "confirmation_mae_cross": cross_metrics["mae"], "confirmation_rmse_cross": cross_metrics["rmse"],
           "confirmation_r2_cross": cross_metrics["r2"], "confirmation_prediction_correlation": cross_metrics["correlation"],
           "normalized_mae_cross": cross_metrics["mae"] / scale,
           "confirmation_mae_self": self_metrics["mae"], "confirmation_rmse_self": self_metrics["rmse"],
           "confirmation_r2_self": self_metrics["r2"], "median_baseline_mae": median_metrics["mae"],
           "median_baseline_r2": median_metrics["r2"], "cross_improvement_over_median": improvement,
           "cross_predictable": not reasons, "rejection_reason": ";".join(reasons),
           "alpha_selected_on_discovery_only": True, "confirmation_reselected_alpha": False}
    return row, {"cross_alpha_search": cross_search, "self_alpha_search": self_search}


def mechanism_status(qualified_count: int) -> str:
    return "general_conditional_candidate" if qualified_count >= 3 else (
        "narrow_conditional_diagnostic" if qualified_count else "no_conditional_mechanism")


def robust_residual_reference(residual: np.ndarray, floor: float) -> dict[str, float]:
    absolute = np.abs(np.asarray(residual, dtype=float))
    center = float(np.median(absolute))
    raw_scale = float(1.4826 * np.median(np.abs(absolute - center)))
    return {"center": center, "raw_scale": raw_scale, "scale": max(raw_scale, floor), "floor": float(floor)}


def residual_score(actual: np.ndarray, prediction: np.ndarray, reference: dict[str, float]) -> np.ndarray:
    return np.maximum((np.abs(np.asarray(actual) - np.asarray(prediction)) - reference["center"]) / reference["scale"], 0.0)


def refit_frozen_target(values: np.ndarray, feature_names: list[str], target_index: int,
                        train_indices: np.ndarray, alpha_cross: float, alpha_self: float,
                        floor: float) -> TargetModels:
    x_cross, y, predictor_indices = cross_design(values, target_index, train_indices)
    x_self, self_y = self_history_design(values, target_index, train_indices)
    cross = Ridge(alpha=alpha_cross, solver="lsqr").fit(x_cross, y)
    self_model = Ridge(alpha=alpha_self, solver="lsqr").fit(x_self, self_y)
    labels = predictor_labels(feature_names, target_index)
    return TargetModels(feature_names[target_index], target_index, predictor_indices,
                        [f"{name}@lag{lag}" for name, lag in labels], alpha_cross, alpha_self,
                        cross, self_model,
                        robust_residual_reference(y - cross.predict(x_cross), floor),
                        robust_residual_reference(self_y - self_model.predict(x_self), floor))


def score_models(values: np.ndarray, indices: np.ndarray, models: list[TargetModels]) -> tuple[np.ndarray, np.ndarray]:
    cross_scores, self_scores = [], []
    for model in models:
        x_cross, y, _ = cross_design(values, model.target_index, indices)
        x_self, sy = self_history_design(values, model.target_index, indices)
        cross_scores.append(residual_score(y, model.cross.predict(x_cross), model.residual_reference_cross))
        self_scores.append(residual_score(sy, model.self_history.predict(x_self), model.residual_reference_self))
    return np.column_stack(cross_scores), np.column_stack(self_scores)


def aggregate_target_scores(scores: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(scores, dtype=float)
    applicable = np.isfinite(values); count = applicable.sum(axis=1)
    maximum = np.full(len(values), np.nan); top3 = np.full(len(values), np.nan)
    for index in range(len(values)):
        current = values[index, applicable[index]]
        if len(current):
            maximum[index] = np.max(current)
            top3[index] = np.sort(current)[-min(3, len(current)):].mean()
    return {"max_target_score": maximum, "top3_mean_target_score": top3,
            "applicable_target_count": count, "coverage": count / max(values.shape[1], 1)}


def validation_split(validity: np.ndarray, start: int, end: int, guard_points: int = 30) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    candidates = eligible_indices(validity, start, end, tuple(sorted(set(CROSS_LAGS + SELF_LAGS))))
    usable = len(candidates) - guard_points; half = usable // 2
    threshold = candidates[:half]
    guard = candidates[half:half + guard_points]
    controlled = candidates[half + guard_points:half + guard_points + half]
    threshold_raw = set(np.concatenate([threshold - lag for lag in CROSS_LAGS + SELF_LAGS]))
    controlled_raw = set(np.concatenate([controlled - lag for lag in CROSS_LAGS + SELF_LAGS]))
    if threshold_raw & controlled_raw:
        raise ValueError("Validation threshold and controlled histories overlap")
    manifest = {"threshold_score_count": len(threshold), "controlled_score_count": len(controlled),
                "history_guard_points": len(guard), "raw_history_overlap": False,
                "threshold_uses_injected_samples": False}
    return manifest, {"threshold": threshold, "guard": guard, "controlled": controlled}


def circular_target_injection(window: np.ndarray, target_index: int, shift: int) -> np.ndarray:
    result = np.asarray(window).copy()
    result[:, target_index] = np.roll(result[:, target_index], int(shift))
    return result


def fixed_threshold_metrics(normal: np.ndarray, anomaly: np.ndarray, threshold_values: np.ndarray,
                            quantile: float = .99) -> dict[str, float]:
    normal, anomaly = np.asarray(normal), np.asarray(anomaly)
    normal, anomaly = normal[np.isfinite(normal)], anomaly[np.isfinite(anomaly)]
    threshold_values = np.asarray(threshold_values); threshold_values = threshold_values[np.isfinite(threshold_values)]
    if not len(normal) or not len(anomaly) or not len(threshold_values):
        return {name: np.nan for name in ("threshold", "auroc", "aupr", "precision", "recall", "f1", "fpr")}
    threshold = float(np.quantile(threshold_values, quantile))
    labels = np.r_[np.zeros(len(normal), int), np.ones(len(anomaly), int)]; scores = np.r_[normal, anomaly]
    alarm = scores > threshold; attack = labels == 1
    tp, fp = int((alarm & attack).sum()), int((alarm & ~attack).sum())
    fn, tn = int((~alarm & attack).sum()), int((~alarm & ~attack).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"threshold": threshold, "auroc": float(roc_auc_score(labels, scores)),
            "aupr": float(average_precision_score(labels, scores)), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12), "fpr": fp / max(fp + tn, 1)}


def select_primary(controlled: pd.DataFrame, qualified_count: int, minimum_auroc: float = .80,
                   minimum_advantage: float = .10, minimum_coverage: float = .95) -> dict[str, Any]:
    if qualified_count < 3 or controlled.empty:
        return {"primary_frozen": False, "aggregation": None, "selection_used_test": False}
    rows = controlled.set_index("score_name")
    rec_auc = float(rows.loc["rec_score", "auroc"]); eligible = []
    for name in ("cross_max", "cross_top3_mean"):
        self_name = name.replace("cross", "self"); row = rows.loc[name]
        if (row.auroc >= minimum_auroc and row.auroc - rows.loc[self_name, "auroc"] >= minimum_advantage and
                row.auroc - rec_auc >= minimum_advantage and row.coverage >= minimum_coverage and
                bool(row.marginal_statistics_preserved)):
            eligible.append((name, float(row.auroc)))
    if not eligible:
        return {"primary_frozen": False, "aggregation": None, "selection_used_test": False}
    best = max(value for _, value in eligible); close = [name for name, value in eligible if value >= best - .01]
    selected = "cross_top3_mean" if "cross_top3_mean" in close else close[0]
    return {"primary_frozen": True, "aggregation": selected, "controlled_auroc": float(rows.loc[selected, "auroc"]),
            "selection_used_test": False, "selection_rule": "controlled AUROC; within 0.01 prefer top3 mean"}
