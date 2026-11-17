from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

SELF_LAGS = (1, 2, 4, 8, 16, 32)
SOURCE_LAGS = (0, 1, 2, 4, 8, 16, 32)


def lag_matrix(values: np.ndarray, target: int, lags=SELF_LAGS, start: int = 32) -> np.ndarray:
    return np.column_stack([values[start-lag:len(values)-lag, target] for lag in lags])


def cross_matrix(values: np.ndarray, source: int, target: int, start: int = 32) -> np.ndarray:
    return np.column_stack([lag_matrix(values, target, SELF_LAGS, start),
                            *[values[start-lag:len(values)-lag, source] for lag in SOURCE_LAGS]])


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Ridge:
    return Ridge(alpha=alpha, fit_intercept=True, solver="cholesky").fit(x, y)


def gain(self_error: np.ndarray, cross_error: np.ndarray, epsilon: float = 1e-12) -> float:
    self_mse, cross_mse = np.mean(np.square(self_error)), np.mean(np.square(cross_error))
    return float((self_mse - cross_mse) / (self_mse + epsilon))


def dependency_loss(self_error: np.ndarray, cross_error: np.ndarray) -> np.ndarray:
    a, b = np.square(self_error), np.square(cross_error)
    return (b - a) / (b + a + 1e-12)


def causal_rolling_mean(values: np.ndarray, window: int = 16) -> np.ndarray:
    x = np.asarray(values, float)
    c = np.r_[0.0, np.cumsum(x)]
    out = np.full(len(x), np.nan)
    out[window-1:] = (c[window:] - c[:-window]) / window
    return out


def robust_reference(values: np.ndarray, epsilon: float = 1e-6):
    x = np.asarray(values, float); x = x[np.isfinite(x)]
    median = float(np.median(x)); mad = 1.4826*np.median(np.abs(x-median))
    iqr = (np.quantile(x,.75)-np.quantile(x,.25))/1.349; std=float(np.std(x))
    if mad > epsilon: return median, float(mad), "mad", float(iqr), std
    if iqr > epsilon: return median, float(iqr), "iqr", float(iqr), std
    if std > epsilon: return median, std, "std", float(iqr), std
    return median, epsilon, "epsilon", float(iqr), std


def aggregate_edges(scores: np.ndarray, method: str) -> np.ndarray:
    if method == "dynamic_edge_max": return np.max(scores, axis=-1)
    if method == "dynamic_edge_top2_mean":
        k=min(2,scores.shape[-1]); return np.partition(scores,scores.shape[-1]-k,axis=-1)[...,-k:].mean(axis=-1)
    raise ValueError(method)
