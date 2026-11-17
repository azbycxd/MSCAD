from __future__ import annotations

import numpy as np
import torch


CAUSAL_VARIANTS = ("window_mean_current", "endpoint_mean", "recent8_mean", "recent16_mean")
ALL_VARIANTS = (*CAUSAL_VARIANTS, "offline_overlap_point_mean")
CHANNEL_SCORE_VARIANTS = (
    "endpoint_raw_mean",
    "endpoint_raw_top5",
    "endpoint_calibrated_mean",
    "endpoint_calibrated_top3",
    "endpoint_calibrated_top5",
    "endpoint_calibrated_top10",
    "endpoint_calibrated_max",
)


def temporal_reconstruction_scores(x: torch.Tensor, x_hat: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if x.shape != x_hat.shape or x.ndim != 3:
        raise ValueError("Reconstruction inputs must have matching [batch,time,channels] shapes")
    point_error = (x_hat - x).abs().mean(dim=2)
    scores = {
        # Preserve the historical joint reduction order exactly.
        "window_mean_current": (x_hat - x).abs().mean(dim=(1, 2)),
        "endpoint_mean": point_error[:, -1],
        "recent8_mean": point_error[:, -8:].mean(dim=1),
        "recent16_mean": point_error[:, -16:].mean(dim=1),
    }
    return scores, point_error


def add_overlap_errors(error_sum: np.ndarray, coverage_count: np.ndarray, window_starts: np.ndarray,
                       point_errors: np.ndarray) -> None:
    starts = np.asarray(window_starts, dtype=np.int64)
    errors = np.asarray(point_errors, dtype=np.float64)
    if errors.ndim != 2 or len(starts) != len(errors):
        raise ValueError("point_errors must be [windows,time] and align with window_starts")
    for offset in range(errors.shape[1]):
        rows = starts + offset
        if len(rows) and (rows.min() < 0 or rows.max() >= len(error_sum)):
            raise IndexError("Overlap accumulation would cross the sequence boundary")
        error_sum[rows] += errors[:, offset]
        coverage_count[rows] += 1


def finalize_overlap_scores(error_sum: np.ndarray, coverage_count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covered = np.asarray(coverage_count) > 0
    scores = np.full(len(error_sum), np.nan, dtype=np.float64)
    np.divide(error_sum, coverage_count, out=scores, where=covered)
    return scores, covered


def endpoint_channel_errors(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    if x.shape != x_hat.shape or x.ndim != 3:
        raise ValueError("Reconstruction inputs must have matching [batch,time,channels] shapes")
    return (x_hat[:, -1, :] - x[:, -1, :]).abs()


def fit_channel_calibration(train_errors: np.ndarray, minimum_scale: float = 1e-8) -> dict[str, np.ndarray]:
    errors = np.asarray(train_errors)
    if errors.ndim != 2 or not len(errors) or not np.isfinite(errors).all():
        raise ValueError("train_errors must be a finite non-empty [samples,channels] array")
    median = np.median(errors, axis=0)
    mean = errors.mean(axis=0)
    std = errors.std(axis=0)
    mad = np.median(np.abs(errors - median), axis=0)
    robust = 1.4826 * mad
    q25, q75 = np.quantile(errors, [0.25, 0.75], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    effective = robust.copy()
    method = np.full(errors.shape[1], "mad", dtype="U8")
    use_iqr = effective < minimum_scale
    effective[use_iqr] = iqr_scale[use_iqr]
    method[use_iqr] = "iqr"
    use_std = effective < minimum_scale
    effective[use_std] = std[use_std]
    method[use_std] = "std"
    use_unit = effective < minimum_scale
    effective[use_unit] = 1.0
    method[use_unit] = "unit"
    return {
        "residual_median": median,
        "residual_mean": mean,
        "residual_std": std,
        "residual_mad": mad,
        "robust_scale": robust,
        "residual_q90": np.quantile(errors, 0.90, axis=0),
        "residual_q95": np.quantile(errors, 0.95, axis=0),
        "residual_q99": np.quantile(errors, 0.99, axis=0),
        "zero_mad": mad == 0,
        "fallback_method": method,
        "effective_scale": effective,
    }


def calibrate_endpoint_errors(errors: np.ndarray, calibration: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(errors, dtype=np.float64)
    calibrated = np.maximum(0.0, values - calibration["residual_median"]) / calibration["effective_scale"]
    if not np.isfinite(calibrated).all():
        raise ValueError("Channel calibration produced non-finite values")
    return calibrated


def _topk(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2 or not 1 <= k <= values.shape[1]:
        raise ValueError("top-k input must be [samples,channels] with a valid k")
    indices = np.argpartition(values, -k, axis=1)[:, -k:]
    selected = np.take_along_axis(values, indices, axis=1)
    order = np.argsort(selected, axis=1)[:, ::-1]
    indices = np.take_along_axis(indices, order, axis=1)
    selected = np.take_along_axis(selected, order, axis=1)
    return selected.mean(axis=1), indices, selected


def aggregate_endpoint_channel_scores(raw_errors: np.ndarray, calibrated_errors: np.ndarray) -> tuple[
        dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw = np.asarray(raw_errors, dtype=np.float64)
    calibrated = np.asarray(calibrated_errors, dtype=np.float64)
    if raw.shape != calibrated.shape or raw.ndim != 2:
        raise ValueError("raw and calibrated errors must have matching [samples,channels] shapes")
    raw5, raw5_idx, raw5_values = _topk(raw, 5)
    cal3, cal3_idx, cal3_values = _topk(calibrated, 3)
    cal5, cal5_idx, cal5_values = _topk(calibrated, 5)
    cal10, cal10_idx, cal10_values = _topk(calibrated, 10)
    scores = {
        "endpoint_raw_mean": raw.mean(axis=1),
        "endpoint_raw_top5": raw5,
        "endpoint_calibrated_mean": calibrated.mean(axis=1),
        "endpoint_calibrated_top3": cal3,
        "endpoint_calibrated_top5": cal5,
        "endpoint_calibrated_top10": cal10,
        "endpoint_calibrated_max": cal3_values[:, 0],
    }
    details = {
        "raw_top5_indices": raw5_idx, "raw_top5_values": raw5_values,
        "calibrated_top3_indices": cal3_idx, "calibrated_top3_values": cal3_values,
        "calibrated_top5_indices": cal5_idx, "calibrated_top5_values": cal5_values,
        "calibrated_top10_indices": cal10_idx, "calibrated_top10_values": cal10_values,
    }
    return ({name: value.astype(np.float32) for name, value in scores.items()}, details)
