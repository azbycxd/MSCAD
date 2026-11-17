from __future__ import annotations

import numpy as np


POINTWISE_POLICIES = (
    "pointwise_baseline",
    "consecutive_2",
    "consecutive_3",
    "vote_2_of_3",
    "vote_3_of_5",
)
HYSTERESIS_POLICY = "hysteresis_q995_q99_exit3"
ALL_POLICIES = (*POINTWISE_POLICIES, HYSTERESIS_POLICY)


def _binary(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.int8)
    if result.ndim != 1 or not np.isin(result, [0, 1]).all():
        raise ValueError("Expected a one-dimensional binary array")
    return result


def pointwise_alarm(scores: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("scores and threshold must be finite")
    return (values > threshold).astype(np.int8)


def consecutive_alarm(raw_exceed: np.ndarray, confirmation_points: int) -> np.ndarray:
    exceed = _binary(raw_exceed)
    if confirmation_points not in (2, 3):
        raise ValueError("Only pre-registered consecutive_2/consecutive_3 are allowed")
    result = np.zeros_like(exceed)
    run = 0
    for index, value in enumerate(exceed):
        run = run + 1 if value else 0
        result[index] = int(run >= confirmation_points)
    return result


def causal_vote_alarm(raw_exceed: np.ndarray, votes_required: int, window_points: int) -> np.ndarray:
    exceed = _binary(raw_exceed)
    if (votes_required, window_points) not in ((2, 3), (3, 5)):
        raise ValueError("Only pre-registered 2-of-3 and 3-of-5 votes are allowed")
    result = np.zeros_like(exceed)
    running = 0
    for index, value in enumerate(exceed):
        running += int(value)
        if index >= window_points:
            running -= int(exceed[index - window_points])
        available = min(index + 1, window_points)
        result[index] = int(available >= votes_required and running >= votes_required)
    return result


def hysteresis_alarm(scores: np.ndarray, enter_threshold: float, stay_threshold: float,
                     exit_below_points: int = 3) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite one-dimensional array")
    if not np.isfinite(enter_threshold) or not np.isfinite(stay_threshold) or enter_threshold < stay_threshold:
        raise ValueError("Hysteresis requires finite enter_threshold >= stay_threshold")
    if exit_below_points != 3:
        raise ValueError("Only the pre-registered exit3 policy is allowed")
    result = np.zeros(len(values), dtype=np.int8)
    active = False
    below_count = 0
    for index, score in enumerate(values):
        if not active:
            if score > enter_threshold:
                active = True
                below_count = 0
                result[index] = 1
            continue
        if score > stay_threshold:
            below_count = 0
            result[index] = 1
        else:
            below_count += 1
            if below_count < exit_below_points:
                result[index] = 1
            else:
                active = False
                below_count = 0
    return result


def apply_policy(scores: np.ndarray, policy_name: str, threshold: float | None = None,
                 enter_threshold: float | None = None, stay_threshold: float | None = None) -> np.ndarray:
    if policy_name == HYSTERESIS_POLICY:
        if enter_threshold is None or stay_threshold is None:
            raise ValueError("Hysteresis requires enter and stay thresholds")
        return hysteresis_alarm(scores, enter_threshold, stay_threshold)
    if threshold is None:
        raise ValueError("A pointwise threshold is required")
    exceed = pointwise_alarm(scores, threshold)
    if policy_name == "pointwise_baseline":
        return exceed
    if policy_name == "consecutive_2":
        return consecutive_alarm(exceed, 2)
    if policy_name == "consecutive_3":
        return consecutive_alarm(exceed, 3)
    if policy_name == "vote_2_of_3":
        return causal_vote_alarm(exceed, 2, 3)
    if policy_name == "vote_3_of_5":
        return causal_vote_alarm(exceed, 3, 5)
    raise ValueError(f"Unknown or non-registered policy: {policy_name}")


def binary_segments(alarm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = _binary(alarm)
    padded = np.r_[0, values, 0]
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    return starts.astype(np.int64), ends.astype(np.int64)


def maximum_continuous_alarm_length(alarm: np.ndarray) -> int:
    starts, ends = binary_segments(alarm)
    return int((ends - starts + 1).max()) if len(starts) else 0
