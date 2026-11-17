import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd


@dataclass
class Event:
    event_id: int
    start: int
    end: int
    kind: str
    variables: List[str]
    params: Dict[str, Any]


def _choose_windows(rng: np.random.Generator, n: int, window: int, count: int, min_gap: int) -> List[int]:
    if n <= window:
        raise ValueError("Series length must be greater than window length.")
    candidates = list(range(0, n - window + 1))
    rng.shuffle(candidates)
    chosen: List[int] = []
    for s in candidates:
        if all(abs(s - c) >= min_gap for c in chosen):
            chosen.append(s)
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        raise ValueError("Unable to place requested events with given min_gap.")
    chosen.sort()
    return chosen


def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if np.std(x) < 1e-8:
        return 0.0, float(np.mean(y))
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def _apply_ratio_mismatch(
    X: np.ndarray,
    start: int,
    end: int,
    i: int,
    j: int,
    rng: np.random.Generator,
    strength: float,
    blend: float,
) -> Dict[str, Any]:
    xi = X[start:end, i]
    xj = X[start:end, j]
    a, b = _linear_fit(xi, xj)
    delta = rng.uniform(0.2, strength) * (1.0 if rng.random() > 0.5 else -1.0)
    a_prime = a * (1.0 + delta) if abs(a) > 1e-8 else delta
    xj_new = a_prime * xi + b
    X[start:end, j] = (1.0 - blend) * xj + blend * xj_new
    return {"a": a, "b": b, "a_prime": a_prime, "delta": delta, "blend": blend}


def _apply_decorrelation(
    X: np.ndarray,
    start: int,
    end: int,
    vars_idx: List[int],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    for idx in vars_idx:
        window = X[start:end, idx].copy()
        rng.shuffle(window)
        X[start:end, idx] = window
    return {"shuffle": True}


def _apply_lag_shift(
    X: np.ndarray,
    start: int,
    end: int,
    idx: int,
    rng: np.random.Generator,
    max_lag: int,
) -> Dict[str, Any]:
    length = end - start
    lag = int(rng.integers(2, max(3, min(max_lag, length // 2))))
    window = X[start:end, idx].copy()
    X[start:end, idx] = np.roll(window, lag)
    return {"lag": lag}


def _random_rotation_matrix(rng: np.random.Generator, k: int) -> np.ndarray:
    A = rng.normal(size=(k, k))
    Q, _ = np.linalg.qr(A)
    return Q


def _apply_subgroup_rotation(
    X: np.ndarray,
    start: int,
    end: int,
    vars_idx: List[int],
    rng: np.random.Generator,
    alpha: float,
) -> Dict[str, Any]:
    window = X[start:end][:, vars_idx]
    mu = window.mean(axis=0)
    sigma = window.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    z = (window - mu) / sigma
    k = len(vars_idx)
    if k == 2:
        theta = rng.uniform(math.radians(10), math.radians(35))
        if rng.random() > 0.5:
            theta = -theta
        rot = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
            dtype=float,
        )
    else:
        rot = _random_rotation_matrix(rng, k)
    mix = (1.0 - alpha) * np.eye(k) + alpha * rot
    z_new = z @ mix
    # Match marginal mean/std to preserve univariate distribution roughly.
    z_new = (z_new - z_new.mean(axis=0)) / (z_new.std(axis=0) + 1e-6)
    window_new = z_new * sigma + mu
    X[start:end][:, vars_idx] = window_new
    return {"alpha": alpha, "k": k}


def synthesize(
    df: pd.DataFrame,
    numeric_cols: List[str],
    window: int,
    count: int,
    min_gap: int,
    seed: int,
    ratio_strength: float,
    ratio_blend: float,
    max_lag: int,
    rotation_alpha: float,
    types: List[str],
) -> Tuple[pd.DataFrame, List[Event]]:
    rng = np.random.default_rng(seed)
    X = df[numeric_cols].to_numpy(dtype=np.float32, copy=True)
    n = X.shape[0]
    starts = _choose_windows(rng, n, window, count, min_gap)
    events: List[Event] = []
    anomaly = np.zeros(n, dtype=np.int8)
    for event_id, start in enumerate(starts):
        end = start + window
        kind = rng.choice(types)
        if kind == "ratio_mismatch":
            i, j = rng.choice(len(numeric_cols), size=2, replace=False)
            params = _apply_ratio_mismatch(X, start, end, i, j, rng, ratio_strength, ratio_blend)
            variables = [numeric_cols[i], numeric_cols[j]]
        elif kind == "decorrelation":
            vars_idx = rng.choice(len(numeric_cols), size=3, replace=False).tolist()
            params = _apply_decorrelation(X, start, end, vars_idx, rng)
            variables = [numeric_cols[i] for i in vars_idx]
        elif kind == "lag_shift":
            idx = int(rng.integers(0, len(numeric_cols)))
            params = _apply_lag_shift(X, start, end, idx, rng, max_lag)
            variables = [numeric_cols[idx]]
        elif kind == "subgroup_rotation":
            vars_idx = rng.choice(len(numeric_cols), size=3, replace=False).tolist()
            params = _apply_subgroup_rotation(X, start, end, vars_idx, rng, rotation_alpha)
            variables = [numeric_cols[i] for i in vars_idx]
        else:
            raise ValueError(f"Unknown anomaly type: {kind}")

        anomaly[start:end] = 1
        events.append(
            Event(
                event_id=event_id,
                start=start,
                end=end,
                kind=kind,
                variables=variables,
                params=params,
            )
        )

    out_df = df.copy()
    out_df[numeric_cols] = X
    out_df["SyntheticAnomaly"] = anomaly
    return out_df, events


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize correlation anomalies on multivariate time series.")
    parser.add_argument("--input", required=True, help="Input CSV (normal data).")
    parser.add_argument("--output", required=True, help="Output CSV with synthetic anomalies.")
    parser.add_argument("--meta", required=True, help="Output JSON metadata for injected events.")
    parser.add_argument("--window", type=int, default=100, help="Window length for each anomaly.")
    parser.add_argument("--count", type=int, default=50, help="Number of anomaly windows to inject.")
    parser.add_argument("--min-gap", type=int, default=200, help="Minimum gap between anomaly windows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--ratio-strength", type=float, default=0.6, help="Max relative change for ratio mismatch.")
    parser.add_argument("--ratio-blend", type=float, default=0.7, help="Blend ratio for ratio mismatch.")
    parser.add_argument("--max-lag", type=int, default=12, help="Max lag for lag_shift.")
    parser.add_argument("--rotation-alpha", type=float, default=0.6, help="Mixing strength for subgroup rotation.")
    parser.add_argument(
        "--types",
        default="ratio_mismatch,decorrelation,lag_shift,subgroup_rotation",
        help="Comma-separated anomaly types.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    exclude = {"Timestamp", "Normal/Attack", "SyntheticAnomaly"}
    numeric_cols = [c for c in df.columns if c not in exclude]
    if not numeric_cols:
        raise ValueError("No numeric columns found after excluding Timestamp/Normal/Attack.")

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    out_df, events = synthesize(
        df=df,
        numeric_cols=numeric_cols,
        window=args.window,
        count=args.count,
        min_gap=args.min_gap,
        seed=args.seed,
        ratio_strength=args.ratio_strength,
        ratio_blend=args.ratio_blend,
        max_lag=args.max_lag,
        rotation_alpha=args.rotation_alpha,
        types=types,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.meta), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in events], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
