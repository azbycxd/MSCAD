from __future__ import annotations

import numpy as np

try:
    from scipy.stats import genpareto
except Exception:  # pragma: no cover
    genpareto = None


class POTThreshold:
    def __init__(self, q: float = 0.98, target_prob: float = 0.001):
        self.q = q
        self.target_prob = target_prob
        self.u: float | None = None
        self.tau: float | None = None

    def fit(self, scores: np.ndarray) -> "POTThreshold":
        s = np.asarray(scores, dtype=float)
        self.u = float(np.quantile(s, self.q))
        excess = s[s > self.u] - self.u
        if len(excess) < 10:
            self.tau = float(np.quantile(s, min(0.999, max(self.q, 0.99))))
            return self
        if genpareto is None:
            # Fallback when scipy is unavailable.
            self.tau = float(np.quantile(s, min(0.999, max(self.q, 0.995))))
            return self
        c, loc, scale = genpareto.fit(excess, floc=0.0)
        p_excess = max(1e-8, len(excess) / len(s))
        p_tail_target = self.target_prob / p_excess
        p_tail_target = float(np.clip(p_tail_target, 1e-8, 1 - 1e-8))
        y = genpareto.ppf(1.0 - p_tail_target, c, loc=loc, scale=scale)
        self.tau = float(self.u + y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.tau is None:
            raise RuntimeError("POTThreshold must be fitted before predict.")
        return (np.asarray(scores) > self.tau).astype(int)
