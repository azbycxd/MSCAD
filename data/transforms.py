from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    raw_std_: np.ndarray | None = None
    effective_scale_: np.ndarray | None = None
    eps: float = 1e-6

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.raw_std_ = x.std(axis=0, keepdims=True)
        self.effective_scale_ = np.where(self.raw_std_ < self.eps, 1.0, self.raw_std_)
        self.std_ = self.effective_scale_
        return self

    def fit_many(self, arrays: list[np.ndarray]) -> "StandardScaler":
        valid = [np.asarray(x) for x in arrays if len(x)]
        if not valid:
            raise ValueError("StandardScaler.fit_many requires at least one non-empty array.")
        width = valid[0].shape[1]
        if any(x.ndim != 2 or x.shape[1] != width for x in valid):
            raise ValueError("All arrays passed to StandardScaler.fit_many must be 2D with equal width.")
        count = sum(len(x) for x in valid)
        total = sum((x.sum(axis=0, dtype=np.float64) for x in valid), start=np.zeros(width, dtype=np.float64))
        mean64 = total / count
        squared = sum((((x.astype(np.float64) - mean64) ** 2).sum(axis=0) for x in valid), start=np.zeros(width, dtype=np.float64))
        raw64 = np.sqrt(squared / count)
        dtype = valid[0].dtype if np.issubdtype(valid[0].dtype, np.floating) else np.float64
        self.mean_ = mean64.astype(dtype, copy=False)[None, :]
        self.raw_std_ = raw64.astype(dtype, copy=False)[None, :]
        self.effective_scale_ = np.where(self.raw_std_ < self.eps, 1.0, self.raw_std_).astype(dtype, copy=False)
        self.std_ = self.effective_scale_
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.effective_scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform.")
        return (x - self.mean_) / self.effective_scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
