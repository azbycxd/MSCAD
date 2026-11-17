from __future__ import annotations

import numpy as np


def source_temporal_shift(window: np.ndarray, source: int, shift: int = 16):
    out=window.copy(); out[:,source]=np.roll(out[:,source],shift); return out


def source_block_permutation(window: np.ndarray, source: int):
    out=window.copy(); blocks=np.split(out[:,source],4); out[:,source]=np.concatenate([blocks[i] for i in (2,0,3,1)]); return out


def target_endpoint_spike(window: np.ndarray, target: int, scale: float):
    out=window.copy(); out[-1,target]+=3*scale; return out


def target_short_level_shift(window: np.ndarray, target: int, scale: float):
    out=window.copy(); out[-8:,target]+=2*scale; return out


def replace_channels(window: np.ndarray, replacement: np.ndarray, channels: list[int]):
    out=window.copy(); out[:,channels]=replacement[:,channels]; return out


def same_multiset(a,b): return bool(np.array_equal(np.sort(a),np.sort(b)))
