"""Loading PTB-XL arrays and turning a recording into model input."""
from __future__ import annotations

import json
import pathlib

import numpy as np

from .swaps import CLASSES, apply, canonical


def load(data_dir) -> tuple[np.ndarray, list[dict], np.ndarray]:
    d = pathlib.Path(data_dir)
    X = np.load(d / "X100.npy").astype(np.float32)
    meta = json.load(open(d / "meta.json"))["records"]
    X = np.stack([canonical(x) for x in X])
    return X, meta, np.array([m["fold"] for m in meta])


def finish(x: np.ndarray) -> np.ndarray:
    """Remove each lead's baseline offset and clip extreme values (mV)."""
    x = x - np.median(x, axis=-1, keepdims=True)
    return np.clip(x, -5.0, 5.0).astype(np.float32)


def variants(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """The recording under every interchange class, in CLASSES order."""
    rng = np.random.default_rng(seed)
    return np.stack([finish(apply(x, k, rng)) for k in CLASSES])


def features(x: np.ndarray) -> np.ndarray:
    """Inter-lead correlations plus per-lead amplitude summaries, for the baseline model."""
    c = np.nan_to_num(np.corrcoef(x))
    iu = np.triu_indices(12, 1)
    return np.concatenate([c[iu], x.max(1), x.min(1), np.abs(x).mean(1)]).astype(np.float32)
