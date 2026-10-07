"""Proportions with Wilson intervals and a rank-based AUROC (no scikit-learn needed)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

MIN_N = 30


def wilson(x: int, n: int, z: float = 1.96):
    if n == 0:
        return [None, None]
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def prop(mask) -> dict:
    """x/n, rate and Wilson interval of a boolean array; rate and interval are None below MIN_N."""
    mask = np.asarray(mask, bool)
    x, n = int(mask.sum()), int(mask.size)
    if n < MIN_N:
        return {"x": x, "n": n, "rate": None, "ci": [None, None]}
    return {"x": x, "n": n, "rate": x / n, "ci": wilson(x, n)}


def auroc(y, s):
    """Rank-based AUROC with average ranks for ties; None when only one class is present."""
    y = np.asarray(y)
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def macro_auroc(Y, P):
    vals = [auroc(Y[:, k], P[:, k]) for k in range(Y.shape[1])]
    return None if any(v is None for v in vals) else float(np.mean(vals))
