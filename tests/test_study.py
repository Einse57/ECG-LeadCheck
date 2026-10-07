import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leadcheck.data import variants  # noqa: E402
from leadcheck.stats import auroc, prop, wilson  # noqa: E402
from leadcheck.swaps import APPROXIMATE, CLASSES, INVERSE, apply, canonical  # noqa: E402


def ecg(seed=0):
    rng = np.random.default_rng(seed)
    return canonical(rng.normal(size=(12, 400)).cumsum(axis=1).astype(np.float32) * 0.01)


def test_variants_are_deterministic():
    x = ecg()
    assert np.array_equal(variants(x, seed=3), variants(x, seed=3))


def test_inverse_undoes_every_exact_interchange():
    x = ecg(1)
    for k in CLASSES[1:]:
        if k not in APPROXIMATE:
            assert np.allclose(apply(apply(x, k), INVERSE.get(k, k)), x, atol=1e-6), k


def test_auroc_matches_definition():
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.8])
    pairs = [(a, b) for a in s[y == 1] for b in s[y == 0]]
    expected = np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs])
    assert abs(auroc(y, s) - expected) < 1e-12
    assert auroc(np.ones(4), np.arange(4)) is None


def test_wilson_and_small_counts():
    lo, hi = wilson(0, 850)
    assert lo == 0.0 and 0.003 < hi < 0.006
    assert prop(np.ones(10, bool))["rate"] is None
    assert prop(np.r_[np.ones(10), np.zeros(40)].astype(bool))["x"] == 10
