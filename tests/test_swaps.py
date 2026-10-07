import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leadcheck.swaps import APPROXIMATE, CLASSES, LIMB, PRECORDIAL, apply, canonical  # noqa: E402


def ecg(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(12, 500)).cumsum(axis=1) * 0.01
    return canonical(x)


def einthoven_holds(y):
    i, ii, iii, avr, avl, avf = y[:6]
    return (np.allclose(iii, ii - i) and np.allclose(avr, -(i + ii) / 2)
            and np.allclose(avl, i - ii / 2) and np.allclose(avf, ii - i / 2))


def test_every_class_keeps_lead_identities():
    x = ecg()
    for kind in CLASSES:
        assert einthoven_holds(apply(x, kind)), kind


def test_pairwise_swaps_are_involutions():
    x = ecg(1)
    for kind in ["LA/RA", "LA/LL", "RA/LL", *PRECORDIAL]:
        assert np.allclose(apply(apply(x, kind), kind), x), kind


def test_rotations_have_order_three():
    x = ecg(2)
    for kind in ["rotation RA>LA>LL", "rotation RA>LL>LA"]:
        assert np.allclose(apply(apply(apply(x, kind), kind), kind), x), kind


def test_limb_permutations_leave_precordial_leads_unchanged():
    x = ecg(3)
    for kind in LIMB:
        if kind not in APPROXIMATE:
            assert np.allclose(apply(x, kind)[6:], x[6:]), kind


def test_textbook_signatures():
    x = ecg(4)
    i, ii, iii, avr, avl, avf = x[:6]
    y = apply(x, "LA/RA")
    assert np.allclose(y[0], -i) and np.allclose(y[1], iii) and np.allclose(y[3], avl)
    y = apply(x, "RA/RL")
    assert np.abs(y[1]).max() < 0.2 * np.abs(ii).max() + 0.05   # lead II almost flat
