"""Electrode interchanges applied to a 12-lead ECG.

Leads are ordered I, II, III, aVR, aVL, aVF, V1-V6. Limb-cable interchanges are computed from the
electrode potentials (RA taken as zero, LA = I, LL = II), so they are exact for the three limb
electrodes. Interchanges with the right-leg (ground) electrode need its potential, which a 12-lead
ECG does not record; it is approximated as the left-leg potential plus a small perturbation.
"""
from __future__ import annotations

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# cable -> site it was attached to
LIMB = {
    "LA/RA": {"RA": "LA", "LA": "RA"},
    "LA/LL": {"LA": "LL", "LL": "LA"},
    "RA/LL": {"RA": "LL", "LL": "RA"},
    "rotation RA>LA>LL": {"RA": "LA", "LA": "LL", "LL": "RA"},
    "rotation RA>LL>LA": {"RA": "LL", "LL": "LA", "LA": "RA"},
    "RA/RL": {"RA": "RL", "RL": "RA"},
    "LA/RL": {"LA": "RL", "RL": "LA"},
}
PRECORDIAL = {f"V{i}/V{i + 1}": (5 + i, 6 + i) for i in range(1, 6)}
CLASSES = ["none"] + list(LIMB) + list(PRECORDIAL)
APPROXIMATE = {"RA/RL", "LA/RL"}
# undoing an interchange: pairwise swaps are their own inverse, the two rotations invert each other
INVERSE = {"rotation RA>LA>LL": "rotation RA>LL>LA", "rotation RA>LL>LA": "rotation RA>LA>LL"}


def limb_leads(ra: np.ndarray, la: np.ndarray, ll: np.ndarray) -> np.ndarray:
    """The six limb leads from three electrode potentials."""
    return np.stack([la - ra, ll - ra, ll - la, ra - (la + ll) / 2, la - (ra + ll) / 2, ll - (ra + la) / 2])


def canonical(x: np.ndarray) -> np.ndarray:
    """Recompute III, aVR, aVL and aVF from I and II, so every record obeys Einthoven's law exactly."""
    y = x.copy()
    y[:6] = limb_leads(np.zeros_like(x[0]), x[0], x[1])
    return y


def apply(x: np.ndarray, kind: str, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return the ECG that would have been recorded with the given interchange."""
    if kind == "none":
        return x.copy()
    if kind in PRECORDIAL:
        a, b = PRECORDIAL[kind]
        y = x.copy()
        y[[a, b]] = y[[b, a]]
        return y
    rng = rng or np.random.default_rng(0)
    site = {"RA": np.zeros_like(x[0]), "LA": x[0], "LL": x[1]}
    beta = rng.uniform(0.0, 0.1)
    site["RL"] = (1 - beta) * site["LL"] + beta * (site["RA"] + site["LA"]) / 2 + rng.normal(0, 0.01, x.shape[1])
    cable = {c: site[LIMB[kind].get(c, c)] for c in ("RA", "LA", "LL")}
    y = x.copy()
    y[:6] = limb_leads(cable["RA"], cable["LA"], cable["LL"])
    wct_true = (site["RA"] + site["LA"] + site["LL"]) / 3
    wct_seen = (cable["RA"] + cable["LA"] + cable["LL"]) / 3
    y[6:] = x[6:] + wct_true - wct_seen
    return y
