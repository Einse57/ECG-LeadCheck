"""Check electrode placement with LeadCheck, then delineate with OpenECG.

usage: python examples/leadcheck_then_openecg.py RECORD [--backend onnx|openvino|torch] [--device CPU]
RECORD is a WFDB 12-lead record path without extension (e.g. PTB-XL records500/00000/00009_hr).

Format conversions:
- LeadCheck expects 12 leads in the order I, II, III, aVR, aVL, aVF, V1-V6, 10 s at 100 Hz, in mV.
  The record is resampled to 100 Hz (scipy.signal.resample_poly) and its first 10 s are used.
- OpenECG's report() takes one lead at its native rate (fs=500 here); lead II is passed unchanged.
If LeadCheck flags a limb or right-leg interchange, the OpenECG result for lead II is marked as unreliable,
since those interchanges change lead II. A chest (V1-V6) interchange leaves lead II unchanged, so lead-II
delineation is kept, but the 12-lead record should still be re-recorded or corrected.
Needs: wfdb, scipy, openecg[report].
"""
import argparse
import json
import pathlib
import sys
from math import gcd

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leadcheck import backends  # noqa: E402
from leadcheck.data import finish  # noqa: E402
from leadcheck.swaps import CLASSES, LEADS, canonical  # noqa: E402


def read_record(path):
    import wfdb
    rec = wfdb.rdrecord(path)
    names = [n.upper() for n in rec.sig_name]
    order = [names.index(n.upper()) for n in LEADS]
    return rec.p_signal[:, order].T.astype(np.float32), int(rec.fs)


def to_leadcheck(sig, fs):
    from scipy.signal import resample_poly
    g = gcd(100, fs)
    x = resample_poly(sig, 100 // g, fs // g, axis=-1)[:, :backends.SAMPLES]
    if x.shape[-1] < backends.SAMPLES:
        x = np.pad(x, ((0, 0), (0, backends.SAMPLES - x.shape[-1])))
    return finish(canonical(x.astype(np.float32)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--backend", default="onnx", choices=["onnx", "openvino", "torch"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt")
    a = ap.parse_args()
    sig, fs = read_record(a.record)
    thr = json.loads((ROOT / "model/threshold.json").read_text())["threshold"]
    p = backends.softmax(backends.load(a.backend, a.device, ckpt=a.ckpt)(to_leadcheck(sig, fs)[None]))[0]
    alarm = bool(1 - p[0] >= thr)
    k = int(p[1:].argmax()) + 1
    lead_check = {"alarm": alarm, "score": round(float(1 - p[0]), 4), "threshold": round(thr, 4),
                  "most_likely_interchange": CLASSES[k] if alarm else "none"}
    import openecg
    rep = openecg.report(sig[LEADS.index("II")], fs=fs)
    print(json.dumps({"record": a.record, "fs": fs, "lead_check": lead_check,
                      "openecg_lead_ii_reliable": not alarm or CLASSES[k].startswith("V"),
                      "openecg": json.loads(rep.to_json())["heart_rate"],
                      "openecg_summary": rep.summary}, indent=1))


if __name__ == "__main__":
    main()
