"""Apply the pre-registered LA/RA criteria to flagged recordings. usage: check_flags.py DATA_DIR SCAN.json THRESHOLD"""
import json
import pathlib
import sys

import numpy as np
from scipy.signal import find_peaks

data, scan, thr = pathlib.Path(sys.argv[1]), json.load(open(sys.argv[2])), float(sys.argv[3])
X = np.load(data / "X100.npy").astype(np.float32)
meta = json.load(open(data / "meta.json"))["records"]
row = {m["ecg_id"]: i for i, m in enumerate(meta)}


def template(x):
    x = x - np.median(x, axis=1, keepdims=True)
    energy = np.abs(np.diff(x, axis=1)).sum(0)
    peaks, _ = find_peaks(energy, distance=35, height=np.percentile(energy, 98) * 0.5)
    beats = [x[:, p - 30:p + 30] for p in peaks if 30 <= p < x.shape[1] - 30]
    return np.median(beats, axis=0) if len(beats) >= 3 else None   # 60 samples, R near index 30


out = []
for r in scan:
    if r["score"] < thr or r["type"] != "LA/RA":
        continue
    t = template(X[row[r["ecg_id"]]])
    if t is None:
        continue
    base = t[:, 0:3].mean(1)
    qrs_I = t[0, 25:36].sum() - 11 * base[0]
    p_I = t[0, 8:20].mean() - base[0]
    r_amp = t[6:10, 25:36].max(1) - base[6:10]               # V1..V4
    normal_progression = bool(r_amp[3] > r_amp[0] and r_amp[2] > r_amp[0])
    meets = bool(qrs_I < 0 and p_I < 0 and normal_progression)
    out.append({**r, "qrs_I": round(float(qrs_I), 3), "p_I": round(float(p_I), 3),
                "normal_R_progression": normal_progression, "meets_criteria": meets,
                "report": meta[row[r["ecg_id"]]]["report"][:90]})
for o in out:
    print(o["ecg_id"], f"score {o['score']:.4f}", "MEETS" if o["meets_criteria"] else "-", f"QRS_I {o['qrs_I']:+.2f} P_I {o['p_I']:+.3f} Rprog {o['normal_R_progression']}", o["report"][:60])
print(len(out), "flagged as LA/RA;", sum(o["meets_criteria"] for o in out), "meet all criteria")
json.dump(out, open(pathlib.Path(sys.argv[2]).with_name("lara_flags.json"), "w"), indent=1)
