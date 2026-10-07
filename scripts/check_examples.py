"""Run every bundled example under every interchange through the ONNX model (demo sanity check)."""
import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leadcheck.data import variants  # noqa: E402
from leadcheck.swaps import CLASSES, canonical  # noqa: E402

opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
sess = ort.InferenceSession(str(ROOT / "model/leadcheck.onnx"), opts, providers=["CPUExecutionProvider"])
thr = json.loads((ROOT / "model/threshold.json").read_text())["threshold"]
d = np.load(ROOT / "data/examples.npz")
false_alarm, missed, wrong_type = 0, 0, 0
for x, name in zip(d["X"].astype(np.float32), d["names"]):
    V = variants(canonical(x))
    logits = sess.run(None, {"ecg": V})[0]
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    alarm = 1 - p[:, 0] >= thr
    false_alarm += alarm[0]
    missed += (~alarm[1:]).sum()
    wrong_type += (alarm[1:] & (p[1:, 1:].argmax(1) + 1 != np.arange(1, len(CLASSES)))).sum()
    if alarm[0] or (~alarm[1:]).any():
        print(name, "| false alarm" if alarm[0] else "", "| missed:", [CLASSES[k + 1] for k in np.where(~alarm[1:])[0]])
print(f"{len(d['X'])} examples: false alarms {false_alarm}, missed {missed}/{len(d['X']) * 12}, wrong type {wrong_type}")
