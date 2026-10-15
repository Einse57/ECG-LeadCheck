"""Compare backends against the ONNX Runtime reference on PTB-XL records under every interchange.

usage: agreement.py DATA_DIR --folds 10 [--ckpt leadcheck_main.pt] [--ov-device CPU] [--out agreement.json]
DATA_DIR is the output of scripts/build_dataset.py. Reports, per backend: max |logit difference|,
agreement of the predicted class per input, and agreement of the alarm decision at model/threshold.json.
"""
import argparse
import json
import pathlib
import platform
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leadcheck import backends  # noqa: E402
from leadcheck.data import load, variants  # noqa: E402


def compare_chunks(chunks, thr):
    ref = np.concatenate([c[0] for c in chunks])
    got = np.concatenate([c[1] for c in chunks])
    return compare(ref, got, thr)


def compare(ref, got, thr):
    pr, pg = backends.softmax(ref), backends.softmax(got)
    return {"n_inputs": int(len(ref)),
            "max_abs_logit_diff": float(np.abs(ref - got).max()),
            "class_agreement": float((ref.argmax(1) == got.argmax(1)).mean()),
            "alarm_agreement": float(((1 - pr[:, 0] >= thr) == (1 - pg[:, 0] >= thr)).mean()),
            "max_abs_prob_diff": float(np.abs(pr - pg).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--folds", default="10")
    ap.add_argument("--limit", type=int, default=0, help="first N records only (0 = all)")
    ap.add_argument("--ckpt")
    ap.add_argument("--ov-device", default="CPU")
    ap.add_argument("--torch-device", default="auto")
    ap.add_argument("--out")
    ap.add_argument("--save-probs", help="npz with softmax outputs per backend (records, variants, classes), "
                                         "readable by scripts/evaluate.py")
    a = ap.parse_args()
    thr = json.loads((ROOT / "model/threshold.json").read_text())["threshold"]
    X, meta, folds = load(a.data)
    idx = np.where(np.isin(folds, [int(f) for f in a.folds.split(",")]))[0]
    if a.limit:
        idx = idx[:a.limit]
    ref_f = backends.load("onnx")
    runs = {}
    if a.ckpt:
        tb = backends.load("torch", a.torch_device, ckpt=a.ckpt)
        runs[f"torch-{tb.device}"] = tb
    try:
        runs[f"openvino-{a.ov_device}"] = backends.load("openvino", a.ov_device)
        if a.ov_device.upper() == "CPU":
            runs["openvino-CPU-bf16"] = backends.load("openvino", "CPU", precision="bf16")
    except ImportError:
        print("openvino not installed; skipping")
    res = {"records": int(len(idx)), "folds": a.folds, "inputs": int(len(idx)) * 13,
           "host": platform.processor() or platform.machine(),
           "reference": "onnxruntime CPU, model/leadcheck.onnx", "backends": {}}
    chunks = {name: [] for name in runs}
    for s in range(0, len(idx), 40):  # stream in small chunks to bound memory
        V = np.concatenate([variants(X[i], seed=int(i)) for i in idx[s:s + 40]]).astype(np.float32)
        ref = ref_f(V)
        for name, f in runs.items():
            chunks[name].append((ref, f(V)))
    for name in runs:
        res["backends"][name] = compare_chunks(chunks[name], thr)
        print(name, json.dumps(res["backends"][name]), flush=True)
    if a.save_probs:
        n = len(idx)
        probs = {"fold": folds[idx], "ecg_id": np.array([meta[i]["ecg_id"] for i in idx]),
                 "onnx": backends.softmax(np.concatenate([c[0] for c in next(iter(chunks.values()))]))
                 .reshape(n, 13, -1)}
        for name in runs:
            probs[name] = backends.softmax(np.concatenate([c[1] for c in chunks[name]])).reshape(n, 13, -1)
        np.savez_compressed(a.save_probs, **probs)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
