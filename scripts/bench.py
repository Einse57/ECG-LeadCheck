"""Latency and peak memory of one backend. usage: bench.py --backend openvino --device CPU [--batch 13]

Runs the bundled examples (data/examples.npz) under every interchange (13 inputs per recording, the
demo's workload) and prints median / p90 latency per batch and peak resident memory of the process.
"""
import argparse
import json
import pathlib
import platform
import resource
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leadcheck import backends  # noqa: E402
from leadcheck.data import variants  # noqa: E402
from leadcheck.swaps import canonical  # noqa: E402


def peak_rss_mb():
    try:
        import psutil
        p = psutil.Process()
        if hasattr(p.memory_info(), "peak_wset"):  # Windows
            return p.memory_info().peak_wset / 2**20
    except ImportError:
        pass
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if sys.platform != "darwin" else r / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="onnx", choices=["onnx", "torch", "openvino"])
    ap.add_argument("--device", default=None, help="torch: auto|xpu|cpu; openvino: CPU|GPU|NPU")
    ap.add_argument("--ckpt")
    ap.add_argument("--batch", type=int, default=13)
    ap.add_argument("--precision", help="openvino INFERENCE_PRECISION_HINT, e.g. f32 (default: device default)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    d = np.load(ROOT / "data/examples.npz")
    V = np.concatenate([variants(canonical(x)) for x in d["X"].astype(np.float32)])
    kw = {"batch": a.batch, "precision": a.precision} if a.backend == "openvino" else {}
    t0 = time.perf_counter()
    f = backends.load(a.backend, a.device, ckpt=a.ckpt, **kw)
    load_s = time.perf_counter() - t0
    batches = [V[s:s + a.batch] for s in range(0, len(V) - a.batch + 1, a.batch)]
    sync = None
    if a.backend == "torch" and getattr(f, "device", "") == "xpu":
        import torch
        sync = torch.xpu.synchronize
    for i in range(a.warmup):
        f(batches[i % len(batches)])
    ts = []
    for i in range(a.iters):
        t = time.perf_counter()
        f(batches[i % len(batches)])
        if sync:
            sync()
        ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts)
    out = {"backend": a.backend, "device": str(getattr(f, "device", a.device)), "batch": a.batch,
           "precision": a.precision,
           "load_compile_s": round(load_s, 3), "median_ms": round(float(np.median(ts)), 3),
           "p90_ms": round(float(np.percentile(ts, 90)), 3), "per_ecg_ms": round(float(np.median(ts)) / a.batch, 3),
           "peak_rss_mb": round(peak_rss_mb(), 1), "iters": a.iters, "host": platform.platform()}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
