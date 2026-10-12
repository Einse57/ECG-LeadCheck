import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openvino")
pytest.importorskip("onnxruntime")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leadcheck import backends  # noqa: E402
from leadcheck.data import variants  # noqa: E402
from leadcheck.swaps import canonical  # noqa: E402


def test_openvino_cpu_matches_onnxruntime():
    d = np.load(ROOT / "data/examples.npz")
    V = np.concatenate([variants(canonical(x)) for x in d["X"][:4].astype(np.float32)])
    ref = backends.load("onnx")(V)
    got = backends.load("openvino", "CPU", precision="f32")(V)
    thr = json.loads((ROOT / "model/threshold.json").read_text())["threshold"]
    assert np.abs(ref - got).max() < 1e-3
    assert (ref.argmax(1) == got.argmax(1)).all()
    pr, pg = backends.softmax(ref), backends.softmax(got)
    assert ((1 - pr[:, 0] >= thr) == (1 - pg[:, 0] >= thr)).all()


def test_openvino_fixed_batch_pads_remainder():
    d = np.load(ROOT / "data/examples.npz")
    V = variants(canonical(d["X"][0].astype(np.float32)))  # 13 inputs
    got = backends.load("openvino", "CPU", batch=4, precision="f32")(V)
    ref = backends.load("onnx")(V)
    assert got.shape == ref.shape and np.abs(ref - got).max() < 1e-3
