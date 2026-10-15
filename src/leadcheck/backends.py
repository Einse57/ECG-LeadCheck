"""Inference backends for the interchange detector.

All backends take a float32 batch shaped (batch, 12, samples) and return logits shaped (batch, classes).

- ``onnx``: ONNX Runtime on CPU with ``model/leadcheck.onnx`` (the reference used by the demo).
- ``torch``: the PyTorch checkpoint (GitHub release ``v2``). Runs on ``torch.xpu`` when an Intel GPU is
  available, otherwise on CPU. ``device`` can force ``xpu`` or ``cpu``.
- ``openvino``: OpenVINO on ``CPU``, ``GPU`` or ``NPU``, reading ``model/leadcheck.onnx`` directly.
  CPU runs in f32 by default (``precision`` overrides). Optional: ``pip install -r requirements-openvino.txt``.
"""
from __future__ import annotations

import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
ONNX_PATH = ROOT / "model" / "leadcheck.onnx"
SAMPLES = 1000  # 10 s at 100 Hz


def torch_device(requested: str | None = None) -> str:
    """``xpu`` if requested or available, else ``cpu``. Falls back to CPU if XPU is missing."""
    import torch

    xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()
    if requested in (None, "auto"):
        return "xpu" if xpu_ok else "cpu"
    requested = requested.lower()
    if requested == "xpu" and not xpu_ok:
        import warnings

        warnings.warn("torch.xpu is not available; falling back to CPU", RuntimeWarning, stacklevel=2)
        return "cpu"
    return requested


class OnnxBackend:
    def __init__(self, model_path=ONNX_PATH, threads: int | None = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.device = "CPU"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {"ecg": np.ascontiguousarray(x, np.float32)})[0]


class TorchBackend:
    def __init__(self, ckpt, device: str | None = None):
        import torch

        from .model import LeadNet
        from .swaps import CLASSES

        self.torch = torch
        self.device = torch_device(device)
        ck = torch.load(ckpt, map_location="cpu")
        assert ck["classes"] == CLASSES, "checkpoint class order differs from swaps.CLASSES"
        self.model = LeadNet(len(CLASSES)).eval()
        self.model.load_state_dict(ck["state"])
        self.model.to(self.device)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            t = self.torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(self.device)
            return self.model(t).float().cpu().numpy()


class OpenVINOBackend:
    """OpenVINO compiled model. NPU needs static shapes, so the batch size is fixed there and inputs are
    padded/split to it; CPU and GPU keep a dynamic batch."""

    def __init__(self, device: str = "CPU", model_path=ONNX_PATH, batch: int | None = None,
                 precision: str | None = None):
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(model_path))
        self.device = device.upper()
        self.batch = batch or (1 if self.device == "NPU" else None)
        if self.batch is not None or self.device == "NPU":
            model.reshape({model.input(0): ov.PartialShape([self.batch, 12, SAMPLES])})
        # CPUs with BF16/AMX default to bf16 inference, which moves logits enough to flip a few alarms;
        # keep the CPU at f32 unless asked otherwise. GPU/NPU use their device default (f16) unless set.
        if precision is None and self.device == "CPU":
            precision = "f32"
        config = {}
        if precision:
            config["INFERENCE_PRECISION_HINT"] = precision
        self.compiled = core.compile_model(model, self.device, config)
        self.request = self.compiled.create_infer_request()

    def _run(self, x):
        self.request.infer({0: x})
        return self.request.get_output_tensor(0).data.copy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, np.float32)
        if self.batch is None:
            return self._run(x)
        out = []
        for s in range(0, len(x), self.batch):
            chunk = x[s:s + self.batch]
            n = len(chunk)
            if n < self.batch:
                chunk = np.concatenate([chunk, np.zeros((self.batch - n, *chunk.shape[1:]), np.float32)])
            out.append(self._run(chunk)[:n])
        return np.concatenate(out)


def load(backend: str = "onnx", device: str | None = None, ckpt=None, **kw):
    """Return a callable ``logits = f(batch)`` for the named backend."""
    if backend == "onnx":
        return OnnxBackend(**kw)
    if backend == "torch":
        if ckpt is None:
            raise ValueError("the torch backend needs --ckpt (leadcheck_main.pt from release v2)")
        return TorchBackend(ckpt, device)
    if backend == "openvino":
        return OpenVINOBackend(device or "CPU", **kw)
    raise ValueError(f"unknown backend {backend!r}")


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)
