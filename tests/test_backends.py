import sys
import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from leadcheck.backends import torch_device  # noqa: E402


def test_xpu_request_falls_back_to_cpu_when_unavailable(monkeypatch):
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert torch_device("xpu") == "cpu"
    assert any("falling back" in str(x.message) for x in w)
    assert torch_device("auto") == "cpu"
