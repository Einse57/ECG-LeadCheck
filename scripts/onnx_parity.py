"""Check that the exported ONNX model matches the PyTorch checkpoint. usage: onnx_parity.py DATA CKPT.pt MODEL.onnx"""
import pathlib
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from leadcheck.data import load, variants  # noqa: E402
from leadcheck.model import LeadNet  # noqa: E402
from leadcheck.swaps import CLASSES  # noqa: E402

X, meta, folds = load(sys.argv[1])
ck = torch.load(sys.argv[2], map_location="cpu")
assert ck["classes"] == CLASSES
model = LeadNet(len(CLASSES)).eval()
model.load_state_dict(ck["state"])
idx = np.where(folds == 9)[0][:20]
V = np.concatenate([variants(X[i], seed=int(i)) for i in idx]).astype(np.float32)
with torch.no_grad():
    ref = model(torch.from_numpy(V)).numpy()
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
got = ort.InferenceSession(sys.argv[3], opts, providers=["CPUExecutionProvider"]).run(None, {"ecg": V})[0]
diff = float(np.abs(ref - got).max())
print(f"max |logit difference| over {len(V)} inputs: {diff:.2e}")
assert diff < 1e-3, "ONNX export does not match the checkpoint"
