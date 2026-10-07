"""Export a checkpoint to ONNX for the demo app. usage: export_onnx.py CKPT.pt OUT.onnx"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from leadcheck.model import LeadNet  # noqa: E402

ck = torch.load(sys.argv[1], map_location="cpu")
model = LeadNet(len(ck["classes"])).eval()
model.load_state_dict(ck["state"])
torch.onnx.export(model, torch.zeros(1, 12, 1000), sys.argv[2], input_names=["ecg"], output_names=["logits"],
                  dynamic_axes={"ecg": {0: "batch", 2: "time"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
print("exported", sys.argv[2])
