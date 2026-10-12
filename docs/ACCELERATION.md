# Running LeadCheck with PyTorch XPU or OpenVINO

LeadCheck can run as a lead-placement quality check before ECG delineation, for example with
[OpenECG](https://github.com/vitaldb/openecg): a recording flagged for an electrode interchange is
corrected or re-recorded before its waves are delineated.

The demo and `scripts/check_examples.py` use ONNX Runtime on CPU. Two optional backends are available
through `src/leadcheck/backends.py`:

| backend | install | devices | model file |
|---|---|---|---|
| `onnx` (default, reference) | `requirements.txt` | CPU | `model/leadcheck.onnx` |
| `torch` | `requirements-xpu.txt` (XPU wheel index) or any PyTorch | `xpu` if available, else `cpu` | `leadcheck_main.pt` from release `v2` |
| `openvino` | `requirements-openvino.txt` | `CPU`, `GPU`, `NPU` | `model/leadcheck.onnx` |

- `torch`: `--device auto` picks `xpu` when `torch.xpu.is_available()`, otherwise CPU. Asking for `xpu`
  without an XPU device warns and runs on CPU.
- `openvino`: CPU runs with an f32 precision hint by default. On CPUs with BF16 support OpenVINO would
  otherwise pick bf16, which changed the alarm decision on 0.08 % of inputs in the check below.
  GPU and NPU use their device default precision. NPU needs a static shape, so the batch size is fixed
  (default 1) and the last partial batch is zero-padded.

## Commands

```bash
pip install -r requirements.txt -r requirements-openvino.txt
python scripts/bench.py --backend openvino --device CPU          # latency and peak memory
python scripts/bench.py --backend torch --device auto --ckpt leadcheck_main.pt
python scripts/agreement.py DATA_DIR --folds 9,10 --ckpt leadcheck_main.pt --ov-device CPU
python -m pytest -q tests                                        # OpenVINO test is skipped without openvino
python examples/leadcheck_then_openecg.py records500/00000/00009_hr --backend openvino --device CPU
```

`DATA_DIR` is the output of `scripts/build_dataset.py` (see `run_all.sh`, step 1).
`bench.py` runs the bundled examples under every interchange (13 inputs per recording) and prints
median and p90 latency per batch and the process's peak resident memory.

## Agreement with the ONNX Runtime reference

PTB-XL folds 9 and 10, every recording under every interchange: 4,381 recordings, 56,953 inputs.
Reference: ONNX Runtime 1.30.0 CPU. Tested on an Intel Xeon cloud VM (8 vCPU, Linux), OpenVINO 2026.4.1,
PyTorch 2.10.0 (CPU).

| backend | max abs logit difference | same predicted class | same alarm decision |
|---|---|---|---|
| PyTorch CPU (`leadcheck_main.pt`) | 4.7e-6 | 100 % | 100 % |
| OpenVINO CPU, f32 (default) | 7.6e-6 | 100 % | 100 % |
| OpenVINO CPU, bf16 | 6.29 | 99.95 % | 99.92 % |

With the fold-9 threshold re-derived from each backend's outputs, the fold-10 results from
`scripts/evaluate.py` are identical for ONNX Runtime, PyTorch CPU and OpenVINO CPU f32, and match
`results/fold10_metrics.json` at the precision shown in `GATE_RESULT.md`.

GPU, NPU and XPU agreement has not been measured yet.

## Chaining with OpenECG

`examples/leadcheck_then_openecg.py` reads a 12-lead WFDB record and:

1. reorders leads to I, II, III, aVR, aVL, aVF, V1-V6, resamples to 100 Hz with
   `scipy.signal.resample_poly`, and takes the first 10 s, which is LeadCheck's input format;
2. runs LeadCheck and reports the alarm and the most likely interchange;
3. passes lead II at its original sampling rate to `openecg.report()` (which accepts `fs`; OpenECG's
   `Inference` delineator instead expects a single lead at 250 Hz).

Chest (V1-V6) interchanges do not change lead II, so lead-II delineation is kept for them. Limb and
right-leg interchanges change lead II, so the example marks the OpenECG result as unreliable.
It needs `wfdb`, `scipy` and `openecg[report]`.
