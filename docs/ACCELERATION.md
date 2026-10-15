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
Reference: ONNX Runtime 1.30.0 on the same machine's CPU. Software: OpenVINO 2026.4.1, PyTorch 2.10.0
(`+cpu` wheel; `+xpu` wheel for the Arc GPU), Python 3.12.

| machine | backend / device | max abs logit difference | same predicted class | same alarm decision |
|---|---|---|---|---|
| Core Ultra 9 285H, Windows 11 | PyTorch CPU | 4.8e-6 | 100 % | 100 % |
| | PyTorch XPU, Arc 140T GPU | 5.7e-6 | 100 % | 100 % |
| | OpenVINO CPU, f32 (default) | 7.6e-6 | 100 % | 100 % |
| | OpenVINO CPU, bf16 | 0.070 | 99.995 % | 99.97 % |
| | OpenVINO GPU, Arc 140T (default precision) | 0.018 | 100 % | 99.995 % |
| | OpenVINO NPU (default precision, batch 1) | 0.016 | 100 % | 99.991 % |
| Core Ultra 7 265F, Windows 11 | PyTorch CPU | 4.3e-6 | 100 % | 100 % |
| | OpenVINO CPU, f32 (default) | 7.6e-6 | 100 % | 100 % |
| | OpenVINO CPU, bf16 | 0.070 | 99.995 % | 99.97 % |
| | OpenVINO NPU (default precision, batch 1) | 0.016 | 100 % | 99.991 % |
| Intel Xeon cloud VM (8 vCPU), Linux | PyTorch CPU | 4.7e-6 | 100 % | 100 % |
| | OpenVINO CPU, f32 (default) | 7.6e-6 | 100 % | 100 % |
| | OpenVINO CPU, bf16 | 6.29 | 99.95 % | 99.92 % |

On the Xeon VM, the fold-10 results from `scripts/evaluate.py` (with the fold-9 threshold re-derived from
each backend's outputs) are identical for ONNX Runtime, PyTorch CPU and OpenVINO CPU f32, and match
`results/fold10_metrics.json` at the precision shown in `GATE_RESULT.md`.

## Latency

`scripts/bench.py`, 200 iterations after 10 warm-up runs on the bundled examples. Batch 13 is one
recording under every interchange. On the Windows machines the runs were started as a scheduled task
in the interactive desktop session. Peak memory is the process peak resident set (peak working set on
Windows) and includes the Python runtime and the framework.

| machine | backend / device | batch | median ms | p90 ms | peak memory MB |
|---|---|---|---|---|---|
| Core Ultra 9 285H | ONNX Runtime CPU | 13 | 40.5 | 71.7 | 105 |
| | PyTorch CPU | 13 / 1 | 32.5 / 5.58 | 38.2 / 6.21 | 256 |
| | PyTorch XPU, Arc 140T | 13 / 1 | 8.42 / 2.15 | 8.72 / 2.41 | 1,144 |
| | OpenVINO CPU f32 | 13 / 1 | 15.2 / 1.61 | 16.1 / 2.25 | 134 |
| | OpenVINO GPU, Arc 140T | 13 / 1 | 1.01 / 0.335 | 1.10 / 0.342 | 426 |
| | OpenVINO NPU | 13 / 1 | 7.01 / 1.11 | 7.15 / 1.17 | 157 |
| Core Ultra 7 265F | ONNX Runtime CPU | 13 | 25.4 | 26.6 | 106 |
| | PyTorch CPU | 13 / 1 | 14.8 / 3.06 | 15.3 / 3.47 | 257 |
| | OpenVINO CPU f32 | 13 / 1 | 12.7 / 1.13 | 13.5 / 1.33 | 131 |
| | OpenVINO NPU | 13 / 1 | 6.93 / 1.09 | 7.09 / 1.13 | 165 |
| Intel Xeon cloud VM (8 vCPU, shared) | ONNX Runtime CPU | 13 | 47.3 | 64.8 | 110 |
| | PyTorch CPU | 13 | 26.7 | 37.4 | 286 |
| | OpenVINO CPU f32 | 13 | 13.5 | 20.1 | 153 |

On the 265F machine, OpenVINO's `GPU` device is a discrete non-Intel card; it was not tested.

### Core Ultra 9 285H, single session (2026-10-14)

All paths in one scheduled-task session, batch 13 (one recording under every interchange), median of
200 iterations after 10 warm-up runs. ONNX Runtime 1.30.0 (`CPUExecutionProvider`), OpenVINO 2026.4.1.
CPU load before the session: about 3 %.

| backend / device | settings | median ms | p90 ms |
|---|---|---|---|
| ONNX Runtime CPU | default threads | 34.6 | 40.4 |
| ONNX Runtime CPU | 2 threads, as in `app.py` | 62.8 | 63.7 |
| OpenVINO GPU, Arc 140T | device default precision | 1.04 | 1.16 |
| OpenVINO NPU | device default precision | 6.98 | 7.12 |
| OpenVINO CPU | f32 | 14.3 | 14.7 |

A quick agreement check in the same session (first 200 fold-10 recordings, 2,600 inputs) gave 100 %
class and alarm agreement with ONNX Runtime on GPU, NPU and CPU f32.

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
