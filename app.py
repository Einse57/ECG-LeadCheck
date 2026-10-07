"""ECG LeadCheck demo: simulate an electrode interchange on a real ECG and see whether the model catches it."""
import json
import pathlib
import sys

import matplotlib
import numpy as np
import onnxruntime as ort
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from leadcheck.data import finish  # noqa: E402
from leadcheck.swaps import APPROXIMATE, CLASSES, apply, canonical  # noqa: E402

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
NAMES = {
    "none": "No mistake",
    "LA/RA": "Left and right arm swapped (LA/RA)",
    "LA/LL": "Left arm and left leg swapped (LA/LL)",
    "RA/LL": "Right arm and left leg swapped (RA/LL)",
    "rotation RA>LA>LL": "Limb electrodes rotated (RA→LA→LL)",
    "rotation RA>LL>LA": "Limb electrodes rotated (RA→LL→LA)",
    "RA/RL": "Right arm and right leg swapped (RA/RL)*",
    "LA/RL": "Left arm and right leg swapped (LA/RL)*",
    **{c: f"Chest electrodes {c} swapped" for c in CLASSES if c.startswith("V")},
}

st.set_page_config(page_title="ECG LeadCheck", page_icon=":material/monitor_heart:", layout="wide")


@st.cache_resource
def session():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(str(ROOT / "model/leadcheck.onnx"), opts, providers=["CPUExecutionProvider"])


@st.cache_data
def examples():
    d = np.load(ROOT / "data/examples.npz")
    return d["X"].astype(np.float32), [str(n) for n in d["names"]]


THRESHOLD = json.loads((ROOT / "model/threshold.json").read_text())["threshold"]


def probs(batch):
    logits = session().run(None, {"ecg": batch.astype(np.float32)})[0]
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def lead_importance(x, k):
    """Drop in the predicted class probability when each lead is blanked out."""
    blanked = np.repeat(x[None], 12, 0)
    for j in range(12):
        blanked[j, j] = 0
    p = probs(np.concatenate([x[None], blanked]))[:, k]
    return np.clip(p[0] - p[1:], 0, None)


def plot(x, highlight):
    fig, axes = plt.subplots(6, 2, figsize=(12, 7.2), sharex=True)
    t = np.arange(x.shape[1]) / 100
    for j, ax in enumerate(axes.T.flat):
        ax.plot(t, x[j], color="#1f2d3d", lw=0.8)
        ax.set_ylim(-2, 2)
        ax.set_yticks([])
        ax.text(0.005, 0.8, LEADS[j], transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.grid(True, color="#f3c5c5", lw=0.5)
        if highlight[j] > 0:
            ax.set_facecolor((1, 0.85, 0.85, min(1.0, 0.25 + highlight[j])))
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("seconds")
    fig.tight_layout(h_pad=0.2)
    return fig


st.title("ECG LeadCheck")
st.markdown("Swapped ECG electrodes can mimic a heart attack or hide one. This model checks a 12-lead ECG for "
            "**electrode interchanges**: pick a real ECG, simulate a mistake, and see whether it is caught.")
st.info("ECGs are real recordings from the PTB-XL test set (CC BY 4.0) that the model never trained on. "
        "Research prototype, not a medical device.", icon=":material/info:")

X, names = examples()
left, right = st.columns([1, 2.4])
with left:
    q = st.query_params
    i = st.selectbox("ECG", range(len(names)), index=int(q.get("ecg", 0)), format_func=lambda k: names[k])
    labels = [NAMES[c] for c in CLASSES]
    kind = CLASSES[labels.index(st.radio("Simulate an electrode mistake", labels,
                                         index=CLASSES.index(q.get("swap", "none"))))]
    uploaded = st.file_uploader("Or upload your own (CSV, 12 columns I..V6 in mV, 100 Hz, 10 s)", type="csv")

x = canonical(X[i])
title = names[i]
if uploaded is not None:
    try:
        raw = np.genfromtxt(uploaded, delimiter=",", skip_header=1)[:1000, :12].T
        assert raw.shape == (12, 1000) and np.isfinite(raw).all()
        x, title, kind = canonical(raw.astype(np.float32)), "Uploaded ECG", "none"
    except Exception:
        st.error("Could not read that file: expected 1,000 rows (10 s at 100 Hz) and 12 columns in mV.")
x = finish(apply(x, kind, np.random.default_rng(0)))
p = probs(x[None])[0]
alarm = 1 - p[0] >= THRESHOLD
k = int(p[1:].argmax()) + 1 if alarm else 0
imp = lead_importance(x, k) if alarm else np.zeros(12)
imp = imp / imp.max() if imp.max() > 0 else imp

with right:
    if alarm:
        st.error(f"**Likely electrode mistake: {NAMES[CLASSES[k]]}** ({p[k]:.0%}). Shaded leads gave it away.")
    else:
        st.success(f"**Electrodes look correctly placed** (interchange score {1 - p[0]:.0%}, alarm at {THRESHOLD:.0%}).")
    if kind != "none":
        st.caption(f"Simulated: {NAMES[kind]}" + (" · *approximate simulation" if kind in APPROXIMATE else ""))
    st.pyplot(plot(x, imp), clear_figure=True)

with st.expander("How it works and how well"):
    st.markdown((ROOT / "model/summary.md").read_text() if (ROOT / "model/summary.md").exists() else "")
st.caption("Code and evaluation: https://github.com/hxlvvm/ECG-LeadCheck · Data: PTB-XL, Wagner et al. 2020, "
           "PhysioNet, CC BY 4.0")
