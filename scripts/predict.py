"""Score every interchange variant of the chosen folds with a CNN checkpoint and/or the baseline.

usage: predict.py DATA_DIR OUT.npz --cnn ckpt.pt --baseline model.joblib --folds 9,10
Writes probabilities with shape (records, variants, classes) per model.
"""
import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from leadcheck.data import features, load, variants  # noqa: E402
from leadcheck.model import LeadNet  # noqa: E402
from leadcheck.swaps import CLASSES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"), ap.add_argument("out")
    ap.add_argument("--cnn"), ap.add_argument("--baseline"), ap.add_argument("--folds", default="9,10")
    ap.add_argument("--originals-only", action="store_true")
    a = ap.parse_args()
    X, meta, folds = load(a.data)
    idx = np.where(np.isin(folds, [int(f) for f in a.folds.split(",")]))[0]
    V = np.stack([variants(X[i], seed=int(i))[:1 if a.originals_only else None] for i in idx])
    out = {"idx": idx, "ecg_id": np.array([meta[i]["ecg_id"] for i in idx]), "fold": folds[idx]}
    if a.cnn:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(a.cnn, map_location=dev)
        model = LeadNet(len(CLASSES)).to(dev).eval()
        model.load_state_dict(ck["state"])
        flat = torch.from_numpy(V.reshape(-1, 12, V.shape[-1]))
        probs = []
        with torch.no_grad():
            for s in range(0, len(flat), 1024):
                probs.append(torch.softmax(model(flat[s:s + 1024].to(dev)).float(), 1).cpu().numpy())
        out["cnn"] = np.concatenate(probs).reshape(V.shape[0], V.shape[1], len(CLASSES))
    if a.baseline:
        import joblib
        gbm = joblib.load(a.baseline)
        F = np.stack([features(v) for v in V.reshape(-1, 12, V.shape[-1])])
        out["baseline"] = gbm.predict_proba(F).reshape(V.shape[0], V.shape[1], len(CLASSES))
    np.savez_compressed(a.out, **out)
    print("saved", a.out, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
