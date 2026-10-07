"""Electrode interchanges vs ECG diagnosis models, with and without LeadCheck detect-and-correct.

usage: diag_study.py DATA SCP LEADCHECK.pt THRESHOLD.json OUT.json DIAG.pt [DIAG.pt ...]
"""
import json
import pathlib
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from leadcheck.data import finish, load, variants  # noqa: E402
from leadcheck.diagnosis import SUPER, build, superclass_labels  # noqa: E402
from leadcheck.model import LeadNet  # noqa: E402
from leadcheck.swaps import APPROXIMATE, CLASSES, apply  # noqa: E402

INVERSE = {"rotation RA>LA>LL": "rotation RA>LL>LA", "rotation RA>LL>LA": "rotation RA>LA>LL"}
MI = SUPER.index("MI")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def run(model, V, sigmoid):
    flat = torch.from_numpy(V.reshape(-1, 12, V.shape[-1]))
    out = []
    with torch.no_grad():
        for s in range(0, len(flat), 1024):
            z = model(flat[s:s + 1024].to(DEV)).float()
            out.append((torch.sigmoid(z) if sigmoid else torch.softmax(z, 1)).cpu().numpy())
    return np.concatenate(out).reshape(V.shape[0], V.shape[1], -1)


def boot(values, n=1000, seed=0):
    values = np.asarray(values, float)
    if len(values) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def metrics(P, thr, Y, clean_pred, keep=None):
    """P: (n, 5) probabilities for one variant; clean_pred: (n, 5) bool predictions on the clean recording."""
    keep = np.ones(len(P), bool) if keep is None else keep
    pred = P >= thr
    norm_only = (Y == np.array([1, 0, 0, 0, 0])).all(1) & ~clean_pred[:, MI] & keep
    mi_hit = (Y[:, MI] == 1) & clean_pred[:, MI] & keep
    flips = (pred != clean_pred).any(1)[keep]
    false_mi = pred[norm_only, MI]
    missed = ~pred[mi_hit, MI]
    aucs = [roc_auc_score(Y[keep, k], P[keep, k]) for k in range(5)]
    return {"flip": float(flips.mean()), "flip_n": int(flips.sum()), "flip_ci": boot(flips),
            "false_mi": float(false_mi.mean()), "false_mi_n": f"{int(false_mi.sum())}/{len(false_mi)}", "false_mi_ci": boot(false_mi),
            "missed_mi": float(missed.mean()), "missed_mi_n": f"{int(missed.sum())}/{len(missed)}", "missed_mi_ci": boot(missed),
            "macro_auroc": float(np.mean(aucs))}


def main():
    data, scp, lc_ckpt, thr_json, out_json, diag_ckpts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:]
    X, meta, folds = load(data)
    Y = superclass_labels(meta, scp)
    lc = LeadNet(len(CLASSES)).to(DEV).eval()
    lc.load_state_dict(torch.load(lc_ckpt, map_location=DEV)["state"])
    lc_thr = json.load(open(thr_json))["threshold"]
    res = {"leadcheck_threshold": lc_thr, "models": {}}
    sets = {}
    for f in (9, 10):
        idx = np.where((folds == f) & (Y.sum(1) > 0))[0]
        V = np.stack([variants(X[i], seed=int(i)) for i in idx])
        L = run(lc, V, sigmoid=False)
        flagged = 1 - L[:, :, 0] >= lc_thr
        typed = L[:, :, 1:].argmax(2) + 1
        C = V.copy()
        rerecord = np.zeros(flagged.shape, bool)
        for a in range(V.shape[0]):
            for b in range(V.shape[1]):
                if flagged[a, b]:
                    k = CLASSES[typed[a, b]]
                    if k in APPROXIMATE:
                        rerecord[a, b] = True
                    else:
                        C[a, b] = finish(apply(V[a, b], INVERSE.get(k, k)))
        sets[f] = (idx, V, C, flagged, typed, rerecord)
        print(f"fold {f}: {len(idx)} recordings; flagged clean {flagged[:, 0].mean():.4f}", flush=True)
    idx10, V10, C10, fl10, ty10, rr10 = sets[10]
    res["leadcheck"] = {"clean_flagged": float(fl10[:, 0].mean()),
                        "per_class": {CLASSES[k]: {"flagged": float(fl10[:, k].mean()), "typed_right": float((ty10[fl10[:, k], k] == k).mean()) if fl10[:, k].any() else None,
                                                   "rerecord": float(rr10[:, k].mean())} for k in range(1, len(CLASSES))}}
    for ck in diag_ckpts:
        st = torch.load(ck, map_location=DEV)
        model = build(st["arch"]).to(DEV).eval()
        model.load_state_dict(st["state"])
        idx9, V9 = sets[9][0], sets[9][1]
        P9 = run(model, V9[:, :1], sigmoid=True)[:, 0]
        thr = np.array([roc_curve(Y[idx9, k], P9[:, k])[2][np.argmax(np.subtract(*roc_curve(Y[idx9, k], P9[:, k])[1::-1]))] for k in range(5)])
        P = run(model, V10, sigmoid=True)
        PC = run(model, C10, sigmoid=True)
        Y10 = Y[idx10]
        clean_pred = P[:, 0] >= thr
        name = pathlib.Path(ck).stem
        r = {"arch": st["arch"], "swap_aug": st["swap_aug"], "clean_macro_auroc": float(np.mean([roc_auc_score(Y10[:, k], P[:, 0, k]) for k in range(5)])),
             "thresholds": thr.tolist(), "none": {}, "detect_correct": {}}
        for k in range(1, len(CLASSES)):
            r["none"][CLASSES[k]] = metrics(P[:, k], thr, Y10, clean_pred)
            m = metrics(PC[:, k], thr, Y10, clean_pred, keep=~rr10[:, k])
            m["rerecord"] = float(rr10[:, k].mean())
            r["detect_correct"][CLASSES[k]] = m
        clean_corr = PC[:, 0] >= thr
        r["detect_correct_clean_cost"] = {"flip": float((clean_corr != clean_pred).any(1)[~rr10[:, 0]].mean()),
                                          "rerecord": float(rr10[:, 0].mean()),
                                          "macro_auroc": float(np.mean([roc_auc_score(Y10[~rr10[:, 0], j], PC[~rr10[:, 0], 0, j]) for j in range(5)]))}
        res["models"][name] = r
        la = r["none"]["LA/RA"]
        print(f"{name}: clean AUROC {r['clean_macro_auroc']:.4f} | LA/RA flip {la['flip']:.3f} false MI {la['false_mi']:.3f} ({la['false_mi_n']}) "
              f"-> corrected flip {r['detect_correct']['LA/RA']['flip']:.3f}", flush=True)
    json.dump(res, open(out_json, "w"), indent=1)
    print("saved", out_json)


if __name__ == "__main__":
    main()
