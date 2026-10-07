"""Fold-10 results at the 99 % specificity threshold chosen on fold 9. usage: evaluate.py PRED.npz"""
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0] + "/src")
from leadcheck.swaps import CLASSES  # noqa: E402

GROUPS = {
    "exact limb (LA/RA, RA/LL, rotations)": ["LA/RA", "RA/LL", "rotation RA>LA>LL", "rotation RA>LL>LA"],
    "LA/LL": ["LA/LL"],
    "right leg (approximate)": ["RA/RL", "LA/RL"],
    "adjacent precordial": [c for c in CLASSES if c.startswith("V")],
}
TARGETS = {"exact limb (LA/RA, RA/LL, rotations)": 0.95, "LA/LL": 0.80, "right leg (approximate)": 0.97,
           "adjacent precordial": 0.90}


def report(P, fold):
    any_score = 1 - P[:, :, 0]
    thr = np.quantile(any_score[fold == 9, 0], 0.99)
    t = fold == 10
    flagged = any_score[t] >= thr
    res = {"threshold": float(thr), "specificity": float(1 - flagged[:, 0].mean()), "n_test": int(t.sum()), "classes": {}}
    for k, c in enumerate(CLASSES[1:], 1):
        auc = roc_auc_score(np.r_[np.zeros(t.sum()), np.ones(t.sum())], np.r_[any_score[t, 0], any_score[t, k]])
        typed = P[t, k, 1:].argmax(1) + 1 == k
        res["classes"][c] = {"sensitivity": float(flagged[:, k].mean()), "auroc": float(auc),
                             "type_correct_when_flagged": float(typed[flagged[:, k]].mean()) if flagged[:, k].any() else None}
    res["groups"] = {}
    for g, cs in GROUPS.items():
        ks = [CLASSES.index(c) for c in cs]
        sens = float(flagged[:, ks].mean())
        res["groups"][g] = {"sensitivity": sens, "target": TARGETS[g], "pass": sens >= TARGETS[g]}
    ks = list(range(1, len(CLASSES)))
    typed = (P[t][:, ks, 1:].argmax(2) + 1) == np.array(ks)
    res["type_correct_when_flagged"] = float(typed[flagged[:, ks]].mean())
    res["pooled_sensitivity"] = float(flagged[:, ks].mean())
    return res


def main():
    d = np.load(sys.argv[1])
    out = {m: report(d[m], d["fold"]) for m in ("cnn", "baseline") if m in d.files}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
