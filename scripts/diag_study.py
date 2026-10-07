"""Electrode interchanges vs ECG diagnosis models, with and without LeadCheck detect-and-correct (plan amendment 1).

usage: diag_study.py DATA SCP LEADCHECK.pt THRESHOLD.json OUT.json DIAG.pt [DIAG.pt ...]
"""
import hashlib
import json
import pathlib
import platform
import subprocess
import sys

import numpy as np
import scipy
import sklearn
import torch
from sklearn.metrics import roc_curve

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from leadcheck.data import finish, load, variants  # noqa: E402
from leadcheck.diagnosis import SUPER, build, superclass_labels  # noqa: E402
from leadcheck.model import LeadNet  # noqa: E402
from leadcheck.stats import MIN_N, auroc, macro_auroc, prop  # noqa: E402
from leadcheck.swaps import APPROXIMATE, CLASSES, INVERSE, apply  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
MI = SUPER.index("MI")
BOOT = 500
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def run(model, V, sigmoid):
    flat = torch.from_numpy(np.ascontiguousarray(V.reshape(-1, 12, V.shape[-1])))
    out = []
    with torch.no_grad():
        for s in range(0, len(flat), 1024):
            z = model(flat[s:s + 1024].to(DEV)).float()
            out.append((torch.sigmoid(z) if sigmoid else torch.softmax(z, 1)).cpu().numpy())
    return np.concatenate(out).reshape(V.shape[0], V.shape[1], -1)


def null_variant(x, rng):
    """Benign perturbation: 0.01 mV Gaussian noise and a 30 ms (3-sample) shift."""
    y = np.concatenate([x[:, 3:], np.repeat(x[:, -1:], 3, axis=1)], axis=1)
    return finish(y + rng.normal(0, 0.01, y.shape))


def thresholds(Y9, P9):
    youden, spec95 = [], []
    for k in range(Y9.shape[1]):
        fpr, tpr, thr = roc_curve(Y9[:, k], P9[:, k])
        youden.append(float(thr[np.argmax(tpr - fpr)]))
        spec95.append(float(np.quantile(P9[Y9[:, k] == 0, k], 0.95)))
    return {"youden": youden, "spec95": spec95}


def endpoint(P, thr, Y, clean_pred, keep):
    pred = P >= thr
    norm_only = (Y == np.array([1, 0, 0, 0, 0])).all(1) & ~clean_pred[:, MI] & keep
    mi_hit = (Y[:, MI] == 1) & clean_pred[:, MI] & keep
    return {"flip": prop((pred != clean_pred).any(1)[keep]), "false_mi": prop(pred[norm_only, MI]),
            "missed_mi": prop(~pred[mi_hit, MI])}


def boot_ci(stat, groups, rng):
    """Paired percentile bootstrap that resamples patients; stat(idx) -> float or None."""
    vals = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        v = stat(idx)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    data, scp, lc_ckpt, thr_json, out_json, diag_ckpts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:]
    X, meta, folds = load(data)
    Y = superclass_labels(meta, scp)
    ck = torch.load(lc_ckpt, map_location=DEV)
    assert ck["classes"] == CLASSES, "LeadCheck checkpoint class order differs from swaps.CLASSES"
    lc = LeadNet(len(CLASSES)).to(DEV).eval()
    lc.load_state_dict(ck["state"])
    lc_thr = json.load(open(thr_json))["threshold"]
    rng = np.random.default_rng(2026)
    res = {"plan": "study/PLAN.md, amendment 1", "leadcheck_threshold": lc_thr, "models": {}}
    sets = {}
    for f in (9, 10):
        idx = np.where((folds == f) & (Y.sum(1) > 0))[0]
        V = np.stack([variants(X[i], seed=int(i)) for i in idx])
        nullV = np.stack([null_variant(V[a, 0], np.random.default_rng(int(i))) for a, i in enumerate(idx)])[:, None]
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
        null_flag = 1 - run(lc, nullV, sigmoid=False)[:, 0, 0] >= lc_thr
        sets[f] = dict(idx=idx, V=V, C=C, null=nullV, flagged=flagged, typed=typed, rerecord=rerecord, null_flag=null_flag)
        print(f"fold {f}: {len(idx)} labelled recordings; LeadCheck flags {flagged[:, 0].mean():.4f} of unmodified, "
              f"{null_flag.mean():.4f} of null-perturbed", flush=True)
    rate9 = sets[9]["flagged"][:, 0].mean()
    assert 0.005 <= rate9 <= 0.015, f"fold-9 clean flag rate {rate9:.4f}: threshold does not belong to this checkpoint"
    s10 = sets[10]
    idx10, Y10 = s10["idx"], Y[s10["idx"]]
    patients = np.array([meta[i]["patient_id"] for i in idx10])
    groups = [np.where(patients == p)[0] for p in np.unique(patients)]
    fl, ty, rr = s10["flagged"], s10["typed"], s10["rerecord"]
    res["data"] = {"fold10_labelled_recordings": int(len(idx10)), "fold10_patients": len(groups)}
    exact = [k for k in range(1, len(CLASSES)) if CLASSES[k] not in APPROXIMATE]
    res["leadcheck"] = {
        "clean_flagged": prop(fl[:, 0]), "null_flagged": prop(s10["null_flag"]),
        "pooled_exact_sensitivity": prop(fl[:, exact].ravel()),
        "per_class": {CLASSES[k]: {"flagged": prop(fl[:, k]), "typed_right_when_flagged": prop(ty[fl[:, k], k] == k),
                                   "wrong_type_corrections": int((fl[:, k] & (ty[:, k] != k) & ~rr[:, k]).sum()),
                                   "rerecord": prop(rr[:, k]), "approximate": CLASSES[k] in APPROXIMATE}
                      for k in range(1, len(CLASSES))}}
    per_model_P = {}
    for ckp in diag_ckpts:
        st = torch.load(ckp, map_location=DEV)
        model = build(st["arch"]).to(DEV).eval()
        model.load_state_dict(st["state"])
        P9 = run(model, sets[9]["V"][:, :1], sigmoid=True)[:, 0]
        thr = thresholds(Y[sets[9]["idx"]], P9)
        P = run(model, s10["V"], sigmoid=True)
        PC = run(model, s10["C"], sigmoid=True)
        PN = run(model, s10["null"], sigmoid=True)[:, 0]
        name = pathlib.Path(ckp).stem
        per_model_P[name] = (P, PC, st["swap_aug"])
        clean_auc = macro_auroc(Y10, P[:, 0])
        r = {"arch": st["arch"], "swap_aug": st["swap_aug"], "seed": st["seed"], "best_epoch": st.get("best_epoch"),
             "selection": st.get("selection"), "clean_macro_auroc": clean_auc, "clean_mi_auroc": auroc(Y10[:, MI], P[:, 0, MI]),
             "thresholds": thr, "null": {"macro_auroc": macro_auroc(Y10, PN)}, "none": {}, "detect_correct": {}}
        for op, t in thr.items():
            t = np.array(t)
            clean_pred = P[:, 0] >= t
            r["null"][op] = endpoint(PN, t, Y10, clean_pred, np.ones(len(Y10), bool))
            for k in range(1, len(CLASSES)):
                kc = CLASSES[k]
                keep = ~rr[:, k]
                r["none"].setdefault(kc, {})[op] = endpoint(P[:, k], t, Y10, clean_pred, np.ones(len(Y10), bool))
                r["detect_correct"].setdefault(kc, {})[op] = endpoint(PC[:, k], t, Y10, clean_pred, keep)
            r["detect_correct_clean_cost"] = r.get("detect_correct_clean_cost", {})
            r["detect_correct_clean_cost"][op] = endpoint(PC[:, 0], t, Y10, clean_pred, ~rr[:, 0])
        for k in range(1, len(CLASSES)):
            kc = CLASSES[k]
            keep = ~rr[:, k]
            r["none"][kc]["macro_auroc"] = macro_auroc(Y10, P[:, k])
            r["none"][kc]["mi_auroc"] = auroc(Y10[:, MI], P[:, k, MI])
            r["detect_correct"][kc]["macro_auroc"] = macro_auroc(Y10[keep], PC[keep, k]) if keep.sum() >= MIN_N else None
            r["detect_correct"][kc]["n_kept"] = int(keep.sum())
        la = CLASSES.index("LA/RA")
        r["bootstrap"] = {
            "la_ra_macro_auroc_drop": {
                "estimate": clean_auc - r["none"]["LA/RA"]["macro_auroc"],
                "ci": boot_ci(lambda i: (lambda a, b: None if a is None or b is None else a - b)(
                    macro_auroc(Y10[i], P[i, 0]), macro_auroc(Y10[i], P[i, la])), groups, rng)}}
        t95 = np.array(thr["spec95"])
        cp = P[:, 0] >= t95
        no = (Y10 == np.array([1, 0, 0, 0, 0])).all(1) & ~cp[:, MI]

        def fm_diff(i):
            m = no[i]
            if m.sum() < MIN_N:
                return None
            return float((P[i, la, MI][m] >= t95[MI]).mean() - (PC[i, la, MI][m] >= t95[MI]).mean())

        r["bootstrap"]["la_ra_false_mi_spec95_none_minus_corrected"] = {"estimate": fm_diff(np.arange(len(Y10))),
                                                                       "ci": boot_ci(fm_diff, groups, rng)}
        r["checkpoint_sha256"] = sha256(ckp)
        res["models"][name] = r
        print(f"{name}: clean {clean_auc:.4f}, LA/RA drop {r['bootstrap']['la_ra_macro_auroc_drop']['estimate']:.4f} "
              f"{r['bootstrap']['la_ra_macro_auroc_drop']['ci']}", flush=True)
    std = [n for n, v in per_model_P.items() if not v[2]]
    aug = [n for n, v in per_model_P.items() if v[2]]
    if std and aug:
        la = CLASSES.index("LA/RA")

        def arm_gap(i):
            drops = {}
            for arm, names in (("std", std), ("aug", aug)):
                vals = []
                for n in names:
                    P = per_model_P[n][0]
                    a, b = macro_auroc(Y10[i], P[i, 0]), macro_auroc(Y10[i], P[i, la])
                    if a is None or b is None:
                        return None
                    vals.append(a - b)
                drops[arm] = np.mean(vals)
            return float(drops["std"] - drops["aug"])

        res["arm_comparison"] = {"la_ra_macro_auroc_drop_standard_minus_augmented": {
            "estimate": arm_gap(np.arange(len(Y10))), "ci": boot_ci(arm_gap, groups, rng)}}
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    res["provenance"] = {"git_commit": git, "git_dirty": dirty, "python": platform.python_version(), "torch": torch.__version__,
                         "numpy": np.__version__, "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
                         "leadcheck_checkpoint_sha256": sha256(lc_ckpt), "threshold_file": str(thr_json),
                         "X100_sha256": sha256(pathlib.Path(data) / "X100.npy"),
                         "meta_sha256": sha256(pathlib.Path(data) / "meta.json"), "bootstrap_replicates": BOOT}
    json.dump(clean(res), open(out_json, "w"), indent=1, allow_nan=False)
    print("saved", out_json)


if __name__ == "__main__":
    main()
