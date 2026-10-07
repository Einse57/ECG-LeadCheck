"""Collect the out-of-fold scores of all unmodified recordings. usage: scan.py OOF_DIR DATA_DIR OUT.json"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from leadcheck.swaps import CLASSES  # noqa: E402

oof, data, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
meta = {m["ecg_id"]: m for m in json.load(open(data / "meta.json"))["records"]}
rows = []
for k in range(1, 11):
    d = np.load(oof / f"fold{k}.npz")
    P = d["cnn"][:, 0, :]
    for eid, p in zip(d["ecg_id"], P):
        top = int(p[1:].argmax()) + 1
        rows.append({"ecg_id": int(eid), "fold": k, "score": float(1 - p[0]), "type": CLASSES[top], "p_type": float(p[top])})
rows.sort(key=lambda r: -r["score"])
json.dump(rows, open(out, "w"))
scores = np.array([r["score"] for r in rows])
print(len(rows), "recordings;", {t: int((scores >= t).sum()) for t in (0.97, 0.99, 0.999, 0.9999)})
probes = json.load(open(pathlib.Path(__file__).resolve().parents[1] / "data/probes/real_world.json"))
rank = {r["ecg_id"]: (n + 1, r) for n, r in enumerate(rows)}
for p in probes:
    n, r = rank[p["ecg_id"]]
    print(f"probe {p['ecg_id']} ({p['meaning']}): rank {n}/{len(rows)}, score {r['score']:.4f}, type {r['type']} {r['p_type']:.3f}")
from collections import Counter
print("top 40 types:", Counter(r["type"] for r in rows[:40]))
for r in rows[:25]:
    m = meta[r["ecg_id"]]
    print(f"  {r['ecg_id']:>6} f{r['fold']:<2} {r['score']:.5f} {r['type']:<20} {m['report'][:70]!r}")
