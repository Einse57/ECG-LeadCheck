"""Read the PTB-XL 100 Hz records into one float16 array (records x 12 leads x 1000 samples, mV)."""
import csv
import json
import pathlib
import sys

import numpy as np

raw, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rows = sorted(csv.DictReader(open(raw / "ptbxl_database.csv")), key=lambda r: int(r["ecg_id"]))
X = np.zeros((len(rows), 12, 1000), np.float16)
lead_order = None
for k, r in enumerate(rows):
    base = raw / r["filename_lr"]
    header = open(f"{base}.hea").read().split("\n")
    names = [line.split()[-1] for line in header[1:13]]
    gains = [float(line.split()[2].split("(")[0].split("/")[0]) for line in header[1:13]]
    assert lead_order in (None, names), (r["ecg_id"], names)
    lead_order = names
    sig = np.fromfile(f"{base}.dat", dtype="<i2").reshape(-1, 12)[:1000] / np.array(gains)
    X[k] = sig.T.astype(np.float16)
meta = [{"ecg_id": int(r["ecg_id"]), "patient_id": r["patient_id"], "fold": int(r["strat_fold"]),
         "validated_by_human": r["validated_by_human"], "report": r["report"], "scp_codes": r["scp_codes"],
         "electrodes_problems": r["electrodes_problems"], "pacemaker": r["pacemaker"]} for r in rows]
np.save(out / "X100.npy", X)
json.dump({"leads": lead_order, "records": meta}, open(out / "meta.json", "w"))
iii_err = np.abs(X[:, 2].astype(np.float32) - (X[:, 1].astype(np.float32) - X[:, 0].astype(np.float32)))
print(len(rows), "records, leads", lead_order, "| median |III-(II-I)|", float(np.median(iii_err)), "mV, p99", float(np.percentile(iii_err, 99)))
