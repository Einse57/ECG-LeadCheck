"""Bundle a few fold-10 recordings (PTB-XL, CC BY 4.0) for the demo. usage: make_examples.py DATA_DIR OUT.npz"""
import ast
import json
import pathlib
import sys

import numpy as np

data, out = pathlib.Path(sys.argv[1]), sys.argv[2]
X = np.load(data / "X100.npy")
meta = json.load(open(data / "meta.json"))["records"]
labels = {"NORM": "normal ECG", "IMI": "inferior myocardial infarction", "AMI": "anterior myocardial infarction",
          "LVH": "left ventricular hypertrophy", "CLBBB": "left bundle branch block", "CRBBB": "right bundle branch block",
          "AFIB": "atrial fibrillation", "LAFB": "left anterior fascicular block", "1AVB": "first-degree AV block"}
rng = np.random.default_rng(7)
picked, names = [], []
for code, label in labels.items():
    pool = [i for i, m in enumerate(meta) if m["fold"] == 10 and m["validated_by_human"] == "True"
            and ast.literal_eval(m["scp_codes"]).get(code, 0) >= 80 and not m["pacemaker"]]
    for i in rng.choice(pool, size=min(2 if code != "NORM" else 4, len(pool)), replace=False):
        picked.append(int(i))
        names.append(f"PTB-XL #{meta[i]['ecg_id']} · {label}")
np.savez_compressed(out, X=X[picked].astype(np.float16), names=np.array(names), ecg_id=np.array([meta[i]["ecg_id"] for i in picked]))
print(len(picked), "examples ->", out)
