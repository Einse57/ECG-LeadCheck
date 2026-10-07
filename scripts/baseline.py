"""Gradient boosting on inter-lead correlations and amplitudes. usage: baseline.py DATA_DIR OUT.joblib"""
import pathlib
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from leadcheck.data import features, finish, load  # noqa: E402
from leadcheck.swaps import CLASSES, apply  # noqa: E402

X, meta, folds = load(sys.argv[1])
rng = np.random.default_rng(0)
F, y = [], []
for i in np.where(folds <= 8)[0]:
    for _ in range(3):
        k = 0 if rng.random() < 0.4 else int(rng.integers(1, len(CLASSES)))
        F.append(features(finish(apply(X[i], CLASSES[k], rng))))
        y.append(k)
model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_leaf_nodes=63, random_state=0)
model.fit(np.stack(F), np.array(y))
joblib.dump(model, sys.argv[2])
print("trained on", len(y), "samples; saved", sys.argv[2])
