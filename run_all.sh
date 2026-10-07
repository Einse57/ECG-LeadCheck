#!/usr/bin/env bash
# Rebuild everything from PTB-XL: data, interchange detector, baseline, diagnosis models, study.
# usage: W=/path/to/workdir ./run_all.sh          (training steps need a CUDA GPU; Python 3.12, requirements-train.txt)
# The released checkpoints (GitHub release v2) reproduce the published numbers exactly; retraining reproduces them
# within run-to-run variation (training uses seeded augmentation and deterministic cuDNN kernels).
set -euo pipefail
W=${W:?set W to a working directory}
R=$(cd "$(dirname "$0")" && pwd)
P=${PYTHON:-python3}
B=https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3
mkdir -p "$W"/raw/ptbxl "$W"/data "$W"/models/diag_v2 "$W"/preds

# 1. PTB-XL 1.0.3, 100 Hz records (CC BY 4.0), from PhysioNet's open-data mirror
cd "$W"/raw/ptbxl
for f in ptbxl_database.csv scp_statements.csv LICENSE.txt; do curl -sS -f --retry 3 -o "$f" "$B/$f"; done
"$P" -c "import csv; [print(r['filename_lr']) for r in csv.DictReader(open('ptbxl_database.csv'))]" > lr_list.txt
{ sed 's/$/.hea/' lr_list.txt; sed 's/$/.dat/' lr_list.txt; } > files.txt
cut -d/ -f1-2 lr_list.txt | sort -u | xargs mkdir -p
xargs -P 16 -I{} curl -sS -f --retry 3 -o {} "$B/{}" < files.txt
cd "$R"
"$P" scripts/build_dataset.py "$W"/raw/ptbxl "$W"/data
(cd "$W" && grep -v "models/" "$R"/CHECKSUMS.sha256 | sha256sum -c -)

# 2. Interchange detector (folds 1-8), threshold on fold 9, results on fold 10
"$P" scripts/train.py "$W"/data "$W"/models/main.pt --folds 1-8 --epochs 40 --seed 0
"$P" scripts/baseline.py "$W"/data "$W"/models/baseline.joblib
"$P" scripts/predict.py "$W"/data "$W"/preds/main.npz --cnn "$W"/models/main.pt --baseline "$W"/models/baseline.joblib --folds 9,10
"$P" scripts/evaluate.py "$W"/preds/main.npz | tee results/fold10_metrics.json
"$P" -c "import json; d=json.load(open('results/fold10_metrics.json'))['cnn']; json.dump({'threshold': d['threshold'], 'specificity_fold10': d['specificity'], 'source': 'fold 9, 99% specificity'}, open('model/threshold.json', 'w'), indent=1)"
"$P" scripts/export_onnx.py "$W"/models/main.pt model/leadcheck.onnx
"$P" scripts/onnx_parity.py "$W"/data "$W"/models/main.pt model/leadcheck.onnx

# 3. Study: eight diagnosis models, then the interchange study
for arch in resnet inception; do for seed in 0 1; do
  "$P" scripts/diag_train.py "$W"/data "$W"/raw/ptbxl/scp_statements.csv "$W"/models/diag_v2/diag_${arch}_s${seed}.pt --arch $arch --seed $seed
  "$P" scripts/diag_train.py "$W"/data "$W"/raw/ptbxl/scp_statements.csv "$W"/models/diag_v2/diag_${arch}_s${seed}_aug.pt --arch $arch --seed $seed --swap-aug
done; done
"$P" scripts/diag_study.py "$W"/data "$W"/raw/ptbxl/scp_statements.csv "$W"/models/main.pt model/threshold.json study/results.json "$W"/models/diag_v2/*.pt
"$P" scripts/study_report.py
