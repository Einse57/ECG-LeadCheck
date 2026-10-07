# Evaluation plan (written 2026-10-07, before any model was trained or fold 10 was read)

Data: PTB-XL 1.0.3, 100 Hz. Official stratified folds: train on 1-8, choose thresholds on 9, test on 10.
Every interchange class is applied to every fold-10 ECG, so each class is tested on the same 2,198
recordings. Limb leads III, aVR, aVL and aVF are recomputed from I and II for every record, swapped
or not, so a model cannot detect "recomputed" leads instead of physiology.

## Operating point

One alarm threshold on the "any interchange" score, set so that 1 % of the unmodified fold-9 ECGs are
flagged (99 % specificity). The same rule is used for every model.

## Targets for the CNN on fold 10

| group | classes | sensitivity at the fold-9 threshold |
|---|---|---|
| exact limb interchanges | LA/RA, RA/LL, both rotations | >= 95 % pooled |
| hardest limb interchange | LA/LL | >= 80 % (reported separately) |
| right-leg interchanges (approximate simulation) | RA/RL, LA/RL | >= 97 % pooled |
| adjacent precordial swaps | V1/V2 ... V5/V6 | >= 90 % pooled |
| type | flagged interchanged ECGs given the correct type | >= 90 % |
| specificity | unmodified fold-10 ECGs not flagged | reported, expected near 99 % |

## Baseline

A gradient-boosting model on inter-lead correlations and lead amplitudes, in the spirit of the
correlation method of Jekova et al. 2016 (Comput Methods Programs Biomed 134:31-41), trained on the
same folds and scored with the same threshold rule. Reference values from that paper, on a different
test set: limb 95.5-96.9 %, adjacent precordial 93.8-97.8 %, specificity > 99 %.

## Decision rule

- The CNN is reported against the baseline on identical data. If it does not beat the baseline's
  pooled sensitivity by at least 3 points, the README says the simpler model is as good.
- Real-world probes (`data/probes/real_world.json`, records whose reports mention interchanged
  electrodes) are scored by a model that never trained on their fold. No pass/fail: the reports are
  suspicions, not confirmed labels.
- Scan for unreported interchanges: out-of-fold scores only. A flagged record counts as "consistent
  with LA/RA interchange" only if lead I shows a negative P wave and negative QRS while precordial
  R-wave progression is normal (which argues against dextrocardia). Results are stated as "flagged,
  not confirmed by a cardiologist", by record ID only.
