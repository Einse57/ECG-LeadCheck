# Results (2026-10-07)

The plan and targets are in `GATE.md`, committed before training. All numbers are on PTB-XL fold 10
(2,198 recordings the model never saw). Every interchange was applied to every recording. The alarm
threshold was fixed on fold 9 at 99 % specificity. Raw output: `results/fold10_metrics.json`.

## Sensitivity at the fold-9 threshold

| group | CNN | correlation-feature baseline | target | CNN verdict |
|---|---|---|---|---|
| exact limb (LA/RA, RA/LL, rotations) | **99.6 %** | 96.0 % | 95 % | pass |
| LA/LL | **94.0 %** | 24.2 % | 80 % | pass |
| right leg, approximate simulation (RA/RL, LA/RL) | 100 % | 100 % | 97 % | pass |
| adjacent precordial (V1/V2 ... V5/V6) | **93.0 %** | 78.0 % | 90 % | pass |
| correct type when flagged | **99.5 %** | 92.9 % | 90 % | pass |
| specificity on unmodified fold-10 recordings | 98.9 % | 98.7 % | ~99 % | as expected |
| all interchanges pooled | **96.4 %** | 83.2 % | baseline + 3 | pass (+13.2) |

**Per class (CNN):**

| class | sensitivity | AUROC |
|---|---|---|
| LA/RA | 99.1 % | 0.998 |
| LA/LL | 94.0 % | 0.995 |
| RA/LL | 99.6 % | 0.999 |
| rotation RA>LA>LL | 100 % | 0.999 |
| rotation RA>LL>LA | 99.8 % | 0.999 |
| RA/RL | 100 % | 1.000 |
| LA/RL | 100 % | 1.000 |
| V1/V2 | 97.0 % | 0.997 |
| V2/V3 | 94.9 % | 0.996 |
| V3/V4 | 96.0 % | 0.997 |
| V4/V5 | 88.4 % | 0.992 |
| V5/V6 | 88.5 % | 0.991 |

**Context, not a head-to-head comparison.** Jekova et al. (2016) report, on a different test set,
95.5-96.9 % for limb reversals and 93.8-97.8 % for adjacent precordial swaps at more than 99 %
specificity. The baseline here uses inter-lead correlations in the same spirit but is not a
reimplementation of their rules, and it is much weaker on LA/LL. The fair reading: the CNN is in the
same range as the published method, and clearly better than a correlation-feature model on identical
data.

## Real-world probes (out-of-fold models)

The alarm threshold was 0.970.

| record | what the report says | model |
|---|---|---|
| 1853 | possible V1/V2 lead interchange (named as a differential) | **flagged**: score 0.995, type V1/V2, rank 43 of 21,799 |
| 11641 | arm electrodes possibly interchanged | not flagged; score 0.318, leaning LA/RA, rank 647 |
| 21572 | arm electrodes possibly interchanged | not flagged (0.029) |
| 10646 | electrodes interchanged? (annotator note) | not flagged (0.030) |
| 2646 | tall R in V2; electrode position or old posterior infarct | not flagged (0.024) |

## Scan for unreported interchanges

- **Flag rate.** Out-of-fold scores flagged 208 of 21,799 recordings (0.95 %), which matches the
  expected 1 % false-alarm rate.
- **LA/RA flags.** Five were typed LA/RA. Two meet the pre-registered criteria (records 14144 and
  13160), but both reports describe extreme right-axis deviation with a fascicular block. That is a known
  mimic of LA/RA interchange, and their P-wave inversion in lead I is under 0.02 mV.
- **Conclusion.** There is no evidence of hidden interchanges beyond the false-alarm rate. Nothing here
  is confirmed by a cardiologist.
