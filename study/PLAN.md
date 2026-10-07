# Study plan: do swapped electrodes fool ECG-AI, and does detect-and-correct fix it?

Written 2026-10-07, before any diagnosis model was trained.

## Diagnosis models

- **Task.** The standard PTB-XL superdiagnostic task: NORM, MI, STTC, CD, HYP, multi-label. Labels are
  aggregated from `scp_statements.csv` codes with `diagnostic == 1`, as in PTB-XL's example code.
- **Folds.** Train on 1-8, set thresholds on 9, test on 10. Training uses only unmodified recordings.
- **Architectures.** Two: the 1D ResNet used for LeadCheck, and an InceptionTime-style 1D network.
  Two seeds each.
- **Sanity bar.** Clean macro-AUROC of at least 0.90 on fold 10. Published benchmarks are about 0.93.
- **Decision thresholds.** Per class, chosen on clean fold 9 by maximising Youden's J.

## Interchanges

All 12 classes from `src/leadcheck/swaps.py`, applied to every fold-10 recording. LA/RA, the most
common real-world interchange, is reported first.

## Primary endpoints, per interchange, on fold 10

1. **Flip rate.** The share of recordings whose set of predicted classes changes relative to the same
   model's clean prediction. Comparing to the model's own clean output isolates the interchange from label noise.
2. **False MI.** Among recordings labelled NORM only, the share newly called MI.
3. **Missed MI.** Among MI-labelled recordings correctly called MI when clean, the share no longer called MI.
4. **Macro-AUROC drop** against the true labels.

Reporting: counts beside every percentage, with paired bootstrap 95 % CIs (1,000 resamples of recordings).

## Defences, compared on the same recordings

- **None:** the diagnosis model alone.
- **Detect and correct:** LeadCheck flags the recording at its fold-9 threshold, and the predicted
  interchange is inverted when it is an exact class. RL interchanges are routed to "re-record", because
  information is lost. Unflagged recordings pass unchanged.
- **Swap-augmented training:** a diagnosis model trained with random interchanges and the true labels.

## Costs, reported explicitly

- Clean recordings flagged by LeadCheck (about 1 %) that get "corrected" into a swapped recording.
- Wrong-type corrections.
- The net change in diagnostic accuracy on clean recordings.

## What counts as the finding

The harm measurement under "none" is the finding. Detect-and-correct is the engineering response;
for exact interchanges its recovery is bounded by LeadCheck's detection and typing accuracy, and that
is stated plainly.

## Amendment 1 (2026-10-07, after the first run and an independent review, before re-running)

The first run is archived unchanged in `study/archive_v1/`. A methodological review found no leakage and no
metric error, but identified the problems below. These changes are fixed before the second run.

1. **Training selection.** The diagnosis models are saved at the epoch with the best fold-9 macro-AUROC
   (fold 10 still untouched). In the first run the epoch-40 checkpoints were past their fold-9 peak, and
   one model (ResNet, seed 0) reached 0.8998 clean macro-AUROC, just under the 0.90 sanity bar.
2. **Seeds.** Training augmentation is seeded per data-loader worker, and the swap-augmented arm gets two
   seeds per architecture, like the standard arm.
3. **Operating points.**
   - The threshold-free macro-AUROC drop is the primary endpoint.
   - The Youden thresholds stay as pre-registered.
   - A second operating point is added: per class, the threshold giving 95 % specificity on clean fold 9.
     Flip, false-MI and missed-MI rates depend strongly on the threshold, so both are reported.
4. **Null arm.** A benign perturbation (Gaussian noise of 0.01 mV plus a 30 ms shift) calibrates how much a
   flip rate means.
5. **Statistics.**
   - Every proportion is reported as x/n with a Wilson 95 % interval.
   - The headline differences (AUROC drop under LA/RA, false MI without vs with LeadCheck) get paired
     bootstrap intervals that resample patients, not recordings.
   - Rows with fewer than 30 records are not reported.
6. **Reporting.**
   - Interchange typing accuracy and wrong-type corrections.
   - Positive predictive value of a LeadCheck flag at 0.5, 2 and 5 % interchange prevalence.
   - Provenance (git commit, library versions, checksums of checkpoints and data).
   - Right-leg rows labelled as approximate simulations.
