# Prior-art check (2026-10-07)

**Question.** Has anyone measured how electrode interchanges change the output of deep-learning ECG
*diagnosis* models, and whether detecting and undoing the interchange restores the diagnosis?

**Searches.** PubMed E-utilities, the arXiv API and web search. Semantic Scholar did not answer.

PubMed queries and hit counts:

| query | hits |
|---|---|
| "lead reversal" deep learning diagnosis | 0 |
| electrode misplacement artificial intelligence ECG interpretation | 4 |
| "lead misplacement" robustness neural network ECG | 0 |
| ECG classifier robustness lead swap | 0 |
| "electrode interchange" myocardial infarction misclassification | 0 |
| ECG deep learning electrode misplacement diagnostic accuracy | 2 |
| lead reversal automated ECG interpretation error | 1 |

arXiv returned nothing on topic for any of these queries.

**Closest work** (abstracts read):
- **Bond, Finlay, Al-Zaiti, Macfarlane**, *J Electrocardiol* 2021;69S:1-6. A commentary calling for
  guidelines and best practices for "stress testing" ECG machine-learning algorithms. It is a call, with
  no measurement.
- **Rafiei, ..., Clifford**, *J Electrocardiol* 2026;95:154204. A personalised, uncertainty-aware method
  for *detecting and estimating* lead misplacement in reduced-lead devices. Detection, not the effect on
  diagnosis models.
- **Jekova et al.**, *Comput Methods Programs Biomed* 2016;134:31-41. Inter-lead correlation rules that
  *detect* cable reversals.
- **Hedén et al.** 1995 (PMID 7733003), and a 2020 systematic review of machine learning for misplacement
  detection (PMID 32866909). Detection only.
- **JMIR Med Inform 2021** (doi 10.2196/25347). Deep-learning *detection* of misplaced V1/V2 electrodes.
  It cites misplacement causing wrong diagnoses in 17-24 % of patients (clinical readers).

**Conclusion.** These searches found no study that measures the effect of electrode interchanges on
deep-learning diagnosis models, or a detect-and-correct pipeline in front of one. The wording used
anywhere else is "no study found in these searches", not "first".

## Update after an adversarial review (2026-10-08)

A second search found work the first search missed. Abstracts were read for all of the following:

- **Pandey, Reyna, Koscova, Clifford, Sameni**, "Detecting and Correcting Electrocardiogram Limb-Lead
  Electrode Swapping via Deep Learning", CinC 2026 (accepted, preliminary programme):
  https://cinc.org/2026/Program/accepted/280.html
  - Deep-learning detection and typing of the six limb swaps, trained on synthetic swaps.
  - Closed-form correction.
  - AUROC 0.99 on synthetically swapped PTB-XL; flags 4.34 % of unmodified PTB-XL.
  - **The detect-and-correct idea is therefore concurrent work, not new here.** It does not measure the
    effect on diagnosis models.
- **Cao, Shi, Wu**, "Reversal No Longer Matters: Attention-Based Arrhythmia Detection With Lead-Reversal ECG
  Data", ICASSP 2020: https://rc.signalprocessingsociety.org/conferences/icassp-2020/spsicassp20vid0773
  - States that lead reversals cause a significant accuracy decline in automated arrhythmia algorithms
    (no figures in the abstract), and proposes a reversal-robust network.
  - Arrhythmia only, not MI.
- **Detection only, no effect on diagnosis models:**
  - Huang et al., Physiol Meas 2024 (PMID 38663434): lightweight deep-learning detection of misplacement.
  - de Bie et al., J Electrocardiol 2014 (PMID 25213624): pre-acquisition warning system, 90 % average
    sensitivity at 99.8 % specificity, but 22 % for LA-LL.
  - Hedén et al., Am J Cardiol 1996 (PMID 8806356): neural networks for lead reversal.
  - Bukhari et al., Comput Biol Med 2025 (PMID 40695024): artefact and misplacement detection in UK
    Biobank and CRIC.

**Revised claim.** That electrode reversals degrade automated ECG analysis is known, and deep-learning
detect-and-correct exists concurrently. These searches found no study that, per interchange type,
measures how often deep-learning diagnosis models on PTB-XL newly call or lose myocardial infarction at a
fixed operating point, or that compares a detect-and-correct guard with swap-augmented training. That
narrower measurement is this study's contribution.
