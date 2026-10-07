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
