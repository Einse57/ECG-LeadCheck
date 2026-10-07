- **Training data.** The model is a 1D ResNet trained on real 12-lead ECGs from PTB-XL folds 1-8.
  Electrode interchanges were simulated from the physics of the leads: limb swaps are recomputed from
  the electrode potentials, and chest swaps exchange channels. No hand labelling was needed.
- **Test set.** PTB-XL fold 10: 2,198 recordings it never saw, each tested under every interchange.
  The alarm threshold gives one false alarm per 100 correctly recorded ECGs.

| interchange | caught |
|---|---|
| limb electrodes (LA/RA, RA/LL, rotations) | 99.6 % |
| left arm / left leg, the hardest | 94.0 % |
| right-leg (ground) interchanges, approximate simulation | 100 % |
| adjacent chest electrodes | 93.0 % |
| correct type named, when flagged | 99.5 % |

- **Baseline.** A correlation-feature baseline on the same data caught 83 % overall, against 96 % for this model.
- **Limits.** Only interchanges are covered; electrodes shifted to the wrong position cannot be simulated
  from 12 leads. Right-leg cases are an approximation. Simulated test cases, not prospective clinical data.
