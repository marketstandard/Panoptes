# Panoptes-v0 iteration log

Every training run is appended with its config, data hash, metrics, statistical tests,
the decision taken, and the next refinement. Newest last.

## Iteration 1 — 2026-08-12T20:43:37Z

**Config**: evidential MLP (feature branch only; sequence branch gated OFF), Dirichlet evidence head, AdamW lr 3e-4 wd 1e-2, batch 16, <=200 epochs, early stop patience 20 on grouped-val ECE, seeds {13, 42, 87}.

**Data**: panoptes-verified-corpus, n=104, sha256 `a1d4c0c46d5bb57674932aef71e0185488fad172426d561d2c1077c5f6d03d9b`.
**Device**: cuda. **Power gate**: n=104 >= 3140 required (two-proportion worst case, mde=0.05, alpha=0.05, power=0.8): gate FAILS.

**Out-of-fold metrics by seed** (GroupKFold by prompt):

| Seed | AUROC | Brier | ECE | TPR@1%FPR |
|---|---|---|---|---|
| 13 | 0.751 | 0.075 | 0.081 | 0.229 |
| 42 | 0.712 | 0.076 | 0.091 | 0.135 |
| 87 | 0.730 | 0.076 | 0.080 | 0.438 |
| **mean** | **0.754** | **0.075** | **0.084** | **0.365** |

AUROC 95% CI (ensemble OOF): [0.608, 0.887]

**Comparison battery** (BH-corrected):

| Pair | Test | Statistic | p | q | Significant |
|---|---|---|---|---|---|
| panoptes-v0 vs logistic-tier0 | mcnemar | 1.000 | 0.3750 | 0.5572 | no |
| panoptes-v0 vs logistic-tier0 | delong | -0.732 | 0.4644 | 0.5572 | no |
| panoptes-v0 vs heuristic | mcnemar | 1.000 | 0.2188 | 0.5572 | no |
| panoptes-v0 vs heuristic | delong | 0.823 | 0.4106 | 0.5572 | no |
| logistic-tier0 vs heuristic | mcnemar | 1.000 | 1.0000 | 1.0000 | no |
| logistic-tier0 vs heuristic | delong | 1.385 | 0.1661 | 0.5572 | no |

Conformal (alpha=0.1): empirical coverage 0.923, abstention rate 0.000.

**Decision**: ship as *exploratory* — the corpus is below the neural power gate, so no comparative claim is made. Weights saved locally with SHA-256 sidecars; Hugging Face release pending a corpus that passes the gate.

**Next refinements**: (1) grow human controls and community datasets until the gate passes; (2) enable the char-sequence branch and re-run this battery; (3) per-kind (code) evidential head once code controls exist.

## Iteration 2 — 2026-08-12T21:45:39Z

**Config**: evidential MLP (char-sequence branch ENABLED (power gate passed)), Dirichlet evidence head, AdamW lr=3e-4 wd=1e-2 batch=512, <=40 epochs, patience 6 on grouped-validation ECE, seeds [13, 42, 87].

**Data**: defactify-text (Roy et al. 2026, arXiv:2510.22874; hygiene-filtered, hash-pinned), n=71666, sha256 `2669f1cae481b304c05cd8a9c95501bf6109774b8f12cb6c30a34618257fe65d`.
**Device**: cuda. **Power gate**: n=71666 >= 3140 required (two-proportion worst case, mde=0.05, alpha=0.05, power=0.8): gate passes.

**Story groups**: 61006 reconstructed (threshold 0.45, mean size 1.17, max 1401). Official-split leakage audit: 1236/10769 test rows share a story with train (11.5%).

**Out-of-fold metrics by seed** (GroupKFold by story group):

| Seed | AUROC | Brier | ECE | TPR@1%FPR |
|---|---|---|---|---|
| 13 | 0.998 | 0.009 | 0.018 | 0.973 |
| 42 | 0.998 | 0.009 | 0.018 | 0.973 |
| 87 | 0.998 | 0.009 | 0.018 | 0.969 |
| **mean** | **0.998** | **0.009** | **0.019** | **0.973** |

AUROC 95% CI (ensemble OOF): [0.998, 0.999]

**Comparison battery** (BH-corrected):

| Pair | Test | Statistic | p | q | Significant |
|---|---|---|---|---|---|
| panoptes-v0 vs logistic-tier0 | mcnemar | 560.462 | 0.0000 | 0.0000 | yes |
| panoptes-v0 vs logistic-tier0 | delong | 14.860 | 0.0000 | 0.0000 | yes |
| panoptes-v0 vs heuristic | mcnemar | 8083.895 | 0.0000 | 0.0000 | yes |
| panoptes-v0 vs heuristic | delong | 75.451 | 0.0000 | 0.0000 | yes |
| logistic-tier0 vs heuristic | mcnemar | 6870.393 | 0.0000 | 0.0000 | yes |
| logistic-tier0 vs heuristic | delong | 71.518 | 0.0000 | 0.0000 | yes |
| panoptes-v0 vs gbm-tier1 | mcnemar | 0.351 | 0.5538 | 0.5538 | no |
| panoptes-v0 vs gbm-tier1 | delong | -3.242 | 0.0012 | 0.0013 | yes |
| gbm-tier1 vs logistic-tier0 | mcnemar | 550.305 | 0.0000 | 0.0000 | yes |
| gbm-tier1 vs logistic-tier0 | delong | 15.010 | 0.0000 | 0.0000 | yes |

Conformal (alpha=0.1): empirical coverage 0.900, abstention rate 0.000.

**Cross-domain** (Defactify-trained -> project corpus, n=104): AUROC 0.471 (95% CI 0.282-0.708), Brier 0.525.

**Decision**: ship as the *comparative* Defactify-trained model — the power gate passes, the sequence branch is enabled, and the battery above is statistically licensed. Weights saved locally with SHA-256 sidecars; Hugging Face release pending.

**Next refinements**: (1) close the domain gap measured in cross_domain with more human controls; (2) per-kind (code) evidential head once code controls exist; (3) publish weights with the signed card once the release checklist clears.
