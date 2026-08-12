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
