# Evaluation methodology

Panoptes evaluation must answer four questions separately:

1. Does the generic detector distinguish human-written and AI-generated prose?
2. Does the code detector distinguish human-written, AI-refined, and AI-generated code?
3. Do known watermark tests control false positives and retain power at realistic lengths?
4. Does source-family attribution reject unknown generators instead of forcing a label?

## Data discipline

- Keep prompt, domain, model, and language groups separated across train/calibration/test splits.
- Prefer group-aware cross-validation or explicit held-out groups.
- Do not tune release thresholds on test data.
- Do not bundle datasets with unclear redistribution terms.
- Record dataset version, detector version, calibration bundle, feature schema, and artifact hash.

## Metrics

Binary detection:

- AUROC;
- AUPRC;
- Brier score;
- expected calibration error;
- TPR at fixed 1% and 5% FPR;
- minimum token count for useful power.

Source attribution:

- top-1 and top-3 conditional accuracy among supported families;
- open-set rejection AUROC;
- conformal coverage;
- unknown rate for truly unseen generators.

Watermark tests:

- empirical false-positive rate on unwatermarked controls;
- true-positive rate versus eligible token count;
- retention after 10%, 25%, and 50% edits;
- segment-localization error;
- q-value calibration under multiple testing.

Code robustness:

- formatter-only changes;
- identifier renaming;
- comment removal;
- dead-code insertion;
- AST-preserving transformations;
- cross-language performance.

## Current status

The shipped calibration artifact (`backend/artifacts/baseline-calibration.json`) is fitted on the verified reference corpus — 104 hash-verified records (96 AI outputs across 6 model families, 8 human controls) — via isotonic regression with grouped cross-validation, corpus-fitted source-family geometry, and conformal thresholds. The methodology report (`backend/artifacts/methodology-report.json`) records the VIF feature screening, pre-registered hypothesis tests (H1–H6) with Benjamini–Hochberg q-values, and the econometric specification battery. The synthetic development artifact remains reproducible via `python research/calibration.py --synthetic` for pipeline testing.

Honest statistical caveats: at n=104 the corpus supports calibration and tier-0/tier-1 modeling, but hypothesis tests are underpowered for small effects (≈23% power for a d=0.5 two-group difference), and the neural tier (Panoptes-v0) is gated accordingly. Growing the corpus through community baseline submissions directly increases what the methodology can conclude. A release-quality evaluation should additionally incorporate larger external corpora (e.g. RAID-derived) through the signed pointer-manifest mechanism.
