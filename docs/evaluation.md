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

The checked-in research scripts generate a synthetic development artifact and benchmark card. These validate the pipeline and artifact formats, but they are not release-quality scientific results. A release must replace them with RAID-derived calibration and a documented code benchmark.
