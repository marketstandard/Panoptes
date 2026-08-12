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

The shipped calibration artifact (`backend/artifacts/baseline-calibration.json`) is fitted on the verified reference corpus — 104 hash-verified records (96 AI outputs across 6 model families, 8 human controls) — via isotonic regression with grouped cross-validation, corpus-fitted source-family geometry, and conformal thresholds. The methodology report (`backend/artifacts/methodology-report.json`) records the VIF feature screening, pre-registered hypothesis tests (H1–H6) with Benjamini–Hochberg q-values, and the econometric specification battery, per cohort (`cohorts.corpus` and `cohorts.defactify`). The synthetic development artifact remains reproducible via `python research/calibration.py --synthetic` for pipeline testing.

Honest statistical caveats: at n=104 the corpus supports calibration and tier-0/tier-1 modeling, but hypothesis tests are underpowered for small effects (≈23% power for a d=0.5 two-group difference), and on that corpus the neural tier (Panoptes-v0) is gated accordingly. Growing the corpus through community baseline submissions directly increases what the methodology can conclude.

### Defactify external validation (n=71,666)

The release-quality external evaluation now exists: the bench runs on the Defactify_Text_Dataset (Roy et al. 2026, arXiv:2510.22874), fetched and hygiene-filtered by `research/fetch_defactify.py` (73,193 raw → 71,666 clean: 412 API-error artifacts, 73 exact duplicates, and 957 sub-50-token texts dropped). Raw text stays local and gitignored; a signed pointer manifest lives at `datasets/manifests/defactify-text.json`.

- **Leakage audit.** TF-IDF story-group reconstruction (cosine threshold 0.45) finds 61,006 groups; 11.5% of the official test split shares a story with official train. All reported numbers use story-grouped GroupKFold, not the official splits.
- **Detection (out-of-fold).** Logistic tier-0 AUROC 0.989 (95% CI 0.988–0.990, ECE 0.006); GBM tier-1 AUROC 0.999 (ECE 0.005); Panoptes-v0 with the sequence branch admitted AUROC 0.998 (ECE 0.019). The dataset authors' baselines score 53–58% accuracy.
- **Domain shift.** The shipped runtime (heuristic + corpus-fitted calibration, zero-shot on Defactify) attains AUROC 0.648 (95% CI 0.643–0.652, ECE 0.029) — signed into `backend/artifacts/cards/defactify-external-validation.json`.
- **Hypotheses at full power.** The pre-registered H1–H6 battery re-runs unchanged: H1–H5 nulls rejected (q ≤ 0.0024), H6 (segment autocorrelation) not rejected (q = 1.0).
- **Exploratory attribution.** Seven-class source attribution (human + six LLM families) with multinomial logistic and a K=7 Dirichlet Panoptes-v0 variant; per-family F1 against the authors' 5–9% band in `backend/artifacts/cards/attribution-defactify.json`.
- **Dual calibration.** A second signed artifact (`backend/artifacts/defactify-calibration.json`) fits the isotonic map, reliability bins, conformal threshold, and seven-family Mahalanobis geometry on Defactify. It is opt-in: `PANOPTES_CALIBRATION_BUNDLE=defactify-calibration.json` (allowlist-validated, falls back to the corpus-fitted default).
