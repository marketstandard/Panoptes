# Methodology report — baseline corpus

Generated 2026-08-12T20:28:24Z from 104 hash-verified documents (8 human, 96 AI) across families: claude-opus-5-max, gemini-3.1-pro, glm-5.2-max, gpt-5.6-sol-max, grok-4.6-extra-high, human, kimi-k3-max.

All p-values adjusted with Benjamini-Hochberg within the hypothesis registry. Decisions use q <= 0.05.

## Variable selection (multicollinearity screen)

| Feature | VIF | Verdict |
|---|---|---|
| long_words | 1.66 | keep |
| connectors | 1.09 | keep |
| unique_ratio | 2.71 | keep |
| short_sentences | 6.67 | investigate |
| structured | 4.39 | keep |
| digits | 3.29 | keep |
| balanced_lines | 2.83 | keep |
| token_entropy | 4.33 | keep |
| word_length_var | 2.64 | keep |
| line_sd | 3.06 | keep |
| sentence_len_mean | 7.20 | investigate |
| sentence_len_sd | 1.63 | keep |
| log_words | 405.38 | **excluded** |
| hapax_ratio | 39.91 | **excluded** |
| punctuation | 18.59 | **excluded** |
| mean_word_length | 14.86 | **excluded** |
| avg_line_len | 14.62 | **excluded** |

Condition number (standardized, retained set): 7.6

### Exclusion justifications

- **log_words**: VIF 405.4 exceeds 10; the feature is a near-linear combination of the retained set and its coefficient would be uninterpretable (variance inflation).
- **hapax_ratio**: VIF 39.9 exceeds 10; the feature is a near-linear combination of the retained set and its coefficient would be uninterpretable (variance inflation).
- **punctuation**: VIF 18.6 exceeds 10; the feature is a near-linear combination of the retained set and its coefficient would be uninterpretable (variance inflation).
- **mean_word_length**: VIF 14.9 exceeds 10; the feature is a near-linear combination of the retained set and its coefficient would be uninterpretable (variance inflation).
- **avg_line_len**: VIF 14.6 exceeds 10; the feature is a near-linear combination of the retained set and its coefficient would be uninterpretable (variance inflation).

## Hypothesis tests (pre-registered)

| ID | Test | Statistic | p | q (BH) | Effect | 95% CI | Null decision |
|---|---|---|---|---|---|---|---|
| H1 | welch_t | 1.000 | 0.1599 | 0.1919 | cohens_d = 0.1058 | [-2.941e-05, 8.91e-05] | **not rejected** |
| H2 | mann_whitney | 188.000 | 0.0085 | 0.0513 | rank_biserial = 0.5104 | [0.2031, 0.7526] | **not rejected** |
| H3 | logistic_lr | 5.510 | 0.0189 | 0.0567 | odds_ratio = 1.85e-05 | [2.061e-09, 1.539] | **not rejected** |
| H4 | permutation_manova | 0.524 | 0.0925 | 0.1387 | wilks_lambda = 0.5244 | — | **not rejected** |
| H5 | welch_t | 2.010 | 0.0320 | 0.0640 | cohens_d = 0.5475 | [-0.002966, 0.09218] | **not rejected** |
| H6 | durbin_watson_permutation | 2.558 | 0.2884 | 0.2884 | durbin_watson = 2.558 | [0, 4] | **not rejected** |

- **H1**: Connector rate (however/therefore/moreover/additionally/overall/furthermore per token) is higher in AI-generated text than in human text.
- **H2**: Token entropy (Shannon entropy over word tokens) is lower in AI-generated text than in human text.
- **H3**: Unique-token ratio discriminates AI from human text beyond document length.
- **H4**: Per-family feature centroids (connectors, unique_ratio, long_words, token_entropy) are separated across source families, including human.
- **H5**: Long-word rate (fraction of tokens with >= 7 characters) is higher in AI prose than in human prose.
- **H6**: Segment-level detector residuals are uncorrelated within documents (the segment-independence assumption holds).

## Specification tests (binary logistic model)

Model: penalized logistic regression (IRLS, ridge 1e-6) on 12 screened, standardized features.

| Test | Statistic | p | Verdict |
|---|---|---|---|
| link_test | 0.229 | 0.6325 | adequate |
| reset_test | 4.667 | 0.0970 | adequate |
| hosmer_lemeshow | 4.365 | 0.8228 | adequate |
| breusch_pagan | 14.019 | 0.2995 | adequate |
| jarque_bera | 469.437 | 0.0000 | **concern** |

- Durbin-Watson (ingestion-ordered residuals): 2.087 (2.0 = no serial correlation)
- Cook's distance: max 0.2445; 11 points above 4/n
- Pseudo-R^2: McFadden 0.358, Tjur 0.236

Artifact SHA-256: `48eebe82333dc09322899200b10034e4bb2f11130b9dec199089cca88d603779`
