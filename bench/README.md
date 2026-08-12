# Panoptes bench

The training and test bench for the Panoptes corpus. Community members use it to

- reproduce the project's reference models on the hash-verified corpus,
- evaluate their own datasets against shipped models,
- contribute datasets as signed hash-pointer manifests (raw data never leaves your machine),
- train and inspect **Panoptes-v0**, the project's evidential neural detector.

Everything here is local-first. The corpus is re-hashed against its run manifests on every load — tampered data is rejected, not silently consumed.

## Setup

```bash
pip install -e backend[dev]   # repo root; provides panoptes + scientific stack
pip install pandas pyarrow    # Defactify loader only (parquet); the runtime never imports these
# optional neural tier (Panoptes-v0):
pip install torch             # CUDA build recommended; CPU works
```

## Commands

```bash
python -m bench train --model logistic --data corpus      # tier-0 baseline
python -m bench train --model gbm --data corpus           # tier-1 (needs n>=300)
python -m bench train --model panoptes-v0 --data corpus   # tier-2 (needs torch)
python -m bench train --model logistic --data defactify   # any tier on the Defactify bench
python -m bench external-validate --data defactify        # shipped runtime vs a dataset it never saw
python -m bench attribute --data defactify                # exploratory 7-class source attribution
python -m bench evaluate --model models/logistic-tier0/model.pkl
python -m bench validate --dataset your.csv               # your data vs shipped model
python -m bench contribute --dataset your.csv --name my-dataset
python -m bench predict --model models/logistic-tier0/model.pkl --text "..."
```

`--data` accepts `corpus` (the hash-verified project corpus), `defactify` (the
Defactify_Text_Dataset; see below), or a path to your own CSV/JSONL.

## The Defactify bench dataset

`--data defactify` loads the [Defactify_Text_Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Text_Dataset)
(Roy et al. 2026, arXiv:2510.22874): NYT human articles vs single-prompt rewrites by six LLM
families. Fetch it first:

```bash
python research/fetch_defactify.py
```

The fetcher downloads the three parquet splits from Hugging Face, verifies each against a pinned
SHA-256, applies hygiene filters, and writes clean parquet to `datasets/local/defactify/`
(gitignored) plus a signed pointer manifest to `datasets/manifests/defactify-text.json`.

**Hygiene filters** (per-split counts in `datasets/local/defactify/fetch-manifest.json`):

- **error artifacts** — rows whose text is an API/client error string (the upstream data contains
  "Error communicating with OpenAI…" rows labeled as GPT-4o output): 412 dropped;
- **exact duplicates** — SHA-256 of normalized text: 73 dropped;
- **under 50 word-tokens** — the runtime support threshold: 957 dropped.

73,193 raw rows become **71,666** clean records (9,483 human, 62,183 AI).

**Story groups and the leakage audit.** Each human story has up to six AI near-rewrites, and the
upstream splits are not documented as story-disjoint. The loader reconstructs story groups at load
time — TF-IDF (word 1–2 g, min-df 2) cosine nearest-neighbor connected components at threshold
0.45 → 61,006 groups — and audits the official splits: **1,236 of 10,769 official test rows
(11.5%) share a story with official train**. All bench metrics therefore use story-grouped
GroupKFold, never the official splits. Group reconstruction is cached under
`datasets/local/defactify/story-groups-t*.json` (keyed by dataset hash; delete to rebuild).

**Attribution experiment** (`python -m bench attribute --data defactify`): exploratory 7-class
source attribution (human + six LLM families) with two contenders — multinomial logistic on the
stylometric vector and a K=7 Dirichlet variant of Panoptes-v0 — evaluated out-of-fold with
macro-F1, per-family F1, and a confusion matrix, against the 5–9% attribution accuracy Roy et al.
report for their baselines. The signed card (`backend/artifacts/cards/attribution-defactify.json`)
also derives binary detection metrics as P(AI) = 1 − P(human). `--skip-dirichlet` runs the
logistic contender only.

**External validation** (`python -m bench external-validate --data defactify`): scores the dataset
with the *shipped* runtime — heuristic raw score plus corpus-fitted isotonic calibration — which
never saw Defactify. The signed benchmark card
(`backend/artifacts/cards/defactify-external-validation.json`) records the domain-shift cost
(AUROC 0.648) as a first-class artifact.

## The tier gate

Models are admitted by statistical power, not hype:

| Tier | Model | Requirement |
|---|---|---|
| 0 | Penalized logistic regression | always |
| 1 | Gradient boosting | n ≥ 300 |
| 2 | Neural (Panoptes-v0) | power gate: n ≥ ~3,140 to detect a 5-point gain at 80% power |

Below the gate, neural results are labeled **exploratory** on the model card. The gate rationale is computed, not asserted, and is printed at train time and stored on every card.

## Evaluation protocol

All metrics are **out-of-fold** under `GroupKFold` by prompt group, so a topic never appears in both train and test. Reported: AUROC (with bootstrap 95% CI), Brier, ECE, TPR at 1%/5% FPR, reliability bins, coverage-vs-abstention curve, split-conformal set behavior, and fairness slices by length bucket, kind, and family.

## Contributing a dataset

1. Format your data as CSV or JSONL matching `schemas/bench-dataset.schema.json` (required: `text`, `label`; optional: `family`, `kind`, `group`).
2. `python -m bench validate --dataset your.csv` to see how shipped models do on it.
3. `python -m bench contribute --dataset your.csv --name my-dataset` writes a signed pointer manifest to `datasets/manifests/my-dataset.json` containing **only hashes and counts**.
4. Open a PR adding that manifest. Your raw text never enters the repository.

## Panoptes-v0

An evidential deep-learning detector (Dirichlet evidence head) that natively produces **vacuity** and **dissonance** uncertainty, driving the SUPPORTED/INSUFFICIENT evidence states. See `research/findings/panoptes-v0.md` for the full iteration log. Weights are local (`models/panoptes-v0/` and `models/panoptes-v0-defactify/`, gitignored); **open weights on Hugging Face are coming soon**.

On the Defactify bench the power gate passes (n=71,666 ≥ 3,140), so the character-sequence branch trains: out-of-fold AUROC 0.998 (95% CI 0.998–0.999), ECE 0.019 under story-grouped GroupKFold. The Defactify-trained card (`backend/artifacts/panoptes-v0-card.json`) also carries a `cross_domain` block (the Defactify ensemble scoring the 104-record project corpus) and a `corpus_trained` block preserving the previous sub-gate iteration for comparison.
