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
# optional neural tier (Panoptes-v0):
pip install torch             # CUDA build recommended; CPU works
```

## Commands

```bash
python -m bench train --model logistic --data corpus      # tier-0 baseline
python -m bench train --model gbm --data corpus           # tier-1 (needs n>=300)
python -m bench train --model panoptes-v0 --data corpus   # tier-2 (needs torch)
python -m bench evaluate --model models/logistic-tier0/model.pkl
python -m bench validate --dataset your.csv               # your data vs shipped model
python -m bench contribute --dataset your.csv --name my-dataset
python -m bench predict --model models/logistic-tier0/model.pkl --text "..."
```

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

An evidential deep-learning detector (Dirichlet evidence head) that natively produces **vacuity** and **dissonance** uncertainty, driving the SUPPORTED/INSUFFICIENT evidence states. See `research/findings/panoptes-v0.md` for the full iteration log. Weights are local (`models/panoptes-v0/`, gitignored); **open weights on Hugging Face are coming soon**.
