# Dataset manifests

This directory contains pointer manifests for datasets used in Panoptes calibration and evaluation.

Do not commit raw third-party datasets. Manifests describe how authorized users can obtain data, how splits are grouped, which labels are available, and which hashes identify the exact dataset revision.

## Rules

- `raw_text_in_repo` must be `false`.
- Non-redistributable datasets are pointer-only.
- Small license-clear fixtures belong in `fixtures/`, not here.
- Every new dataset requires a `NOTICE` entry and a maintainer review.
- Group keys must prevent leakage across train/calibration/test splits.

See `docs/contributing-markers.md` for the full workflow.
