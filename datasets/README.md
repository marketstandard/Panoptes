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

## Version 2 datasets (not fetched here)

Pointer manifests for these should be added when hashes and access instructions exist. Do not copy raw text into git.

| Dataset | Why | Status |
| --- | --- | --- |
| RAID (Dugan et al. 2024) | Unseen models, domains, decoding, attacks | Not fetched. Tracked in `docs/`. |
| M4 / M4GT-Bench (Wang et al.) | Multi-domain, multilingual, multi-generator | Not fetched. |
| EvoBench | Generator-generation shift | Not fetched. |
| Multi-author human panel (500–2,000 authors) | Author-disjoint stylometry | Not collected. |
| Recorded human–AI coauthoring | Mixture-rate ground truth | Token-splice proxies only. |

The Defactify pointer is already at `datasets/manifests/defactify-text.json`.
