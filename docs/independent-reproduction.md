# Independent reproduction (version 2)

This is the procedure an **outside researcher** should follow. The authors' own run of `python -m bench.reproduce` is a first-party self-check (`independent: false` in `backend/artifacts/reproduction-selfcheck.json`). It does not satisfy RQ4.

## What to receive

A git tag or commit SHA for Panoptes v1.0, plus (if you cannot fetch Defactify yourself) the signed pointer manifest `datasets/manifests/defactify-text.json`. You do **not** need raw community text. You do need network access if you want to re-fetch Defactify.

## Commands

```bash
git clone https://github.com/marketstandard/Panoptes.git
cd Panoptes
git checkout <v1-tag>
python -m pip install -e "backend[dev]"
python -m pip install pandas pyarrow
python baselines/baseline.py verify-catalog
python -m bench.validate_submission bench/protocol.json
python -m bench.reproduce
python -m pytest backend bench baselines
python -m bench measure --data corpus
```

Optional, large:

```bash
python -m bench.fetch_defactify --check
python -m bench measure --data defactify
```

## What to report

Fill this table. `original` is the number in the v1 paper / signed card. `independent` is the number you recomputed. `delta` is the absolute difference.

| Metric | Artifact | Original | Independent | \|delta\| |
| --- | --- | --- | --- | --- |
| Corpus n | corpus-summary.json | 104 |  |  |
| Protocol heuristic AUROC | cards/measurement-protocol.json | 0.624 |  |  |
| Protocol logistic AUROC | cards/measurement-protocol.json | 0.686 |  |  |
| Mixture slope (AI prefix) | cards/measurement-protocol.json | 0.029 |  |  |
| Unknown-rejection AUROC | cards/measurement-protocol.json | 0.515 |  |  |
| Shipped heuristic on Defactify AUROC | cards/defactify-external-validation.json | 0.648 |  |  |
| Panoptes-v0 in-domain AUROC | panoptes-v0-card.json | 0.998 |  |  |
| Panoptes-v0 → corpus AUROC | panoptes-v0-card.json cross_domain | 0.471 |  |  |
| Catalog runs verified | verify-catalog | 12 |  |  |

Open a pull request that adds `docs/reproductions/<your-lab>-reproduction/` with the filled table and the SHA of the commit you checked out. Do not commit raw Defactify text.

## What not to do

- Do not retune calibration on the test groups.
- Do not treat a hash match as a metric match; recompute the scores.
- Do not blend watermark tests into the passive-attribution numbers.
