# Contributing to Panoptes

Thanks for contributing. Panoptes is an evidence workbench, so correctness and restraint matter more than impressive-sounding claims.

## Start here

1. Read `docs/system-design.md` and `docs/interpretation.md`.
2. Run the fixture profile before changing statistical code.
3. Add or update tests for every behavior change.
4. Keep the analysis response schema versioned.
5. Do not present a detector score as proof of authorship.

## Development setup

```bash
python -m pip install -e backend[dev]
npm --prefix frontend install
python -m pytest backend/tests
npm --prefix frontend test
```

## Pull request checklist

- [ ] I updated the JSON Schema if the response changed.
- [ ] I added backend or frontend tests.
- [ ] I explained scientific limitations in user-facing language.
- [ ] I did not log raw submitted text.
- [ ] I preserved abstention behavior for unsupported domains.
- [ ] I updated `NOTICE` for any new dataset, model, or substantial dependency.
- [ ] I did not commit raw baseline outputs outside `baselines/reference/` (catalog entries are hashes only).

## Coding standards

- Python: Ruff-clean, typed public interfaces, no hidden network calls.
- TypeScript: strict mode, accessible controls, no unsafe HTML rendering.
- Statistics: state assumptions, cohorts, uncertainty, and failure modes.
- Documentation: plain-language summary first, technical detail second.

## Contribution ladder

- Reporter: file reproducible issues with fixture or sanitized input.
- Contributor: submit tested documentation, UI, API, detector, or calibration fixes.
- Reviewer: review statistical interpretation, security, accessibility, and developer experience.
- Maintainer: approve releases, schema changes, detector registry changes, and calibration bundles.

## Marker and calibration data contributions

Panoptes accepts marker/watermark evidence through signed pointer manifests and hashed artifacts, not raw third-party corpora.

- Add a dataset pointer manifest under `datasets/manifests/` when introducing a new cohort.
- Add contributor artifacts under `docs/reproductions/<contribution-id>/`.
- Keep raw text out of git unless it is a tiny license-clear fixture under `fixtures/`.
- Validate schemas and hashes with:

```bash
python -m bench.validate_submission path/to/artifact.json
```

See `docs/contributing-markers.md` for the full workflow and acceptance checklist.

## Watermark-removal and external-system evaluations

- To run or extend the watermark-removal robustness evaluation (attack battery, retention metrics), see `docs/watermark-removal.md`.
- To evaluate an external watermark scheme, remover, or detector straight from its git repository — or to contribute a `panoptes.adapter.json` / `panoptes_adapter.py` for one — see `docs/testing-external-repos.md`. Only run repositories you trust; the harness executes cloned code in a subprocess.

## Baseline run submissions

You can also contribute known-model baseline outputs for the canonical prompt set:

1. Run the prompts (manual, API, or agent-assisted) and `finalize` the run — see `baselines/README.md`.
2. Optionally `anchor` the manifest with OpenTimestamps.
3. `submit` the run and open a PR containing **only** the new `baselines/catalog/registry.jsonl` line and the manifest under `baselines/catalog/manifests/`.

Raw model outputs live in `baselines/runs/`, which is gitignored: the catalog carries hashes only. The one exception is the maintainer-curated `baselines/reference/` set.
