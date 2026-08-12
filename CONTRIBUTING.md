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
