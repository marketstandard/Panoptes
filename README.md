# Panoptes

Panoptes is a local-first evidence workbench for AI-generated text, source code, public statistical watermarks, source-family similarity, and signed file provenance.

It is designed to be understandable by non-specialists and auditable by technical users. Panoptes does not produce authorship verdicts and should not be used alone for punitive decisions.

## Quick start

Requirements: Python 3.12 and Node.js 20+.

```bash
python -m pip install -e backend[dev]
npm --prefix frontend install
panoptes up --profile fixture
```

Open http://127.0.0.1:8000.

For development with hot reload:

```bash
python -m uvicorn panoptes.main:app --app-dir backend --reload
npm --prefix frontend run dev
```

## CLI

```bash
panoptes doctor
panoptes up --profile fixture
panoptes analyze path/to/input.txt
panoptes fixtures
panoptes models list
panoptes models verify
```

## Tests

```bash
python -m pytest backend/tests
npm --prefix frontend test
npm --prefix frontend run build
```

For local development, run backend tests from `backend/` so the editable package is importable.

## Scientific position

- Generic AI detection is probabilistic and can be wrong.
- A negative watermark test is not evidence of human authorship.
- Source-family similarity is not exact model attribution.
- Signed provenance is not authorship.
- Short, translated, paraphrased, heavily edited, or mixed-origin content can invalidate assumptions.

See `docs/system-design.md`, `docs/math.md`, `docs/interpretation.md`, `docs/evaluation.md`, and `docs/deployment.md`.

## License

MIT. Third-party model and dataset terms are recorded in `NOTICE` and detector registry metadata.
