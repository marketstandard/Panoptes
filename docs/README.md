# Panoptes documentation index

Entry points by question. The paper itself lives at `frontend/public/paper.html`
(served in-app at `/paper.html`).

## Protocol and governance

- [research-protocol-v2-1.md](research-protocol-v2-1.md) — active protocol: claims,
  non-claims, evaluation matrix, watermark-intelligence addendum.
- [research-protocol.md](research-protocol.md) — v2.0 protocol, superseded by v2.1.
- [threat-model.md](threat-model.md) — adversary classes, trust boundaries, plugin
  trust model, watermark contamination threat.
- [contributing-markers.md](contributing-markers.md) — how to contribute baseline
  runs and datasets, including `--watermark` contamination declarations.

## Architecture and evidence

- [system-design.md](system-design.md) — channel separation, plugin loader,
  contamination data-flow, storage layout.
- [math.md](math.md) — feature definitions, watermark statistics (including
  temperature-scaled embedding and inheritance z), calibration math.
- [interpretation.md](interpretation.md) — how to read every channel, including
  contamination flags, radioactivity results, and plugin watermarks.
- [ui-evidence-map.md](ui-evidence-map.md) — schema field → UI surface mapping,
  including the signed research cards.

## Evaluation and reproduction

- [evaluation.md](evaluation.md) — evaluation matrix, external-repo results,
  watermark intelligence cards.
- [independent-reproduction.md](independent-reproduction.md) — how to reproduce
  the signed artifacts from a clean checkout.
- [testing-external-repos.md](testing-external-repos.md) — harness for running
  third-party detectors through Panoptes.
- [benchmark-card.json](benchmark-card.json) — the frozen benchmark card.

## Watermark intelligence

- [watermark-contamination.md](watermark-contamination.md) — contamination threat
  model, manifest fields, operator guidance.
- [watermark-removal.md](watermark-removal.md) — removal robustness evaluation
  (paraphrase / neutralization arms) and related cards.

## Operations

- [deployment.md](deployment.md) — running the backend and frontend.
