# Watermark contamination governance

Panoptes treats production LLM watermarks as a **calibration confound**, not as
authorship evidence. Once providers (notably Anthropic with SynthID-Text under
the EU AI Act) emit watermarked text into the public web and into API baselines,
any corpus trained or calibrated on those outputs can absorb a statistical bias
that later looks like "Claude-like" stylometry or a weak watermark hit.

## Threat model (lineage vs. use)

| Observation | What it can mean | What it does *not* mean |
| --- | --- | --- |
| Declared-active / suspected baseline run | The generator may have watermarked sampling enabled | That a downstream author "used Claude" |
| Positive public-scheme screen on AI text | Demo-key green-list density above chance | That a private vendor key is present |
| Positive public-scheme screen on human controls | Possible corpus integrity problem | Authorship of the control text |
| Student model inherits a watermark (radioactivity) | Training lineage passed through watermarked teacher outputs | Unauthorized distillation without further context |

Web-scale scraping creates a second path: watermarked text enters general
pretraining mixes, so *any* model trained on the open web can become weakly
"radioactive." A detector hit is therefore **lineage-compatible evidence**, not
proof of intent.

## How Panoptes records contamination

1. **Baseline runs** (`schemas/baseline-run.schema.json`) may carry an optional
   `watermark` block: `declared-none`, `declared-active`, `suspected`, or
   `unknown`. Contributors set this with `--watermark` on
   `baselines/baseline.py finalize|run`. Anthropic runs on/after 2026-08-02
   default to `suspected` unless overridden.
2. **Dataset manifests** may declare the same block for external corpora.
3. **Corpus summary** aggregates per-cohort `watermark_status` and lists
   `contaminated_cohorts`. A known-scheme smoke screen
   (`bench/watermark_screening.py`) runs the public KGW demo adapter and the
   Unicode zero-width reference; it cannot read private vendor keys.
   On the frozen 104-record corpus the screen flags 3 of 8 human-control
   records at α = 0.05 under the demo key (marginal z ≈ 2.2–2.3), while the
   92 AI records are symmetric about zero (mean z = 0.02, 10 above and 11
   below |z| > 1.645). We read this as a small-n integrity signal to monitor
   as the corpus grows — the adapter itself is well-calibrated on the AI
   records — not as an authorship claim about the controls.
4. **Runtime** (`analyze`) appends a calibration limitation when the loaded
   calibration bundle carries `watermark_note` or `contaminated_cohorts`.

## What operators should do

- Declare watermark status when generating new Anthropic (or other watermarked)
  baselines; do not leave post-transition Claude runs as silent `unknown`.
- Prefer `declared-none` only when the provider/session is known to disable
  watermarking, or when outputs predate the rollout.
- Treat radioactivity and contamination flags as distribution-shift diagnostics
  for calibration validity — never as courtroom proof of distillation.
- Keep watermark hypothesis tests, passive AI participation, and C2PA
  provenance in separate evidence channels (see the conservative evidence ADR).

## Related

- [watermark-removal.md](watermark-removal.md) — edit/removal robustness of the green-list family
- Temperature and radioactivity cards (bench runners) — sampling power and distillation inheritance
- [contributing-markers.md](contributing-markers.md) — manifest contribution rules
