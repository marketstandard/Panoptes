# Threat model and limitations

## Adversarial transformations

Panoptes must assume users may transform text before analysis:

- copy and paste;
- synonym substitution;
- machine or human paraphrase;
- translation;
- human/AI stitching;
- prompt-induced style changes;
- second-model rewriting;
- code formatting;
- variable renaming;
- dead-code insertion;
- AST-preserving code transformations.

The system reports robustness metrics but must not claim immunity.

## Statistical misuse risks

- Treating a p-value as the probability that text is watermarked.
- Treating detector confidence as proof of authorship.
- Treating watermark absence as proof that content is human-written.
- Treating source-family similarity as exact model attribution.
- Applying calibrated English prose results to unsupported languages or domains.
- Using an analysis on very short input without an abstention warning.

## Security threats

- Malicious uploads intended to exhaust memory or parser resources.
- Archive bombs or decompression attacks.
- HTML/JavaScript injection through displayed text or provenance metadata.
- Malicious plugin paths or model artifacts.
- Dependency confusion and compromised model files.
- Accidental logging of sensitive submitted text.

## Controls

- Request and upload size limits.
- Explicit file type checks and parser timeouts.
- No raw text in logs by default.
- HTML sanitization before display.
- Hash/signature verification for pinned artifacts.
- Plugin loading only from explicit local paths.
- Dependency pinning and CI license/secret checks.
- Deterministic fixture mode that requires no external calls.

## Ethical limits

Panoptes must not present itself as a tool for automated punishment. Any academic, legal, employment, or moderation decision requires independent evidence and human review.
