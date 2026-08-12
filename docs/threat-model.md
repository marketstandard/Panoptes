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

## Community baseline submissions

The baseline catalog (`baselines/catalog/`) lets anyone register hashes of model outputs for the canonical prompt set. This introduces a distinct trust surface:

- **Fabricated or mislabeled claims.** A registry entry asserts that a named model produced outputs with given hashes. Nothing can cryptographically prove which model answered; `reported_version` is an unverified contributor claim. Entries must therefore never be presented as verified provenance or exact model attribution — confidence comes only from independent replication across contributors.
- **Agent-harness contamination.** `agent-chat` runs pass through an agent harness (system prompts, tools). They are labeled as a separate interface cohort and must never be blended with `api`/`chat-ui` runs in calibration.
- **Ledger poisoning and spam.** Registry lines and manifests are schema-validated data only (bounded strings, hex hashes, no free-form markup, no executable content). `verify-catalog` recomputes every canonical hash, rejects duplicates, tampering, and unreferenced manifests, and runs in CI. Maintainers review every catalog PR per `docs/contributing-markers.md`.
- **Raw text leakage.** Community outputs stay in gitignored `baselines/runs/`; only SHA-256 manifests enter the repository. The validator rejects baseline manifests that embed output text, mirroring the `raw_text_in_repo` rule for datasets.
- **Timestamp trust.** Optional OpenTimestamps proofs anchor a manifest's existence and integrity in the Bitcoin blockchain. They prove *when* a claim existed, not that the claim is true.

## Ethical limits

Panoptes must not present itself as a tool for automated punishment. Any academic, legal, employment, or moderation decision requires independent evidence and human review. Baseline catalogs measure model behavior for calibration research; they are not leaderboards and must not be used to rank models for punitive purposes or to attribute authorship.
