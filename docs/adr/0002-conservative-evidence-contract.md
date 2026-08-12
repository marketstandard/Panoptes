# ADR 0002: Conservative evidence contract

## Status

Accepted

## Context

AI-text detection and watermark evidence are frequently misinterpreted. Model attribution is especially vulnerable to overclaiming.

## Decision

Panoptes reports evidence states, calibrated probabilities, hypothesis-test details, source-family similarity, and unknown scores. It does not report authorship verdicts or exact unsupported model identity. Abstention is a valid result.

## Consequences

- The API requires explicit limitations and capability metadata.
- UI copy must distinguish probability, hypothesis tests, similarity, and provenance.
- Detectors must declare cohorts, minimum token counts, licenses, and known limitations.
- Releases must include benchmark cards and false-positive operating points.
