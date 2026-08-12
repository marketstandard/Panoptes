# UI evidence map

This document maps every user-visible surface to the versioned analysis response contract and explains how to interpret it.

## Global response metadata

| Field | UI surface | Interpretation |
| --- | --- | --- |
| `schema_version` | hidden diagnostics | API contract version used by the report |
| `report_id` | hidden diagnostics | Unique identifier for support/debugging; not persisted by default |
| `runtime.profile` | top navigation | `fixture`, `local-cpu`, `local-gpu`, `cloud-cpu`, or `cloud-gpu` |
| `runtime.device` | top navigation | Compute device reported by the profile |
| `runtime.models_loaded` | technical lab heading | Detector/model identifiers active for the report |
| `runtime.calibration_bundles` | technical lab | Calibration bundles applied to scores |

## Input diagnostics

| Field | UI surface | Interpretation |
| --- | --- | --- |
| `input.content_hash` | technical diagnostics | SHA-256-style content fingerprint for reproducibility |
| `input.content_type` | answer state strip | `prose`, `code`, `mixed`, or `file` routing result |
| `input.language` | answer state strip | Detected or user-selected language cohort |
| `input.token_count` | answer state strip | Number of tokens used by the detector/watermark pipeline |
| `input.character_count` | source panel metadata | Original text length |
| `input.segment_count` | hero instrument | Number of synchronized analysis segments |
| `input.user_overrode_type` | limitations | Indicates the user bypassed automatic routing |

## Summary and answer strip

| Field | UI surface | Interpretation |
| --- | --- | --- |
| `summary.plain_language` | primary heading | Non-technical synthesis of evidence; not a verdict |
| `summary.evidence_state` | state pill | Whether the evidence is supported, insufficient, unsupported, or unavailable |
| `summary.confidence_label` | state pill | Coarse reliability label derived from evidence quantity and separation |
| `summary.overall.human` | probability bar | Calibrated probability assigned to human-written outcome |
| `summary.overall.ai_generated` | probability bar | Calibrated probability assigned to AI-generated outcome |
| `summary.overall.ai_refined_or_mixed` | probability bar | Calibrated probability assigned to mixed/refined participation |

## Posterior decomposition

| Field | UI surface | Interpretation |
| --- | --- | --- |
| `posterior.prior_odds` | statistic | User-selected prior odds \(O_0\) |
| `posterior.likelihood_ratio` | statistic | Detector evidence multiplier |
| `posterior.posterior_odds` | statistic | \(O_0 \times \mathrm{LR}\) |
| `posterior.calibration_bundle` | statistic detail | Bundle used to map scores to probabilities |
| `posterior.reliability_error` | statistic | Expected calibration error for the cohort when known |
| `posterior.cohort` | lab note | Population/domain represented by calibration |

## Watermark evidence

| Field | UI surface | Interpretation |
| --- | --- | --- |
| `watermarks[].scheme` | metric card / lab block | Watermark scheme identifier |
| `watermarks[].status` | lab block | `tested`, `insufficient_data`, `adapter_unavailable`, or `not_applicable` |
| `watermarks[].eligible_tokens` | stat | Number of tokens eligible for testing |
| `watermarks[].green_tokens` | stat | Count \(G\) of green-list-compatible tokens |
| `watermarks[].expected_green` | stat | Null expectation \(\gamma n\) |
| `watermarks[].green_rate` | stat | Observed \(G/n\) |
| `watermarks[].green_rate_interval` | stat | Wilson confidence interval for observed rate |
| `watermarks[].dilution_estimate` | stat | Fraction of original watermarked tokens estimated to remain |
| `watermarks[].z` | stat | Standardized excess under null |
| `watermarks[].p_value` | stat | One-sided p-value; not P(watermarked) |
| `watermarks[].q_value` | metric detail | Benjamini-Hochberg FDR-adjusted p-value |
| `watermarks[].effect` | stat | Observed green-rate excess |
| `watermarks[].power` | stat | Approximate power of test at current \(n\) |
| `watermarks[].tokens[]` | source overlay | `{start, end, green}` spans used for local rendering only |

## Source-family panel

| Field | UI surface | Interpretation |
| --- | --- | --- |
| `source_families.conditional_on_ai[].family` | family row | Calibrated source-family candidate |
| `source_families.conditional_on_ai[].probability` | family row bar | Conditional similarity among supported candidates |
| `source_families.unknown_score` | unknown meter | Open-set evidence that no calibrated family is a good match |
| `source_families.interpretation` | panel note | Guardrail explaining conditional similarity |

## Provenance panel

| Field | UI surface | Interpretation |
| --- | --- | --- |
| `provenance.status` | status tone | `verified`, `tampered`, `not_present`, `unsupported_file`, `error`, or `not_applicable` |
| `provenance.summary` | paragraph | Plain-language file provenance result |
| `provenance.issuer` | definition list | Signing identity when available |
| `provenance.timestamp` | definition list | Signed timestamp when available |
| `provenance.actions[]` | definition list | Recorded C2PA actions such as creation/edit/export |

## Segment synchronization

| Field | UI surface | Interpretation |
| --- | --- | --- |
| `segments[].id` | segment strip / matrix columns | Stable segment label |
| `segments[].start` / `end` | token overlay selection | Character offsets into submitted text |
| `segments[].token_count` | segment metadata | Tokens in segment |
| `segments[].kind` | segment metadata | Segment content type |
| `segments[].posterior` | segment chip fill | Segment-local calibrated outcome |
| `segments[].watermark_evidence` | watermark matrix | Scheme-specific segment evidence |
| `segments[].source_family` | source matrix | Source-family similarity by segment |
| `segments[].anomaly_percentile` | segment metadata | Percentile-like evidence of unusual detector score |

## Matrices and waterfall

| Field | UI surface | Interpretation |
| --- | --- | --- |
| `matrices.source_by_segment` | source matrix | Family similarity conditioned on AI participation |
| `matrices.watermark_evidence_by_segment` | watermark matrix | Scheme-by-segment evidence strength |
| `matrices.contribution_waterfall[]` | waterfall chart | Prior, detector evidence, penalties, and final posterior contribution |

## Math definitions

| Field | UI surface | Interpretation |
| --- | --- | --- |
| `math[].name` | equation card title | Human-readable calculation name |
| `math[].meaning` | equation card paragraph | What the calculation means |
| `math[].formula` | KaTeX equation | Rendered scientific notation |
| `math[].units` | definition list | Units of output |
| `math[].assumptions[]` | definition list | Conditions required for validity |
| `math[].limitations[]` | definition list | Misuse and failure modes |
| `math[].kind` | hidden style semantics | Calibrated evidence, hypothesis test, or descriptive context |
