# How to interpret a Panoptes report

Panoptes reports evidence, not a verdict about a person.

## The short answer

The answer panel summarizes:

- whether the input has enough analyzable evidence;
- the calibrated probability that the content is human-written, AI-generated, or AI-refined/mixed;
- whether any known watermark scheme was detected;
- whether signed file provenance was present;
- whether source-family attribution is reliable enough to display.

A high probability is still probabilistic. A low probability is not proof of human authorship.

## Evidence states

| State | Meaning |
|---|---|
| `supported` | The input was long enough and in a supported domain for at least one analysis path. |
| `insufficient_data` | The input is too short or sparse for the requested statistic. |
| `unsupported_language` | The generic detector is not calibrated for the detected language. |
| `unsupported_content` | The input type is outside the detector's supported content domain. |
| `out_of_distribution` | The input differs materially from the calibration cohort. |
| `not_applicable` | The evidence type does not apply to this input. |
| `adapter_unavailable` | A vendor or configuration needed for a test is unavailable. |

## Watermark evidence

A known watermark test asks whether token choices line up with a configured pseudo-random scheme more often than expected by chance. Panoptes reports:

- eligible token count;
- observed green count;
- expected null count;
- z-score;
- one-sided p-value;
- false-discovery-corrected q-value;
- per-segment evidence.

A p-value is the probability of evidence at least this extreme **if no watermark were present**. It is not the probability that the text is watermarked.

## AI participation probability

The generic detector output is recalibrated on held-out data. The displayed probability depends on:

- detector score;
- content type;
- language;
- length bucket;
- selected prior odds;
- calibration bundle.

The default prior is neutral. Technical users can inspect and adjust prior odds to see how assumptions affect posterior probability. ECE is shown as a calibration diagnostic; it is not mixed into the posterior.

## Source-family similarity

Source-family bars answer a narrow question:

> If this is AI-generated and the source is one of the calibrated candidate families, which families are most similar?

The `unknown` score measures whether the input looks unlike all supported candidates. High `unknown` evidence means Panoptes should not force a family label.

## Code results

Code uses a separate detector and code-aligned windows. Formatting, variable renaming, dead-code insertion, and syntax-preserving edits can weaken or remove watermark and stylistic evidence. Treat short snippets and heavily transformed code as low-evidence inputs.

## Signed provenance

A verified C2PA manifest means a supported file carried a cryptographically signed record. It does not mean:

- the text itself is watermarked;
- the file was not later edited;
- the ideas originated with an AI;
- the current uploader is the original author.

## Responsible use

Panoptes should be used as one signal in a human review process. It should not be used alone for accusations, discipline, censorship, identity attribution, or high-stakes automated decisions.
