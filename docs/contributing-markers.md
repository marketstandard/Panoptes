# Contributing marker and calibration data

Panoptes supports community-contributed watermark schemes, calibration cohorts, benchmark results, and detector adapters while avoiding redistribution of raw third-party corpora.

The contribution model is **signed pointer manifests plus hashed artifacts**:

- Datasets are described by pointer manifests.
- Calibration and evaluation results are hashed, schema-validated artifacts.
- Raw text is committed only when it is a small, license-clear fixture.
- Reviewers must be able to validate schemas, hashes, splits, licenses, and limitations without downloading the dataset.

## 1. Choose the contribution type

### Dataset pointer

Use this when proposing a new evaluation or calibration cohort, for example a RAID slice, DroidCollection slice, a new watermark corpus, or a domain-specific prose corpus.

Create:

```text
datasets/manifests/<dataset-id>.json
```

Validate against `schemas/dataset-manifest.schema.json`.

### Calibration artifact

Use this when you have trained or fitted a calibration bundle from a licensed dataset.

Create:

```text
research/submissions/<contribution-id>/calibration-artifact.json
```

Validate against `schemas/calibration-artifact.schema.json`.

### Benchmark card

Use this when reporting benchmark metrics without adding a calibration bundle.

Create:

```text
research/submissions/<contribution-id>/benchmark-card.json
```

Validate against `schemas/benchmark-card.schema.json`.

### Watermark evaluation card

Use this when adding or updating a watermark adapter.

Create:

```text
research/submissions/<contribution-id>/watermark-eval-card.json
```

Validate against `schemas/watermark-eval-card.schema.json`.

## 2. Dataset manifest requirements

A dataset manifest must include:

- stable dataset ID;
- kind (`prose`, `code`, `watermark`, or `mixed`);
- SPDX license and redistribution status;
- citation;
- source URL/version/access instructions;
- SHA-256 content hash;
- train/calibration/test split counts and grouping keys;
- label schema;
- privacy classification;
- limitations;
- canonical `artifact_sha256`.

Rules:

- `privacy.raw_text_in_repo` must be `false`.
- `splits.group_keys` must identify leakage groups such as source conversation, generator run, project repository, or author.
- Train, calibration, and test groups must not overlap.
- Datasets with high PII risk require a maintainer discussion before review.

## 3. Calibration artifact requirements

A calibration artifact must include:

- bundle ID and cohort;
- dataset manifest ID;
- detector ID and version;
- feature schema;
- reproduction command and seed;
- code commit or immutable source reference;
- group-aware cross-validation method;
- AUROC, Brier score, ECE, TPR@1%FPR, and TPR@5%FPR;
- binary calibrator thresholds;
- optional source geometry;
- optional conformal threshold;
- canonical artifact hash.

## 4. Watermark adapter requirements

A watermark adapter must:

- implement `WatermarkAdapter`;
- declare scheme ID and version;
- return an abstention status for unsupported or short inputs;
- compute eligible token count, green count, expected count, \(z\), p-value, effect, and power;
- return token spans only as `{start, end, green}` offsets;
- include an evaluation card with empirical FPR on unwatermarked text, TPR by token bucket, and robustness after edits.

## 5. Validation

From the repository root:

```bash
python research/validate_submission.py \
  datasets/manifests/example.json \
  research/submissions/example/calibration-artifact.json \
  research/submissions/example/benchmark-card.json
```

The validator checks:

- JSON schema validity;
- canonical artifact hashes;
- privacy policy;
- required contribution metadata.

## 6. Review checklist

Maintainers review:

- license and redistribution terms;
- PII and provenance of source data;
- group-aware split integrity;
- evaluation metric soundness;
- false-positive behavior;
- schema compatibility;
- documentation clarity;
- privacy implications of token overlays;
- whether the detector registry should advertise the new capability.

## 7. Privacy and security notes

- Do not include secret watermark keys.
- Do not include raw third-party data in pull requests.
- Token spans are offsets and labels only.
- Calibration bundles must state cohort and limitations.
- Detection outputs must remain probabilistic and must not claim authorship.
