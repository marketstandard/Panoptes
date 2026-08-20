# Panoptes research protocol v2.1 (addendum)

Registered 2026-08-19. Machine-readable, self-hashed copy: [`bench/protocol-v2.1.json`](../bench/protocol-v2.1.json). This addendum extends the frozen [v1 protocol](research-protocol.md) and is committed **before any outcome-bearing run** of the v2.1 experiments. Every v1 hard rule still applies.

## What changes in v2.1

v2.1 narrows and strengthens the claim. The signature result is **evidence transportability** — how the evidential meaning of a score changes as the calibration population changes — measured with a calibration-transfer matrix. A genuinely strong neural detector is trained and shipped, but as an *instrument* for that measurement and a useful product tier, not as proof that in-domain accuracy generalizes.

Three things are new relative to v1:

1. **A four-way data firewall** — train → development → calibration → test — for the neural detector.
2. **A preregistered candidate-selection pilot** that never touches final test.
3. **A one-shot final-test lock and a gated public model release.**

## Label ontology (never blended)

| Channel | Meaning | Kind |
| --- | --- | --- |
| `ai_participation` | P(any substantial AI contribution) | statistical, population-conditional |
| `ai_majority_generation` | P(majority AI-generated \| participation) | statistical, population-conditional |
| `ai_contribution_fraction` | real-valued share of final text from AI (0–1) | statistical; ground truth only on CoAuthor |
| `source_family` | conditional similarity among generator families + unknown | statistical |
| `watermark` | embedded-scheme evidence (KGW, Unicode) | statistical, separate power/multiplicity |
| `provenance` | cryptographic/C2PA attestation | population-independent; only as strong as the signing chain |

Hard rule: these channels are **never** collapsed into one score. Participation, majority-generation, and contribution-fraction are distinct targets and are never equated.

## Cohort roles

| Role | Cohort(s) | Use |
| --- | --- | --- |
| `reference` | Reference Community Corpus v0 (104 records) | contribution/hashing/calibration worked example; transport target; default calibration prior. **Not** a powered evaluation cohort; excluded from pooled neural training. |
| `training` | license-cleared cohorts (MAGE train, RAID clean, conditionally Defactify/EvoBench) | pooled training pool + leave-one-cohort-out |
| `evaluation` | untouched test cohorts | final scoring. A cohort in pooled training is a seen-cohort holdout, never "external." |
| `robustness` | RAID attacks, MAGE OOD/paraphrase | trained clean, scored attacked |
| `mixed_task` | CoAuthor | author-disjoint participation/contribution; never binary fully-AI positives |

## Data firewall

```text
train → development → calibration → test
```

- All four partitions are disjoint on the leakage-control group (prompt/story/source/author/duplicate-cluster); author and source/story disjointness are enforced when present.
- A **global** exact-hash and near-duplicate cluster index spans every cohort before groups are assigned; aligned human/machine source pairs share a group.
- **Final-test lock:** final test labels are evaluated once, after architecture, objective, hyperparameters, windowing, aggregation, seeds, calibrators, and acceptance gates are frozen and committed. A failed gate is reported; it never triggers test-driven retuning.
- Final-test labels are hidden from training, selection, calibration, and aggregation code paths.

## Endpoints

**Primary** (one multiplicity family, BH q ≤ 0.05): worst-cohort AUROC; group-bootstrap Brier; calibration slope; TPR at a calibration-fixed 1% FPR; calibration-transfer delta; selective risk at calibration-fixed coverage.

**Secondary** (descriptive): adaptive ECE, AUPRC, log loss, TPR at 0.1%/5% FPR, attack degradation, contribution-fraction MAE/Spearman/slope, source-family macro-F1, unknown-rejection AUROC.

## Candidate grid and selection

Encoders: `microsoft/deberta-v3-base` (stable primary), `answerdotai/ModernBERT-base` (long-context challenger, needs `transformers>=4.48`), `microsoft/deberta-v3-small` (latency/CPU control). Longformer is excluded by default. Objectives: ERM, group-balanced, GroupDRO. Aggregation: overlap-corrected logit mean vs a small hierarchical summary head.

**Selection rule (development only, lexicographic):** maximize worst-cohort AUROC → minimize worst-cohort Brier → minimize latency/memory. No final test informs any choice. Every pilot run, including failures, is recorded in a signed selection card.

**Seeds:** 13, 42, 87.

## Uncertainty

- Group/author/story bootstrap 95% CIs; never i.i.d. document bootstrap under grouped data.
- **Split conformal:** class-conditional nonconformity thresholds fit on calibration; coverage/set-size evaluated on untouched test. Coverage is **not** guaranteed under shift and is reported diagnostically.
- Selective-prediction and low-FPR thresholds are fixed on calibration; test-ranked curves are descriptive only.

## Release gate

The neural detector becomes the runtime default and its weights are published to Hugging Face only if every gate passes: performance, calibration, license (derived-weight redistribution audit), reproducibility (clean-clone recomputation), safety, and latency. Failure keeps the model experimental and the calibrated logistic tier as default; the negative result is reported.

## What we will not claim

- Panoptes is a highly accurate AI detector.
- AI text is universally identifiable.
- In-domain neural performance implies cross-domain transport.
- Conformal coverage is guaranteed under distribution shift.
- A negative watermark test is evidence of human authorship.
- Valid provenance proves metaphysical authorship rather than a signing chain.
- Source-family similarity is exact model identity.
- A cohort in the pooled training set is an external evaluation.
- Watermark radioactivity proves unauthorized distillation (it is lineage-compatible evidence under web contamination).
- Private vendor watermarks (e.g. Anthropic SynthID-Text) are detectable without their published detector.

## Watermark intelligence addendum (2026-08-20)

Registered evaluations that extend the watermark channel without blending it into passive attribution:

| Eval | Runner | Signed card | Question |
| --- | --- | --- | --- |
| Temperature sweep | `python -m bench.run_watermark_temperature` | `watermark-temperature.json` | How does sampling temperature (incl. greedy T=0) affect detection power? |
| Radioactivity | `python -m bench.run_radioactivity` | `radioactivity.json` | Does a student inherit green-list bias from watermarked teachers, and do paraphrase/neutralization remove it? |
| Contamination screening | `python -m bench.baseline_corpus` | embedded in `corpus-summary.json` | Which baseline cohorts are declared/suspected watermarked? |

See [watermark-contamination.md](watermark-contamination.md), [watermark-removal.md](watermark-removal.md).
