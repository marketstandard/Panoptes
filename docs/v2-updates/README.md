# v2 updates: work this environment cannot complete

Open items referenced by the paper (§21) and the evaluation notes. Each entry
states what is required and why it is out of reach for the current build
environment. Everything else in the v2.1 protocol is complete and signed.

## 1. Dedicated 500–2,000-author human panel

- **What:** a human-control corpus of 500–2,000 distinct authors with
  demographic spread, written under the study prompts, to replace the current
  8-author control cohort.
- **Why open:** requires recruitment, consent, and terms-cleared collection at
  scale; not producible in this environment. The current corpus supports
  calibration and tier modeling but is underpowered for small effects
  (≈23% power for d = 0.5; see `docs/evaluation.md`).
- **Unblocks:** author-leakage controls beyond GroupKFold-by-prompt, narrower
  calibration intervals, and a defensible mixed-authorship contribution-fraction
  head (currently external-evaluation-only).

## 2. DetectGPT / Binoculars weight release

- **What:** publishing the trained external-baseline weights alongside the
  frozen Panoptes neural detector.
- **Why open:** upstream licensing and redistribution terms for those
  baselines have not been cleared in this environment.
- **Unblocks:** byte-identical third-party reruns of the external-baseline
  comparison without retraining.

## 3. Outside reproduction

- **What:** an independent group rerunning the pipeline from a clean checkout
  and confirming the signed artifact hashes (`independent: true` in
  `bench/reproduce`).
- **Why open:** by definition cannot be performed by the authoring
  environment; current verification is first-party (`independent: false`).
- **Unblocks:** the reproduction claim upgrading from "self-consistent" to
  "independently reproduced". See `docs/independent-reproduction.md` for the
  exact steps a reproducer should follow.

## Related

- [evaluation.md](evaluation.md) — what is measured and its caveats
- [independent-reproduction.md](independent-reproduction.md) — reproduction protocol
- [research-protocol-v2-1.md](research-protocol-v2-1.md) — active claims and non-claims
