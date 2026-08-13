# Version 2 updates

Work that **version 1.0 of the paper and system cannot complete from this environment**, listed so it is not silently dropped. Version 1 froze the protocol, ran every experiment the hash-verified corpus supports, and reported the honest nulls. Version 2 is the remaining publication checklist.

Nothing here is an invitation to train a larger Panoptes-v1. The scientific center stays calibration, shift, transportability, and reproducibility.

## Status against the pre-submission checklist

| Checklist item | v1 status | v2 action |
| --- | --- | --- |
| Human corpus with hundreds of independent authors (target 500–2,000) | **Not done.** Eight human controls. | Collect license-clear multi-document authors; enforce author-disjoint splits. |
| Author-, prompt-, and source-disjoint evaluation | **Partial.** Prompt/group disjoint is enforced; author disjoint only when `author` is supplied. | Populate real author IDs; do not infer authors from labels. |
| Train / calibration / test separated | **Done** (`python -m bench measure`). | Keep the hard rule on every new dataset. |
| Strong external detector baselines (Binoculars, DetectGPT, Fast-DetectGPT, transformer classifier) | **Registered, unavailable.** No weights in this environment. | Install extras, score the frozen splits, write cards. Do not substitute the heuristic. |
| Unseen-domain and unseen-generator experiments | **Partial.** Leave-one-family-out is chance (AUROC 0.515) on n=104. Defactify transport is the real domain result. | RAID / M4GT / EvoBench. |
| Adversarial editing and paraphrase | **Proxy only** (truncate, drop, punctuation, shuffle). | DIPPER, RAID attacks, human rewrite, translation. |
| Human–AI coauthoring | **Token-splice pilot** (48 pairs, flat slope). | Recorded workflows: grammar edit, outline→draft→revise, alternate sections, heavy human edit. |
| Calibration (Brier, ECE, reliability, slope, prior sensitivity) | **Done** on the project corpus protocol card. | Repeat on large author-disjoint data. |
| Conformal coverage by length, domain, generator, class | **Partial.** Pooled coverage only. | Slice coverage on a large set. |
| Evidence dependence modeled | **Done as a pilot** (naive vs correlated vs document). | Hierarchical model on long documents. |
| Power analysis on primary metrics | **Done** for accuracy MDE; calibration/transport still underpowered at n=104. | Recompute power for Brier/ECE/transport. |
| Preregistration timestamped | **Done** (`research/protocol.json`, 2026-08-13). | Do not retune against test outcomes. |
| Watermark results separate from passive detection | **Done.** Separate card; Type I on unwatermarked text. | Controlled watermarked generations + degradation. |
| Unknown-generator rejection | **Run; chance on this corpus.** | Unseen families at Defactify/RAID scale. |
| Cryptographic artifacts reproducible | **First-party self-check passes.** | Outside reproduction. |
| Independent reproduction | **Not done.** Template: [independent-reproduction.md](independent-reproduction.md). | Outside researcher, original vs recomputed deltas. |
| Bibliography machine-checked | **Partial.** Numbered list in `paper.html`. | DOI/Crossref pass. |
| RAID | **Not fetched.** | Pointer manifest + grouped evaluation. |
| M4GT-Bench | **Not fetched.** | Multilingual/multi-domain transport matrix. |
| EvoBench | **Not fetched.** | Generator-generation shift. |
| Hugging Face weights release | **Not done.** Local gitignored weights only. | After comparison battery on a powered corpus. |

## Datasets to add (do not copy raw text into git)

Add pointer manifests under `datasets/manifests/` following `datasets/README.md`:

1. RAID (Dugan et al. 2024) — unseen models, domains, decoding, attacks.
2. M4 / M4GT-Bench (Wang et al.) — multi-domain, multilingual, multi-generator.
3. A coauthoring set with documented mix rates and author IDs.
4. A human set with 500–2,000 authors and several documents each.

## Detectors to score on frozen splits

See `bench/external_baselines.py`. When weights exist:

```bash
python -m bench measure --data <dataset>   # already scores heuristic + logistic
# then score binoculars / detectgpt / transformer on the same ProtocolSplit objects
```

Never fit their calibration on the Panoptes test groups.

## Paper v2 (after the data exist)

- Replace the mixture pilot with real coauthoring curves.
- Replace the robustness proxy with RAID/DIPPER.
- Report an outside reproduction table (original vs independent, absolute delta).
- Keep the Defactify transport collapse as a central result.
- Do not center the paper on in-domain AUROC.

## Commands that already work for v1

```bash
python -m bench measure --data corpus
python research/run_v1_experiments.py
python research/reproduce.py
python research/make_figures.py
python research/validate_submission.py research/protocol.json backend/artifacts/cards/measurement-protocol.json
python baselines/baseline.py verify-catalog
```
