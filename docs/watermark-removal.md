# Watermark-removal robustness

This evaluation answers the adversarial question: **does a removal attack defeat the watermark?** It is the complement to the watermark subsystem's Type&nbsp;I measurements — instead of asking "does the detector false-positive on ordinary text?", it asks "given genuinely watermarked text, how much detection survives an attempt to remove it?"

The result is a signed card, `backend/artifacts/cards/watermark-removal.json`, reported in paper v2.0 (Table 12).

## Why this matters now

Anthropic announced in August 2026 that Claude's text watermark is a version of Google DeepMind's **SynthID-Text** — a member of the Aaronson (2022) green-list family, the same family Panoptes's `KGWReferenceAdapter` models. Anthropic's production key is private, so we cannot test that specific key. What we *can* characterize is the **robustness of the family**, using a green-list scheme whose key we control. Anthropic states that light editing probably will not remove the watermark but a complete rewrite will; this evaluation reproduces that behavior directly.

## Two watermark families

Removers are family-specific, so the eval tests both major text-watermark families:

| Family | Scheme | Embedded by | Detected by |
| --- | --- | --- | --- |
| Statistical | KGW green-list (`kgw-v1`) | `bench/watermark_gen.py` (logit bias under the demo key) | `KGWReferenceAdapter` (binomial z-test) |
| Unicode | Zero-width signature (`unicode-zerowidth-v1`) | `bench/watermark_unicode.py` | `detect_unicode_watermark` |

The two have **disjoint attack surfaces**: Unicode hygiene destroys the zero-width watermark but leaves the statistical one intact; synonym substitution does the reverse.

## Pipeline

```bash
# 1. GPU: generate ground-truth watermarked + control passages (KGW green-list, demo key)
python research/run_watermark_generation.py

# 2. GPU: LLM complete-rewrite attack (the strongest removal — a paraphrase model that
#    does NOT know the watermark key re-rolls word choices off the green list)
python research/run_watermark_paraphrase.py

# 3. CPU, deterministic: apply the attack battery and measure retention
python research/run_watermark_removal.py
```

Steps 1–2 are one-time GPU steps that write signed cards (`watermarked-generations.json`, `watermarked-paraphrases.json`). Step 3 is CPU-only and deterministic; it reads those cards and writes `watermark-removal.json`. Keeping generation on the GPU and the eval on the CPU means the reported numbers are reproducible without a GPU.

## Attack battery

`bench/watermark_attacks.py`:

- `unicode_scrub` — strip zero-width/invisible characters (mirrors a remover's "Unicode hygiene").
- `synonym_substitute` (30% / 60%) — deterministic synonym swaps that perturb word choice.
- `truncate`, `drop_tokens`, `shuffle_sentences`, `lowercase`, `strip_punctuation` — reuse the proxy edits.
- `llm_paraphrase` — a complete rewrite by an instruction-tuned LM (precomputed in step 2).

## Reading the card

For each attack, the card reports **retention**:

- `kgw.per_attack[*].detection_rate_before/after` — fraction of watermarked passages still flagged at p&nbsp;&lt;&nbsp;0.05, and `mean_z_before/after`.
- `unicode.per_attack[*].present_rate_before/after` — fraction of zero-width signatures surviving.
- `passive_evasion.per_attack[*]` — heuristic-detector AUROC on the corpus under the same attack (does the removal *also* evade passive detection?).
- `external_repos` — any external removal tools evaluated via [`evaluate-repo`](testing-external-repos.md).

Headline v2.0 findings: the statistical watermark survives hygiene, truncation, token drop, shuffle, case, and punctuation edits (detection stays ~1.0); only the complete LLM rewrite moves it (100%&nbsp;→&nbsp;42%, mean green-rate 0.82&nbsp;→&nbsp;0.56). The Unicode watermark is fragile — hygiene, truncate, drop, shuffle, and strip-punctuation all destroy it. Watermark removal and passive evasion are orthogonal: synonym paraphrase craters passive AUROC (0.67&nbsp;→&nbsp;0.43) while barely denting the statistical watermark.

## Limitations

- The generator uses a small LM (GPT-2 by default) for speed; the green-list statistics, not the prose quality, are what the eval measures.
- The demo key is public (`panoptes-demo-key`). A vendor's production key is private; results characterize the *family*, not any specific deployment.
- Paraphrase fidelity is not separately scored; the card measures only whether the watermark survives rewriting.
