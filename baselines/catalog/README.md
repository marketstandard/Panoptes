# Baseline catalog

This directory is the community ledger for model baseline runs. It contains **hashes only** — never raw model outputs.

- `registry.jsonl` — append-only ledger. One JSON object per line: `{run_id, model_slug, kind, prompts_sha256, manifest_sha256, merkle_root, submitted_utc, contributor, ots_proof?}`.
- `manifests/` — one `panoptes-baseline-run-v1` manifest per submitted run, named by its canonical hash (`<manifest_sha256>.json`), plus optional `<manifest_sha256>.ots` OpenTimestamps proofs.

## Submit a run

```bash
python baselines/baseline.py scaffold --model <model-slug> --kind text   # or: run / agent-assisted
# ... fill in outputs ...
python baselines/baseline.py finalize --run baselines/runs/<model-slug>_text --interface chat-ui
python baselines/baseline.py anchor --run baselines/runs/<model-slug>_text        # optional
python baselines/baseline.py submit --run baselines/runs/<model-slug>_text --contributor <your-handle>
```

Then open a pull request containing **only** the new `registry.jsonl` line and the new file(s) under `manifests/`. Your `baselines/runs/` folder is gitignored and must never be committed.

## Verify the catalog

```bash
python baselines/baseline.py verify-catalog
```

This re-validates every manifest against its JSON Schema, recomputes every canonical hash, cross-checks registry lines against manifests, and rejects duplicates and unreferenced files. CI runs the same check.

## Trust model

A registry line is a **claim**: "contributor X states that model M produced outputs with these hashes for prompts version P, at time T." An optional `.ots` proof anchors that claim in the Bitcoin blockchain, proving the manifest existed at that time and has not changed since. Nothing here can prove an output genuinely came from the claimed model — confidence comes from independent replication: multiple contributors running the same prompts version against the same model and reporting statistically similar outputs. See `baselines/README.md` and `docs/threat-model.md`.
