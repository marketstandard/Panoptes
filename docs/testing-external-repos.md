# Testing external systems from a git repo

`python -m bench evaluate-repo` points Panoptes at **any git repository** and returns a signed evaluation card. The external system is described by a small *adapter contract* and executed in a subprocess, so cloned code never runs inside the Panoptes process. Third-party claims become reproducible signed cards instead of anecdotes.

```bash
python -m bench evaluate-repo <git-url> --kind watermark-remover \
    [--ref <branch|tag|sha>] [--adapter-path panoptes_adapter.py] \
    [--docker] [--timeout 180] [--out card.json]
```

Kinds: `watermark-remover`, `watermark-scheme`, `detector`.

## Adapter contract

A repository opts in with a `panoptes.adapter.json` at its root:

```json
{
  "kind": "watermark-remover",
  "name": "my-tool",
  "version": "1.0",
  "entry": {"type": "python-function", "module": "panoptes_adapter", "callable": "transform"},
  "requires_network": false
}
```

If no manifest is present, the harness auto-detects a `panoptes_adapter.py` at the root exposing the conventional callable for the kind. If the repo ships neither, supply a thin adapter yourself with `--adapter-path path/to/panoptes_adapter.py`; it is copied into the clone before running.

### Kind contracts

| Kind | Callable | Signature |
| --- | --- | --- |
| `watermark-remover` | `transform` | `transform(text: str) -> str` |
| `watermark-scheme` | `detect` | `detect(text: str) -> {"score": float, "p_value": float}` |
| `detector` | `score` | `score(text: str) -> float` |

Each kind routes to the matching evaluation:

- **watermark-remover** → the [removal-retention eval](watermark-removal.md): the transform is applied to the watermarked generations and Unicode-embedded controls, and detection is measured before vs. after.
- **watermark-scheme** → the scheme's `detect` is run over the watermarked and control passages → TPR / FPR.
- **detector** → the score is run over the verified corpus and evaluated under the frozen protocol (AUROC and friends) via `PrecomputedScoreDetector` + `evaluate_protocol`.

## Security

**This clones and executes arbitrary code. Only run repositories you trust.**

- The adapter runs in a **subprocess** (`bench/_repo_adapter_runner.py`), never in the Panoptes process.
- The subprocess environment is **scrubbed**: common credential variables (`OPENAI*`, `ANTHROPIC*`, `AWS_*`, `GITHUB_*`, `*TOKEN*`, `*API_KEY*`, …) are dropped so a malicious repo cannot read them from its process environment.
- A **wall-clock limit** (`--timeout`, default 180s) bounds the run.
- Network is **not** blocked by default. Pass `--docker` to run the adapter in a network-disabled container (`docker run --network=none`); the image must already contain the repo's dependencies, and the repo is mounted read-only.
- Clones go to `.panoptes/repos/<sha>/` (gitignored) and are reused across runs.

The harness prints a trusted-repo warning on every invocation. For a hostile or unknown repo, use `--docker` and review the code first.

## Reference adapter: watermarks-remover

A worked example ships under [`integrations/watermarks_remover/`](../integrations/watermarks_remover/). It wraps the deterministic "Layer A" Unicode-hygiene transform from [`guillaumemeyer/watermarks-remover`](https://github.com/guillaumemeyer/watermarks-remover) (MIT) — no API key needed — as a `watermark-remover`:

```bash
python -m bench evaluate-repo https://github.com/guillaumemeyer/watermarks-remover \
    --kind watermark-remover \
    --adapter-path integrations/watermarks_remover/panoptes_adapter.py
```

Result (signed `backend/artifacts/cards/external-repo-watermark-remover.json`): the tool's Unicode hygiene drives the zero-width watermark from 100% present to 0%, while leaving the statistical KGW watermark fully detectable (100% → 100%) — a concrete demonstration that removers are family-specific. The repo's network-dependent "Layer B" statistical rewrite is not exercised by the reference adapter.
