# Baseline run checklist — code

Protocol (full version in baselines/prompts/):

1. Fresh session per prompt; no history, memory, system prompt, or custom instructions.
2. Provider default sampling settings; single turn; no browsing or tools.
3. Paste the prompt verbatim from baselines/prompts/code.md.
4. Save the raw, unedited reply into the file named below.
5. Record the exact model version string shown by the product in `_run.json`.

Files:

- [ ] `code-01.md` — Log level counter
- [ ] `code-02.md` — LRU cache
- [ ] `code-03.md` — Countdown timer component
- [ ] `code-04.md` — Largest files CLI
- [ ] `code-05.md` — Second-largest bug fix
- [ ] `code-06.md` — Clamp test suite
- [ ] `code-07.md` — Customer order summary query
- [ ] `code-08.md` — TTL cache class

When every box is checked:

```bash
python baselines/baseline.py finalize --run <this folder>
```
