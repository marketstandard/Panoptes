# Baseline run checklist — text

Protocol (full version in baselines/prompts/):

1. Fresh session per prompt; no history, memory, system prompt, or custom instructions.
2. Provider default sampling settings; single turn; no browsing or tools.
3. Paste the prompt verbatim from baselines/prompts/text.md.
4. Save the raw, unedited reply into the file named below.
5. Record the exact model version string shown by the product in `_run.json`.

Files:

- [ ] `text-01.md` — Neighbor plant note
- [ ] `text-02.md` — Hash map explainer
- [ ] `text-03.md` — Public libraries essay
- [ ] `text-04.md` — Lighthouse mystery opening
- [ ] `text-05.md` — Constrained delay email
- [ ] `text-06.md` — Composting summary
- [ ] `text-07.md` — Personal finance op-ed
- [ ] `text-08.md` — Cast-iron how-to

When every box is checked:

```bash
python baselines/baseline.py finalize --run <this folder>
```
