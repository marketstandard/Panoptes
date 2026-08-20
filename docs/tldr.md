# Panoptes, in plain language

*A five-minute explanation of what this project does and why — no math, no jargon.
For the full technical treatment, read the [research paper](https://marketstandard.github.io/Panoptes/paper.html).*

## The problem

Paste text into most "AI detectors" and you get a confident answer: *98%
AI-generated*. What you don't get is any indication of when that answer falls
apart — and it falls apart often. These tools are usually trained to separate
"human" from "machine" writing using examples from a handful of models and
websites. Show them writing that looks different — a newer model, a paraphrased
passage, a non-native English speaker, a human editing AI drafts — and the
confidence stays high even as the accuracy quietly collapses.

People have been harmed by that gap: students falsely accused, freelancers
fired, moderation decisions made on numbers that didn't mean what they appeared
to mean.

## What Panoptes does instead

Panoptes is an open-source tool that analyzes text and tells you **what the
evidence supports, how confident it should be, and when to abstain** — instead
of pretending every question has a crisp yes/no answer.

Three rules separate it from a typical detector:

1. **A score is not a probability.** A raw detector score gets statistically
   calibrated against a verified reference corpus before it becomes a
   percentage — and the quality of that calibration is printed on every report.
2. **Context changes the answer.** The same document should be judged
   differently in a classroom where 5% of submissions are AI-assisted than in a
   bot farm where 95% are. Panoptes asks for that context (a "prior") and shows
   how much it moves the conclusion.
3. **"We don't know" is a valid answer.** When text resembles nothing the
   system was calibrated on, Panoptes says so, rather than forcing a guess.

## The four kinds of evidence

Think of Panoptes as four independent witnesses who only compare notes at the
end:

- **Writing-style analysis.** Statistical and neural models estimate how much
  of the text looks machine-written — including partial answers like "mostly
  human with some machine assistance," which binary tools can't express.
- **Hidden watermarks.** Some AI providers embed subtle statistical patterns
  into their models' output. Panoptes tests for publicly documented watermark
  schemes and shows exactly which tokens triggered the result.
- **Family resemblance.** If the text looks like the output of a specific model
  family the system knows, it says so — and if it matches none of them, it says
  *that*.
- **Digital receipts.** For files that carry cryptographically signed
  provenance (C2PA), Panoptes verifies the signature rather than guessing.

## What we learned building it

A few findings from the research behind v2.1, in plain terms:

- **The hard part isn't accuracy, it's travel.** Almost any detector works on
  text similar to its training data. Ours is evaluated on whether it keeps
  working on datasets it never saw — and the neural model does (roughly 0.80
  vs 0.64–0.69 for classical approaches, on a standard ranking-quality
  measure).
- **Watermarks have a blind spot: the temperature dial.** Watermarking works by
  nudging random word choices. Turn an AI's randomness down to zero ("greedy"
  decoding) and there is nothing left to nudge — the watermark can't embed at
  all. We measured exactly where detection power falls off.
- **Watermarks are inherited.** Train a new model on a watermarked model's
  output and the student picks up a faint trace of the teacher's watermark.
  That makes watermark signals evidence about *training lineage* — not proof
  that a particular person used a particular tool.
- **Watermarks contaminate datasets.** Once watermarked text circulates on the
  public web, any corpus built from that text can absorb the bias. Panoptes
  tracks declared watermark status on its own calibration data and warns when a
  flagged cohort is in use.

## What Panoptes will not tell you

Just as important as what it does:

- It **cannot detect private, undisclosed watermarks** (including Anthropic's
  production watermark) — only publicly documented schemes.
- It **never claims proof of authorship.** Its output is calibrated evidence
  with stated limits, suitable for research and review — not a verdict to
  punish someone with.
- It is **honest about small data.** The current human reference corpus is
  small, and the paper quantifies exactly how much that limits statistical
  power.

## Who it's for

- **Researchers** who want reproducible, hash-signed evaluation artifacts
  instead of another leaderboard number.
- **Educators and editors** who need to understand the strength — and the
  limits — of AI-text evidence before acting on it.
- **Platform and trust-and-safety teams** who want local-first analysis (your
  text never leaves your machine) with an auditable report.
- **Developers** who want to plug in their own detectors or watermark schemes.

## Where to go next

- **Read the paper:** [marketstandard.github.io/Panoptes/paper.html](https://marketstandard.github.io/Panoptes/paper.html)
- **Get the code:** [github.com/marketstandard/Panoptes](https://github.com/marketstandard/Panoptes) — the README has a five-command quick start.
- **Understand the evidence UI:** [docs/interpretation.md](interpretation.md)
- **See what's measured:** [docs/evaluation.md](evaluation.md)

*Panoptes is built by Carrington Junior and Trey Huffine. Questions, critiques,
and reproductions are welcome as GitHub issues.*
