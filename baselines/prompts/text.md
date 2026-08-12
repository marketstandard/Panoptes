# Panoptes baseline prompts — text

Version 1.0.0. The machine-readable source of truth is `prompts.manifest.json`; this document is the human-friendly copy-paste view. Keep the two in sync (a test in `baselines/tests/` checks it).

## Run protocol — read before running

1. Start a **fresh session** for every prompt: no chat history, memory, or prior context.
2. Use **product defaults**: no system prompt, no custom instructions, default temperature.
3. **Single turn**: send the prompt verbatim, record the complete first reply. No follow-ups, no regeneration.
4. Disable browsing, code execution, retrieval, and plugins where the interface allows.
5. Save the **raw, unedited reply** as `<prompt-id>.md` (for example `text-01.md`) inside the run folder.
6. Record the exact **model version string** shown by the product, the interface you used (`chat-ui`, `api`, or `agent-chat`), and the date.

Fastest path: run `python baselines/baseline.py scaffold --model <model-slug> --kind text` to create a ready-to-fill run folder, paste each prompt below into the model, save each reply, then `python baselines/baseline.py finalize --run baselines/runs/<model-slug>_text`.

---

## text-01 — Neighbor plant note

- Target length: ~100 words
- Tags: `genre:note`, `length:50-149`
- Output file: `text-01.md`

```text
Write a short note to my neighbor Dana asking her to water my two tomato plants while I am away for a long weekend. Keep it friendly, mention that the watering can is by the back door, and stay under 120 words.
```

## text-02 — Hash map explainer

- Target length: ~300 words
- Tags: `genre:explainer`, `length:150-499`
- Output file: `text-02.md`

```text
Explain how a hash map works to a first-year computer science student. Cover buckets, hash collisions, and resizing. Use one small real-world analogy. Aim for about 300 words.
```

## text-03 — Public libraries essay

- Target length: ~800 words
- Tags: `genre:essay`, `length:500plus`
- Output file: `text-03.md`

```text
Write an essay of about 800 words discussing how public libraries have changed over the last fifty years and what role they might play in the next fifty. Include at least one concrete example of a service libraries offer today that they did not offer twenty years ago.
```

## text-04 — Lighthouse mystery opening

- Target length: ~250 words
- Tags: `genre:creative`, `length:150-499`
- Output file: `text-04.md`

```text
Write the opening page of a mystery novel, about 250 words. The story begins with a lighthouse keeper on the Oregon coast noticing that the light from the neighboring lighthouse, unmanned for thirty years, has started blinking again in a regular pattern.
```

## text-05 — Constrained delay email

- Target length: ~150 words
- Tags: `genre:email`, `length:150-499`
- Output file: `text-05.md`

```text
Write a professional email to a project team announcing that the launch of a mobile banking app is delayed by three weeks. Requirements: exactly three bullet points listing the reasons, a revised launch date of October 14, and a closing sentence thanking the team. About 150 words.
```

## text-06 — Composting summary

- Target length: ~60 words (intentionally short: exercises abstention behavior)
- Tags: `genre:summary`, `length:lt50`
- Output file: `text-06.md`

```text
Summarize the following passage in exactly three sentences. Do not add information that is not in the passage.

Passage:
Composting turns kitchen scraps and yard waste into a dark, crumbly soil amendment called humus. The process depends on a balance of "greens," such as vegetable peels and fresh grass clippings, which supply nitrogen, and "browns," such as dried leaves and cardboard, which supply carbon. A widely cited rule of thumb is to keep roughly three parts browns to one part greens by volume. Microorganisms do most of the work, and they need moisture and oxygen to thrive, so a pile should feel like a wrung-out sponge and be turned every week or two. Under good conditions a hot pile can finish in two to three months, while a neglected cold pile may take a year. Meat, dairy, and oily foods are usually excluded from open piles because they attract pests, though enclosed systems can handle them. Finished compost smells earthy rather than sour, and none of the original material should be recognizable.
```

## text-07 — Personal finance op-ed

- Target length: ~400 words
- Tags: `genre:opinion`, `length:150-499`
- Output file: `text-07.md`

```text
Write a persuasive opinion piece of about 400 words arguing that every high school student should complete a personal finance course before graduation. Address one common counterargument and respond to it.
```

## text-08 — Cast-iron how-to

- Target length: ~300 words
- Tags: `genre:howto`, `length:150-499`
- Output file: `text-08.md`

```text
Write a step-by-step guide to seasoning and caring for a cast-iron skillet. Format it as a numbered list of exactly seven steps. Give each step a short bolded title followed by one or two sentences.
```
