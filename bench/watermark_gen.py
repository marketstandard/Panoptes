"""KGW/Aaronson green-list watermark *generation* (the SynthID-Text family).

The detector (:class:`panoptes.analysis.watermarks.KGWReferenceAdapter`) flags
text whose consecutive regex-token pairs land on the key's green list more often
than chance. To build ground-truth watermarked text for the removal evaluation
we generate with a small causal LM and bias each next-token choice toward
green-listed tokens — the standard KGW logit bias (add ``delta`` to the logits
of green-list candidates before sampling). Controls are generated with
``delta=0`` (no bias), so the same model produces matched unwatermarked text.

Generation is word/punctuation level so that each sampled step advances the
detector's regex-token stream by exactly one token and the green-list stays
aligned with ``green_for(previous, token)``. The LM is only a fluency prior;
the watermark is the controlled variable. The torch/transformers imports are
lazy so the module (and its pure helpers) stay importable without the extras.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from panoptes.analysis.watermarks import green_for

_WORD_RE = re.compile(r"^\w+$")
_LAST_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_PUNCT = {".", ",", "?", "!", ";", ":"}
_SENTENCE_END = {".", "?", "!"}
DEFAULT_MODEL = "gpt2"
DEFAULT_DELTA = 2.0
DEFAULT_KEY_NOTE = "panoptes-demo-key"  # the key baked into green_for


@dataclass(frozen=True)
class CandidateVocab:
    """Token ids that decode to exactly one detector regex token.

    Word candidates carry a leading space (`` word``) so they slot after the
    preceding token; punctuation candidates carry none so they attach to the
    preceding word. Each id therefore corresponds to exactly one ``\\w+`` or
    ``[^\\w\\s]`` token in the detector's stream.
    """

    ids: list[int]
    tokens: list[str]


def candidate_vocabulary(tokenizer) -> CandidateVocab:
    ids: list[int] = []
    tokens: list[str] = []
    for token_id in range(len(tokenizer)):
        try:
            decoded = tokenizer.decode([token_id])
        except Exception:  # pragma: no cover - defensive against odd byte tokens
            continue
        if decoded.startswith(" "):
            core = decoded[1:]
            if _WORD_RE.match(core):
                ids.append(token_id)
                tokens.append(core)
        elif decoded in _PUNCT:
            ids.append(token_id)
            tokens.append(decoded)
    return CandidateVocab(ids=ids, tokens=tokens)


def biased_pick(
    cand_logits: np.ndarray,
    cand_tokens: list[str],
    previous: str,
    delta: float,
    rng: np.random.Generator,
    top_k: int | None = None,
    temperature: float = 1.0,
) -> int:
    """Pick a candidate index, biasing green-listed (previous, token) pairs.

    Scales logits by ``1/temperature`` (temperature > 0), then adds ``delta`` to
    the logit of each green-listed candidate (KGW logit bias), then samples from
    the restricted softmax. ``temperature=0`` selects argmax after the green-list
    bias — documenting the greedy dead zone where sampling randomness vanishes.
    Pure numpy so it is testable without a model.
    """
    logits = np.asarray(cand_logits, dtype=float)
    order = np.argsort(-logits, kind="stable")
    if top_k is not None:
        order = order[: max(1, top_k)]
    sel = logits[order].copy()
    if temperature <= 0:
        # Greedy: apply bias, then take argmax. Bias still changes the ranking
        # when two candidates are close; there is no sampling randomness left.
        for j, idx in enumerate(order):
            if green_for(previous, cand_tokens[int(idx)]):
                sel[j] += delta
        return int(order[int(np.argmax(sel))])
    sel = sel / float(temperature)
    for j, idx in enumerate(order):
        if green_for(previous, cand_tokens[int(idx)]):
            sel[j] += delta
    sel -= sel.max()
    probs = np.exp(sel)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return int(order[0])
    probs /= total
    choice = int(rng.choice(len(order), p=probs))
    return int(order[choice])


def _last_regex_token(text: str) -> str:
    tokens = _LAST_TOKEN_RE.findall(text)
    return tokens[-1] if tokens else ""


def _load_lm(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_watermarked(
    prompts: list[str],
    *,
    model_name: str = DEFAULT_MODEL,
    delta: float = DEFAULT_DELTA,
    max_tokens: int = 120,
    top_k: int = 64,
    seed: int = 0,
    device: str | None = None,
    progress: bool = False,
    temperature: float = 1.0,
) -> list[str]:
    """Generate one passage per prompt with KGW green-list biasing.

    ``delta=0`` yields matched unwatermarked controls (same model, no bias).
    ``temperature`` scales candidate logits before the bias (0 = greedy).
    Returns the full text (prompt + generated continuation) per prompt.
    """
    import torch

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer, model = _load_lm(model_name, device)
    cand = candidate_vocabulary(tokenizer)
    cand_ids = torch.tensor(cand.ids, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed)

    outputs: list[str] = []
    for p_idx, prompt in enumerate(prompts):
        text = prompt.rstrip()
        previous = _last_regex_token(text)
        emitted = 0
        while emitted < max_tokens:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                logits = model(input_ids).logits[0, -1]
            cand_logits = logits[cand_ids].float().cpu().numpy()
            pick = biased_pick(
                cand_logits,
                cand.tokens,
                previous,
                delta,
                rng,
                top_k=top_k,
                temperature=temperature,
            )
            token = cand.tokens[pick]
            text = (text + " " + token) if _WORD_RE.match(token) else (text + token)
            previous = token
            emitted += 1
            del input_ids, logits, cand_logits
        outputs.append(text)
        if progress and (p_idx % 10 == 0 or p_idx == len(prompts) - 1):
            print(f"    generated {p_idx + 1}/{len(prompts)}", flush=True)
    del model
    import gc

    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs
