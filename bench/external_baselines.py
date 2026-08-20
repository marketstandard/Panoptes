"""Optional external detector adapters.

Binoculars, DetectGPT / Fast-DetectGPT, and large transformer classifiers
require model weights and (usually) a GPU. This module exposes a uniform
surface so they can be scored on the same protocol splits when the
operator installs the extras. When weights are absent the adapters report
`unavailable` rather than silently substituting a different detector.

The GPU scorers import torch/transformers lazily inside functions so the
module stays importable (and the runtime backend untouched) without the
extras. Scoring is precomputed per dataset over the protocol splits'
calibration ∪ test partitions; evaluation then runs through
bench.evaluate.evaluate_protocol with a lookup detector, so isotonic
calibration is fitted on the calibration partition only — never on test
groups.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

UNAVAILABLE = {
    "binoculars": {
        "name": "Binoculars",
        "citation": "Hans et al. 2024, ICML",
        "requires": "transformers + two observer/performer LMs",
        "status": "unavailable",
    },
    "detectgpt": {
        "name": "DetectGPT",
        "citation": "Mitchell et al. 2023, ICML",
        "requires": "source LM + perturbation model",
        "status": "unavailable",
    },
    "fast_detectgpt": {
        "name": "Fast-DetectGPT",
        "citation": "Bao et al. 2024, ICLR",
        "requires": "source/reference LM pair",
        "status": "unavailable",
    },
    "transformer_classifier": {
        "name": "Transformer classifier (RoBERTa-style)",
        "citation": "standard fine-tuned detector",
        "requires": "fine-tuned classifier weights + GPU",
        "status": "unavailable",
    },
}


def catalog() -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in UNAVAILABLE.items()}


def status_card() -> dict:
    return {
        "schema": "panoptes-external-baselines-v1",
        "detectors": catalog(),
        "note": (
            "External zero-shot and neural detectors are registered here so that "
            "a later run can score them on the frozen protocol splits. This "
            "environment does not ship their weights. Do not substitute the "
            "Panoptes heuristic for an unavailable external detector."
        ),
    }


class PrecomputedScoreDetector:
    """Protocol detector surface over precomputed zero-shot scores.

    fit() is a no-op: external baselines are never trained on the bench
    datasets. Scores are squashed with a fixed logistic map solely to land
    in (0, 1); the map is monotone, so AUROC is invariant and the protocol's
    isotonic layer (fitted on the calibration partition only) learns the
    actual calibration curve.
    """

    name = "external-precomputed"

    def __init__(self, scores: dict[int, float], temperature: float = 1.0):
        self.scores = scores
        self.temperature = temperature

    def fit(self, dataset, idx):
        return self

    def predict_proba(self, dataset, idx) -> np.ndarray:
        out = []
        for i in idx:
            raw = self.scores[int(i)] / self.temperature
            out.append(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, raw)))))
        return np.array(out)


def _load_causal_lm(name: str, torch_dtype: str = "bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=getattr(torch, torch_dtype), device_map="cuda:0"
    )
    model.eval()
    return tokenizer, model


def _free_model() -> None:
    """Release cached GPU blocks back to the driver.

    The caller must drop its own model reference (``del model``) before calling
    this: ``del`` on a bare parameter inside a helper only unbinds the helper's
    local name, leaving the caller's reference — and the ~14 GB of weights —
    alive. That bug kept both Binoculars models resident (28 GB > 24 GB VRAM)
    and forced WDDM paging.
    """
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _batches(texts: list[str], batch_size: int):
    for start in range(0, len(texts), batch_size):
        yield start, texts[start : start + batch_size]


def _chunked_target_logp(logits, targets, chunk: int = 128):
    """Log-prob of each target token, with the softmax computed in sequence
    chunks so the fp32 vocab-wide tensor never exceeds a few hundred MB.

    The full-vocab fp32 log_softmax over (batch, seq, 65k) is several GB; on a
    24 GB card already holding a 7B model that tips the WDDM driver into
    paging GPU memory, collapsing throughput. Chunking keeps peak activation
    memory small at negligible speed cost."""
    import torch

    out = torch.zeros(logits.shape[0], logits.shape[1], dtype=torch.float32, device=logits.device)
    for s in range(0, logits.shape[1], chunk):
        logp = torch.log_softmax(logits[:, s : s + chunk].float(), dim=-1)
        out[:, s : s + chunk] = logp.gather(-1, targets[:, s : s + chunk].unsqueeze(-1)).squeeze(-1)
    return out


def _length_sorted_indices(texts: list[str]) -> np.ndarray:
    """Sort by character count (cheap token-length proxy) so batches have
    uniform padded shapes. Uniform shapes let the CUDA caching allocator reuse
    blocks; variable shapes fragment reserved memory until the WDDM driver
    pages GPU memory through system RAM (observed: 30x slowdown)."""
    return np.argsort([len(t) for t in texts], kind="stable")


def _token_budget_batches(
    sorted_texts: list[str], max_batch: int, token_budget: int
) -> list[tuple[int, list[str]]]:
    """Pack length-sorted texts into batches bounded by a token budget.

    Peak activation memory scales with batch_size x padded_length; a fixed
    batch size either wastes compute on short texts (small batches) or pages
    VRAM on long ones (big batches x long sequences). Packing to
    batch_size x max_length <= token_budget keeps peak memory bounded while
    keeping short-text batches large. Estimated tokens are ceil(chars / 4),
    a conservative proxy for English prose."""
    batches: list[tuple[int, list[str]]] = []
    current: list[str] = []
    start = 0
    max_est = 0
    for text in sorted_texts:
        est = max(1, (len(text) + 3) // 4)
        new_max = max(max_est, est)
        if current and (
            len(current) + 1 > max_batch or (len(current) + 1) * new_max > token_budget
        ):
            batches.append((start, current))
            start += len(current)
            current = [text]
            max_est = est
        else:
            current.append(text)
            max_est = new_max
    if current:
        batches.append((start, current))
    return batches


def _mean_token_loglik(
    model,
    tokenizer,
    texts: list[str],
    max_tokens: int,
    batch_size: int = 16,
    progress: bool = False,
) -> np.ndarray:
    """Mean per-token log-likelihood of each text under the model (<= 0)."""
    import torch

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    order = _length_sorted_indices(texts)
    sorted_texts = [texts[int(i)] for i in order]
    out = np.zeros(len(texts))
    batches = _token_budget_batches(sorted_texts, max_batch=batch_size, token_budget=8192)
    with torch.no_grad():
        for step, (start, batch) in enumerate(batches):
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens
            ).to("cuda:0")
            logits = model(**encoded).logits[:, :-1]
            targets = encoded["input_ids"][:, 1:]
            mask = encoded["attention_mask"][:, 1:].float()
            token_logp = _chunked_target_logp(logits, targets)
            scores = (token_logp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            out[order[start : start + len(batch)]] = scores.cpu().numpy()
            # The CUDA caching allocator never returns reserved blocks on its own;
            # over thousands of variable-shaped batches the reserved pool climbs past
            # physical VRAM and the WDDM driver pages GPU memory (observed: 50x
            # slowdown). empty_cache only releases *unreferenced* blocks, so the big
            # per-batch tensors must be deleted first — otherwise they stay live until
            # reassignment and the cache can never shrink.
            del encoded, logits, targets, mask, token_logp, scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if progress and (step % 25 == 0 or step == len(batches) - 1):
                print(f"    batch {step + 1}/{len(batches)}", flush=True)
    return out


def binoculars_scores(
    texts: list[str],
    observer: str = "tiiuae/falcon-7b",
    performer: str = "tiiuae/falcon-7b-instruct",
    max_tokens: int = 1024,
    batch_size: int = 16,
) -> np.ndarray:
    """Binoculars score (Hans et al. 2024): log PPL_observer / cross-PPL.

    s(x) = (-L_observer) / exp(-L_performer), where L is the mean per-token
    log-likelihood. Higher = more machine-like. The two 7B models are loaded
    sequentially so the pair fits a single 24 GB GPU.
    """
    tokenizer, observer_model = _load_causal_lm(observer)
    loglik_observer = _mean_token_loglik(
        observer_model, tokenizer, texts, max_tokens, batch_size, progress=True
    )
    del observer_model  # drop the last reference so the weights actually free
    _free_model()

    tokenizer, performer_model = _load_causal_lm(performer)
    loglik_performer = _mean_token_loglik(
        performer_model, tokenizer, texts, max_tokens, batch_size, progress=True
    )
    del performer_model
    _free_model()

    log_ppl_observer = -loglik_observer
    cross_ppl = np.exp(-loglik_performer)
    return log_ppl_observer / np.maximum(cross_ppl, 1e-8)


def fast_detectgpt_scores(
    texts: list[str],
    model_name: str = "EleutherAI/gpt-j-6B",
    n_samples: int = 5,
    max_tokens: int = 1024,
    batch_size: int = 16,
    seed: int = 13,
) -> np.ndarray:
    """Fast-DetectGPT conditional-curvature score (Bao et al. 2024).

    d(x) = mean_i [ log q(x_i | x_<i) - mean_{x~q(.|x_<i)} log q(x~ | x_<i) ],
    estimated with `n_samples` draws per position from the model's own
    conditional distribution — one forward pass per text, no perturbation
    model. Higher = more machine-like.
    """
    import torch

    tokenizer, model = _load_causal_lm(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    generator = None
    order = _length_sorted_indices(texts)
    sorted_texts = [texts[int(i)] for i in order]
    out = np.zeros(len(texts))
    batches = _token_budget_batches(sorted_texts, max_batch=batch_size, token_budget=8192)
    with torch.no_grad():
        for step, (start, batch) in enumerate(batches):
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens
            ).to("cuda:0")
            logits = model(**encoded).logits[:, :-1]
            targets = encoded["input_ids"][:, 1:]
            mask = encoded["attention_mask"][:, 1:].float()
            if generator is None:
                generator = torch.Generator(device=logits.device).manual_seed(seed)
            actual = torch.zeros(
                logits.shape[0], logits.shape[1], dtype=torch.float32, device=logits.device
            )
            sampled = torch.zeros_like(actual)
            # Vocab-wide softmax/sampling in sequence chunks: the full fp32
            # tensors are several GB and tip a 24 GB card into WDDM paging.
            for s in range(0, logits.shape[1], 128):
                logp = torch.log_softmax(logits[:, s : s + 128].float(), dim=-1)
                actual[:, s : s + 128] = logp.gather(
                    -1, targets[:, s : s + 128].unsqueeze(-1)
                ).squeeze(-1)
                flat_probs = logp.exp().reshape(-1, logp.shape[-1])
                flat_logp = logp.reshape(-1, logp.shape[-1])
                draws = torch.multinomial(
                    flat_probs, n_samples, replacement=True, generator=generator
                )
                sampled[:, s : s + 128] = (
                    flat_logp.gather(-1, draws).mean(dim=-1).reshape(logp.shape[0], logp.shape[1])
                )
            discrepancy = (actual - sampled) * mask
            scores = discrepancy.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            out[order[start : start + len(batch)]] = scores.cpu().numpy()
            # See _mean_token_loglik: delete the live per-batch tensors so
            # empty_cache can actually return their blocks to the driver.
            del encoded, logits, targets, mask, actual, sampled, discrepancy, scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if step % 25 == 0 or step == len(batches) - 1:
                print(f"    batch {step + 1}/{len(batches)}", flush=True)
    del model
    _free_model()
    return out
