"""Tests for bench/external_baselines.py with mocked weights.

The GPU scorers are exercised on CPU by monkeypatching the model loader with
small fake causal LMs whose log-likelihoods are computable by hand, so the
masking, normalization, sampling, and score-composition logic is verified
without downloading weights.

Run from the repository root: python -m pytest bench/tests/test_external_baselines.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="external-baseline extras not installed")

from bench import external_baselines as eb  # noqa: E402


class _FakeEncoded(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    """Character-level tokenizer over a fixed vocab, HF-shaped surface."""

    def __init__(self, vocab: int):
        self.vocab = vocab
        self.pad_token = None  # exercise the pad_token=None branch
        self.eos_token = "<eos>"

    def __call__(self, batch, return_tensors, padding, truncation, max_length):
        rows = [
            [ord(c) % (self.vocab - 1) + 1 for c in text][:max_length] for text in batch
        ]
        longest = max(len(row) for row in rows)
        input_ids = torch.zeros(len(batch), longest, dtype=torch.long)
        mask = torch.zeros(len(batch), longest, dtype=torch.long)
        for r, row in enumerate(rows):
            input_ids[r, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[r, : len(row)] = 1
        return _FakeEncoded(input_ids=input_ids, attention_mask=mask)


class _FakeOut:
    def __init__(self, logits):
        self.logits = logits


class _UniformModel:
    """Every position: uniform distribution over the vocab."""

    def __init__(self, vocab: int):
        self.vocab = vocab

    def __call__(self, input_ids, attention_mask):
        batch, length = input_ids.shape
        return _FakeOut(torch.zeros(batch, length, self.vocab))


class _PeakedModel:
    """Every position: all mass on `token` (logit `peak`), uniform elsewhere."""

    def __init__(self, vocab: int, token: int, peak: float = 10.0):
        self.vocab = vocab
        self.token = token
        self.peak = peak

    def __call__(self, input_ids, attention_mask):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab)
        logits[:, :, self.token] = self.peak
        return _FakeOut(logits)


class _EchoModel:
    """Every position: all mass on the actual next input token."""

    def __init__(self, vocab: int, peak: float = 12.0):
        self.vocab = vocab
        self.peak = peak

    def __call__(self, input_ids, attention_mask):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab)
        targets = input_ids[:, 1:]
        logits[:, :-1].scatter_(-1, targets.unsqueeze(-1), self.peak)
        logits[:, -1, 0] = self.peak
        return _FakeOut(logits)


def test_precomputed_score_detector_surface():
    detector = eb.PrecomputedScoreDetector({0: 2.0, 1: -2.0, 2: 0.0}, temperature=0.5)
    assert detector.fit(object(), [0, 1]) is detector  # no-op fit returns self
    probs = detector.predict_proba(object(), [0, 1, 2])
    assert probs.shape == (3,)
    assert np.all((probs > 0.0) & (probs < 1.0))
    expected = 1.0 / (1.0 + math.exp(-(2.0 / 0.5)))
    assert probs[0] == pytest.approx(expected)
    assert probs[0] > probs[2] > probs[1]  # monotone in the raw score


def test_precomputed_score_detector_clips_extremes():
    detector = eb.PrecomputedScoreDetector({0: 1e9, 1: -1e9})
    probs = detector.predict_proba(object(), [0, 1])
    assert np.isfinite(probs).all()
    assert probs[0] > 0.999999
    assert probs[1] < 0.000001


def test_precomputed_score_detector_missing_index_raises():
    detector = eb.PrecomputedScoreDetector({0: 1.0})
    with pytest.raises(KeyError):
        detector.predict_proba(object(), [7])


def test_batches_chunking():
    batches = list(eb._batches(list("abcdefg"), 3))
    assert batches == [(0, ["a", "b", "c"]), (3, ["d", "e", "f"]), (6, ["g"])]


def test_token_budget_batches_respect_budget_and_order():
    # 4 chars per estimated token; budget 10 tokens -> 40 chars.
    texts = ["a" * 20, "b" * 20, "c" * 36, "d" * 4, "e" * 80]
    batches = eb._token_budget_batches(texts, max_batch=4, token_budget=10)
    assert [b[0] for b in batches] == [0, 2, 3, 4]  # cumulative starts
    assert [len(b[1]) for b in batches] == [2, 1, 1, 1]
    # Every batch satisfies size x max_est_tokens <= budget (except a lone overlong text).
    for _, batch in batches:
        est_max = max((len(t) + 3) // 4 for t in batch)
        assert len(batch) * est_max <= 10 or len(batch) == 1
    # Concatenating batches reproduces the input order (input is pre-sorted).
    assert [t for _, batch in batches for t in batch] == texts


def test_token_budget_batches_empty():
    assert eb._token_budget_batches([], max_batch=4, token_budget=10) == []


def test_mean_token_loglik_uniform_model():
    vocab = 64
    tokenizer = _FakeTokenizer(vocab)
    model = _UniformModel(vocab)
    texts = ["hello world", "a much longer piece of text with many characters"]
    scores = eb._mean_token_loglik(model, tokenizer, texts, max_tokens=128, batch_size=2)
    # Uniform logits: every real token contributes -log(vocab); padding excluded.
    assert scores.shape == (2,)
    assert scores[0] == pytest.approx(-math.log(vocab), abs=1e-5)
    assert scores[1] == pytest.approx(-math.log(vocab), abs=1e-5)


def test_mean_token_loglik_masking_excludes_padding():
    vocab = 64
    peak_token = 7
    tokenizer = _FakeTokenizer(vocab)
    model = _PeakedModel(vocab, token=peak_token, peak=10.0)
    texts = ["aaaa", "aaaaaaaaaaaa"]  # different lengths -> padding in the same batch
    scores = eb._mean_token_loglik(model, tokenizer, texts, max_tokens=64, batch_size=2)
    # Hand-compute: target tokens are input_ids shifted left; each row's mean is
    # over its own real tokens only.
    z = math.log(math.exp(10.0) + vocab - 1)
    logp_hit = 10.0 - z
    logp_miss = -z
    for text, score in zip(texts, scores, strict=True):
        ids = [ord(c) % (vocab - 1) + 1 for c in text]
        targets = ids[1:]
        expected = np.mean([logp_hit if t == peak_token else logp_miss for t in targets])
        assert score == pytest.approx(expected, abs=1e-4)


def test_binoculars_score_composition(monkeypatch):
    vocab = 64
    tokenizer = _FakeTokenizer(vocab)
    # Observer: uniform (L_obs = -log V). Performer: peaked (L_perf = logp of targets).
    logliks = [np.array([-math.log(vocab), -math.log(vocab)]), np.array([-2.0, -3.0])]
    calls = {"load": 0, "ll": 0}

    def fake_load(name, torch_dtype="bfloat16"):
        calls["load"] += 1
        return tokenizer, object()

    def fake_loglik(model, tok, texts, max_tokens, batch_size=8, progress=False):
        array = logliks[calls["ll"]]
        calls["ll"] += 1
        assert len(texts) == 2
        return array

    monkeypatch.setattr(eb, "_load_causal_lm", fake_load)
    monkeypatch.setattr(eb, "_mean_token_loglik", fake_loglik)
    scores = eb.binoculars_scores(["text one", "text two"])
    # s = (-L_obs) / exp(-L_perf)
    expected = math.log(vocab) / np.exp(np.array([2.0, 3.0]))
    assert scores.shape == (2,)
    assert np.allclose(scores, expected)
    assert calls == {"load": 2, "ll": 2}  # observer and performer each loaded once


def test_fast_detectgpt_perfect_prediction_scores_zero(monkeypatch):
    vocab = 64
    tokenizer = _FakeTokenizer(vocab)
    monkeypatch.setattr(
        eb, "_load_causal_lm", lambda name, torch_dtype="bfloat16": (tokenizer, _EchoModel(vocab))
    )
    texts = ["some text here", "different content entirely"]
    scores = eb.fast_detectgpt_scores(texts, n_samples=3, max_tokens=64, batch_size=2)
    assert scores.shape == (2,)
    # Model puts ~all mass on the actual next token: actual ~= sampled ~= 0.
    assert np.all(np.abs(scores) < 1e-3)


def test_fast_detectgpt_seeded_sampling_is_deterministic(monkeypatch):
    vocab = 64
    tokenizer = _FakeTokenizer(vocab)
    monkeypatch.setattr(
        eb, "_load_causal_lm", lambda name, torch_dtype="bfloat16": (tokenizer, _PeakedModel(vocab, token=5, peak=8.0))
    )
    texts = ["text with several tokens", "another one here"]
    first = eb.fast_detectgpt_scores(texts, n_samples=4, max_tokens=64, batch_size=2, seed=13)
    second = eb.fast_detectgpt_scores(texts, n_samples=4, max_tokens=64, batch_size=2, seed=13)
    assert np.allclose(first, second)  # seeded sampling is reproducible
    # A peaked model: positions whose target is the peak token score > 0
    # (actual logp >> expected logp under the model's own distribution).
    assert np.all(np.isfinite(first))
