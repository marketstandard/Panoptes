"""Tests for the NeuralTrainableDetector (Phase 6 LOCO neural path).

Injects a tiny config-built BERT and a whitespace mock tokenizer so no
pretrained weights are downloaded and training runs on CPU in seconds. Requires
torch/transformers; skips on interpreters without them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401

    _HAS_ML = True
except Exception:
    _HAS_ML = False

if not _HAS_ML:
    pytest.skip("torch/transformers unavailable", allow_module_level=True)

from transformers import BertConfig, BertModel  # noqa: E402

from bench.datasets import Dataset  # noqa: E402
from bench.neural import detector as detector_mod  # noqa: E402
from bench.neural.model import WindowEncoder  # noqa: E402


class MockTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0
    bos_token_id = 101
    eos_token_id = 102

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        words = text.split()
        ids = [2 + (hash(w) % 50) for w in words]
        offsets = []
        pos = 0
        for w in words:
            start = text.index(w, pos)
            offsets.append((start, start + len(w)))
            pos = start + len(w)
        return {"input_ids": ids, "offset_mapping": offsets}


def _tiny_window_encoder(hidden=24, vocab=200):
    cfg = BertConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=48,
        max_position_embeddings=64,
    )
    return WindowEncoder(encoder=BertModel(cfg), hidden_size=hidden, num_labels=2)


def _dataset(n=24, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    ai = ["utilize", "furthermore", "consequently", "comprehensive", "demonstrate"]
    hu = ["the", "cat", "sat", "on", "mat", "red"]
    texts, labels = [], []
    for i in range(n):
        label = i % 2
        vocab = ai if label else hu
        texts.append(" ".join(rng.choice(vocab, size=12)))
        labels.append(label)
    return Dataset(
        texts=texts,
        labels=np.array(labels, dtype=int),
        families=["gen" if label else "human" for label in labels],
        kinds=["text"] * n,
        groups=[f"g{i // 4}" for i in range(n)],
        buckets=["short"] * n,
        provenance="tiny",
        sha256="0" * 64,
        domains=["d0"] * n,
    )


_REAL_IMPORT_STACK = detector_mod._import_stack


def _patched_stack():
    stack = _REAL_IMPORT_STACK()
    stack["WindowEncoder"] = lambda hf: _tiny_window_encoder()
    stack["AutoTokenizer"] = type(
        "AT", (), {"from_pretrained": staticmethod(lambda hf: MockTokenizer())}
    )
    return stack


def test_trainable_detector_fit_predict_cpu(monkeypatch):
    monkeypatch.setattr(detector_mod, "_import_stack", _patched_stack)
    winner = {
        "encoder": "tiny",
        "hf": "tiny",
        "objective": "erm",
        "aggregation": "overlap_corrected_logit_mean",
        "max_length": 24,
        "overlap": 4,
    }
    det = detector_mod.NeuralTrainableDetector(
        winner,
        device="cpu",
        max_windows=4,
        config_overrides={"max_epochs": 1, "batch_size": 8, "grad_accum": 1},
    )
    ds = _dataset()
    idx = np.arange(len(ds))
    det.fit(ds, idx)
    probs = det.predict_proba(ds, idx)
    assert probs.shape == (len(ds),)
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_trainable_detector_requires_fit_first():
    winner = {
        "encoder": "tiny",
        "hf": "tiny",
        "objective": "erm",
        "aggregation": "overlap_corrected_logit_mean",
        "max_length": 24,
        "overlap": 4,
    }
    det = detector_mod.NeuralTrainableDetector(winner, device="cpu")
    ds = _dataset()
    with pytest.raises(RuntimeError, match="fit"):
        det.predict_proba(ds, np.arange(len(ds)))
