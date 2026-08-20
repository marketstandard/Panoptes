"""Tests for the FrozenNeuralDetector bench adapter (Phase 5 -> Phase 6 bridge).

Builds a tiny, fully offline ensemble artifact (config-built BERT + word-level
tokenizer) so no pretrained weights are downloaded. Requires torch/transformers;
skips on interpreters without them.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    from safetensors.torch import save_file

    _HAS_ML = True
except Exception:
    _HAS_ML = False

if not _HAS_ML:
    pytest.skip("torch/transformers unavailable", allow_module_level=True)

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from bench.datasets import Dataset  # noqa: E402
from bench.neural.frozen import FrozenNeuralDetector  # noqa: E402
from bench.neural.model import WindowEncoder  # noqa: E402

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_tiny_artifact(root: Path, hidden: int = 16) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2}
    for i, w in enumerate(WORDS):
        vocab[w] = 3 + i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[PAD]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tok,
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[PAD]",
    ).save_pretrained(root)
    cfg = BertConfig(
        vocab_size=len(vocab),
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    cfg.save_pretrained(root)
    seed_dir = root / "seed-1"
    seed_dir.mkdir()
    model = WindowEncoder(encoder=BertModel(cfg), hidden_size=hidden, num_labels=2)
    enc_path = seed_dir / "encoder.safetensors"
    save_file(model.state_dict(), str(enc_path))
    manifest = {
        "schema": "panoptes-neural-ensemble-v1",
        "winner": {
            "encoder": "tiny",
            "hf": "tiny",
            "objective": "erm",
            "aggregation": "overlap_corrected_logit_mean",
            "max_length": 16,
            "overlap": 4,
        },
        "seeds": [
            {
                "seed": 1,
                "encoder": "seed-1/encoder.safetensors",
                "encoder_sha256": _sha256(enc_path),
                "summary_head": None,
                "summary_head_sha256": None,
            }
        ],
        "calibration": {
            "binary_calibrator": {"x_thresholds": [0.0, 1.0], "y_thresholds": [0.0, 1.0]}
        },
        "config_sha256": _sha256(root / "config.json"),
    }
    (root / "ensemble_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _dataset(n: int = 12) -> Dataset:
    rng = np.random.default_rng(0)
    texts = [" ".join(rng.choice(WORDS, size=10)) for _ in range(n)]
    labels = np.array([i % 2 for i in range(n)], dtype=int)
    return Dataset(
        texts=texts,
        labels=labels,
        families=["gen" if label else "human" for i, label in enumerate(labels)],
        kinds=["text"] * n,
        groups=[f"g{i // 4}" for i in range(n)],
        buckets=["short"] * n,
        provenance="tiny",
        sha256="0" * 64,
        domains=["d0"] * n,
    )


def test_frozen_neural_detector_interface(tmp_path):
    artifact = _build_tiny_artifact(tmp_path / "art")
    det = FrozenNeuralDetector(artifact_dir=str(artifact), device="cpu")
    assert det.available() is True
    ds = _dataset()
    idx = np.arange(len(ds))
    out = det.fit(ds, idx)  # no-op
    assert out is det
    probs = det.predict_proba(ds, idx)
    assert probs.shape == (len(ds),)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_frozen_neural_detector_unavailable_raises_on_score(tmp_path):
    det = FrozenNeuralDetector(artifact_dir=str(tmp_path / "missing"), device="cpu")
    assert det.available() is False
    ds = _dataset()
    with pytest.raises(FileNotFoundError):
        det.predict_proba(ds, np.arange(len(ds)))
