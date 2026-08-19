"""Tests for the neural runtime (NeuralModelManager + NeuralProseDetector).

Builds a tiny, fully offline artifact (config-built BERT + word-level tokenizer)
so no pretrained weights are downloaded. Requires torch/transformers; the module
skips on interpreters without them.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

from bench.neural.model import WindowEncoder  # noqa: E402
from panoptes.analysis.neural_runtime import (  # noqa: E402
    NeuralModelManager,
    NeuralProseDetector,
    NeuralRuntimeError,
)
from panoptes.schemas import ContentType  # noqa: E402

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_tiny_artifact(root: Path, *, hidden: int = 16, vocab_words: list[str] | None = None) -> Path:
    """Write a self-contained tiny ensemble artifact and return its directory."""
    root.mkdir(parents=True, exist_ok=True)
    vocab_words = vocab_words or WORDS
    vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2}
    for i, w in enumerate(vocab_words):
        vocab[w] = 3 + i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[PAD]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]", unk_token="[PAD]"
    )
    fast.save_pretrained(root)

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
            "encoder": "tiny-bert",
            "hf": "tiny-bert",
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
            "binary_calibrator": {"x_thresholds": [0.0, 0.5, 1.0], "y_thresholds": [0.05, 0.5, 0.95]}
        },
        "config_sha256": _sha256(root / "config.json"),
    }
    (root / "ensemble_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def _long_text() -> str:
    return " ".join(WORDS * 12)


def test_available_absent(tmp_path):
    manager = NeuralModelManager(artifact_dir=str(tmp_path / "nope"))
    assert manager.available() is False


def test_available_malformed(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "ensemble_manifest.json").write_text("not json", encoding="utf-8")
    assert NeuralModelManager(artifact_dir=str(bad)).available() is False


def test_load_and_score(tmp_path):
    artifact = _build_tiny_artifact(tmp_path / "art")
    manager = NeuralModelManager(artifact_dir=str(artifact), device="cpu")
    assert manager.available() is True
    out = manager.score_text(_long_text())
    assert 0.0 <= out["raw_participation"] <= 1.0
    assert 0.0 <= out["calibrated_participation"] <= 1.0
    assert out["n_windows"] >= 1
    assert out["n_seeds"] == 1
    assert out["seed_probabilities"] and len(out["seed_probabilities"]) == 1


def test_hash_mismatch_raises(tmp_path):
    artifact = _build_tiny_artifact(tmp_path / "art")
    # Tamper with the checkpoint after the manifest hash was recorded.
    enc = artifact / "seed-1" / "encoder.safetensors"
    enc.write_bytes(enc.read_bytes() + b"tamper")
    manager = NeuralModelManager(artifact_dir=str(artifact), device="cpu")
    with pytest.raises(NeuralRuntimeError, match="hash mismatch"):
        manager.ensemble()


def test_missing_checkpoint_raises(tmp_path):
    artifact = _build_tiny_artifact(tmp_path / "art")
    (artifact / "seed-1" / "encoder.safetensors").unlink()
    manager = NeuralModelManager(artifact_dir=str(artifact), device="cpu")
    with pytest.raises(NeuralRuntimeError, match="missing checkpoint"):
        manager.ensemble()


def test_adapter_abstains_non_english(tmp_path):
    artifact = _build_tiny_artifact(tmp_path / "art")
    adapter = NeuralProseDetector(NeuralModelManager(artifact_dir=str(artifact), device="cpu"))
    score = adapter.score(_long_text(), ContentType.PROSE, "es")
    assert score.abstain_reason is not None
    assert "English" in score.abstain_reason


def test_adapter_score_ternary(tmp_path):
    artifact = _build_tiny_artifact(tmp_path / "art")
    adapter = NeuralProseDetector(NeuralModelManager(artifact_dir=str(artifact), device="cpu"))
    score = adapter.score(_long_text(), ContentType.PROSE, "en")
    assert score.abstain_reason is None
    assert score.detector_id == "neural-ensemble-v1"
    dist = score.distribution
    total = dist.human + dist.ai_refined_or_mixed + dist.ai_generated
    assert abs(total - 1.0) < 1e-6
    assert 0.0 <= dist.human <= 1.0
