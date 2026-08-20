"""Unit tests for the neural pilot infrastructure.

These tests are network-free: windowing/aggregation/objective logic uses a
whitespace mock tokenizer, and the model/training tests inject a tiny
config-built BERT so no pretrained weights are downloaded. They require torch
and transformers; on environments without them (e.g. the lightweight bench
interpreter) the whole module skips.
"""

from __future__ import annotations

import numpy as np
import pytest

try:  # the lightweight bench interpreter may lack a working torch/transformers
    import torch  # noqa: F401
    import transformers  # noqa: F401

    _HAS_ML = True
except Exception:  # ImportError or a broken-dependency ValueError
    _HAS_ML = False

if not _HAS_ML:
    pytest.skip("torch/transformers unavailable", allow_module_level=True)

from transformers import BertConfig, BertModel  # noqa: E402

from bench.neural import aggregate, data, objectives, windowing  # noqa: E402
from bench.neural.model import HierarchicalSummaryHead, WindowEncoder  # noqa: E402
from bench.neural.train import (  # noqa: E402
    PilotConfig,
    cohort_metrics,
    encode_corpus,
    train_window_encoder,
)


class MockTokenizer:
    """Whitespace tokenizer: one token per word, with char offsets."""

    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0
    bos_token_id = 101
    eos_token_id = 102

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        words = text.split()
        ids = list(range(2, 2 + len(words)))
        offsets = []
        pos = 0
        for w in words:
            start = text.index(w, pos)
            offsets.append((start, start + len(w)))
            pos = start + len(w)
        return {"input_ids": ids, "offset_mapping": offsets}


def _tiny_encoder(hidden=32, vocab=200):
    cfg = BertConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    return BertModel(cfg)


# ---------------------------------------------------------------- windowing


def test_windowing_tiles_short_document_single_window():
    tok = MockTokenizer()
    text = "one two three four five"
    wins = windowing.document_windows(text, tok, max_length=16, overlap=2)
    assert len(wins) == 1
    w = wins[0]
    # content tokens are wrapped by cls/sep
    assert w.input_ids[0] == tok.cls_token_id and w.input_ids[-1] == tok.sep_token_id
    assert (w.token_start, w.token_end) == (0, 5)
    assert text[w.char_start : w.char_end] == text


def test_windowing_overlap_and_spans_tile_long_document():
    tok = MockTokenizer()
    text = " ".join(f"w{i}" for i in range(40))
    max_length, overlap = 12, 4  # content budget 10, stride 6
    wins = windowing.document_windows(text, tok, max_length=max_length, overlap=overlap)
    assert len(wins) > 1
    # First window starts at token 0; last ends at the final token.
    assert wins[0].token_start == 0
    assert wins[-1].token_end == 40
    # Consecutive windows overlap by `overlap` content tokens.
    for a, b in zip(wins, wins[1:], strict=False):
        assert b.token_start == a.token_start + (max_length - 2 - overlap)
        assert b.token_start < a.token_end  # overlap region is shared
    # Char offsets are monotone and within the text.
    for w in wins:
        assert 0 <= w.char_start < w.char_end <= len(text)


def test_windowing_empty_text_yields_single_window():
    tok = MockTokenizer()
    wins = windowing.document_windows("   ", tok, max_length=16, overlap=2)
    assert len(wins) == 1
    assert (wins[0].token_start, wins[0].token_end) == (0, 0)


def test_pad_windows_pads_to_max_length():
    tok = MockTokenizer()
    wins = windowing.document_windows("a b c", tok, max_length=16, overlap=2)
    padded = windowing.pad_windows(wins, pad_id=0, max_length=16)
    assert all(len(w.input_ids) == 16 for w in padded)
    assert all(len(w.attention_mask) == 16 for w in padded)
    assert padded[0].attention_mask[-1] == 0  # padding is masked out


# --------------------------------------------------------------- aggregation


def test_overlap_corrected_single_window_equals_its_logodds():
    probs = aggregate.overlap_corrected_logodds(np.array([2.0]), [(0, 10)], n_tokens=10)
    assert probs == pytest.approx(2.0)


def test_overlap_corrected_no_double_count_on_overlap():
    # Two windows both covering tokens 4..8; the overlap must be averaged, not summed.
    window_lo = np.array([1.0, 3.0])
    spans = [(0, 8), (4, 12)]
    # tokens 0-3 -> 1.0 ; tokens 4-7 -> mean(1,3)=2 ; tokens 8-11 -> 3.0
    got = aggregate.overlap_corrected_logodds(window_lo, spans, n_tokens=12)
    expected = (4 * 1.0 + 4 * 2.0 + 4 * 3.0) / 12
    assert got == pytest.approx(expected)


def test_overlap_corrected_disjoint_windows_is_plain_mean():
    window_lo = np.array([0.0, 2.0])
    spans = [(0, 5), (5, 10)]
    got = aggregate.overlap_corrected_logodds(window_lo, spans, n_tokens=10)
    assert got == pytest.approx(1.0)


def test_log_odds_and_sigmoid():
    logits = np.array([[0.0, 2.0], [3.0, 1.0]])
    lo = aggregate.log_odds(logits)
    assert lo[0] == pytest.approx(2.0)
    assert lo[1] == pytest.approx(-2.0)
    assert float(aggregate.sigmoid(0.0)) == pytest.approx(0.5)


# ---------------------------------------------------------------- objectives


def test_group_key_format():
    assert data.group_key("cmv", "gpt-4", 1) == "cmv|gpt-4|1"


def test_group_balanced_weights_inverse_frequency():
    gk = ["a", "a", "a", "b"]  # group a:3, b:1
    w = objectives.group_balanced_weights(gk)
    # b (rare) is upweighted relative to a
    assert w[3] > w[0]
    assert np.mean(w) == pytest.approx(1.0)


def test_group_dro_shifts_weight_toward_worst_group():
    gk = ["a"] * 4 + ["b"] * 4
    dro = objectives.GroupDRO(gk, step_size=0.5)
    # group "b" has higher loss
    per = torch.tensor([0.1, 0.1, 0.1, 0.1, 2.0, 2.0, 2.0, 2.0])
    dro.loss(per, gk)
    assert dro.q[dro.g2i["b"]] > dro.q[dro.g2i["a"]]
    assert dro.q.sum() == pytest.approx(1.0)


def test_make_objective_kinds():
    gk = ["a", "b", "a"]
    name, payload = objectives.make_objective("erm", gk)
    assert name == "erm" and payload is None
    name, payload = objectives.make_objective("group_balanced", gk)
    assert payload.shape == (3,)
    name, payload = objectives.make_objective("group_dro", gk)
    assert isinstance(payload, objectives.GroupDRO)
    with pytest.raises(ValueError):
        objectives.make_objective("nope", gk)


# ------------------------------------------------------------------- metrics


def test_cohort_metrics_worst_cohort():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    # cohort x: perfect; cohort y: reversed (bad)
    probs = np.array([0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1])
    cohorts = ["x", "x", "x", "x", "y", "y", "y", "y"]
    m = cohort_metrics(labels, probs, cohorts)
    assert m["per_cohort"]["x"]["auroc"] == pytest.approx(1.0)
    assert m["per_cohort"]["y"]["auroc"] == pytest.approx(0.0)
    assert m["worst_cohort_auroc"] == pytest.approx(0.0)
    assert m["n_cohorts"] == 2


# ------------------------------------------------------- model + train (tiny)


def _tiny_window_encoder(hidden=32):
    return WindowEncoder(encoder=_tiny_encoder(hidden), hidden_size=hidden, num_labels=2)


def test_window_encoder_forward_shapes():
    model = _tiny_window_encoder()
    ids = torch.randint(0, 200, (3, 10))
    mask = torch.ones(3, 10, dtype=torch.long)
    logits, cls = model(ids, mask)
    assert logits.shape == (3, 2)
    assert cls.shape == (3, 32)


def test_summary_head_forward_masked():
    head = HierarchicalSummaryHead(hidden=32, nhead=4)
    embeds = torch.randn(2, 5, 32)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    logits = head(embeds, mask)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def _make_tiny_corpus(labels, n_tokens=8):
    """Build a WindowedCorpus of single-window docs directly (no tokenizer)."""
    from bench.neural.windowing import Window

    windows = []
    for _ in labels:
        ids = [101] + list(np.random.randint(2, 200, size=n_tokens)) + [102]
        windows.append([Window(ids, [1] * len(ids), 0, n_tokens, 0, n_tokens * 2)])
    return data.WindowedCorpus(
        windows=windows,
        labels=np.array(labels, dtype=np.int64),
        groups=[f"g{i}" for i in range(len(labels))],
        domains=["dA" if i % 2 == 0 else "dB" for i in range(len(labels))],
        families=["human" if label == 0 else "gen" for label in labels],
        group_keys=[f"d|f|{label}" for label in labels],
        n_tokens=[n_tokens] * len(labels),
        max_length=n_tokens + 2,
        overlap=0,
    )


def test_train_window_encoder_tiny_cpu_learnable():
    # Separable docs: label correlates with a token-id offset baked into embeds.
    labels = [0] * 12 + [1] * 12
    train = _make_tiny_corpus(labels)
    dev = _make_tiny_corpus(labels)
    model = _tiny_window_encoder()
    cfg = PilotConfig(batch_size=8, grad_accum=1, max_epochs=1, eval_batch_size=16)
    model, history = train_window_encoder(
        model, train, dev, "erm", None, cfg, device="cpu", log_prefix="[tiny] "
    )
    assert history["best"] is not None
    out = encode_corpus(model, dev, "cpu", 16)
    assert out.window_logits.shape[0] == len(labels)
    probs = out.doc_probabilities_logit_mean()
    assert probs.shape == (len(labels),)
    assert np.all((probs >= 0) & (probs <= 1))
