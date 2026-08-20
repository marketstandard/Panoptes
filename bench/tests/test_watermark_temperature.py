"""Tests for temperature-scaled KGW sampling (no model download)."""

from __future__ import annotations

import numpy as np
from panoptes.analysis.watermarks import green_for

from bench.run_watermark_temperature import build_card, run_synthetic_sweep, synthesize_passage
from bench.watermark_gen import biased_pick


def test_temperature_zero_is_deterministic_argmax() -> None:
    previous = "the"
    tokens = ["alpha", "bravo", "charlie", "delta"]
    logits = np.array([1.0, 0.5, 0.2, 0.1])
    rng = np.random.default_rng(0)
    picks = {
        biased_pick(logits, tokens, previous, delta=0.0, rng=rng, temperature=0.0)
        for _ in range(20)
    }
    assert picks == {0}


def test_temperature_zero_bias_can_flip_argmax() -> None:
    previous = "the"
    tokens = ["alpha", "bravo", "charlie", "delta"]
    # Make a red candidate slightly preferred before bias; large delta should
    # flip to a green candidate under greedy selection.
    green_flags = [green_for(previous, t) for t in tokens]
    assert any(green_flags) and not all(green_flags)
    logits = np.array([2.0 if not g else 1.5 for g in green_flags], dtype=float)
    rng = np.random.default_rng(1)
    pick = biased_pick(logits, tokens, previous, delta=5.0, rng=rng, temperature=0.0)
    assert green_flags[pick]


def test_high_temperature_spreads_more_than_low() -> None:
    previous = "the"
    tokens = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    logits = np.linspace(3.0, 0.0, num=len(tokens))
    rng_low = np.random.default_rng(2)
    rng_high = np.random.default_rng(2)
    low = [
        biased_pick(logits, tokens, previous, delta=0.0, rng=rng_low, temperature=0.3)
        for _ in range(400)
    ]
    high = [
        biased_pick(logits, tokens, previous, delta=0.0, rng=rng_high, temperature=1.5)
        for _ in range(400)
    ]
    assert len(set(high)) >= len(set(low))


def test_synthesize_passage_and_sweep_card_shape() -> None:
    text = synthesize_passage(n_tokens=60, delta=2.0, temperature=1.0, seed=0)
    assert len(text.split()) > 20
    cells = run_synthetic_sweep([0.0, 1.0], [0.0, 2.0], n_passages=8, n_tokens=60)
    assert len(cells) == 4
    watermarked = next(c for c in cells if c["temperature"] == 1.0 and c["delta"] == 2.0)
    control = next(c for c in cells if c["temperature"] == 1.0 and c["delta"] == 0.0)
    assert watermarked["mean_green_rate"] is not None
    assert control["mean_green_rate"] is not None
    assert watermarked["mean_green_rate"] > control["mean_green_rate"]
    card = build_card(cells, mode="synthetic", generator={"key_id": "panoptes-demo-key"})
    assert card["schema"] == "panoptes-watermark-temperature-card-v1"
    assert 0.0 in card["temperatures"]
