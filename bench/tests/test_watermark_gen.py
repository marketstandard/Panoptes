"""Unit tests for the KGW green-list generator helpers (no model download)."""

from __future__ import annotations

import numpy as np

from bench.watermark_gen import biased_pick, candidate_vocabulary
from panoptes.analysis.watermarks import _green_for, green_for


class _FakeTokenizer:
    """Minimal tokenizer: id -> decoded string."""

    def __init__(self, mapping: dict[int, str]):
        self._m = mapping

    def __len__(self) -> int:
        return len(self._m)

    def decode(self, ids) -> str:
        return self._m[ids[0]]


def test_green_for_matches_private_prf() -> None:
    for prev in ["", "the", "cat", "."]:
        for tok in ["dog", "runs", ",", "quickly"]:
            assert green_for(prev, tok) == _green_for(prev, tok)


def test_candidate_vocabulary_filters_to_single_regex_tokens() -> None:
    tok = _FakeTokenizer(
        {
            0: " the",      # word-initial word -> keep ("the")
            1: "ing",       # fragment without leading space -> drop
            2: " ing",      # word-initial fragment -> kept as a word token
            3: ".",         # bare punctuation -> keep
            4: " ,",        # spaced punctuation -> drop (we keep punctuation attached)
            5: " hello world",  # two words -> drop
            6: "",          # empty -> drop
        }
    )
    cand = candidate_vocabulary(tok)
    got = dict(zip(cand.ids, cand.tokens))
    assert got[0] == "the"
    assert got[2] == "ing"
    assert got[3] == "."
    assert 1 not in got
    assert 4 not in got
    assert 5 not in got
    assert 6 not in got


def test_biased_pick_favors_green_candidates() -> None:
    previous = "the"
    tokens = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    green_flags = [green_for(previous, t) for t in tokens]
    # Need at least one green and one red for a meaningful test.
    assert any(green_flags) and not all(green_flags)
    logits = np.zeros(len(tokens))  # all equal -> only the bias separates them
    rng = np.random.default_rng(0)
    picks = [biased_pick(logits, tokens, previous, delta=8.0, rng=rng) for _ in range(300)]
    green_picks = sum(1 for p in picks if green_flags[p])
    assert green_picks / len(picks) > 0.9


def test_biased_pick_no_bias_is_valid_and_spreads() -> None:
    previous = "the"
    tokens = ["alpha", "bravo", "charlie", "delta"]
    logits = np.zeros(len(tokens))
    rng = np.random.default_rng(1)
    picks = [biased_pick(logits, tokens, previous, delta=0.0, rng=rng) for _ in range(400)]
    assert all(0 <= p < len(tokens) for p in picks)
    # With no bias and equal logits, every candidate should be picked sometimes.
    assert len(set(picks)) == len(tokens)


def test_biased_pick_top_k_restricts_pool() -> None:
    previous = "x"
    tokens = [f"w{i}" for i in range(10)]
    # Make candidate 0 overwhelmingly the top logit.
    logits = np.array([100.0] + [0.0] * 9)
    rng = np.random.default_rng(2)
    picks = {biased_pick(logits, tokens, previous, delta=0.0, rng=rng, top_k=1) for _ in range(20)}
    assert picks == {0}
