"""Unit tests for the unicode watermark scheme and the removal attack battery."""

from __future__ import annotations

from bench.watermark_attacks import ATTACKS, synonym_substitute, unicode_scrub
from bench.watermark_unicode import (
    detect_unicode_watermark,
    embed_unicode_watermark,
    read_marks,
    strip_invisible,
)

PLAIN = "The committee released its report on Tuesday and members praised the work."


def test_unicode_roundtrip_detects() -> None:
    marked = embed_unicode_watermark(PLAIN)
    res = detect_unicode_watermark(marked)
    assert res["present"] is True
    assert res["n_marks"] > 0
    assert res["payload_match"] >= 0.9


def test_plain_text_not_detected() -> None:
    res = detect_unicode_watermark(PLAIN)
    assert res["present"] is False
    assert res["n_marks"] == 0


def test_strip_invisible_removes_marks() -> None:
    marked = embed_unicode_watermark(PLAIN)
    assert read_marks(marked)  # marks present
    scrubbed = strip_invisible(marked)
    assert read_marks(scrubbed) == []
    assert detect_unicode_watermark(scrubbed)["present"] is False


def test_unicode_scrub_kills_unicode_watermark() -> None:
    marked = embed_unicode_watermark(PLAIN)
    assert detect_unicode_watermark(unicode_scrub(marked))["present"] is False


def test_unicode_scrub_noop_on_clean_ascii() -> None:
    assert unicode_scrub(PLAIN) == PLAIN


def test_synonym_substitute_changes_some_tokens_deterministically() -> None:
    text = "The report said the new study found many important results that people use."
    out1 = synonym_substitute(text, rate=1.0, seed=1)
    out2 = synonym_substitute(text, rate=1.0, seed=1)
    assert out1 == out2  # deterministic
    assert out1 != text  # something changed
    # rate=1.0 replaces every mappable word; 'said' -> 'stated' is in the map
    assert "stated" in out1


def test_synonym_substitute_rate_zero_is_identity() -> None:
    text = "The report said the new study found results."
    assert synonym_substitute(text, rate=0.0, seed=1) == text


def test_synonym_substitute_preserves_zero_width_marks() -> None:
    marked = embed_unicode_watermark(PLAIN)
    out = synonym_substitute(marked, rate=0.5, seed=2)
    # The attack swaps words but does not strip invisible characters.
    assert len(read_marks(out)) == len(read_marks(marked))


def test_attacks_registry_has_expected_keys() -> None:
    for key in ["identity", "unicode_scrub", "synonym_30", "synonym_60", "truncate_50"]:
        assert key in ATTACKS
    assert ATTACKS["identity"]("abc", 0) == "abc"
