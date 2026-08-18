"""Unicode (zero-width) provenance watermark — the second family that tools like
watermarks-remover claim to strip.

Distinct from the statistical KGW green-list watermark (which lives in word
*choice*), a Unicode watermark hides information in invisible characters
interleaved between words. It is the scheme targeted by "Unicode hygiene"
removal. We embed a fixed binary signature using two zero-width codepoints and
detect it by reading the codepoints back. This family is known to be fragile:
any transform that strips or normalizes invisible characters destroys it.
"""

from __future__ import annotations

# ZWNJ (U+200C) encodes bit 0, ZWSP (U+200B) encodes bit 1.
_ZERO = "\u200c"
_ONE = "\u200b"
_MARK_CHARS = frozenset({_ZERO, _ONE})

# A fixed 16-bit signature so detection can verify payload structure, not just
# the presence of stray invisible characters.
SIGNATURE = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0]

# Common invisible / format codepoints a hygiene pass strips.
_INVISIBLE = {
    "​",  # ZWSP
    "‌",  # ZWNJ
    "‍",  # ZWJ
    "﻿",  # BOM / zero-width no-break space
    "⁠",  # word joiner
    "­",  # soft hyphen
    "‎",  # LRM
    "‏",  # RLM
}


def embed_unicode_watermark(text: str, signature: list[int] | None = None) -> str:
    """Prefix each whitespace-separated word with one zero-width bit, cycling the
    signature. Prefixing keeps sentence-ending punctuation at word boundaries so
    sentence-level edits (e.g. shuffle) still operate on the text."""
    sig = signature or SIGNATURE
    words = text.split(" ")
    out = []
    for i, word in enumerate(words):
        bit = sig[i % len(sig)]
        out.append((_ONE if bit else _ZERO) + word)
    return " ".join(out)


def read_marks(text: str) -> list[int]:
    """Extract the ordered zero-width bit stream from text."""
    return [1 if ch == _ONE else 0 for ch in text if ch in _MARK_CHARS]


def detect_unicode_watermark(text: str, signature: list[int] | None = None) -> dict:
    """Detect the embedded signature. Returns presence, mark count, and the
    fraction of extracted bits matching the expected repeating signature."""
    sig = signature or SIGNATURE
    bits = read_marks(text)
    n = len(bits)
    # Need enough marks to be confident; short passages carry fewer words than
    # signature bits, so score the match over whatever marks are present.
    if n < 8:
        return {"present": False, "n_marks": n, "payload_match": 0.0}
    matches = sum(1 for i, b in enumerate(bits) if b == sig[i % len(sig)])
    frac = matches / n
    return {"present": frac >= 0.9, "n_marks": n, "payload_match": frac}


def strip_invisible(text: str) -> str:
    """Remove zero-width / format characters (the 'Unicode hygiene' attack)."""
    return "".join(ch for ch in text if ch not in _INVISIBLE)
