"""Watermark-removal attack battery.

Each transform maps text -> text and mirrors a real removal technique. Some
target Unicode (zero-width) watermarks, others target statistical (green-list)
watermarks, and the proxy edits stress both. The removal evaluation applies
each attack to watermarked passages and measures how much detection survives.

The transforms are deterministic (seeded) so the signed removal card is
reproducible. ``llm_paraphrase`` is the strongest statistical attack but needs
a generation model; it is provided as an opt-in hook, not run by default.
"""

from __future__ import annotations

import re

import numpy as np

from bench.robustness import (
    drop_tokens,
    lowercase,
    shuffle_sentences,
    strip_punctuation,
    truncate,
)
from bench.watermark_unicode import strip_invisible

# A small set of Latin lookalikes (Cyrillic/Greek) that a hygiene pass folds
# back to ASCII. The generated corpus is ASCII, so this is a no-op there; it
# matters for real-world text that carries homoglyph-based marks.
_HOMOGLYPHS = {
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "х": "x",
    "у": "y",
    "н": "h",
    "к": "k",
    "м": "m",
    "т": "t",
    "в": "b",
    "ο": "o",
    "α": "a",
    "ε": "e",
    "ρ": "p",
    "ν": "v",
}

# Deterministic synonym map for the statistical-rewrite attack. Swapping a
# token changes both its own green-list membership and the next token's (the
# green list is keyed on the previous token), so even a modest rate disrupts a
# green-list watermark.
_SYNONYMS = {
    "said": "stated",
    "say": "suggest",
    "says": "suggests",
    "new": "recent",
    "more": "additional",
    "also": "likewise",
    "big": "large",
    "small": "little",
    "show": "demonstrate",
    "shows": "demonstrates",
    "find": "discover",
    "found": "discovered",
    "use": "utilize",
    "used": "utilized",
    "help": "assist",
    "make": "create",
    "made": "created",
    "get": "obtain",
    "need": "require",
    "important": "significant",
    "many": "numerous",
    "good": "beneficial",
    "bad": "harmful",
    "great": "excellent",
    "first": "initial",
    "last": "final",
    "long": "extended",
    "short": "brief",
    "high": "elevated",
    "low": "reduced",
    "old": "ancient",
    "young": "youthful",
    "people": "individuals",
    "city": "municipality",
    "team": "squad",
    "study": "investigation",
    "report": "account",
    "according": "per",
    "about": "regarding",
    "after": "following",
    "before": "preceding",
    "during": "throughout",
    "because": "since",
    "but": "however",
    "so": "therefore",
    "if": "provided",
    "when": "upon",
    "while": "whereas",
    "this": "the",
    "that": "which",
    "these": "those",
    "their": "its",
    "they": "these",
    "them": "those",
    "have": "possess",
    "has": "possesses",
    "will": "shall",
    "would": "might",
    "could": "might",
    "can": "may",
    "over": "throughout",
    "under": "beneath",
}

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WORD_ONLY = re.compile(r"^\w+$")


def unicode_scrub(text: str) -> str:
    """Strip zero-width/format characters and fold homoglyphs to ASCII.

    Mirrors watermarks-remover's 'Unicode hygiene'. Destroys Unicode watermarks
    but leaves word choice — and therefore statistical watermarks — untouched.
    """
    scrubbed = strip_invisible(text)
    return "".join(_HOMOGLYPHS.get(ch, ch) for ch in scrubbed)


def synonym_substitute(text: str, rate: float = 0.3, seed: int = 13) -> str:
    """Deterministically replace a fraction of mappable words with synonyms.

    A stand-in for the 'statistical rewrite' / paraphrase attack: it perturbs
    word choice (breaking green-list runs) while roughly preserving meaning.
    Operates on regex tokens and preserves any trailing zero-width marks, so it
    does not by itself strip a Unicode watermark.
    """
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return text
    rng = np.random.default_rng(seed)
    out = []
    for token in tokens:
        synonym = _SYNONYMS.get(token.lower())
        if synonym is not None and _WORD_ONLY.match(token) and rng.random() < rate:
            out.append(synonym)
        else:
            out.append(token)
    # Re-render: words spaced, punctuation attached to the preceding token.
    parts: list[str] = []
    for token in out:
        if _WORD_ONLY.match(token):
            parts.append(" " + token if parts else token)
        else:
            parts.append(token)
    return "".join(parts)


def llm_paraphrase(text: str, *, seed: int = 13) -> str:  # pragma: no cover - opt-in
    """Strongest statistical attack: a full rewrite. Requires a generation
    backend; not run by default. Anthropic notes a complete rewrite defeats a
    statistical watermark. Provided as a documented hook for the repo harness.
    """
    raise NotImplementedError(
        "llm_paraphrase requires a generation backend; run it via the external "
        "repo harness (bench evaluate-repo) with a paraphrasing tool."
    )


# name -> transform(text, index)  (index seeds per-passage determinism)
ATTACKS = {
    "identity": lambda text, i: text,
    "unicode_scrub": lambda text, i: unicode_scrub(text),
    "synonym_30": lambda text, i: synonym_substitute(text, rate=0.30, seed=13 + i),
    "synonym_60": lambda text, i: synonym_substitute(text, rate=0.60, seed=13 + i),
    "truncate_50": lambda text, i: truncate(text, 0.50),
    "drop_20": lambda text, i: drop_tokens(text, 0.20, seed=13 + i),
    "shuffle_sentences": lambda text, i: shuffle_sentences(text, seed=13 + i),
    "lowercase": lambda text, i: lowercase(text),
    "strip_punctuation": lambda text, i: strip_punctuation(text),
}
