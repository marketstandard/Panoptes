"""Single-source stylometric feature extraction.

Every part of the project that turns text into numbers — the corpus
builder, the methodology layer, the calibration fit, and the bench model
zoo — uses this module, so train/serve skew is impossible by
construction. The seven names in ``ATTRIBUTION_FEATURES`` are computed
with exactly the same formulas as the backend runtime
(``backend/panoptes/analysis/attribution.py``); a test in
``bench/tests/test_bench.py`` asserts the two implementations agree on
fixture texts.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"\b[\w']+\b")
_SENTENCE_RE = re.compile(r"[.!?]+")
_CONNECTORS = {"however", "therefore", "moreover", "additionally", "overall", "furthermore"}
_STRUCTURE_MARKERS = ("\n-", "\n*", ":", ";", "(", ")", "[", "]")
_PUNCT_CHARS = ".,;:!?()[]{}"

# The seven features the backend source-family geometry consumes, in the
# canonical order used when fitting the calibration artifact.
ATTRIBUTION_FEATURES: tuple[str, ...] = (
    "long_words",
    "connectors",
    "unique_ratio",
    "short_sentences",
    "structured",
    "digits",
    "balanced_lines",
)

# Full bench feature vector, canonical order. Artifacts store this list so
# served models always know which column is which.
FEATURE_NAMES: tuple[str, ...] = (
    *ATTRIBUTION_FEATURES,
    "token_entropy",
    "punctuation",
    "mean_word_length",
    "word_length_var",
    "avg_line_len",
    "line_sd",
    "sentence_len_mean",
    "sentence_len_sd",
    "hapax_ratio",
    "log_words",
)

LENGTH_BUCKETS: tuple[str, ...] = ("lt50", "50-149", "150-499", "500plus")


def word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def length_bucket(word_count: int) -> str:
    if word_count < 50:
        return "lt50"
    if word_count < 150:
        return "50-149"
    if word_count < 500:
        return "150-499"
    return "500plus"


def extract(text: str, kind: str = "text") -> dict[str, float]:
    """Compute the full feature mapping for one document.

    ``kind`` is ``"text"`` or ``"code"``; it only affects ``structured``,
    matching the runtime convention that code is structured by definition.
    """
    words = word_tokens(text)
    if not words:
        return {name: 0.0 for name in FEATURE_NAMES}

    counts = Counter(words)
    lengths = [len(word) for word in words]
    mean_length = sum(lengths) / len(lengths)
    length_var = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)

    sentence_lengths = [
        len(_WORD_RE.findall(sentence)) for sentence in _SENTENCE_RE.split(text) if sentence.strip()
    ]
    sentence_mean = (
        sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
    )
    sentence_sd = _stdev(sentence_lengths)

    lines = [line for line in text.splitlines() if line.strip()]
    line_lengths = [len(line) for line in lines]
    avg_line = sum(line_lengths) / max(len(line_lengths), 1)
    line_sd = _stdev(line_lengths)

    token_entropy = -sum(
        (count / len(words)) * math.log2(count / len(words)) for count in counts.values()
    )

    return {
        "long_words": sum(length >= 7 for length in lengths) / len(lengths),
        "connectors": sum(word in _CONNECTORS for word in words) / len(words),
        "unique_ratio": len(counts) / len(words),
        "short_sentences": (
            sum(length < 14 for length in sentence_lengths) / len(sentence_lengths)
            if sentence_lengths
            else 0.0
        ),
        "structured": 1.0 if kind == "code" else _structure_rate(text),
        "digits": sum(char.isdigit() for char in text) / max(len(text), 1),
        "balanced_lines": max(0.0, 1.0 - line_sd / max(avg_line, 1)),
        "token_entropy": token_entropy,
        "punctuation": sum(char in _PUNCT_CHARS for char in text) / max(len(text), 1),
        "mean_word_length": mean_length,
        "word_length_var": length_var,
        "avg_line_len": avg_line,
        "line_sd": line_sd,
        "sentence_len_mean": sentence_mean,
        "sentence_len_sd": sentence_sd,
        "hapax_ratio": sum(1 for count in counts.values() if count == 1) / len(words),
        "log_words": math.log1p(len(words)),
    }


def vector(text: str, kind: str = "text", names: tuple[str, ...] = FEATURE_NAMES) -> list[float]:
    features = extract(text, kind)
    return [features[name] for name in names]


def matrix(texts: list[str], kinds: list[str] | None = None) -> list[list[float]]:
    if kinds is None:
        kinds = ["text"] * len(texts)
    return [vector(text, kind) for text, kind in zip(texts, kinds, strict=True)]


def heuristic_raw_score(text: str, kind: str = "text") -> float:
    """The shipped runtime detector's raw score for this document.

    Imported from the installed ``panoptes`` package so the calibration
    fit always targets the real deployed heuristic, never a copy.
    """
    try:
        from panoptes.analysis.detectors import HeuristicCodeDetector, HeuristicProseDetector
        from panoptes.schemas import ContentType
    except ImportError as exc:  # pragma: no cover - environment problem
        raise RuntimeError(
            "bench requires the panoptes backend package; run `pip install -e backend` "
            "from the repository root"
        ) from exc

    if kind == "code":
        detector = HeuristicCodeDetector()
        return detector.score(text, ContentType.CODE, "unknown").raw_score
    detector = HeuristicProseDetector()
    return detector.score(text, ContentType.PROSE, "en").raw_score


def _structure_rate(text: str) -> float:
    return sum(text.count(marker) for marker in _STRUCTURE_MARKERS) / max(len(text) / 80, 1)


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
