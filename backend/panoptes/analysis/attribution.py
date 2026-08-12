from __future__ import annotations

import math
import re
from collections import Counter

from panoptes.schemas import ContentType, SourceFamilies, SourceFamilyProbability

_WORD_RE = re.compile(r"\b[\w']+\b")

_FAMILIES = ("llama-like", "mistral-like", "qwen-like", "gpt-like", "gemma-like")


def source_family_distribution(
    text: str,
    content_type: ContentType,
    bundle=None,
) -> SourceFamilies:
    features = _features(text, content_type)
    if bundle is not None:
        from panoptes.analysis.calibration_bundle import geometry_membership

        fitted = geometry_membership(bundle, features)
        if fitted is not None:
            probabilities, unknown = fitted
            if len(text) < 500:
                unknown = min(1.0, unknown + 0.25)
            return SourceFamilies(
                conditional_on_ai=[
                    SourceFamilyProbability(family=family, probability=probability)
                    for family, probability in sorted(
                        probabilities.items(), key=lambda item: item[1], reverse=True
                    )
                ],
                unknown_score=unknown,
                interpretation=(
                    "Conditional similarity among corpus-observed source families, from the "
                    "signed calibration artifact's fitted geometry. This is not proof of exact "
                    "model identity, and unsupported generators should remain unknown."
                ),
                basis="corpus-fitted",
                cohort_size=bundle.n_records,
            )
    logits = {
        "llama-like": -1.0 + 1.4 * features["long_words"] + 0.5 * features["connectors"],
        "mistral-like": -1.1 + 1.1 * features["unique_ratio"] + 0.4 * features["short_sentences"],
        "qwen-like": -1.2 + 1.2 * features["structured"] + 0.5 * features["digits"],
        "gpt-like": -0.9 + 1.3 * features["connectors"] + 0.4 * features["balanced_lines"],
        "gemma-like": -1.3 + 1.0 * features["short_sentences"] + 0.4 * features["unique_ratio"],
    }
    probabilities = _softmax(logits)
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probabilities.values())
    max_entropy = math.log(len(probabilities))
    unknown = min(max(entropy / max_entropy, 0.0), 1.0)
    if len(text) < 500:
        unknown = min(1.0, unknown + 0.25)
    return SourceFamilies(
        conditional_on_ai=[
            SourceFamilyProbability(family=family, probability=probability)
            for family, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        ],
        unknown_score=unknown,
        interpretation=(
            "Conditional similarity among supported source families (hand-tuned heuristic "
            "geometry; no signed corpus artifact loaded). This is not proof of exact "
            "model identity, and unsupported generators should remain unknown."
        ),
        basis="heuristic",
        cohort_size=None,
    )


def _features(text: str, content_type: ContentType) -> dict[str, float]:
    words = _WORD_RE.findall(text.lower())
    if not words:
        return {
            "long_words": 0.0,
            "connectors": 0.0,
            "unique_ratio": 0.0,
            "short_sentences": 0.0,
            "structured": 0.0,
            "digits": 0.0,
            "balanced_lines": 0.0,
        }
    counts = Counter(words)
    sentence_lengths = [
        len(_WORD_RE.findall(sentence)) for sentence in re.split(r"[.!?]+", text) if sentence.strip()
    ]
    lines = [line for line in text.splitlines() if line.strip()]
    avg_line = sum(len(line) for line in lines) / max(len(lines), 1)
    line_sd = _stdev([len(line) for line in lines])
    return {
        "long_words": sum(len(word) >= 7 for word in words) / len(words),
        "connectors": sum(
            word in {"however", "therefore", "moreover", "additionally", "overall", "furthermore"}
            for word in words
        )
        / len(words),
        "unique_ratio": len(counts) / len(words),
        "short_sentences": sum(length < 14 for length in sentence_lengths)
        / max(len(sentence_lengths), 1),
        "structured": 1.0 if content_type == ContentType.CODE else _structure_rate(text),
        "digits": sum(char.isdigit() for char in text) / max(len(text), 1),
        "balanced_lines": max(0.0, 1.0 - line_sd / max(avg_line, 1)),
    }


def _structure_rate(text: str) -> float:
    markers = ("\n-", "\n*", ":", ";", "(", ")", "[", "]")
    return sum(text.count(marker) for marker in markers) / max(len(text) / 80, 1)


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    maximum = max(logits.values())
    exps = {key: math.exp(value - maximum) for key, value in logits.items()}
    total = sum(exps.values())
    return {key: value / total for key, value in exps.items()}


def _stdev(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
