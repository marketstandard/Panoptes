from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from panoptes.schemas import ContentType, OutcomeDistribution

_WORD_RE = re.compile(r"\b[\w']+\b")


@dataclass(frozen=True)
class DetectorScore:
    distribution: OutcomeDistribution
    raw_score: float
    detector_id: str
    abstain_reason: str | None = None


class DetectorAdapter:
    id = "base"
    min_tokens = 1
    content_types: tuple[ContentType, ...] = ()
    languages: tuple[str, ...] = ()

    def score(self, text: str, content_type: ContentType, language: str) -> DetectorScore:
        raise NotImplementedError


class FixtureDetector(DetectorAdapter):
    id = "fixture-detector"

    def score(self, text: str, content_type: ContentType, language: str) -> DetectorScore:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = int.from_bytes(digest[:8], "big") / 2**64
        signal = _fixture_signal(text)
        ai = 0.08 + 0.84 * ((raw + signal) / 2)
        refined = 0.12 if "edited" in text.lower() or "revised" in text.lower() else 0.04
        human = max(0.02, 1 - ai - refined)
        distribution = OutcomeDistribution(
            human=human,
            ai_generated=ai,
            ai_refined_or_mixed=refined,
        ).normalized()
        return DetectorScore(
            distribution=distribution,
            raw_score=ai,
            detector_id=self.id,
            abstain_reason="Fixture mode does not load a statistical detector.",
        )


class HeuristicProseDetector(DetectorAdapter):
    id = "heuristic-prose-detector"
    min_tokens = 40
    content_types = (ContentType.PROSE, ContentType.MIXED)
    languages = ("en",)

    def score(self, text: str, content_type: ContentType, language: str) -> DetectorScore:
        words = _WORD_RE.findall(text.lower())
        if len(words) < self.min_tokens:
            return _abstain(self.id, "Insufficient prose tokens for generic detection.")
        if language != "en":
            return _abstain(self.id, "Generic prose detector is calibrated only for English.")

        lengths = [len(word) for word in words]
        unique_ratio = len(set(words)) / max(len(words), 1)
        avg_word = sum(lengths) / len(lengths)
        sentence_lengths = [
            len(_WORD_RE.findall(sentence))
            for sentence in re.split(r"[.!?]+", text)
            if _WORD_RE.findall(sentence)
        ]
        burstiness = _stdev(sentence_lengths)
        long_word_rate = sum(length >= 7 for length in lengths) / len(lengths)
        connector_rate = sum(
            word in {"however", "therefore", "moreover", "furthermore", "additionally", "overall"}
            for word in words
        ) / len(words)

        score = 0.50
        score += 0.18 * _bounded(avg_word - 4.7, -2, 3)
        score += 0.16 * _bounded(long_word_rate - 0.16, -0.15, 0.25)
        score += 0.15 * _bounded(
            0.25 - burstiness / max(sum(sentence_lengths) / len(sentence_lengths), 1), -0.2, 0.2
        )
        score += 0.12 * _bounded(connector_rate - 0.012, -0.02, 0.06)
        score -= 0.16 * _bounded(unique_ratio - 0.58, -0.2, 0.3)
        score = min(max(score, 0.02), 0.98)

        refined = 0.10 if _looks_edited(text) else 0.05
        ai_generated = max(0.02, score - refined / 2)
        human = max(0.02, 1 - score - refined / 2)
        return DetectorScore(
            distribution=OutcomeDistribution(
                human=human,
                ai_generated=ai_generated,
                ai_refined_or_mixed=refined,
            ).normalized(),
            raw_score=score,
            detector_id=self.id,
        )


class HeuristicCodeDetector(DetectorAdapter):
    id = "heuristic-code-detector"
    min_tokens = 25
    content_types = (ContentType.CODE,)

    def score(self, text: str, content_type: ContentType, language: str) -> DetectorScore:
        tokens = _WORD_RE.findall(text)
        if len(tokens) < self.min_tokens:
            return _abstain(self.id, "Insufficient code tokens for generic detection.")

        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        comment_lines = sum(line.strip().startswith(("#", "//", "/*", "*")) for line in lines)
        comment_ratio = comment_lines / max(len(lines), 1)
        identifiers = [token for token in tokens if token.isidentifier()]
        unique_identifier_ratio = len(set(identifiers)) / max(len(identifiers), 1)
        long_identifier_ratio = sum(len(token) >= 14 for token in identifiers) / max(
            len(identifiers), 1
        )
        indentation_consistency = _indentation_consistency(lines)
        boilerplate_terms = ("example", "usage", "todo", "note:", "this function", "returns")
        boilerplate_rate = sum(term in text.lower() for term in boilerplate_terms) / len(
            boilerplate_terms
        )

        score = 0.48
        score += 0.16 * _bounded(comment_ratio - 0.08, -0.08, 0.25)
        score += 0.13 * _bounded(long_identifier_ratio - 0.08, -0.08, 0.22)
        score += 0.12 * _bounded(indentation_consistency - 0.72, -0.5, 0.25)
        score += 0.10 * _bounded(boilerplate_rate - 0.2, -0.2, 0.5)
        score -= 0.15 * _bounded(unique_identifier_ratio - 0.78, -0.3, 0.2)
        score = min(max(score, 0.02), 0.98)

        refined = 0.12 if _looks_edited(text) else 0.06
        ai_generated = max(0.02, score - refined / 2)
        human = max(0.02, 1 - score - refined / 2)
        return DetectorScore(
            distribution=OutcomeDistribution(
                human=human,
                ai_generated=ai_generated,
                ai_refined_or_mixed=refined,
            ).normalized(),
            raw_score=score,
            detector_id=self.id,
        )


def select_detector(profile: str, content_type: ContentType, settings=None) -> DetectorAdapter:
    if profile == "fixture":
        return FixtureDetector()
    if settings is not None:
        try:
            from panoptes.plugins import get_plugin_registry

            for detector in get_plugin_registry(settings).detectors:
                types = getattr(detector, "content_types", ()) or ()
                if not types or content_type in types or content_type.value in types:
                    return detector
        except Exception:
            pass
    if content_type == ContentType.CODE:
        return HeuristicCodeDetector()
    return HeuristicProseDetector()


def _abstain(detector_id: str, reason: str) -> DetectorScore:
    return DetectorScore(
        distribution=OutcomeDistribution(
            human=1 / 3, ai_generated=1 / 3, ai_refined_or_mixed=1 / 3
        ),
        raw_score=0.5,
        detector_id=detector_id,
        abstain_reason=reason,
    )


def _bounded(value: float, lower: float, upper: float) -> float:
    if upper == lower:
        return 0.0
    return (min(max(value, lower), upper) - lower) / (upper - lower)


def _stdev(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _indentation_consistency(lines: list[str]) -> float:
    indents = [len(line) - len(line.lstrip(" ")) for line in lines if line.startswith(" ")]
    if len(indents) < 2:
        return 0.5
    multiples = sum(indent % 2 == 0 or indent % 4 == 0 for indent in indents)
    return multiples / len(indents)


def _looks_edited(text: str) -> bool:
    lowered = text.lower()
    return any(
        term in lowered for term in ("edited", "revised", "paraphrased", "translated", "proofread")
    )


def _fixture_signal(text: str) -> float:
    lowered = text.lower()
    if "watermark positive" in lowered or "ai-generated" in lowered:
        return 0.9
    if "human-written" in lowered:
        return 0.08
    return 0.45
