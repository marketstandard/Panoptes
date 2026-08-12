from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from statistics import NormalDist

from panoptes.schemas import ContentType, WatermarkResult

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_DEFAULT_GREEN_FRACTION = 0.5
_MIN_TOKENS = 50


@dataclass(frozen=True)
class WatermarkToken:
    token: str
    start: int
    end: int
    green: bool


class WatermarkAdapter:
    id = "watermark-base"

    def detect(self, text: str, content_type: ContentType) -> tuple[WatermarkResult, list[WatermarkToken]]:
        raise NotImplementedError


class KGWReferenceAdapter(WatermarkAdapter):
    id = "kgw-v1"

    def detect(self, text: str, content_type: ContentType) -> tuple[WatermarkResult, list[WatermarkToken]]:
        spans = list(_TOKEN_RE.finditer(text))
        if len(spans) < _MIN_TOKENS:
            return (
                WatermarkResult(
                    scheme=self.id,
                    status="insufficient_data",
                    eligible_tokens=len(spans),
                    green_tokens=None,
                    expected_green=None,
                    z=None,
                    p_value=None,
                    q_value=None,
                    effect=None,
                    power=None,
                ),
                [],
            )

        previous = ""
        tokens: list[WatermarkToken] = []
        for span in spans:
            token = span.group(0)
            green = _green_for(previous, token)
            tokens.append(WatermarkToken(token=token, start=span.start(), end=span.end(), green=green))
            previous = token

        green_count = sum(token.green for token in tokens)
        n = len(tokens)
        expected = n * _DEFAULT_GREEN_FRACTION
        variance = n * _DEFAULT_GREEN_FRACTION * (1 - _DEFAULT_GREEN_FRACTION)
        z = (green_count - expected) / math.sqrt(variance)
        p_value = 1 - NormalDist().cdf(z)
        effect = (green_count / n) - _DEFAULT_GREEN_FRACTION
        power = _power(n, alpha=0.01, observed_rate=green_count / n)
        result = WatermarkResult(
            scheme=self.id,
            status="tested",
            eligible_tokens=n,
            green_tokens=green_count,
            expected_green=expected,
            z=z,
            p_value=max(min(p_value, 1.0), 1e-16),
            q_value=None,
            effect=effect,
            power=power,
        )
        return result, tokens


class ClaudePendingAdapter(WatermarkAdapter):
    id = "claude-text-watermark"

    def detect(self, text: str, content_type: ContentType) -> tuple[WatermarkResult, list[WatermarkToken]]:
        return (
            WatermarkResult(
                scheme=self.id,
                status="adapter_unavailable",
                eligible_tokens=len(_TOKEN_RE.findall(text)),
                green_tokens=None,
                expected_green=None,
                z=None,
                p_value=None,
                q_value=None,
                effect=None,
                power=None,
            ),
            [],
        )


def watermark_adapters() -> list[WatermarkAdapter]:
    return [KGWReferenceAdapter(), ClaudePendingAdapter()]


def apply_fdr(results: list[WatermarkResult]) -> list[WatermarkResult]:
    tested = [(index, result.p_value) for index, result in enumerate(results) if result.p_value is not None]
    if not tested:
        return results
    ordered = sorted(tested, key=lambda item: item[1])
    q_by_index: dict[int, float] = {}
    running = 1.0
    count = len(ordered)
    for rank_from_end, (index, p_value) in enumerate(reversed(ordered), start=1):
        rank = count - rank_from_end + 1
        running = min(running, (count / rank) * p_value)
        q_by_index[index] = min(running, 1.0)
    updated: list[WatermarkResult] = []
    for index, result in enumerate(results):
        if index in q_by_index:
            updated.append(result.model_copy(update={"q_value": q_by_index[index]}))
        else:
            updated.append(result)
    return updated


def evidence_by_segment(segments: list[tuple[int, int]], tokens: list[WatermarkToken]) -> list[float | None]:
    values: list[float | None] = []
    for start, end in segments:
        segment_tokens = [token for token in tokens if token.start >= start and token.end <= end]
        if len(segment_tokens) < 20:
            values.append(None)
            continue
        green = sum(token.green for token in segment_tokens)
        n = len(segment_tokens)
        z = (green - n * _DEFAULT_GREEN_FRACTION) / math.sqrt(
            n * _DEFAULT_GREEN_FRACTION * (1 - _DEFAULT_GREEN_FRACTION)
        )
        p_value = max(1 - NormalDist().cdf(z), 1e-16)
        values.append(-math.log10(p_value))
    return values


def _green_for(previous: str, token: str) -> bool:
    digest = hashlib.sha256(f"panoptes-demo-key::{previous}::{token}".encode("utf-8")).digest()
    return digest[0] < 128


def _power(n: int, alpha: float, observed_rate: float) -> float:
    null_sd = math.sqrt(_DEFAULT_GREEN_FRACTION * (1 - _DEFAULT_GREEN_FRACTION) / n)
    critical = NormalDist().inv_cdf(1 - alpha) * null_sd + _DEFAULT_GREEN_FRACTION
    alt_sd = math.sqrt(max(observed_rate * (1 - observed_rate), 1e-9) / n)
    if alt_sd == 0:
        return 0.0
    return max(min(1 - NormalDist().cdf((critical - observed_rate) / alt_sd), 1.0), 0.0)
