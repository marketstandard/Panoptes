from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from statistics import NormalDist

from panoptes.schemas import (
    ConfidenceInterval,
    ContentType,
    WatermarkResult,
    WatermarkTokenSpan,
)

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_DEFAULT_GREEN_FRACTION = 0.5
_MIN_TOKENS = 50
_TOKEN_OVERLAY_LIMIT = 4_000


@dataclass(frozen=True)
class WatermarkToken:
    token: str
    start: int
    end: int
    green: bool


class WatermarkAdapter:
    id = "watermark-base"

    def detect(
        self,
        text: str,
        content_type: ContentType,
        include_tokens: bool = False,
    ) -> tuple[WatermarkResult, list[WatermarkToken]]:
        raise NotImplementedError


def _empty_result(scheme: str, status: str, eligible_tokens: int) -> WatermarkResult:
    return WatermarkResult(
        scheme=scheme,
        status=status,  # type: ignore[arg-type]
        eligible_tokens=eligible_tokens,
        green_tokens=None,
        expected_green=None,
        green_rate=None,
        green_rate_interval=None,
        dilution_estimate=None,
        z=None,
        p_value=None,
        q_value=None,
        effect=None,
        power=None,
        tokens=None,
    )


class KGWReferenceAdapter(WatermarkAdapter):
    id = "kgw-v1"

    def detect(
        self,
        text: str,
        content_type: ContentType,
        include_tokens: bool = False,
    ) -> tuple[WatermarkResult, list[WatermarkToken]]:
        spans = list(_TOKEN_RE.finditer(text))
        if len(spans) < _MIN_TOKENS:
            return _empty_result(self.id, "insufficient_data", len(spans)), []

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
        green_rate = green_count / n
        effect = green_rate - _DEFAULT_GREEN_FRACTION
        power = _power(n, alpha=0.01, observed_rate=green_rate)
        interval = _wilson_interval(green_count, n)
        dilution = _dilution_estimate(green_rate)
        token_spans = (
            [WatermarkTokenSpan(start=t.start, end=t.end, green=t.green) for t in tokens]
            if include_tokens and n <= _TOKEN_OVERLAY_LIMIT
            else None
        )
        result = WatermarkResult(
            scheme=self.id,
            status="tested",
            eligible_tokens=n,
            green_tokens=green_count,
            expected_green=expected,
            green_rate=green_rate,
            green_rate_interval=interval,
            dilution_estimate=dilution,
            z=z,
            p_value=max(min(p_value, 1.0), 1e-16),
            q_value=None,
            effect=effect,
            power=power,
            tokens=token_spans,
        )
        return result, tokens


class ClaudePendingAdapter(WatermarkAdapter):
    """Placeholder for Anthropic's Claude text watermark.

    Anthropic has announced that Claude's watermark is **SynthID-Text** (Google
    DeepMind, Nature 2024), a member of the Aaronson (2022) green-list family —
    the same family :class:`KGWReferenceAdapter` models. Anthropic's production
    key is private and its detection API is not yet public, so this adapter
    stays ``adapter_unavailable``: we cannot test Anthropic's specific key. What
    we *can* characterize is the robustness of the family it belongs to, which
    the watermark-removal evaluation does (see
    ``backend/artifacts/cards/watermark-removal.json``): light editing and
    Unicode hygiene leave a green-list watermark detectable, while a complete
    LLM rewrite largely removes it — consistent with Anthropic's own statement
    that a full rewrite defeats the watermark but light editing does not.
    """

    id = "claude-text-watermark"

    def detect(
        self,
        text: str,
        content_type: ContentType,
        include_tokens: bool = False,
    ) -> tuple[WatermarkResult, list[WatermarkToken]]:
        return _empty_result(self.id, "adapter_unavailable", len(_TOKEN_RE.findall(text))), []


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


def green_for(previous: str, token: str) -> bool:
    """Public green-list membership test shared by the detector and the KGW
    generator (bench.watermark_gen), so generated text is detectable by
    :class:`KGWReferenceAdapter` under the same key."""
    return _green_for(previous, token)


def _power(n: int, alpha: float, observed_rate: float) -> float:
    null_sd = math.sqrt(_DEFAULT_GREEN_FRACTION * (1 - _DEFAULT_GREEN_FRACTION) / n)
    critical = NormalDist().inv_cdf(1 - alpha) * null_sd + _DEFAULT_GREEN_FRACTION
    alt_sd = math.sqrt(max(observed_rate * (1 - observed_rate), 1e-9) / n)
    if alt_sd == 0:
        return 0.0
    return max(min(1 - NormalDist().cdf((critical - observed_rate) / alt_sd), 1.0), 0.0)


def _wilson_interval(successes: int, n: int, level: float = 0.95) -> ConfidenceInterval:
    if n == 0:
        return ConfidenceInterval(lower=0.0, upper=1.0, level=level)
    z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    p_hat = successes / n
    denominator = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denominator
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denominator
    return ConfidenceInterval(
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
        level=level,
    )


def _dilution_estimate(green_rate: float) -> float:
    excess = green_rate - _DEFAULT_GREEN_FRACTION
    return min(max(excess / (1 - _DEFAULT_GREEN_FRACTION), 0.0), 1.0)
