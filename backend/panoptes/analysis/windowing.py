from __future__ import annotations

import hashlib
import math
import re

from panoptes.schemas import ContentType, OutcomeDistribution, Segment

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_CODE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "javascript",
    ".tsx": "javascript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
}
_CODE_HINTS = (
    "def ",
    "function ",
    "const ",
    "let ",
    "var ",
    "class ",
    "import ",
    "from ",
    "public static",
    "#include",
    "package ",
    "fn ",
    "=>",
)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_spans(text: str) -> list[tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group(0)) for match in _TOKEN_RE.finditer(text)]


def detect_content_type(
    text: str, filename: str | None = None, override: ContentType | None = None
) -> ContentType:
    if override is not None:
        return override
    if filename:
        lowered = filename.lower()
        if any(lowered.endswith(ext) for ext in _CODE_EXTENSIONS):
            return ContentType.CODE
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return ContentType.PROSE
    code_line_count = 0
    for line in lines:
        stripped = line.strip()
        if any(hint in stripped for hint in _CODE_HINTS) or re.search(r"[{}();]\s*$", stripped):
            code_line_count += 1
    ratio = code_line_count / len(lines)
    if ratio >= 0.35:
        return ContentType.CODE
    if 0.12 <= ratio < 0.35:
        return ContentType.MIXED
    return ContentType.PROSE


def detect_language(text: str, content_type: ContentType, filename: str | None = None) -> str:
    if content_type == ContentType.CODE and filename:
        lowered = filename.lower()
        for ext, language in _CODE_EXTENSIONS.items():
            if lowered.endswith(ext):
                return language
    ascii_letters = sum(char.isascii() and char.isalpha() for char in text)
    letters = sum(char.isalpha() for char in text)
    if letters and ascii_letters / letters > 0.85:
        return "en"
    return "unknown"


def make_segments(
    text: str,
    content_type: ContentType,
    target_tokens: int = 160,
    overlap: int = 32,
) -> list[Segment]:
    spans = token_spans(text)
    if not spans:
        return []

    if content_type == ContentType.CODE:
        boundaries = _code_boundaries(text)
        windows = _windows_from_boundaries(text, boundaries)
    else:
        step = max(target_tokens - overlap, 1)
        windows = []
        start_index = 0
        while start_index < len(spans):
            end_index = min(start_index + target_tokens, len(spans))
            windows.append(
                (spans[start_index][0], spans[end_index - 1][1], end_index - start_index)
            )
            if end_index == len(spans):
                break
            start_index += step

    segments: list[Segment] = []
    for index, (start, end, count) in enumerate(windows):
        segments.append(
            Segment(
                id=f"segment-{index + 1}",
                start=start,
                end=end,
                token_count=count,
                kind=content_type if content_type != ContentType.MIXED else ContentType.PROSE,
                posterior=OutcomeDistribution(
                    human=0.34,
                    ai_generated=0.33,
                    ai_refined_or_mixed=0.33,
                ),
                watermark_evidence={},
                source_family={},
                anomaly_percentile=None,
            )
        )
    return segments


def _code_boundaries(text: str) -> list[int]:
    boundaries = [0]
    for match in re.finditer(
        r"\n\s*(def |class |function |const |let |public |private |package |import |from )", text
    ):
        if match.start() > boundaries[-1]:
            boundaries.append(match.start() + 1)
    if len(text) not in boundaries:
        boundaries.append(len(text))
    return boundaries


def _windows_from_boundaries(text: str, boundaries: list[int]) -> list[tuple[int, int, int]]:
    windows: list[tuple[int, int, int]] = []
    current_start = boundaries[0]
    current_tokens = 0
    spans = token_spans(text)
    span_index = 0
    for boundary in boundaries[1:]:
        count = 0
        while span_index < len(spans) and spans[span_index][0] < boundary:
            if spans[span_index][0] >= current_start:
                count += 1
            span_index += 1
        if current_tokens and current_tokens + count > 260:
            windows.append((current_start, boundary, current_tokens))
            current_start = boundary
            current_tokens = 0
        current_tokens += count
    if current_tokens:
        windows.append((current_start, len(text), current_tokens))
    if not windows:
        windows.append((0, len(text), len(spans)))
    return windows


def clamp_probability(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return min(max(value, 0.0), 1.0)
