from panoptes.analysis.windowing import detect_content_type, make_segments, token_spans
from panoptes.schemas import ContentType


def test_detect_code_from_filename() -> None:
    assert detect_content_type("print('hello')", "example.py") == ContentType.CODE


def test_segments_preserve_offsets() -> None:
    text = "One two three. " * 80
    spans = token_spans(text)
    segments = make_segments(text, ContentType.PROSE, target_tokens=40, overlap=5)
    assert segments
    for segment in segments:
        assert text[segment.start : segment.end]
        assert segment.token_count > 0
    assert segments[0].start == spans[0][0]
