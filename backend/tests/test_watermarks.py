from panoptes.analysis.watermarks import KGWReferenceAdapter
from panoptes.schemas import ContentType


def test_watermark_short_text_abstains() -> None:
    result, tokens = KGWReferenceAdapter().detect("short text", ContentType.PROSE)
    assert result.status == "insufficient_data"
    assert tokens == []


def test_watermark_long_text_tests() -> None:
    text = "alpha beta gamma delta " * 40
    result, tokens = KGWReferenceAdapter().detect(text, ContentType.PROSE)
    assert result.status == "tested"
    assert result.eligible_tokens == len(tokens)
    assert result.p_value is not None
    assert result.power is not None
