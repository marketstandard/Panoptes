from panoptes.analysis.pipeline import analyze
from panoptes.schemas import AnalysisRequest
from panoptes.settings import Settings


def test_token_overlay_is_returned_when_requested() -> None:
    text = "AI-generated systematic prose with furthermore and moreover transitions. " * 8
    response = analyze(
        AnalysisRequest(text=text, include_text=True),
        Settings(),
    )
    kgw = next(item for item in response.watermarks if item.scheme == "kgw-v1")
    assert kgw.status == "tested"
    assert kgw.tokens is not None
    assert all(token.end > token.start for token in kgw.tokens)
    assert kgw.green_rate is not None
    assert kgw.green_rate_interval is not None
    assert kgw.dilution_estimate is not None


def test_token_overlay_is_omitted_by_default() -> None:
    text = "AI-generated systematic prose with furthermore and moreover transitions. " * 8
    response = analyze(AnalysisRequest(text=text), Settings())
    kgw = next(item for item in response.watermarks if item.scheme == "kgw-v1")
    assert kgw.tokens is None
