import math

from panoptes.analysis.pipeline import _likelihood_ratio
from panoptes.analysis.watermarks import apply_fdr
from panoptes.schemas import WatermarkResult


def test_likelihood_ratio_bounds() -> None:
    assert math.isclose(_likelihood_ratio(0.5), 1.0)
    assert _likelihood_ratio(0.9) > 1
    assert _likelihood_ratio(0.1) < 1


def test_fdr_adjustment_is_monotonic() -> None:
    results = [
        WatermarkResult(
            scheme="a",
            status="tested",
            eligible_tokens=100,
            green_tokens=70,
            expected_green=50,
            z=2,
            p_value=0.02,
            q_value=None,
            effect=0.2,
            power=0.8,
        ),
        WatermarkResult(
            scheme="b",
            status="tested",
            eligible_tokens=100,
            green_tokens=80,
            expected_green=50,
            z=4,
            p_value=0.001,
            q_value=None,
            effect=0.3,
            power=0.9,
        ),
    ]
    adjusted = apply_fdr(results)
    assert adjusted[0].q_value is not None
    assert adjusted[1].q_value is not None
    assert adjusted[1].q_value <= adjusted[0].q_value
