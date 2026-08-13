import math

import pytest

from panoptes.analysis.pipeline import _calibrate_distribution, _likelihood_ratio
from panoptes.analysis.watermarks import apply_fdr
from panoptes.schemas import EvidenceState, OutcomeDistribution, WatermarkResult


def test_likelihood_ratio_divides_out_cohort_prevalence() -> None:
    assert math.isclose(_likelihood_ratio(0.8, 0.8), 1.0)
    assert math.isclose(_likelihood_ratio(0.8, 0.5), 4.0)


def test_posterior_uses_bayes_not_ece_blend() -> None:
    distribution = OutcomeDistribution(human=0.2, ai_generated=0.8, ai_refined_or_mixed=0.0)
    even = _calibrate_distribution(distribution, EvidenceState.SUPPORTED, prior_odds=1.0, cohort_prevalence=0.5)
    assert even.ai_generated == pytest.approx(0.8, rel=1e-6)
    rare = _calibrate_distribution(distribution, EvidenceState.SUPPORTED, prior_odds=1.0 / 9.0, cohort_prevalence=0.5)
    assert rare.ai_generated < even.ai_generated
    # ECE is not an input; doubling the prior odds must increase the posterior.
    common = _calibrate_distribution(distribution, EvidenceState.SUPPORTED, prior_odds=9.0, cohort_prevalence=0.5)
    assert common.ai_generated > even.ai_generated


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
