"""Tests for the typed evidence ledger (protocol v2.1, Phase 7).

The ledger formalizes the three unlike evidence channels and forbids their
arithmetic fusion. These tests pin the structural invariants: one detail block
per entry, watermark negatives are never human evidence, provenance carries no
detector probability, and the fusion rule is stated.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from panoptes.analysis.detectors import DetectorScore
from panoptes.schemas import (
    EvidenceChannel,
    EvidenceEntry,
    EvidenceLedger,
    EvidenceState,
    EvidenceValidity,
    OutcomeDistribution,
    ProvenanceResult,
    StatisticalEvidence,
    WatermarkEvidence,
    WatermarkResult,
)
from panoptes.analysis.evidence import build_evidence_ledger


def _distribution(ai: float = 0.7) -> OutcomeDistribution:
    return OutcomeDistribution(
        human=round(1.0 - ai, 4),
        ai_refined_or_mixed=0.1,
        ai_generated=round(ai - 0.1, 4),
    )


def _score() -> DetectorScore:
    return DetectorScore(
        detector_id="heuristic-v1",
        raw_score=0.8,
        distribution=_distribution(),
        abstain_reason=None,
    )


def _watermark(scheme: str = "openai-aaronson", *, q: float | None = 0.6) -> WatermarkResult:
    return WatermarkResult(
        scheme=scheme,
        status="tested",
        eligible_tokens=400,
        green_rate=0.51,
        z=0.4,
        p_value=0.34,
        q_value=q,
        power=0.3,
        dilution_estimate=None,
    )


def _provenance(status: str = "not_present") -> ProvenanceResult:
    return ProvenanceResult(status=status, summary=f"provenance {status}")


def _ledger(**overrides) -> EvidenceLedger:
    kwargs = dict(
        state=EvidenceState.SUPPORTED,
        document_score=_score(),
        calibrated=_distribution(),
        bundle=None,
        bundle_id="baseline-calibration.json",
        cohort="prose-en-baseline-v0",
        prevalence=0.5,
        applicability=None,
        watermark_results=[_watermark()],
        provenance=_provenance(),
    )
    kwargs.update(overrides)
    return build_evidence_ledger(**kwargs)


def test_ledger_has_three_channels() -> None:
    ledger = _ledger()
    assert len(ledger.statistical) == 1
    assert len(ledger.watermark) == 1
    assert len(ledger.provenance) == 1
    assert ledger.statistical[0].channel == EvidenceChannel.STATISTICAL
    assert ledger.watermark[0].channel == EvidenceChannel.WATERMARK
    assert ledger.provenance[0].channel == EvidenceChannel.PROVENANCE


def test_entry_requires_exactly_one_detail_block() -> None:
    with pytest.raises(ValidationError):
        EvidenceEntry(
            channel=EvidenceChannel.STATISTICAL,
            target_claim="ai_participation",
            source_identity="heuristic-v1",
            validity=EvidenceValidity.VALID,
            applicability_scope="cohort",
            statistical=StatisticalEvidence(detector_id="heuristic-v1", cohort="c"),
            watermark=WatermarkEvidence(
                scheme="openai-aaronson", status="tested", eligible_tokens=10
            ),
        )


def test_entry_detail_must_match_channel() -> None:
    with pytest.raises(ValidationError):
        EvidenceEntry(
            channel=EvidenceChannel.STATISTICAL,
            target_claim="ai_participation",
            source_identity="heuristic-v1",
            validity=EvidenceValidity.VALID,
            applicability_scope="cohort",
            watermark=WatermarkEvidence(
                scheme="openai-aaronson", status="tested", eligible_tokens=10
            ),
        )


def test_negative_watermark_is_not_human_evidence() -> None:
    ledger = _ledger(watermark_results=[_watermark(q=0.9)])
    entry = ledger.watermark[0]
    assert entry.watermark is not None and entry.watermark.q_value == 0.9
    assert any("not evidence that content is human-written" in lim for lim in entry.limitations)
    assert "not evidence of human authorship" in ledger.channel_summaries["watermark"]


def test_provenance_carries_no_probability() -> None:
    ledger = _ledger(provenance=_provenance("verified"))
    entry = ledger.provenance[0]
    assert entry.validity == EvidenceValidity.VALID
    assert entry.strength is None
    assert any("never inherits" in lim for lim in entry.limitations)


def test_tampered_provenance_is_invalid() -> None:
    ledger = _ledger(provenance=_provenance("tampered"))
    assert ledger.provenance[0].validity == EvidenceValidity.INVALID


def test_statistical_strength_present_only_when_valid() -> None:
    valid = _ledger()
    assert valid.statistical[0].strength is not None
    abstaining = _ledger(
        document_score=DetectorScore(
            detector_id="heuristic-v1",
            raw_score=0.5,
            distribution=_distribution(0.5),
            abstain_reason="too short",
        )
    )
    assert abstaining.statistical[0].validity == EvidenceValidity.WEAKENED
    assert abstaining.statistical[0].strength is None


def test_fusion_note_states_no_fusion_rule() -> None:
    ledger = _ledger()
    assert "never arithmetically fused" in ledger.fusion_note


def test_ledger_serializes_to_json() -> None:
    ledger = _ledger()
    payload = ledger.model_dump(mode="json")
    assert set(payload) >= {"statistical", "watermark", "provenance", "channel_summaries", "fusion_note"}


def test_pipeline_response_includes_ledger() -> None:
    from fastapi.testclient import TestClient

    from panoptes.main import app

    client = TestClient(app)
    response = client.post("/api/v1/analyze", json={"text": "AI-generated " * 80})
    assert response.status_code == 200
    ledger = response.json()["evidence_ledger"]
    assert ledger is not None
    assert len(ledger["statistical"]) == 1
    assert len(ledger["watermark"]) >= 1
    assert len(ledger["provenance"]) == 1
    assert "never arithmetically fused" in ledger["fusion_note"]
