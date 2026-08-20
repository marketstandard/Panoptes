"""Typed evidence ledger construction (protocol v2.1, Phase 7).

Builds an :class:`panoptes.schemas.EvidenceLedger` from the three unlike
evidence channels the pipeline already computes — statistical (calibrated
detector), watermark (hypothesis tests), and provenance (signing chain) —
keeping them separate and never fusing them into one number.

Invariants enforced here:
  - A negative / non-significant watermark test is recorded as *no watermark
    evidence*, never as evidence of human authorship.
  - Provenance attests a signing chain only; it carries no detector probability
    and never alters the statistical channel.
  - Statistical evidence is population-conditional and carries a transport
    warning when the input may be off the calibration cohort.
"""

from __future__ import annotations

from panoptes.analysis.calibration_bundle import CalibrationBundle
from panoptes.analysis.detectors import DetectorScore
from panoptes.schemas import (
    EvidenceChannel,
    EvidenceEntry,
    EvidenceLedger,
    EvidenceState,
    EvidenceValidity,
    OutcomeDistribution,
    ProvenanceEvidence,
    ProvenanceResult,
    StatisticalEvidence,
    WatermarkEvidence,
    WatermarkResult,
)


def _statistical_validity(state: EvidenceState, abstain_reason: str | None) -> EvidenceValidity:
    if state == EvidenceState.SUPPORTED and not abstain_reason:
        return EvidenceValidity.VALID
    if state in (
        EvidenceState.INSUFFICIENT_DATA,
        EvidenceState.UNSUPPORTED_LANGUAGE,
        EvidenceState.UNSUPPORTED_CONTENT,
    ):
        return EvidenceValidity.NOT_APPLICABLE
    if state == EvidenceState.OUT_OF_DISTRIBUTION:
        return EvidenceValidity.WEAKENED
    if abstain_reason:
        return EvidenceValidity.WEAKENED
    return EvidenceValidity.UNKNOWN


def _statistical_entry(
    *,
    state: EvidenceState,
    document_score: DetectorScore,
    calibrated: OutcomeDistribution,
    bundle: CalibrationBundle | None,
    bundle_id: str,
    cohort: str,
    prevalence: float,
    applicability: str | None,
) -> EvidenceEntry:
    validity = _statistical_validity(state, document_score.abstain_reason)
    participation = calibrated.ai_generated + calibrated.ai_refined_or_mixed
    majority = calibrated.ai_generated
    transport_warning = None
    if bundle is not None:
        transport_warning = (
            f"Calibrated on cohort '{cohort}' (prevalence {prevalence:.2f}); the score's "
            "evidential meaning is population-conditional and can shift under domain, "
            "generator, or paraphrase shift."
        )
    assumptions = ["The input is drawn from a population the calibration cohort represents."]
    limitations = [
        "Statistical evidence is population-conditional, not proof of authorship.",
        "Paraphrase, heavy editing, and unseen generators degrade the score.",
    ]
    if document_score.abstain_reason:
        limitations.append(document_score.abstain_reason)
    uncertainty = None
    if bundle is not None:
        ece = bundle.metrics.get("ece")
        if ece is not None:
            uncertainty = f"Calibration ECE {ece:.3f} on the held-out cohort; see reliability bins."
    return EvidenceEntry(
        channel=EvidenceChannel.STATISTICAL,
        target_claim="ai_participation",
        source_identity=document_score.detector_id,
        validity=validity,
        applicability_scope=f"calibration cohort '{cohort}'; English prose statistical route",
        strength=round(participation, 4) if validity == EvidenceValidity.VALID else None,
        uncertainty=uncertainty,
        assumptions=assumptions,
        limitations=limitations,
        statistical=StatisticalEvidence(
            detector_id=document_score.detector_id,
            model_revision=None,
            calibrator_id=bundle_id if bundle is not None else None,
            cohort=cohort,
            cohort_prevalence=prevalence if bundle is not None else None,
            ai_participation=round(participation, 4),
            ai_majority_generation=round(majority, 4),
            contribution_fraction=None,
            applicability=applicability,
            transport_warning=transport_warning,
        ),
    )


def _watermark_entry(result: WatermarkResult) -> EvidenceEntry:
    tested = result.status == "tested" and result.q_value is not None
    significant = tested and result.q_value is not None and result.q_value < 0.05
    if result.status == "insufficient_data" or result.status in (
        "adapter_unavailable",
        "not_applicable",
    ):
        validity = EvidenceValidity.NOT_APPLICABLE
    elif tested:
        validity = EvidenceValidity.VALID
    else:
        validity = EvidenceValidity.UNKNOWN
    strength = None
    if tested and result.q_value is not None:
        # Evidence strength for the watermark's presence, not for authorship.
        strength = round(1.0 - min(max(result.q_value, 0.0), 1.0), 4)
    uncertainty = None
    if tested:
        uncertainty = f"z={result.z:.2f}, q={result.q_value:.3g}" + (
            f", power={result.power:.2f}" if result.power is not None else ""
        )
    limitations = [
        "A negative watermark result is not evidence that content is human-written.",
        "Editing, paraphrase, and short text reduce test power.",
    ]
    if result.power is not None and result.power < 0.5:
        limitations.append(
            "This test had low power at the observed length; a null result is weakly informative."
        )
    return EvidenceEntry(
        channel=EvidenceChannel.WATERMARK,
        target_claim="watermark_present",
        source_identity=result.scheme,
        validity=validity,
        applicability_scope=f"texts generated with the {result.scheme} watermark configuration",
        strength=strength if significant else (strength if tested else None),
        uncertainty=uncertainty,
        assumptions=["The tokenizer and watermark configuration match generation."],
        limitations=limitations,
        watermark=WatermarkEvidence(
            scheme=result.scheme,
            status=result.status,
            eligible_tokens=result.eligible_tokens,
            green_rate=result.green_rate,
            z=result.z,
            p_value=result.p_value,
            q_value=result.q_value,
            power=result.power,
            dilution_estimate=result.dilution_estimate,
        ),
    )


def _provenance_entry(provenance: ProvenanceResult) -> EvidenceEntry:
    status = provenance.status
    if status == "verified":
        validity = EvidenceValidity.VALID
    elif status == "tampered":
        validity = EvidenceValidity.INVALID
    elif status in ("not_present", "not_applicable", "unsupported_file"):
        validity = EvidenceValidity.NOT_APPLICABLE
    else:
        validity = EvidenceValidity.UNKNOWN
    return EvidenceEntry(
        channel=EvidenceChannel.PROVENANCE,
        target_claim="provenance_chain_valid",
        source_identity=provenance.issuer or "none",
        validity=validity,
        applicability_scope="files carrying an embedded C2PA / cryptographic provenance manifest",
        strength=None,  # provenance is a chain attestation, not a probability
        uncertainty=None,
        assumptions=["The signing chain is intact and the issuer is trusted."],
        limitations=[
            "Provenance attests a signing chain, not metaphysical authorship.",
            "Provenance never inherits or alters detector probability.",
        ],
        provenance=ProvenanceEvidence(
            status=status,
            issuer=provenance.issuer,
            timestamp=provenance.timestamp,
            signature_chain=[provenance.issuer] if provenance.issuer else [],
            level=provenance.level,
            actions=list(provenance.actions),
        ),
    )


def build_evidence_ledger(
    *,
    state: EvidenceState,
    document_score: DetectorScore,
    calibrated: OutcomeDistribution,
    bundle: CalibrationBundle | None,
    bundle_id: str,
    cohort: str,
    prevalence: float,
    applicability: str | None,
    watermark_results: list[WatermarkResult],
    provenance: ProvenanceResult,
) -> EvidenceLedger:
    statistical = [
        _statistical_entry(
            state=state,
            document_score=document_score,
            calibrated=calibrated,
            bundle=bundle,
            bundle_id=bundle_id,
            cohort=cohort,
            prevalence=prevalence,
            applicability=applicability,
        )
    ]
    watermark = [_watermark_entry(result) for result in watermark_results]
    provenance_entries = [_provenance_entry(provenance)]

    summaries: dict[str, str] = {}
    participation = calibrated.ai_generated + calibrated.ai_refined_or_mixed
    if state == EvidenceState.SUPPORTED:
        summaries["statistical"] = (
            f"Calibrated statistical evidence estimates P(AI participation) = {participation:.2f} "
            f"for the calibration cohort '{cohort}'."
        )
    else:
        summaries["statistical"] = (
            "Statistical evidence is limited or not applicable for this input "
            f"(state: {state.value})."
        )
    tested = [r for r in watermark_results if r.status == "tested" and r.q_value is not None]
    if any(r.q_value is not None and r.q_value < 0.05 for r in tested):
        summaries["watermark"] = (
            "A configured watermark test found statistically significant evidence."
        )
    elif tested:
        summaries["watermark"] = (
            "No configured watermark test reached significance; "
            "this is not evidence of human authorship."
        )
    else:
        summaries["watermark"] = "No watermark test was applicable to this input."
    if provenance.status == "verified":
        summaries["provenance"] = (
            f"A signing chain from '{provenance.issuer or 'unknown issuer'}' verified; "
            "this attests provenance, not authorship."
        )
    elif provenance.status == "tampered":
        summaries["provenance"] = (
            "A provenance manifest was present but failed verification (tampered)."
        )
    else:
        summaries["provenance"] = "No usable provenance manifest was present."

    return EvidenceLedger(
        statistical=statistical,
        watermark=watermark,
        provenance=provenance_entries,
        channel_summaries=summaries,
    )
