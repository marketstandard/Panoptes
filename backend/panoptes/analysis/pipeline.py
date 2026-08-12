from __future__ import annotations

import math
import uuid
from statistics import NormalDist

from panoptes.analysis.attribution import source_family_distribution
from panoptes.analysis.detectors import select_detector
from panoptes.analysis.provenance import decode_upload, verify_provenance
from panoptes.analysis.watermarks import apply_fdr, evidence_by_segment, watermark_adapters
from panoptes.analysis.windowing import (
    content_hash,
    detect_content_type,
    detect_language,
    make_segments,
    token_spans,
)
from panoptes.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    Capabilities,
    ConfidenceLabel,
    ContentType,
    EvidenceState,
    InputDiagnostics,
    MathDefinition,
    Matrices,
    Matrix,
    OutcomeDistribution,
    PosteriorInfo,
    RuntimeInfo,
    RuntimeProfile,
    Segment,
    Summary,
    WaterfallItem,
)
from panoptes.settings import Settings

MIN_SUPPORTED_TOKENS = 50


def analyze(request: AnalysisRequest, settings: Settings) -> AnalysisResponse:
    text = _request_text(request)
    upload = decode_upload(request.file_base64, request.filename)
    content_type = detect_content_type(text, request.filename, request.content_type)
    language = request.language or detect_language(text, content_type, request.filename)
    spans = token_spans(text)
    segments = make_segments(text, content_type)
    limitations: list[str] = []

    detector = select_detector(settings.profile.value, content_type)
    document_score = detector.score(text, content_type, language)
    if document_score.abstain_reason:
        limitations.append(document_score.abstain_reason)

    state = _evidence_state(len(spans), content_type, language, document_score.abstain_reason)
    calibrated = _calibrate_distribution(document_score.distribution, state, request.prior_odds)
    confidence = _confidence(len(spans), state, calibrated, document_score.abstain_reason)

    watermark_results = []
    tokens_by_scheme: dict[str, list] = {}
    include_overlay = request.include_text
    for adapter in watermark_adapters():
        result, tokens = adapter.detect(text, content_type, include_tokens=include_overlay)
        watermark_results.append(result)
        if tokens:
            tokens_by_scheme[adapter.id] = tokens
    watermark_results = apply_fdr(watermark_results)

    attribution = source_family_distribution(text, content_type)
    provenance = verify_provenance(upload)
    segment_scores = _score_segments(text, segments, detector, content_type, language)
    segment_ranges = [(segment.start, segment.end) for segment in segment_scores]
    evidence_by_scheme = {
        scheme: evidence_by_segment(segment_ranges, tokens)
        for scheme, tokens in tokens_by_scheme.items()
    }
    for index, segment in enumerate(segment_scores):
        segment.watermark_evidence = {
            scheme: values[index] for scheme, values in evidence_by_scheme.items()
        }
        segment.source_family = {
            item.family: item.probability for item in attribution.conditional_on_ai[:3]
        }
        segment.anomaly_percentile = _anomaly_percentile(segment.posterior.ai_generated)

    matrices = _matrices(segment_scores, watermark_results, attribution, request.prior_odds)
    plain_language = _plain_language(state, calibrated, watermark_results, provenance.status)
    limitations.extend(_standard_limitations(state, content_type, language, len(spans), watermark_results))

    return AnalysisResponse(
        report_id=str(uuid.uuid4()),
        runtime=RuntimeInfo(
            profile=settings.profile,
            device=_device_name(settings.profile),
            models_loaded=[document_score.detector_id],
            calibration_bundles=[_calibration_bundle(content_type, language)],
        ),
        input=InputDiagnostics(
            content_hash=content_hash(text) if text else content_hash(upload.content.hex() if upload else ""),
            content_type=content_type,
            language=language,
            token_count=len(spans),
            character_count=len(text),
            segment_count=len(segment_scores),
            user_overrode_type=request.content_type is not None,
        ),
        summary=Summary(
            evidence_state=state,
            plain_language=plain_language,
            confidence_label=confidence,
            overall=calibrated,
        ),
        posterior=PosteriorInfo(
            prior_odds=request.prior_odds,
            likelihood_ratio=_likelihood_ratio(document_score.distribution.ai_generated),
            posterior_odds=request.prior_odds * _likelihood_ratio(document_score.distribution.ai_generated),
            calibration_bundle=_calibration_bundle(content_type, language),
            reliability_error=0.08 if settings.profile != RuntimeProfile.FIXTURE else None,
            cohort="fixture" if settings.profile == RuntimeProfile.FIXTURE else "prose-en/code-baseline-v0",
        ),
        source_families=attribution,
        watermarks=watermark_results,
        provenance=provenance,
        segments=segment_scores,
        matrices=matrices,
        math=_math_definitions(),
        limitations=limitations,
        capabilities=Capabilities(
            languages=["en", "c", "cpp", "csharp", "go", "java", "javascript", "python"],
            model_families=[item.family for item in attribution.conditional_on_ai],
            watermark_schemes=[result.scheme for result in watermark_results],
            content_types=["prose", "code", "mixed", "png", "jpg", "jpeg", "svg"],
            min_tokens=MIN_SUPPORTED_TOKENS,
        ),
        submitted_text=text if request.include_text else None,
    )


def _request_text(request: AnalysisRequest) -> str:
    if request.text is not None:
        return request.text
    if request.fixture:
        return _fixture_text(request.fixture)
    return ""


def _fixture_text(name: str) -> str:
    fixtures = {
        "human-prose": (
            "human-written I drafted this note after checking the roof myself. The shingles were "
            "uneven, and two nails had lifted near the vent. I called a local roofer and took "
            "photos before the rain started again. The repair plan is practical, not dramatic."
        ),
        "ai-prose": (
            "AI-generated effective home maintenance requires a systematic approach to roof "
            "inspection. Furthermore, homeowners should document visible damage, evaluate drainage, "
            "and consult qualified professionals. Overall, timely intervention can reduce repair "
            "costs and preserve structural integrity."
        ),
        "code": (
            "def calculate_repair_cost(area, rate):\n"
            "    if area <= 0:\n"
            "        raise ValueError('area must be positive')\n"
            "    return round(area * rate, 2)\n"
        ),
    }
    return fixtures.get(name, fixtures["ai-prose"])


def _evidence_state(
    token_count: int,
    content_type: ContentType,
    language: str,
    abstain_reason: str | None,
) -> EvidenceState:
    if token_count < MIN_SUPPORTED_TOKENS:
        return EvidenceState.INSUFFICIENT_DATA
    if content_type == ContentType.PROSE and language != "en":
        return EvidenceState.UNSUPPORTED_LANGUAGE
    if abstain_reason and "Insufficient" in abstain_reason:
        return EvidenceState.INSUFFICIENT_DATA
    if abstain_reason and "calibrated only" in abstain_reason:
        return EvidenceState.UNSUPPORTED_LANGUAGE
    return EvidenceState.SUPPORTED


def _calibrate_distribution(
    distribution: OutcomeDistribution,
    state: EvidenceState,
    prior_odds: float,
) -> OutcomeDistribution:
    if state != EvidenceState.SUPPORTED:
        return OutcomeDistribution(
            human=1 / 3,
            ai_generated=1 / 3,
            ai_refined_or_mixed=1 / 3,
        )
    prior_ai = prior_odds / (1 + prior_odds)
    detector_ai = distribution.ai_generated + distribution.ai_refined_or_mixed * 0.45
    blended_ai = (0.72 * detector_ai) + (0.28 * prior_ai)
    refined = distribution.ai_refined_or_mixed * blended_ai
    generated = max(0.0, blended_ai - refined)
    return OutcomeDistribution(
        human=1 - blended_ai,
        ai_generated=generated,
        ai_refined_or_mixed=refined,
    ).normalized()


def _confidence(
    token_count: int,
    state: EvidenceState,
    distribution: OutcomeDistribution,
    abstain_reason: str | None,
) -> ConfidenceLabel:
    if state != EvidenceState.SUPPORTED or abstain_reason:
        return ConfidenceLabel.LOW
    if token_count >= 500 and max(
        distribution.human, distribution.ai_generated, distribution.ai_refined_or_mixed
    ) >= 0.8:
        return ConfidenceLabel.HIGH
    if token_count >= 150:
        return ConfidenceLabel.MEDIUM
    return ConfidenceLabel.LOW


def _score_segments(
    text: str,
    segments: list[Segment],
    detector,
    content_type: ContentType,
    language: str,
) -> list[Segment]:
    scored: list[Segment] = []
    for segment in segments:
        score = detector.score(text[segment.start : segment.end], content_type, language)
        scored.append(segment.model_copy(update={"posterior": score.distribution.normalized()}))
    return scored


def _matrices(
    segments: list[Segment],
    watermarks,
    attribution,
    prior_odds: float,
) -> Matrices:
    columns = [segment.id for segment in segments]
    source_rows = [item.family for item in attribution.conditional_on_ai[:4]] + ["unknown"]
    source_values: list[list[float | None]] = []
    for row in source_rows:
        if row == "unknown":
            source_values.append([1 - sum(segment.source_family.values()) for segment in segments])
        else:
            source_values.append([segment.source_family.get(row, 0.0) for segment in segments])

    watermark_rows = [result.scheme for result in watermarks]
    watermark_values = []
    for row in watermark_rows:
        watermark_values.append([segment.watermark_evidence.get(row) for segment in segments])

    ai_score = sum(segment.posterior.ai_generated for segment in segments) / max(len(segments), 1)
    lr = _likelihood_ratio(ai_score)
    waterfall = [
        WaterfallItem(label="Prior odds", value=prior_odds, kind="prior"),
        WaterfallItem(label="Detector likelihood", value=math.log(max(lr, 1e-12)), kind="increase"),
        WaterfallItem(
            label="Short/unsupported penalty",
            value=-1.0 if len(segments) == 0 else 0.0,
            kind="penalty",
        ),
        WaterfallItem(
            label="Posterior odds",
            value=prior_odds * lr,
            kind="final",
        ),
    ]
    return Matrices(
        source_by_segment=Matrix(
            rows=source_rows,
            columns=columns,
            values=source_values,
            scale="conditional_probability",
            legend="P(source family | AI-generated, supported candidates)",
        ),
        watermark_evidence_by_segment=Matrix(
            rows=watermark_rows,
            columns=columns,
            values=watermark_values,
            scale="neg_log10_q_or_p",
            legend="Watermark evidence strength; null means insufficient evidence",
        ),
        contribution_waterfall=waterfall,
    )


def _plain_language(state: EvidenceState, distribution: OutcomeDistribution, watermarks, provenance_status: str) -> str:
    if state != EvidenceState.SUPPORTED:
        return "There is not enough supported evidence for a reliable statistical conclusion."
    ai = distribution.ai_generated + distribution.ai_refined_or_mixed
    tested = [result for result in watermarks if result.status == "tested" and result.q_value is not None]
    watermark_clause = ""
    if any(result.q_value < 0.01 for result in tested):
        watermark_clause = " A configured public watermark test found strong evidence."
    provenance_clause = ""
    if provenance_status == "verified":
        provenance_clause = " Signed file provenance was also present."
    if ai >= 0.8:
        return "Strong calibrated evidence suggests AI participation." + watermark_clause + provenance_clause
    if ai >= 0.6:
        return "Moderate calibrated evidence suggests AI participation." + watermark_clause + provenance_clause
    if ai <= 0.2:
        return "The calibrated evidence leans human-written, but this is not proof of authorship."
    return "The evidence is mixed or close to the decision boundary." + watermark_clause + provenance_clause


def _standard_limitations(
    state: EvidenceState,
    content_type: ContentType,
    language: str,
    token_count: int,
    watermark_results,
) -> list[str]:
    limitations = [
        "A negative watermark result is not evidence that content is human-written.",
        "Source-family values are conditional similarity, not proof of exact model identity.",
        "Provenance is not authorship.",
        "Reference baselines and the community catalog are calibration and verification infrastructure; they are not inputs to this analysis.",
    ]
    if state == EvidenceState.INSUFFICIENT_DATA:
        limitations.append("The input is below the minimum evidence threshold.")
    if content_type == ContentType.CODE:
        limitations.append("Code formatting and semantic-preserving edits can remove or distort evidence.")
    if content_type == ContentType.PROSE and language != "en":
        limitations.append("Generic prose calibration currently supports English only.")
    if token_count < 150:
        limitations.append("Short inputs have low statistical power.")
    if any(result.status == "adapter_unavailable" for result in watermark_results):
        limitations.append("Claude text watermark detection is unavailable until Anthropic publishes detector details.")
    return limitations


def _math_definitions() -> list[MathDefinition]:
    return [
        MathDefinition(
            name="Watermark z-score",
            meaning="How many standard deviations the green-token count is above the null expectation.",
            formula=r"z=\frac{G-\gamma n}{\sqrt{n\gamma(1-\gamma)}}",
            units="standard deviations",
            assumptions=["The tokenizer and watermark configuration match generation."],
            limitations=["A p-value is not P(watermarked).", "Editing and short text reduce power."],
            kind="hypothesis_test",
        ),
        MathDefinition(
            name="One-sided p-value",
            meaning="Probability of evidence at least this extreme under the null hypothesis.",
            formula=r"p=1-\Phi(z)",
            units="probability",
            assumptions=["Asymptotic normal approximation is adequate, or an exact test is used."],
            limitations=["A p-value is not the probability that text is watermarked."],
            kind="hypothesis_test",
        ),
        MathDefinition(
            name="Benjamini-Hochberg q-value",
            meaning="False-discovery-adjusted significance across schemes and windows.",
            formula=r"q_{(i)}=\min_{j\ge i}\left\{\frac{m}{j}p_{(j)}\right\}",
            units="probability",
            assumptions=["Tests are independent or positively dependent."],
            limitations=["Does not provide a posterior probability of watermarking."],
            kind="hypothesis_test",
        ),
        MathDefinition(
            name="Calibrated posterior",
            meaning="Detector evidence mapped to a probability using held-out calibration data.",
            formula=r"O_1=O_0 \times \mathrm{LR}",
            units="odds",
            assumptions=["The input belongs to the calibration cohort."],
            limitations=["Domain shift, paraphrase, and mixed authorship can invalidate calibration."],
            kind="calibrated_evidence",
        ),
        MathDefinition(
            name="Source-family Mahalanobis distance",
            meaning="Distance from a segment feature vector to each calibrated family centroid.",
            formula=r"d_m^2=(x-\mu_m)^{T}\Sigma^{-1}(x-\mu_m)",
            units="squared standardized distance",
            assumptions=["Feature distributions are approximately stable for the calibration cohort."],
            limitations=["Unsupported generators should remain unknown."],
            kind="descriptive_context",
        ),
        MathDefinition(
            name="Wilson confidence interval",
            meaning="Binomial uncertainty interval for the observed green-token rate.",
            formula=r"\hat{p}\pm z_{1-\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}+\frac{z^2}{4n^2}}",
            units="proportion",
            assumptions=["Eligible token decisions are approximately exchangeable under the null."],
            limitations=["The interval describes rate uncertainty, not authorship."],
            kind="descriptive_context",
        ),
        MathDefinition(
            name="Conformal knownness",
            meaning="Open-set threshold for whether an input resembles any calibrated source family.",
            formula=r"t_\alpha=\mathrm{Quantile}\left(\{s_i\}; \frac{\lceil (n+1)(1-\alpha)\rceil}{n}\right)",
            units="nonconformity score",
            assumptions=["Held-out known-family examples are exchangeable with the input."],
            limitations=["Coverage can degrade under domain shift."],
            kind="calibrated_evidence",
        ),
    ]


def _likelihood_ratio(ai_probability: float) -> float:
    p = min(max(ai_probability, 1e-6), 1 - 1e-6)
    return p / (1 - p)


def _anomaly_percentile(ai_probability: float) -> float:
    z = (ai_probability - 0.35) / 0.18
    return min(max(NormalDist().cdf(z), 0.0), 1.0)


def _device_name(profile: RuntimeProfile) -> str:
    if profile in {RuntimeProfile.LOCAL_GPU, RuntimeProfile.CLOUD_GPU}:
        return "gpu"
    return "cpu"


def _calibration_bundle(content_type: ContentType, language: str) -> str:
    if content_type == ContentType.CODE:
        return f"code-{language}-baseline-v0"
    return f"prose-{language}-baseline-v0"
