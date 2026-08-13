from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeProfile(StrEnum):
    FIXTURE = "fixture"
    LOCAL_CPU = "local-cpu"
    LOCAL_GPU = "local-gpu"
    CLOUD_CPU = "cloud-cpu"
    CLOUD_GPU = "cloud-gpu"


class EvidenceState(StrEnum):
    SUPPORTED = "supported"
    INSUFFICIENT_DATA = "insufficient_data"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    UNSUPPORTED_CONTENT = "unsupported_content"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    NOT_APPLICABLE = "not_applicable"
    ADAPTER_UNAVAILABLE = "adapter_unavailable"


class ContentType(StrEnum):
    PROSE = "prose"
    CODE = "code"
    MIXED = "mixed"
    FILE = "file"


class ConfidenceLabel(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OutcomeDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    human: float = Field(ge=0, le=1)
    ai_generated: float = Field(ge=0, le=1)
    ai_refined_or_mixed: float = Field(ge=0, le=1)

    def normalized(self) -> "OutcomeDistribution":
        total = self.human + self.ai_generated + self.ai_refined_or_mixed
        if total <= 0:
            return OutcomeDistribution(human=1 / 3, ai_generated=1 / 3, ai_refined_or_mixed=1 / 3)
        return OutcomeDistribution(
            human=self.human / total,
            ai_generated=self.ai_generated / total,
            ai_refined_or_mixed=self.ai_refined_or_mixed / total,
        )


class RuntimeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: RuntimeProfile
    device: str
    models_loaded: list[str]
    calibration_bundles: list[str]


class InputDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_hash: str
    content_type: ContentType
    language: str
    token_count: int = Field(ge=0)
    character_count: int = Field(ge=0)
    segment_count: int = Field(ge=0)
    user_overrode_type: bool = False
    # Stylometric features of the submitted text (same definitions as the
    # corpus features), so the UI can show the input against cohort ranges.
    feature_profile: dict[str, float] = {}


class Summary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_state: EvidenceState
    plain_language: str
    confidence_label: ConfidenceLabel
    overall: OutcomeDistribution
    ai_participation: float = Field(ge=0, le=1)
    ai_generation: float = Field(ge=0, le=1)


class PosteriorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prior_odds: float = Field(ge=0)
    likelihood_ratio: float | None = Field(default=None, ge=0)
    posterior_odds: float | None = Field(default=None, ge=0)
    calibration_bundle: str
    reliability_error: float | None = Field(default=None, ge=0)
    cohort: str
    cohort_prevalence: float | None = Field(default=None, ge=0, le=1)


class SourceFamilyProbability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: str
    probability: float = Field(ge=0, le=1)


class SourceFamilies(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conditional_on_ai: list[SourceFamilyProbability]
    unknown_score: float = Field(ge=0, le=1)
    interpretation: str
    basis: str = "heuristic"  # "corpus-fitted" when the signed calibration artifact supplied the geometry
    cohort_size: int | None = Field(default=None, ge=0)


class ReliabilityBin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bin_lo: float = Field(ge=0, le=1)
    bin_hi: float = Field(ge=0, le=1)
    n: int = Field(ge=0)
    mean_predicted: float = Field(ge=0, le=1)
    observed: float = Field(ge=0, le=1)


class CalibrationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle: str
    cohort: str
    n_records: int = Field(ge=0)
    applies_to: str
    ece: float = Field(ge=0, le=1)
    brier: float = Field(ge=0, le=1)
    auroc: float = Field(ge=0, le=1)
    tpr_at_1fpr: float = Field(ge=0, le=1)
    tpr_at_5fpr: float = Field(ge=0, le=1)
    reliability_bins: list[ReliabilityBin]
    conformal_alpha: float = Field(ge=0, le=1)
    conformal_threshold: float = Field(ge=0, le=1)
    artifact_sha256: str


class WatermarkTokenSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    green: bool


class ConfidenceInterval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lower: float = Field(ge=0, le=1)
    upper: float = Field(ge=0, le=1)
    level: float = Field(default=0.95, ge=0, le=1)
    method: str = "wilson"


class WatermarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scheme: str
    status: Literal["tested", "insufficient_data", "adapter_unavailable", "not_applicable"]
    eligible_tokens: int = Field(ge=0)
    green_tokens: int | None = Field(default=None, ge=0)
    expected_green: float | None = Field(default=None, ge=0)
    green_rate: float | None = Field(default=None, ge=0, le=1)
    green_rate_interval: ConfidenceInterval | None = None
    dilution_estimate: float | None = Field(default=None, ge=0, le=1)
    z: float | None = None
    p_value: float | None = Field(default=None, ge=0, le=1)
    q_value: float | None = Field(default=None, ge=0, le=1)
    effect: float | None = None
    power: float | None = Field(default=None, ge=0, le=1)
    tokens: list[WatermarkTokenSpan] | None = None


class ProvenanceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["verified", "tampered", "not_present", "unsupported_file", "error", "not_applicable"]
    summary: str
    issuer: str | None = None
    timestamp: str | None = None
    actions: list[str] = Field(default_factory=list)
    level: Literal["P0", "P1", "P2", "P3", "P4"] = "P0"

    @model_validator(mode="after")
    def assign_default_level(self) -> "ProvenanceResult":
        if self.status == "verified" and self.level == "P0":
            return self.model_copy(update={"level": "P3"})
        return self


class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    token_count: int = Field(ge=0)
    kind: ContentType
    posterior: OutcomeDistribution
    watermark_evidence: dict[str, float | None] = Field(default_factory=dict)
    source_family: dict[str, float] = Field(default_factory=dict)
    anomaly_percentile: float | None = Field(default=None, ge=0, le=1)


class Matrix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[str]
    columns: list[str]
    values: list[list[float | None]]
    scale: str
    legend: str


class WaterfallItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    value: float
    kind: Literal["prior", "increase", "decrease", "penalty", "final"]


class Matrices(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_by_segment: Matrix
    watermark_evidence_by_segment: Matrix
    contribution_waterfall: list[WaterfallItem]


class MathDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    meaning: str
    formula: str
    units: str
    assumptions: list[str]
    limitations: list[str]
    kind: Literal["calibrated_evidence", "hypothesis_test", "descriptive_context"]


class Capabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    languages: list[str]
    model_families: list[str]
    watermark_schemes: list[str]
    content_types: list[str]
    min_tokens: int = Field(ge=0)


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    file_base64: str | None = None
    filename: str | None = None
    content_type: ContentType | None = None
    language: str | None = None
    prior_odds: float = Field(default=1.0, gt=0)
    fixture: str | None = None
    include_text: bool = False


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.2.0"
    report_id: str
    runtime: RuntimeInfo
    input: InputDiagnostics
    summary: Summary
    posterior: PosteriorInfo
    calibration: CalibrationInfo | None = None
    source_families: SourceFamilies
    watermarks: list[WatermarkResult]
    provenance: ProvenanceResult
    segments: list[Segment]
    matrices: Matrices
    math: list[MathDefinition]
    limitations: list[str]
    capabilities: Capabilities
    submitted_text: str | None = Field(default=None, exclude=False)

    def public_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        if data.get("submitted_text") is None:
            data.pop("submitted_text", None)
        return data
