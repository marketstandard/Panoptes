export type EvidenceState =
  | 'supported'
  | 'insufficient_data'
  | 'unsupported_language'
  | 'unsupported_content'
  | 'out_of_distribution'
  | 'not_applicable'
  | 'adapter_unavailable';

export interface OutcomeDistribution {
  human: number;
  ai_generated: number;
  ai_refined_or_mixed: number;
}

export interface Matrix {
  rows: string[];
  columns: string[];
  values: Array<Array<number | null>>;
  scale: string;
  legend: string;
}

export interface Segment {
  id: string;
  start: number;
  end: number;
  token_count: number;
  kind: string;
  posterior: OutcomeDistribution;
  watermark_evidence: Record<string, number | null>;
  source_family: Record<string, number>;
  anomaly_percentile: number | null;
}

export interface WatermarkTokenSpan {
  start: number;
  end: number;
  green: boolean;
}

export interface WatermarkResult {
  scheme: string;
  status: 'tested' | 'insufficient_data' | 'adapter_unavailable' | 'not_applicable';
  eligible_tokens: number;
  green_tokens: number | null;
  expected_green: number | null;
  green_rate: number | null;
  green_rate_interval: {
    lower: number;
    upper: number;
    level: number;
    method: string;
  } | null;
  dilution_estimate: number | null;
  z: number | null;
  p_value: number | null;
  q_value: number | null;
  effect: number | null;
  power: number | null;
  tokens: WatermarkTokenSpan[] | null;
}

export interface AnalysisResponse {
  schema_version: string;
  report_id: string;
  runtime: {
    profile: string;
    device: string;
    models_loaded: string[];
    calibration_bundles: string[];
  };
  input: {
    content_hash: string;
    content_type: string;
    language: string;
    token_count: number;
    character_count: number;
    segment_count: number;
    user_overrode_type: boolean;
    feature_profile: Record<string, number>;
  };
  summary: {
    evidence_state: EvidenceState;
    plain_language: string;
    confidence_label: 'none' | 'low' | 'medium' | 'high';
    overall: OutcomeDistribution;
    ai_participation?: number;
    ai_generation?: number;
  };
  posterior: {
    prior_odds: number;
    likelihood_ratio: number | null;
    posterior_odds: number | null;
    calibration_bundle: string;
    reliability_error: number | null;
    cohort: string;
    cohort_prevalence?: number | null;
  };
  calibration: {
    bundle: string;
    cohort: string;
    n_records: number;
    applies_to: string;
    ece: number;
    brier: number;
    auroc: number;
    tpr_at_1fpr: number;
    tpr_at_5fpr: number;
    reliability_bins: Array<{
      bin_lo: number;
      bin_hi: number;
      n: number;
      mean_predicted: number;
      observed: number;
    }>;
    conformal_alpha: number;
    conformal_threshold: number;
    artifact_sha256: string;
  } | null;
  source_families: {
    conditional_on_ai: Array<{ family: string; probability: number }>;
    unknown_score: number;
    interpretation: string;
    basis: 'heuristic' | 'corpus-fitted';
    cohort_size: number | null;
  };
  watermarks: WatermarkResult[];
  provenance: {
    status: string;
    summary: string;
    issuer?: string | null;
    timestamp?: string | null;
    actions: string[];
    level?: 'P0' | 'P1' | 'P2' | 'P3' | 'P4';
  };
  segments: Segment[];
  matrices: {
    source_by_segment: Matrix;
    watermark_evidence_by_segment: Matrix;
    contribution_waterfall: Array<{ label: string; value: number; kind: string }>;
  };
  math: Array<{
    name: string;
    meaning: string;
    formula: string;
    units: string;
    assumptions: string[];
    limitations: string[];
    kind: string;
  }>;
  limitations: string[];
  capabilities: {
    languages: string[];
    model_families: string[];
    watermark_schemes: string[];
    content_types: string[];
    min_tokens: number;
  };
  submitted_text?: string;
}

export interface AnalysisRequest {
  text?: string;
  fixture?: string;
  prior_odds?: number;
  include_text?: boolean;
}
