import { render, screen } from '@testing-library/react';
import { expect, test, vi } from 'vitest';
import App from './App';
import type { AnalysisResponse } from './types';

vi.mock('./api', () => ({
  analyze: vi.fn(),
  fetchArtifact: vi.fn().mockResolvedValue(null)
}));

test('renders observatory empty state and input controls', () => {
  render(<App />);
  expect(screen.getByText(/See the evidence/i)).toBeTruthy();
  expect(screen.getByLabelText(/Text or code to analyze/i)).toBeTruthy();
  expect(screen.getByText(/Run a fixture/i)).toBeTruthy();
});

test('technical drilldown renders equations and watermark statistics', async () => {
  const { analyze } = await import('./api');
  vi.mocked(analyze).mockResolvedValue(mockResponse());
  render(<App />);
  screen.getByText('AI prose').click();
  expect(await screen.findByText(/Strong calibrated evidence/i)).toBeTruthy();
  expect(screen.getByText(/Known watermark statistics/i)).toBeTruthy();
  expect(screen.getByText(/Posterior decomposition/i)).toBeTruthy();
});

function mockResponse(): AnalysisResponse {
  return {
    schema_version: '1.2.0',
    report_id: 'fixture-report',
    runtime: { profile: 'fixture', device: 'cpu', models_loaded: ['fixture-detector'], calibration_bundles: ['prose-en'] },
    input: {
      content_hash: 'abc',
      content_type: 'prose',
      language: 'en',
      token_count: 120,
      character_count: 700,
      segment_count: 1,
      user_overrode_type: false,
      feature_profile: { long_words: 0.18, connectors: 0.02, unique_ratio: 0.6, short_sentences: 0.4, structured: 0.3, digits: 0.01, balanced_lines: 0.7 }
    },
    summary: {
      evidence_state: 'supported',
      plain_language: 'Strong calibrated evidence suggests AI participation.',
      confidence_label: 'medium',
      overall: { human: 0.2, ai_generated: 0.75, ai_refined_or_mixed: 0.05 },
      ai_participation: 0.8,
      ai_generation: 0.75
    },
    posterior: {
      prior_odds: 1,
      likelihood_ratio: 3,
      posterior_odds: 3,
      calibration_bundle: 'prose-en',
      reliability_error: 0.04,
      cohort: 'fixture',
      cohort_prevalence: 0.5
    },
    calibration: {
      bundle: 'panoptes-reference-corpus-v1',
      cohort: 'panoptes-reference-corpus (6 model families + human controls)',
      n_records: 104,
      applies_to: 'prose',
      ece: 0.127,
      brier: 0.161,
      auroc: 0.589,
      tpr_at_1fpr: 0.44,
      tpr_at_5fpr: 0.44,
      reliability_bins: [
        { bin_lo: 0.4, bin_hi: 0.5, n: 20, mean_predicted: 0.45, observed: 0.2 },
        { bin_lo: 0.5, bin_hi: 0.6, n: 30, mean_predicted: 0.55, observed: 0.7 }
      ],
      conformal_alpha: 0.1,
      conformal_threshold: 0.8,
      artifact_sha256: 'deadbeef'
    },
    source_families: {
      conditional_on_ai: [{ family: 'llama-like', probability: 0.6 }],
      unknown_score: 0.4,
      interpretation: 'Conditional similarity among supported families.',
      basis: 'corpus-fitted',
      cohort_size: 104
    },
    watermarks: [{
      scheme: 'kgw-v1',
      status: 'tested',
      eligible_tokens: 100,
      green_tokens: 62,
      expected_green: 50,
      green_rate: 0.62,
      green_rate_interval: { lower: 0.52, upper: 0.71, level: 0.95, method: 'wilson' },
      dilution_estimate: 0.24,
      z: 2.4,
      p_value: 0.008,
      q_value: 0.016,
      effect: 0.12,
      power: 0.7,
      tokens: [{ start: 0, end: 2, green: true }]
    }],
    provenance: { status: 'not_present', summary: 'No signed provenance manifest was found.', actions: [], level: 'P0' },
    segments: [{
      id: 'segment-1',
      start: 0,
      end: 100,
      token_count: 100,
      kind: 'prose',
      posterior: { human: 0.2, ai_generated: 0.75, ai_refined_or_mixed: 0.05 },
      watermark_evidence: { 'kgw-v1': 2.1 },
      source_family: { 'llama-like': 0.6 },
      anomaly_percentile: 0.8
    }],
    matrices: {
      source_by_segment: { rows: ['llama-like'], columns: ['segment-1'], values: [[0.6]], scale: 'probability', legend: 'Source family probability' },
      watermark_evidence_by_segment: { rows: ['kgw-v1'], columns: ['segment-1'], values: [[2.1]], scale: 'neg_log10_p', legend: 'Watermark evidence' },
      contribution_waterfall: [{ label: 'Prior odds', value: 1, kind: 'prior' }]
    },
    math: [{
      name: 'Watermark z-score',
      meaning: 'Green-token excess.',
      formula: 'z=\\frac{G-\\gamma n}{\\sqrt{n\\gamma(1-\\gamma)}}',
      units: 'standard deviations',
      assumptions: ['Tokenizer matches generation.'],
      limitations: ['A p-value is not P(watermarked).'],
      kind: 'hypothesis_test'
    }],
    limitations: ['Not proof of authorship.'],
    capabilities: {
      languages: ['en'],
      model_families: ['llama-like'],
      watermark_schemes: ['kgw-v1'],
      content_types: ['prose'],
      min_tokens: 50
    },
    submitted_text: 'AI text fixture'
  };
}
