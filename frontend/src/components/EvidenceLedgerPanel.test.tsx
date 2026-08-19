import { render, screen } from '@testing-library/react';
import { expect, test } from 'vitest';
import { EvidenceLedgerPanel } from './EvidenceLedgerPanel';
import type { AnalysisResponse, EvidenceLedger } from '../types';

function ledger(overrides: Partial<EvidenceLedger> = {}): EvidenceLedger {
  return {
    statistical: [
      {
        channel: 'statistical',
        target_claim: 'ai_participation',
        source_identity: 'heuristic-v1',
        validity: 'valid',
        applicability_scope: "calibration cohort 'prose-en-baseline-v0'",
        strength: 0.8,
        uncertainty: 'Calibration ECE 0.04 on the held-out cohort.',
        assumptions: ['The input is drawn from a population the calibration cohort represents.'],
        limitations: ['Statistical evidence is population-conditional, not proof of authorship.'],
        statistical: {
          detector_id: 'heuristic-v1',
          model_revision: null,
          calibrator_id: 'baseline-calibration.json',
          cohort: 'prose-en-baseline-v0',
          cohort_prevalence: 0.5,
          ai_participation: 0.8,
          ai_majority_generation: 0.72,
          contribution_fraction: null,
          applicability: null,
          transport_warning: 'population-conditional'
        },
        watermark: null,
        provenance: null
      }
    ],
    watermark: [
      {
        channel: 'watermark',
        target_claim: 'watermark_present',
        source_identity: 'kgw-v1',
        validity: 'valid',
        applicability_scope: 'texts generated with the kgw-v1 watermark configuration',
        strength: 0.1,
        uncertainty: 'z=0.4, q=0.9, power=0.3',
        assumptions: ['The tokenizer and watermark configuration match generation.'],
        limitations: ['A negative watermark result is not evidence that content is human-written.'],
        statistical: null,
        watermark: {
          scheme: 'kgw-v1',
          status: 'tested',
          eligible_tokens: 400,
          green_rate: 0.51,
          z: 0.4,
          p_value: 0.34,
          q_value: 0.9,
          power: 0.3,
          dilution_estimate: null
        },
        provenance: null
      }
    ],
    provenance: [
      {
        channel: 'provenance',
        target_claim: 'provenance_chain_valid',
        source_identity: 'example-issuer',
        validity: 'valid',
        applicability_scope: 'files carrying an embedded C2PA manifest',
        strength: null,
        uncertainty: null,
        assumptions: ['The signing chain is intact and the issuer is trusted.'],
        limitations: ['Provenance attests a signing chain, not metaphysical authorship.'],
        statistical: null,
        watermark: null,
        provenance: {
          status: 'verified',
          issuer: 'example-issuer',
          timestamp: '2026-08-19T00:00:00Z',
          signature_chain: ['example-issuer'],
          level: 'P3',
          actions: ['c2pa.created']
        }
      }
    ],
    channel_summaries: {
      statistical: 'Calibrated statistical evidence estimates P(AI participation) = 0.80.',
      watermark: 'No configured watermark test reached significance; this is not evidence of human authorship.',
      provenance: "A signing chain from 'example-issuer' verified; this attests provenance, not authorship."
    },
    fusion_note:
      'Statistical, watermark, and provenance evidence are distinct channels and are never arithmetically fused into a single confidence number.',
    ...overrides
  };
}

function responseWith(ledgerValue: EvidenceLedger): AnalysisResponse {
  return { evidence_ledger: ledgerValue } as unknown as AnalysisResponse;
}

test('a negative watermark is never presented as human evidence', () => {
  render(<EvidenceLedgerPanel result={responseWith(ledger())} />);
  expect(screen.getByText(/not evidence of human authorship/i)).toBeTruthy();
  expect(screen.getByText(/not evidence that content is human-written/i)).toBeTruthy();
});

test('valid provenance attests a chain and never alters the statistical posterior', () => {
  render(<EvidenceLedgerPanel result={responseWith(ledger())} />);
  expect(screen.getByText(/attests provenance, not authorship/i)).toBeTruthy();
  expect(screen.getByText(/attests a signing chain, not metaphysical authorship/i)).toBeTruthy();
});

test('the ledger states the no-fusion rule and renders three channels', () => {
  render(<EvidenceLedgerPanel result={responseWith(ledger())} />);
  expect(screen.getByText(/never arithmetically fused/i)).toBeTruthy();
  expect(screen.getByText('Statistical')).toBeTruthy();
  expect(screen.getByText('Watermark')).toBeTruthy();
  expect(screen.getByText('Provenance')).toBeTruthy();
});

test('provenance entries carry no probability strength', () => {
  const { container } = render(<EvidenceLedgerPanel result={responseWith(ledger())} />);
  const provenanceChannel = container.querySelector('[data-channel="provenance"]');
  expect(provenanceChannel).toBeTruthy();
  expect(provenanceChannel!.textContent).not.toMatch(/Strength/);
});
