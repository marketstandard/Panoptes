import { render, screen } from '@testing-library/react';
import { expect, test, vi } from 'vitest';
import { CorpusPanel } from './CorpusPanel';
import { CoverageCurve } from './CoverageCurve';
import { InputProfile } from './InputProfile';
import { ModelCardPanel } from './ModelCardPanel';
import { PosteriorSensitivity } from './PosteriorSensitivity';
import { PowerCurve, powerAt } from './PowerCurve';
import { ReliabilityDiagram } from './ReliabilityDiagram';

vi.mock('../hooks', () => ({
  useArtifact: vi.fn().mockReturnValue({ data: null, loading: false })
}));

test('PosteriorSensitivity renders curve with marker', () => {
  render(<PosteriorSensitivity priorOdds={1} likelihoodRatio={3.2} />);
  expect(screen.getByText(/Posterior sensitivity/i)).toBeTruthy();
  expect(screen.getByText(/LR 3.20/)).toBeTruthy();
});

test('PosteriorSensitivity degrades without a likelihood ratio', () => {
  render(<PosteriorSensitivity priorOdds={1} likelihoodRatio={null} />);
  expect(screen.getByText(/No likelihood ratio/i)).toBeTruthy();
});

test('ReliabilityDiagram renders bins from the calibration block', () => {
  render(
    <ReliabilityDiagram
      calibration={{
        bundle: 'baseline-calibration-v1',
        cohort: 'reference-baselines+human-controls',
        n_records: 104,
        applies_to: 'prose',
        ece: 0.041,
        brier: 0.02,
        auroc: 1.0,
        tpr_at_1fpr: 1,
        tpr_at_5fpr: 1,
        reliability_bins: [{ bin_lo: 0.9, bin_hi: 1, n: 50, mean_predicted: 0.97, observed: 1 }],
        conformal_alpha: 0.1,
        conformal_threshold: 0.05,
        artifact_sha256: 'abc'
      }}
    />
  );
  expect(screen.getByText(/Reliability diagram/i)).toBeTruthy();
  expect(screen.getAllByText(/ECE 0.041/).length).toBeGreaterThan(0);
});

test('ReliabilityDiagram degrades without calibration', () => {
  render(<ReliabilityDiagram calibration={null} />);
  expect(screen.getByText(/No signed calibration artifact/i)).toBeTruthy();
});

test('PowerCurve renders and power is monotone in n', () => {
  render(<PowerCurve currentN={104} />);
  expect(screen.getByText(/Statistical power/i)).toBeTruthy();
  expect(powerAt(3140)).toBeGreaterThan(0.79);
  expect(powerAt(3140)).toBeLessThan(0.81);
  expect(powerAt(5000)).toBeGreaterThan(powerAt(100));
});

test('CoverageCurve shows empty state without a card', () => {
  render(<CoverageCurve />);
  expect(screen.getByText(/Train the reference model/i)).toBeTruthy();
});

test('CorpusPanel shows empty state without the artifact', () => {
  render(<CorpusPanel />);
  expect(screen.getByText(/Corpus summary artifact not found/i)).toBeTruthy();
});

test('InputProfile renders feature rows', () => {
  render(<InputProfile featureProfile={{ long_words: 0.18, connectors: 0.02 }} />);
  expect(screen.getByText(/Input profile/i)).toBeTruthy();
  expect(screen.getByText('long-word rate')).toBeTruthy();
  expect(screen.getByText('0.180')).toBeTruthy();
});

test('InputProfile renders nothing without features', () => {
  const { container } = render(<InputProfile featureProfile={{}} />);
  expect(container.firstChild).toBeNull();
});

test('ModelCardPanel shows empty state without a card', () => {
  render(<ModelCardPanel />);
  expect(screen.getByText(/No Panoptes-v0 card found/i)).toBeTruthy();
});
