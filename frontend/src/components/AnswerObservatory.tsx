import { Activity, Fingerprint, Microscope, ShieldCheck } from 'lucide-react';
import type { AnalysisResponse } from '../types';
import { formatPercent } from '../format';

const outcomes = [
  ['human', 'Human'],
  ['ai_generated', 'AI-generated'],
  ['ai_refined_or_mixed', 'AI-refined / mixed']
] as const;

export function AnswerObservatory({ result }: { result: AnalysisResponse }) {
  const testedWatermarks = result.watermarks.filter((watermark) => watermark.status === 'tested');
  const strongestWatermark = testedWatermarks.reduce(
    (best, watermark) =>
      watermark.q_value !== null && (best?.q_value === null || watermark.q_value < (best.q_value ?? 1))
        ? watermark
        : best,
    testedWatermarks[0] ?? null
  );

  return (
    <section className="observatory-answer" aria-labelledby="answer-heading">
      <div className="answer-main glass-panel">
        <div className="panel-kicker">
          <Microscope size={16} />
          Evidence synthesis
        </div>
        <h2 id="answer-heading">{result.summary.plain_language}</h2>
        <div className="state-row">
          <span className={`state-pill state-${result.summary.evidence_state}`}>
            {result.summary.evidence_state.replaceAll('_', ' ')}
          </span>
          <span className="state-pill">{result.summary.confidence_label} confidence</span>
          <span className="state-pill">{result.input.token_count} tokens</span>
          <span className="state-pill">{result.input.content_type}</span>
        </div>
        <div className="probability-field" role="img" aria-label="Calibrated participation probabilities">
          {outcomes.map(([key, label]) => (
            <div className="probability-row" key={key}>
              <span>{label}</span>
              <div className="probability-track">
                <div
                  className={`probability-fill fill-${key}`}
                  style={{ width: `${result.summary.overall[key] * 100}%` }}
                />
              </div>
              <strong>{formatPercent(result.summary.overall[key])}</strong>
            </div>
          ))}
        </div>
      </div>

      <div className="answer-side">
        <MetricCard
          icon={<Activity />}
          label="Likelihood ratio"
          value={result.posterior.likelihood_ratio?.toFixed(2) ?? 'n/a'}
          detail={result.posterior.calibration_bundle}
        />
        <MetricCard
          icon={<Fingerprint />}
          label="Watermark"
          value={strongestWatermark ? strongestWatermark.scheme : 'none tested'}
          detail={
            strongestWatermark?.q_value !== null && strongestWatermark?.q_value !== undefined
              ? `q=${strongestWatermark.q_value.toExponential(2)}`
              : 'No public scheme detected'
          }
        />
        <MetricCard
          icon={<ShieldCheck />}
          label="Provenance"
          value={result.provenance.status.replaceAll('_', ' ')}
          detail={result.provenance.issuer ?? result.provenance.summary}
        />
      </div>
    </section>
  );
}

function MetricCard({
  icon,
  label,
  value,
  detail
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <article className="metric-card glass-panel">
      <div className="metric-icon">{icon}</div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
    </article>
  );
}
