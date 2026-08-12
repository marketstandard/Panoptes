import { AlertTriangle, Database, GitBranch, Sigma } from 'lucide-react';
import type { AnalysisResponse } from '../types';
import { formatNumber, formatPercent, formatSigned } from '../format';
import { Equation } from './Equation';

export function TechnicalDrilldown({ result }: { result: AnalysisResponse }) {
  return (
    <section className="technical-lab" aria-labelledby="technical-heading">
      <div className="lab-heading">
        <div>
          <p className="panel-kicker">Technical laboratory</p>
          <h2 id="technical-heading">Deep statistical drilldown</h2>
        </div>
        <span>{result.runtime.models_loaded.join(', ')}</span>
      </div>

      <div className="lab-grid">
        <article className="lab-card posterior-card">
          <div className="card-heading">
            <Sigma size={17} />
            <h3>Posterior decomposition</h3>
          </div>
          <Equation formula="O_1 = O_0 \times \mathrm{LR}, \qquad P=\frac{O_1}{1+O_1}" />
          <div className="stat-grid">
            <Stat label="Prior odds" value={formatNumber(result.posterior.prior_odds)} />
            <Stat label="Likelihood ratio" value={formatNumber(result.posterior.likelihood_ratio)} />
            <Stat label="Posterior odds" value={formatNumber(result.posterior.posterior_odds)} />
            <Stat label="Reliability error" value={formatNumber(result.posterior.reliability_error)} />
          </div>
          <p className="lab-note">Calibration cohort: {result.posterior.cohort}</p>
        </article>

        <article className="lab-card watermark-card">
          <div className="card-heading">
            <Database size={17} />
            <h3>Known watermark statistics</h3>
          </div>
          {result.watermarks.map((watermark) => (
            <div className="watermark-block" key={watermark.scheme}>
              <div className="watermark-title">
                <strong>{watermark.scheme}</strong>
                <span>{watermark.status.replaceAll('_', ' ')}</span>
              </div>
              <Equation formula="z=\frac{G-\gamma n}{\sqrt{n\gamma(1-\gamma)}}, \qquad p=1-\Phi(z)" />
              <div className="stat-grid">
                <Stat label="Eligible n" value={String(watermark.eligible_tokens)} />
                <Stat label="Green G" value={watermark.green_tokens === null ? 'n/a' : String(watermark.green_tokens)} />
                <Stat label="Expected" value={formatNumber(watermark.expected_green)} />
                <Stat label="Green rate" value={watermark.green_rate === null ? 'n/a' : formatPercent(watermark.green_rate)} />
                <Stat label="95% interval" value={watermark.green_rate_interval ? `${formatPercent(watermark.green_rate_interval.lower)}–${formatPercent(watermark.green_rate_interval.upper)}` : 'n/a'} />
                <Stat label="Effect" value={formatSigned(watermark.effect)} />
                <Stat label="z" value={formatNumber(watermark.z)} />
                <Stat label="p" value={formatNumber(watermark.p_value)} />
                <Stat label="q" value={formatNumber(watermark.q_value)} />
                <Stat label="Power" value={formatNumber(watermark.power)} />
                <Stat label="Dilution" value={formatNumber(watermark.dilution_estimate)} />
              </div>
            </div>
          ))}
        </article>

        <article className="lab-card waterfall-card">
          <div className="card-heading">
            <GitBranch size={17} />
            <h3>Evidence contribution waterfall</h3>
          </div>
          <div className="waterfall">
            {result.matrices.contribution_waterfall.map((item) => (
              <div className={`waterfall-row kind-${item.kind}`} key={item.label}>
                <span>{item.label}</span>
                <div className="waterfall-track">
                  <div style={{ width: `${Math.min(Math.abs(item.value) * 18, 100)}%` }} />
                </div>
                <strong>{formatSigned(item.value, 2)}</strong>
              </div>
            ))}
          </div>
        </article>

        {result.math.map((item) => (
          <article className="lab-card equation-card" key={item.name}>
            <div className="card-heading">
              <Sigma size={17} />
              <h3>{item.name}</h3>
            </div>
            <Equation formula={item.formula} />
            <p>{item.meaning}</p>
            <dl className="definition-list">
              <div><dt>Units</dt><dd>{item.units}</dd></div>
              <div><dt>Assumptions</dt><dd>{item.assumptions.join(' ')}</dd></div>
              <div><dt>Limits</dt><dd>{item.limitations.join(' ')}</dd></div>
            </dl>
          </article>
        ))}

        <article className="lab-card limits-card">
          <div className="card-heading">
            <AlertTriangle size={17} />
            <h3>Interpretation limits</h3>
          </div>
          <ul>
            {result.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}
          </ul>
        </article>
      </div>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
