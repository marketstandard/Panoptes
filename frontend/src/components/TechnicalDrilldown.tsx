import { AlertTriangle, Database, GitBranch, Sigma } from 'lucide-react';
import type { AnalysisResponse } from '../types';
import { formatNumber, formatPercent, formatSigned } from '../format';
import { Equation } from './Equation';
import { InfoTip } from './InfoTip';

const WATERFALL_TIPS: Record<string, string> = {
  'Prior odds':
    'Starting odds that the document is AI-generated, taken from the calibration cohort before any test evidence is applied.',
  'Detector likelihood':
    'Log likelihood ratio from the detector tests across all segments — how much the test evidence multiplies the odds. Positive pushes toward AI-generated, negative toward human.',
  'Short/unsupported penalty':
    'Penalty applied when the text is too short or lacks supported evidence for a reliable conclusion; it pushes the odds back down because the tests cannot be trusted at this length.',
  'Posterior odds':
    'Final odds after every contribution: prior odds multiplied by the likelihood ratio, with penalties. Converted to the headline probability by P = O / (1 + O).',
};

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
            <InfoTip
              label="Posterior decomposition"
              text="How the headline probability is built: a calibrated detector probability is converted to a prevalence-corrected likelihood ratio, then multiplied by the declared prior odds. ECE is reported alongside and is not mixed into this product."
            />
          </div>
          <Equation formula="O_1 = O_0 \times \mathrm{LR},\quad \mathrm{LR}=\frac{p}{1-p}\cdot\frac{1-\pi}{\pi}" />
          <div className="stat-grid">
            <Stat
              label="Prior odds"
              value={formatNumber(result.posterior.prior_odds)}
              tip="Odds that a document is AI-generated before any test evidence is considered: p / (1 - p), set by the calibration cohort."
            />
            <Stat
              label="Likelihood ratio"
              value={formatNumber(result.posterior.likelihood_ratio)}
              tip="How much the detector tests shift the odds. Above 1 supports AI-generated, below 1 supports human; combined across the document's segments."
            />
            <Stat
              label="Posterior odds"
              value={formatNumber(result.posterior.posterior_odds)}
              tip="Prior odds x likelihood ratio — the odds after the test evidence. P = O / (1 + O) converts this to the headline probability."
            />
            <Stat
              label="Reliability error (ECE)"
              value={formatNumber(result.posterior.reliability_error)}
              tip="Expected calibration error of the detector on the held-out calibration cohort: how far predicted probabilities are from observed frequencies. This is a quality diagnostic. It is not subtracted from or used to rescale the posterior."
            />
            <Stat
              label="Cohort prevalence"
              value={formatNumber(result.posterior.cohort_prevalence)}
              tip="Share of AI-labeled documents in the calibration cohort (π). The likelihood ratio divides out this base rate so the same evidence can be combined with a different real-world prior."
            />
          </div>
          <p className="lab-note">Calibration cohort: {result.posterior.cohort}</p>
        </article>

        <article className="lab-card waterfall-card">
          <div className="card-heading">
            <GitBranch size={17} />
            <h3>Evidence contribution waterfall</h3>
            <InfoTip
              label="Evidence contribution waterfall"
              text="How each step moves the odds, in order: where they start (prior), how the detector evidence shifts them (log likelihood ratio), any penalty for short or unsupported text, and the final odds."
            />
          </div>
          <div className="waterfall">
            {result.matrices.contribution_waterfall.map((item) => (
              <div className={`waterfall-row kind-${item.kind}`} key={item.label}>
                <span>
                  {item.label}
                  <InfoTip
                    label={item.label}
                    text={WATERFALL_TIPS[item.label] ?? "This step's contribution to the final odds."}
                  />
                </span>
                <div className="waterfall-track">
                  <div style={{ width: `${Math.min(Math.abs(item.value) * 18, 100)}%` }} />
                </div>
                <strong>{formatSigned(item.value, 2)}</strong>
              </div>
            ))}
          </div>
          <p className="lab-note">Bar length shows the size of each contribution. Detector evidence is a log likelihood ratio; the other rows are odds.</p>
        </article>

        <article className="lab-card watermark-card">
          <div className="card-heading">
            <Database size={17} />
            <h3>Known watermark statistics</h3>
            <InfoTip
              label="Known watermark statistics"
              text="One-sided binomial tests for known AI watermark schemes. Each test checks whether eligible tokens land in the scheme's pseudorandom 'green' set more often than chance; the status shows whether the test ran or why it could not."
            />
          </div>
          {result.watermarks.map((watermark) => (
            <div className="watermark-block" key={watermark.scheme}>
              <div className="watermark-title">
                <strong>{watermark.scheme}</strong>
                <span>{watermark.status.replaceAll('_', ' ')}</span>
                {watermark.origin === 'plugin' || watermark.scheme.startsWith('plugin:') ? (
                  <span className="badge">plugin</span>
                ) : (
                  <span className="badge">builtin</span>
                )}
              </div>
              <Equation formula="z=\frac{G-\gamma n}{\sqrt{n\gamma(1-\gamma)}}, \qquad p=1-\Phi(z)" />
              <div className="stat-grid">
                <Stat
                  label="Eligible n"
                  value={String(watermark.eligible_tokens)}
                  tip="Number of tokens eligible for this watermark test — the test's sample size. Very small n means the test can say little."
                />
                <Stat
                  label="Green G"
                  value={watermark.green_tokens === null ? 'n/a' : String(watermark.green_tokens)}
                  tip="How many eligible tokens actually fell into the watermark's pseudorandom 'green' set."
                />
                <Stat
                  label="Expected"
                  value={formatNumber(watermark.expected_green)}
                  tip="Green tokens expected by chance if there is no watermark: the green-list fraction gamma multiplied by n."
                />
                <Stat
                  label="Green rate"
                  value={watermark.green_rate === null ? 'n/a' : formatPercent(watermark.green_rate)}
                  tip="Observed fraction of eligible tokens that are green (G / n), compared against the chance rate gamma."
                />
                <Stat
                  label="95% interval"
                  value={watermark.green_rate_interval ? `${formatPercent(watermark.green_rate_interval.lower)}–${formatPercent(watermark.green_rate_interval.upper)}` : 'n/a'}
                  tip="95% confidence interval for the true green rate given this sample size. Wide intervals mean the estimate is imprecise."
                />
                <Stat
                  label="Effect"
                  value={formatSigned(watermark.effect)}
                  tip="Observed green rate minus the chance rate gamma — the raw lift this test detected."
                />
                <Stat
                  label="z"
                  value={formatNumber(watermark.z)}
                  tip="Standardized distance between the observed and expected green counts. Near 0 is consistent with chance; larger positive values indicate a watermark."
                />
                <Stat
                  label="p"
                  value={formatNumber(watermark.p_value)}
                  tip="One-sided p-value: the probability of seeing a green count this high by chance alone. Small p is evidence of a watermark."
                />
                <Stat
                  label="q"
                  value={formatNumber(watermark.q_value)}
                  tip="False-discovery-rate adjusted p-value (Benjamini-Hochberg) across all watermark tests run on this document — the number to quote when several tests were tried."
                />
                <Stat
                  label="Power"
                  value={formatNumber(watermark.power)}
                  tip="The chance this test would detect a watermark of the configured strength at this sample size. Low power means a negative result is only weak evidence of absence."
                />
                <Stat
                  label="Dilution"
                  value={formatNumber(watermark.dilution_estimate)}
                  tip="Estimated fraction of the text carrying the watermark needed to explain the observed lift — low values suggest the text was edited or mixed after generation."
                />
              </div>
            </div>
          ))}
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
            <InfoTip
              label="Interpretation limits"
              text="Read these before acting on the result — the conditions under which this analysis is weak, inconclusive, or should not be relied on."
            />
          </div>
          <ul>
            {result.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}
          </ul>
        </article>
      </div>
    </section>
  );
}

function Stat({ label, value, tip }: { label: string; value: string; tip?: string }) {
  return (
    <div className="stat">
      <span>
        {label}
        {tip ? <InfoTip label={label} text={tip} /> : null}
      </span>
      <strong>{value}</strong>
    </div>
  );
}
