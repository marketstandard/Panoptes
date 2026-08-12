import { InfoTip } from './InfoTip';
import type { AnalysisResponse } from '../types';

interface Props {
  calibration: AnalysisResponse['calibration'];
}

const W = 520;
const H = 280;
const PAD = { left: 52, right: 16, top: 18, bottom: 52 };

export function ReliabilityDiagram({ calibration }: Props) {
  if (!calibration || calibration.reliability_bins.length === 0) {
    return (
      <article className="glass-panel figure-card">
        <h3>
          Reliability diagram
          <InfoTip
            label="Reliability diagram"
            text="Binned predicted probability versus observed AI frequency on held-out corpus folds. Points on the diagonal mean the probabilities mean what they say."
          />
        </h3>
        <p className="figure-empty">No signed calibration artifact is loaded.</p>
      </article>
    );
  }

  const bins = calibration.reliability_bins;
  const maxN = Math.max(...bins.map((bin) => bin.n), 1);
  const x = (value: number) => PAD.left + value * (W - PAD.left - PAD.right);
  const y = (value: number) => H - PAD.bottom - value * (H - PAD.top - PAD.bottom);

  return (
    <article className="glass-panel figure-card">
      <h3>
        Reliability diagram
        <InfoTip
          label="Reliability diagram"
          text={`Held-out calibration of the shipped heuristic on the verified corpus (n=${calibration.n_records}, GroupKFold by prompt). Point size is bin count; the diagonal is perfect calibration. ECE ${calibration.ece.toFixed(3)}. Hover a point for its bin detail.`}
        />
      </h3>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Reliability diagram">
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={x(tick)} y1={y(0)} x2={x(tick)} y2={y(1)} className="grid-line" />
            <text x={x(tick)} y={H - 30} className="axis-label" textAnchor="middle">
              {tick.toFixed(2)}
            </text>
            <text x={PAD.left - 8} y={y(tick) + 4} className="axis-label" textAnchor="end">
              {tick.toFixed(2)}
            </text>
          </g>
        ))}
        <text x={(PAD.left + W - PAD.right) / 2} y={H - 8} className="axis-title" textAnchor="middle">
          Mean predicted probability
        </text>
        <text x={14} y={(PAD.top + H - PAD.bottom) / 2} className="axis-title" textAnchor="middle" transform={`rotate(-90 14 ${(PAD.top + H - PAD.bottom) / 2})`}>
          Observed AI frequency
        </text>
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} className="sensitivity-diagonal" />
        {bins.map((bin) => {
          const label = `bin [${bin.bin_lo.toFixed(2)}, ${bin.bin_hi.toFixed(2)}): predicted ${bin.mean_predicted.toFixed(2)}, observed ${bin.observed.toFixed(2)}, n=${bin.n}`;
          return (
            <g key={bin.bin_lo}>
              <circle cx={x(bin.mean_predicted)} cy={y(bin.observed)} r={4 + 10 * (bin.n / maxN)} className="reliability-point" />
              <circle cx={x(bin.mean_predicted)} cy={y(bin.observed)} r={14 + 10 * (bin.n / maxN)} className="hit-target">
                <title>{label}</title>
              </circle>
            </g>
          );
        })}
      </svg>
      <div className="chart-legend">
        <span><i className="legend-dot legend-teal" /> Held-out bins (size = n)</span>
        <span><i className="legend-line legend-dashed" /> Perfect calibration</span>
      </div>
      <p className="figure-caption">
        {calibration.bundle} · ECE {calibration.ece.toFixed(3)} · Brier {calibration.brier.toFixed(3)} · AUROC{' '}
        {calibration.auroc.toFixed(3)} · conforms at alpha {calibration.conformal_alpha}
      </p>
    </article>
  );
}
