import { InfoTip } from './InfoTip';
import type { AnalysisResponse } from '../types';

interface Props {
  calibration: AnalysisResponse['calibration'];
}

const W = 520;
const H = 260;
const PAD = { left: 44, right: 16, top: 18, bottom: 34 };

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
          text={`Held-out calibration of the shipped heuristic on the verified corpus (n=${calibration.n_records}, GroupKFold by prompt). Point size is bin count; the diagonal is perfect calibration. ECE ${calibration.ece.toFixed(3)}.`}
        />
      </h3>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Reliability diagram">
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={x(tick)} y1={y(0)} x2={x(tick)} y2={y(1)} className="grid-line" />
            <text x={x(tick)} y={H - 14} className="axis-label" textAnchor="middle">
              {tick.toFixed(2)}
            </text>
            <text x={PAD.left - 8} y={y(tick) + 4} className="axis-label" textAnchor="end">
              {tick.toFixed(2)}
            </text>
          </g>
        ))}
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} className="sensitivity-diagonal" />
        {bins.map((bin) => (
          <circle
            key={bin.bin_lo}
            cx={x(bin.mean_predicted)}
            cy={y(bin.observed)}
            r={4 + 10 * (bin.n / maxN)}
            className="reliability-point"
          >
            <title>{`predicted ${bin.mean_predicted.toFixed(2)}, observed ${bin.observed.toFixed(2)}, n=${bin.n}`}</title>
          </circle>
        ))}
      </svg>
      <p className="figure-caption">
        {calibration.bundle} · ECE {calibration.ece.toFixed(3)} · Brier {calibration.brier.toFixed(3)} · AUROC{' '}
        {calibration.auroc.toFixed(3)} · conforms at alpha {calibration.conformal_alpha}
      </p>
    </article>
  );
}
