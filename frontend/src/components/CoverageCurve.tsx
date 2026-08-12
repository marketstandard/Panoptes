import { InfoTip } from './InfoTip';
import { useArtifact } from '../hooks';

interface CoverageRow {
  min_confidence: number;
  coverage: number;
  accuracy: number;
  n_kept: number;
}

interface CardShape {
  model: { name: string };
  evaluation: { coverage_curve: CoverageRow[] };
}

const W = 520;
const H = 280;
const PAD = { left: 52, right: 16, top: 18, bottom: 52 };

export function CoverageCurve() {
  const { data } = useArtifact<CardShape>('logistic-tier0-card');
  const curve = data?.evaluation?.coverage_curve ?? [];

  return (
    <article className="glass-panel figure-card">
      <h3>
        Coverage vs accuracy (abstention)
        <InfoTip
          label="Coverage curve"
          text="Out-of-fold accuracy on the cases the model keeps, as the confidence threshold rises and coverage falls. The value of an abstention-native system: you can trade coverage for accuracy explicitly."
        />
      </h3>
      {curve.length === 0 ? (
        <p className="figure-empty">Train the reference model (python -m bench train --model logistic) to populate this figure.</p>
      ) : (
        <>
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Coverage versus accuracy curve">
            {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
              <g key={tick}>
                <line
                  x1={PAD.left + tick * (W - PAD.left - PAD.right)}
                  y1={H - PAD.bottom}
                  x2={PAD.left + tick * (W - PAD.left - PAD.right)}
                  y2={PAD.top}
                  className="grid-line"
                />
                <text
                  x={PAD.left + tick * (W - PAD.left - PAD.right)}
                  y={H - 30}
                  className="axis-label"
                  textAnchor="middle"
                >
                  {tick.toFixed(2)}
                </text>
                <text
                  x={PAD.left - 8}
                  y={H - PAD.bottom - tick * (H - PAD.top - PAD.bottom) + 4}
                  className="axis-label"
                  textAnchor="end"
                >
                  {tick.toFixed(2)}
                </text>
              </g>
            ))}
            <text x={(PAD.left + W - PAD.right) / 2} y={H - 8} className="axis-title" textAnchor="middle">
              Coverage (fraction of cases kept)
            </text>
            <text x={14} y={(PAD.top + H - PAD.bottom) / 2} className="axis-title" textAnchor="middle" transform={`rotate(-90 14 ${(PAD.top + H - PAD.bottom) / 2})`}>
              Accuracy on kept cases
            </text>
            <path
              d={curve
                .map((row, index) => {
                  const x = PAD.left + row.coverage * (W - PAD.left - PAD.right);
                  const y = H - PAD.bottom - row.accuracy * (H - PAD.top - PAD.bottom);
                  return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
                })
                .join(' ')}
              className="sensitivity-curve"
            />
            {curve.map((row) => {
              const cx = PAD.left + row.coverage * (W - PAD.left - PAD.right);
              const cy = H - PAD.bottom - row.accuracy * (H - PAD.top - PAD.bottom);
              return (
                <g key={row.min_confidence}>
                  <circle cx={cx} cy={cy} r={4} className="reliability-point" />
                  <circle cx={cx} cy={cy} r={13} className="hit-target">
                    <title>{`confidence >= ${row.min_confidence.toFixed(2)}: coverage ${(row.coverage * 100).toFixed(0)}%, accuracy ${(row.accuracy * 100).toFixed(1)}%, n=${row.n_kept}`}</title>
                  </circle>
                </g>
              );
            })}
          </svg>
          <div className="chart-legend">
            <span><i className="legend-line legend-blue" /> Out-of-fold accuracy</span>
            <span><i className="legend-dot legend-teal" /> Confidence thresholds (hover)</span>
          </div>
          <p className="figure-caption">{data?.model.name} · out-of-fold, grouped by prompt</p>
        </>
      )}
    </article>
  );
}
