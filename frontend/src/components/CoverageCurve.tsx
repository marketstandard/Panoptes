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
const H = 260;
const PAD = { left: 44, right: 16, top: 18, bottom: 34 };

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
                <text
                  x={PAD.left + tick * (W - PAD.left - PAD.right)}
                  y={H - 14}
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
            {curve.map((row) => (
              <circle
                key={row.min_confidence}
                cx={PAD.left + row.coverage * (W - PAD.left - PAD.right)}
                cy={H - PAD.bottom - row.accuracy * (H - PAD.top - PAD.bottom)}
                r={4}
                className="reliability-point"
              >
                <title>{`confidence >= ${row.min_confidence.toFixed(2)}: coverage ${(row.coverage * 100).toFixed(0)}%, accuracy ${(row.accuracy * 100).toFixed(1)}%, n=${row.n_kept}`}</title>
              </circle>
            ))}
          </svg>
          <p className="figure-caption">{data?.model.name} · out-of-fold, grouped by prompt</p>
        </>
      )}
    </article>
  );
}
