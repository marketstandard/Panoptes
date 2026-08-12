import { InfoTip } from './InfoTip';

interface Props {
  currentN: number;
}

const W = 520;
const H = 260;
const PAD = { left: 44, right: 16, top: 18, bottom: 34 };
const N_MAX = 5000;
const MDE = 0.05;
const Z_ALPHA = 1.959964;

function normalCdf(z: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const pdf = Math.exp(-(z * z) / 2) / Math.sqrt(2 * Math.PI);
  const cdf = 1 - pdf * poly;
  return z >= 0 ? cdf : 1 - cdf;
}

export function powerAt(n: number): number {
  return normalCdf(MDE * Math.sqrt(n) - Z_ALPHA);
}

export function PowerCurve({ currentN }: Props) {
  const steps = 120;
  const path: string[] = [];
  for (let i = 0; i < steps; i += 1) {
    const n = 20 + (i / (steps - 1)) * (N_MAX - 20);
    const x = PAD.left + ((n - 20) / (N_MAX - 20)) * (W - PAD.left - PAD.right);
    const y = H - PAD.bottom - powerAt(n) * (H - PAD.top - PAD.bottom);
    path.push(`${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`);
  }
  const markerX = PAD.left + ((Math.min(currentN, N_MAX) - 20) / (N_MAX - 20)) * (W - PAD.left - PAD.right);
  const markerY = H - PAD.bottom - powerAt(currentN) * (H - PAD.top - PAD.bottom);
  const y80 = H - PAD.bottom - 0.8 * (H - PAD.top - PAD.bottom);

  return (
    <article className="glass-panel figure-card">
      <h3>
        Statistical power vs corpus size
        <InfoTip
          label="Power curve"
          text="Power to detect a 5-point accuracy difference between two models at alpha 0.05, worst-case variance. The bench admits neural models into the comparison zoo only at 80% power (n about 3,140). The marker is today's verified corpus."
        />
      </h3>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Power versus corpus size">
        {[500, 1000, 2000, 3140, 5000].map((tick) => (
          <text
            key={tick}
            x={PAD.left + ((tick - 20) / (N_MAX - 20)) * (W - PAD.left - PAD.right)}
            y={H - 14}
            className="axis-label"
            textAnchor="middle"
          >
            {tick}
          </text>
        ))}
        {[0, 0.5, 0.8, 1].map((tick) => (
          <text
            key={tick}
            x={PAD.left - 8}
            y={H - PAD.bottom - tick * (H - PAD.top - PAD.bottom) + 4}
            className="axis-label"
            textAnchor="end"
          >
            {tick.toFixed(1)}
          </text>
        ))}
        <line x1={PAD.left} y1={y80} x2={W - PAD.right} y2={y80} className="sensitivity-diagonal" />
        <path d={path.join(' ')} className="sensitivity-curve" />
        <circle cx={markerX} cy={markerY} r={5} className="sensitivity-marker" />
      </svg>
      <p className="figure-caption">
        n = {currentN} · power {(powerAt(currentN) * 100).toFixed(1)}% · 80% power at n = 3,140
      </p>
    </article>
  );
}
