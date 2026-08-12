import { useRef, useState } from 'react';
import { InfoTip } from './InfoTip';

interface Props {
  currentN: number;
  defactifyN?: number;
}

const W = 520;
const H = 280;
const PAD = { left: 52, right: 16, top: 18, bottom: 52 };
const N_MIN = 20;
const N_MAX = 75000;
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

function xScale(n: number): number {
  const lo = Math.log10(N_MIN);
  const hi = Math.log10(N_MAX);
  return PAD.left + ((Math.log10(Math.max(n, N_MIN)) - lo) / (hi - lo)) * (W - PAD.left - PAD.right);
}

function xInvert(x: number): number {
  const lo = Math.log10(N_MIN);
  const hi = Math.log10(N_MAX);
  return 10 ** (lo + ((x - PAD.left) / (W - PAD.left - PAD.right)) * (hi - lo));
}

function yScale(power: number): number {
  return H - PAD.bottom - power * (H - PAD.top - PAD.bottom);
}

export function PowerCurve({ currentN, defactifyN }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<{ n: number; x: number } | null>(null);

  const steps = 160;
  const path: string[] = [];
  for (let i = 0; i < steps; i += 1) {
    const n = 10 ** (Math.log10(N_MIN) + (i / (steps - 1)) * (Math.log10(N_MAX) - Math.log10(N_MIN)));
    path.push(`${i === 0 ? 'M' : 'L'}${xScale(n).toFixed(1)},${yScale(powerAt(n)).toFixed(1)}`);
  }
  const markerX = xScale(Math.min(Math.max(currentN, N_MIN), N_MAX));
  const markerY = yScale(powerAt(currentN));
  const defactifyX = defactifyN ? xScale(Math.min(Math.max(defactifyN, N_MIN), N_MAX)) : null;
  const defactifyY = defactifyN ? yScale(powerAt(defactifyN)) : null;
  const y80 = yScale(0.8);

  const onMove = (event: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const svgX = ((event.clientX - rect.left) / rect.width) * W;
    const clamped = Math.min(Math.max(svgX, PAD.left), W - PAD.right);
    setHover({ n: xInvert(clamped), x: clamped });
  };

  return (
    <article className="glass-panel figure-card">
      <h3>
        Statistical power vs corpus size
        <InfoTip
          label="Power curve"
          text="Power to detect a 5-point accuracy difference between two models at alpha 0.05, worst-case variance. The bench admits neural models into the comparison zoo only at 80% power (n about 3,140). The marker is today's verified corpus. Hover the chart to interrogate any corpus size."
        />
      </h3>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Power versus corpus size"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        className="chart-interactive"
      >
        {[100, 1000, 3140, 10000, 75000].map((tick) => (
          <g key={tick}>
            <line x1={xScale(tick)} y1={yScale(0)} x2={xScale(tick)} y2={yScale(1)} className="grid-line" />
            <text x={xScale(tick)} y={H - 30} className="axis-label" textAnchor="middle">
              {tick.toLocaleString()}
            </text>
          </g>
        ))}
        {[0, 0.5, 0.8, 1].map((tick) => (
          <text key={tick} x={PAD.left - 8} y={yScale(tick) + 4} className="axis-label" textAnchor="end">
            {tick.toFixed(1)}
          </text>
        ))}
        <text x={(PAD.left + W - PAD.right) / 2} y={H - 8} className="axis-title" textAnchor="middle">
          Verified corpus records (n)
        </text>
        <text x={14} y={(PAD.top + H - PAD.bottom) / 2} className="axis-title" textAnchor="middle" transform={`rotate(-90 14 ${(PAD.top + H - PAD.bottom) / 2})`}>
          Power
        </text>
        <line x1={PAD.left} y1={y80} x2={W - PAD.right} y2={y80} className="sensitivity-diagonal" />
        <text x={W - PAD.right - 4} y={y80 - 5} className="axis-label" textAnchor="end">
          80% target
        </text>
        <path d={path.join(' ')} className="sensitivity-curve" />
        {hover ? (
          <g className="hover-layer">
            <line x1={hover.x} y1={yScale(0)} x2={hover.x} y2={yScale(1)} className="hover-guide" />
            <circle cx={hover.x} cy={yScale(powerAt(hover.n))} r={4.5} className="hover-dot" />
          </g>
        ) : null}
        <circle cx={markerX} cy={markerY} r={5} className="sensitivity-marker">
          <title>{`This corpus: n=${currentN}, power ${(powerAt(currentN) * 100).toFixed(1)}%`}</title>
        </circle>
        <text x={markerX} y={markerY - 10} className="axis-label marker-label" textAnchor="middle">
          this corpus
        </text>
        {defactifyX !== null && defactifyY !== null && defactifyN ? (
          <g>
            <circle cx={defactifyX} cy={defactifyY} r={5} className="sensitivity-marker defactify-marker">
              <title>{`Defactify bench: n=${defactifyN.toLocaleString()}, power ${(powerAt(defactifyN) * 100).toFixed(1)}% — the neural gate passes`}</title>
            </circle>
            <text x={defactifyX} y={defactifyY - 10} className="axis-label marker-label" textAnchor="end">
              Defactify bench
            </text>
          </g>
        ) : null}
      </svg>
      <div className="chart-legend">
        <span><i className="legend-line legend-blue" /> Power to detect a 5-pt lift (α = 0.05)</span>
        <span><i className="legend-line legend-dashed" /> 80% target</span>
        <span><i className="legend-dot legend-amber" /> This corpus</span>
        {defactifyN ? <span><i className="legend-dot legend-green" /> Defactify bench</span> : null}
      </div>
      <p className="figure-caption hover-readout">
        {hover
          ? `n = ${Math.round(hover.n).toLocaleString()} → power ${(powerAt(hover.n) * 100).toFixed(1)}%`
          : `n = ${currentN} · power ${(powerAt(currentN) * 100).toFixed(1)}% · 80% power at n = 3,140${defactifyN ? ` · Defactify n = ${defactifyN.toLocaleString()}` : ''}`}
      </p>
    </article>
  );
}
