import { InfoTip } from './InfoTip';
import { useArtifact } from '../hooks';

interface TemperatureCell {
  temperature: number;
  delta: number;
  greedy?: boolean;
  'detection_rate_0.05': number | null;
  mean_z: number | null;
  mean_power: number | null;
  mean_green_rate: number | null;
  n: number;
}

interface TemperatureCard {
  schema: string;
  temperatures: number[];
  deltas: number[];
  cells: TemperatureCell[];
  limitations?: string[];
}

const W = 520;
const H = 260;
const PAD = { left: 48, right: 16, top: 18, bottom: 48 };

export function WatermarkTemperaturePanel() {
  const { data } = useArtifact<TemperatureCard>('watermark-temperature');

  if (!data) {
    return (
      <article className="glass-panel figure-card">
        <h3>Watermark detection vs temperature</h3>
        <p className="figure-empty">
          Temperature card not found. Run <code>python -m bench.run_watermark_temperature</code>.
        </p>
      </article>
    );
  }

  const watermarked = data.cells.filter((c) => c.delta > 0);
  const temps = data.temperatures;
  const rates = watermarked.map((c) => c['detection_rate_0.05'] ?? 0);
  const zs = watermarked.map((c) => c.mean_z ?? 0);
  const maxZ = Math.max(1, ...zs.map(Math.abs));

  const xScale = (t: number) => {
    const lo = Math.min(...temps);
    const hi = Math.max(...temps);
    const span = hi - lo || 1;
    return PAD.left + ((t - lo) / span) * (W - PAD.left - PAD.right);
  };
  const yRate = (r: number) => H - PAD.bottom - r * (H - PAD.top - PAD.bottom);
  const yZ = (z: number) => H - PAD.bottom - ((z + maxZ) / (2 * maxZ)) * (H - PAD.top - PAD.bottom);

  const ratePath = watermarked
    .map((c, i) => `${i === 0 ? 'M' : 'L'}${xScale(c.temperature).toFixed(1)},${yRate(c['detection_rate_0.05'] ?? 0).toFixed(1)}`)
    .join(' ');
  const zPath = watermarked
    .map((c, i) => `${i === 0 ? 'M' : 'L'}${xScale(c.temperature).toFixed(1)},${yZ(c.mean_z ?? 0).toFixed(1)}`)
    .join(' ');
  const greedy = watermarked.find((c) => c.temperature <= 0 || c.greedy);

  return (
    <article className="glass-panel figure-card">
      <h3>
        Watermark detection vs temperature
        <InfoTip
          label="Temperature sweep"
          text="Detection rate and mean z for green-list watermarked text across sampling temperatures. Temperature 0 is greedy decoding — sampling randomness vanishes, so the watermark either embeds deterministically via logit bias or fails to embed. Demo key only; not a vendor private key."
        />
      </h3>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Detection rate versus temperature">
        {[0, 0.5, 1].map((tick) => (
          <text key={tick} x={PAD.left - 8} y={yRate(tick) + 4} className="axis-label" textAnchor="end">
            {tick.toFixed(1)}
          </text>
        ))}
        {temps.map((t) => (
          <text key={t} x={xScale(t)} y={H - 28} className="axis-label" textAnchor="middle">
            {t.toFixed(1)}
          </text>
        ))}
        <text x={(PAD.left + W - PAD.right) / 2} y={H - 8} className="axis-title" textAnchor="middle">
          Sampling temperature
        </text>
        <path d={ratePath} className="sensitivity-curve" />
        <path d={zPath} className="sensitivity-diagonal" />
        {watermarked.map((c) => (
          <circle
            key={`r-${c.temperature}`}
            cx={xScale(c.temperature)}
            cy={yRate(c['detection_rate_0.05'] ?? 0)}
            r={4.5}
            className="sensitivity-marker"
          >
            <title>{`T=${c.temperature}: det=${c['detection_rate_0.05']}, z=${c.mean_z}`}</title>
          </circle>
        ))}
        {greedy ? (
          <g>
            <line
              x1={xScale(greedy.temperature)}
              y1={PAD.top}
              x2={xScale(greedy.temperature)}
              y2={H - PAD.bottom}
              className="hover-guide"
            />
            <text x={xScale(greedy.temperature) + 6} y={PAD.top + 12} className="axis-label">
              greedy (T=0)
            </text>
          </g>
        ) : null}
      </svg>
      <div className="chart-legend">
        <span><i className="legend-line legend-blue" /> Detection rate @ 0.05 (delta&gt;0)</span>
        <span><i className="legend-line legend-dashed" /> Mean z (scaled)</span>
      </div>
      <p className="figure-caption">
        Rates: {rates.map((r) => r.toFixed(2)).join(', ')}. Greedy dead zone annotated at T=0.
      </p>
    </article>
  );
}
