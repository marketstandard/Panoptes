import { useRef, useState } from 'react';
import { InfoTip } from './InfoTip';

interface Props {
  priorOdds: number;
  likelihoodRatio: number | null;
}

const W = 520;
const H = 280;
const PAD = { left: 52, right: 16, top: 18, bottom: 52 };
const LOG_MIN = -2; // prior odds 0.01
const LOG_MAX = 2; // prior odds 100

function posteriorProbability(priorOdds: number, lr: number): number {
  const odds = priorOdds * lr;
  return odds / (1 + odds);
}

function xScale(logOdds: number): number {
  return PAD.left + ((logOdds - LOG_MIN) / (LOG_MAX - LOG_MIN)) * (W - PAD.left - PAD.right);
}

function yScale(probability: number): number {
  return H - PAD.bottom - probability * (H - PAD.top - PAD.bottom);
}

export function PosteriorSensitivity({ priorOdds, likelihoodRatio }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<{ logOdds: number; x: number } | null>(null);

  if (!likelihoodRatio || likelihoodRatio <= 0) {
    return (
      <article className="glass-panel figure-card">
        <h3>
          Posterior sensitivity to the prior
          <InfoTip
            label="Posterior sensitivity"
            text="How the posterior probability of AI participation would move if your prior odds changed, holding the observed likelihood ratio fixed."
          />
        </h3>
        <p className="figure-empty">No likelihood ratio is available for this input.</p>
      </article>
    );
  }

  const steps = 121;
  const curve: string[] = [];
  const diagonal: string[] = [];
  for (let i = 0; i < steps; i += 1) {
    const logOdds = LOG_MIN + (i / (steps - 1)) * (LOG_MAX - LOG_MIN);
    const odds = 10 ** logOdds;
    const x = xScale(logOdds);
    curve.push(`${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${yScale(posteriorProbability(odds, likelihoodRatio)).toFixed(1)}`);
    diagonal.push(`${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${yScale(posteriorProbability(odds, 1)).toFixed(1)}`);
  }
  const markerX = xScale(Math.log10(Math.max(priorOdds, 0.01)));
  const markerY = yScale(posteriorProbability(priorOdds, likelihoodRatio));
  const currentPosterior = posteriorProbability(priorOdds, likelihoodRatio);

  const onMove = (event: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const svgX = ((event.clientX - rect.left) / rect.width) * W;
    const clamped = Math.min(Math.max(svgX, PAD.left), W - PAD.right);
    const logOdds = LOG_MIN + ((clamped - PAD.left) / (W - PAD.left - PAD.right)) * (LOG_MAX - LOG_MIN);
    setHover({ logOdds, x: clamped });
  };

  const hoverOdds = hover ? 10 ** hover.logOdds : null;
  const hoverPosterior = hoverOdds !== null ? posteriorProbability(hoverOdds, likelihoodRatio) : null;

  return (
    <article className="glass-panel figure-card">
      <h3>
        Posterior sensitivity to the prior
        <InfoTip
          label="Posterior sensitivity"
          text="Posterior probability of AI participation as a function of your declared prior odds, holding the observed likelihood ratio fixed. The dashed diagonal is LR = 1 (no evidence). The marker is your current prior. Hover the chart to interrogate any prior."
        />
      </h3>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Posterior probability versus prior odds curve"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        className="chart-interactive"
      >
        {[0.01, 0.1, 1, 10, 100].map((tick) => (
          <g key={tick}>
            <line x1={xScale(Math.log10(tick))} y1={yScale(0)} x2={xScale(Math.log10(tick))} y2={yScale(1)} className="grid-line" />
            <text x={xScale(Math.log10(tick))} y={H - 30} className="axis-label" textAnchor="middle">
              {tick}
            </text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <text key={tick} x={PAD.left - 8} y={yScale(tick) + 4} className="axis-label" textAnchor="end">
            {tick.toFixed(2)}
          </text>
        ))}
        <text x={(PAD.left + W - PAD.right) / 2} y={H - 8} className="axis-title" textAnchor="middle">
          Prior odds (log scale)
        </text>
        <text x={14} y={(PAD.top + H - PAD.bottom) / 2} className="axis-title" textAnchor="middle" transform={`rotate(-90 14 ${(PAD.top + H - PAD.bottom) / 2})`}>
          P(AI-generated)
        </text>
        <path d={diagonal.join(' ')} className="sensitivity-diagonal" />
        <path d={curve.join(' ')} className="sensitivity-curve" />
        {hover && hoverPosterior !== null ? (
          <g className="hover-layer">
            <line x1={hover.x} y1={yScale(0)} x2={hover.x} y2={yScale(1)} className="hover-guide" />
            <circle cx={hover.x} cy={yScale(hoverPosterior)} r={4.5} className="hover-dot" />
          </g>
        ) : null}
        <circle cx={markerX} cy={markerY} r={5} className="sensitivity-marker">
          <title>{`Your prior: odds ${priorOdds}, posterior ${(currentPosterior * 100).toFixed(1)}%`}</title>
        </circle>
      </svg>
      <div className="chart-legend">
        <span><i className="legend-line legend-blue" /> Posterior at observed LR</span>
        <span><i className="legend-line legend-dashed" /> No evidence (LR = 1)</span>
        <span><i className="legend-dot legend-amber" /> Your prior</span>
      </div>
      <p className="figure-caption hover-readout">
        {hover && hoverOdds !== null && hoverPosterior !== null
          ? `prior odds ${hoverOdds.toPrecision(3)} → posterior ${(hoverPosterior * 100).toFixed(1)}%`
          : `LR ${likelihoodRatio.toFixed(2)} · prior odds ${priorOdds} · posterior ${(currentPosterior * 100).toFixed(1)}%`}
      </p>
    </article>
  );
}
