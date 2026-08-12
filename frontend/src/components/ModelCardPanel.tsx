import { InfoTip } from './InfoTip';
import { useArtifact } from '../hooks';

interface CurveRow {
  epoch: number;
  train_loss: number;
  val_ece: number;
}

interface Comparison {
  test: 'mcnemar' | 'delong';
  pair: string;
  p_value: number;
  q_value: number;
  significant_at_0_05: boolean;
  b?: number;
  c?: number;
  auc_a?: number;
  auc_b?: number;
}

interface V0Card {
  model: { name: string; tier: number };
  device: string;
  architecture: {
    feature_branch: string;
    evidence_head: string;
    sequence_branch?: string;
    loss?: string;
    n_features: number;
  };
  evaluation: {
    metrics: { auroc: number; brier: number; ece: number; accuracy: number };
    auroc_ci95: [number, number];
    conformal: { alpha: number; empirical_coverage: number; mean_set_size: number; abstention_rate: number };
  };
  comparison_battery: { comparisons: Comparison[]; gbm_note: string };
  power_gate: { passes: boolean; rationale: string };
  training_curve_seed13: CurveRow[];
  weights: { local: string; release: string };
}

const W = 520;
const H = 200;
const PAD = { left: 44, right: 44, top: 14, bottom: 30 };

function ArchitectureSvg() {
  const layers = ['features (17)', 'Linear 64 · GELU', 'LayerNorm', 'Linear 64 · GELU', 'evidence head', 'Dirichlet α'];
  return (
    <svg viewBox="0 0 520 74" role="img" aria-label="Panoptes-v0 architecture" className="arch-svg">
      {layers.map((label, index) => {
        const x = 8 + index * 86;
        return (
          <g key={label}>
            <rect x={x} y={18} width={76} height={34} rx={6} className={index >= 4 ? 'arch-node head' : 'arch-node'} />
            <text x={x + 38} y={39} textAnchor="middle" className="arch-label">
              {label}
            </text>
            {index < layers.length - 1 ? <line x1={x + 76} y1={35} x2={x + 86} y2={35} className="arch-edge" /> : null}
          </g>
        );
      })}
      <text x={478} y={66} textAnchor="middle" className="arch-label">
        p · vacuity · dissonance
      </text>
    </svg>
  );
}

export function ModelCardPanel() {
  const { data } = useArtifact<V0Card>('panoptes-v0-card');

  return (
    <article className="glass-panel figure-card corpus-panel">
      <h3>
        Panoptes-v0 — evidential detector
        <InfoTip
          label="Panoptes-v0"
          text="A custom evidential neural architecture (Sensoy et al. 2018): the network outputs Dirichlet evidence, so uncertainty (vacuity, dissonance) is native rather than bolted on. Trained on the verified corpus with grouped cross-validation; weights are local, open release on Hugging Face coming soon."
        />
      </h3>
      {!data ? (
        <p className="figure-empty">No Panoptes-v0 card found. Run python -m bench train --model panoptes-v0.</p>
      ) : (
        <>
          <ArchitectureSvg />
          <div className="stat-row">
            <div className="stat">
              <span>AUROC</span>
              <strong>
                {data.evaluation.metrics.auroc.toFixed(3)} [{data.evaluation.auroc_ci95[0].toFixed(3)},{' '}
                {data.evaluation.auroc_ci95[1].toFixed(3)}]
              </strong>
            </div>
            <div className="stat">
              <span>accuracy</span>
              <strong>{data.evaluation.metrics.accuracy.toFixed(3)}</strong>
            </div>
            <div className="stat">
              <span>ECE</span>
              <strong>{data.evaluation.metrics.ece.toFixed(3)}</strong>
            </div>
            <div className="stat">
              <span>conformal coverage</span>
              <strong>{data.evaluation.conformal.empirical_coverage.toFixed(3)}</strong>
            </div>
            <div className="stat">
              <span>device</span>
              <strong>{data.device}</strong>
            </div>
          </div>
          {data.training_curve_seed13.length > 0 ? (
            <>
              <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Training curve" className="training-svg">
                {(() => {
                  const curve = data.training_curve_seed13;
                  const maxLoss = Math.max(...curve.map((row) => row.train_loss), 1e-6);
                  const maxEce = Math.max(...curve.map((row) => row.val_ece), 1e-6);
                  const x = (epoch: number) => PAD.left + (epoch / curve.length) * (W - PAD.left - PAD.right);
                  const yLoss = (loss: number) => H - PAD.bottom - (loss / maxLoss) * (H - PAD.top - PAD.bottom);
                  const yEce = (ece: number) => H - PAD.bottom - (ece / maxEce) * (H - PAD.top - PAD.bottom);
                  const sampled = curve.filter((_, i) => i % Math.max(1, Math.floor(curve.length / 12)) === 0);
                  return (
                    <>
                      <text x={PAD.left - 8} y={yLoss(maxLoss) + 10} className="axis-label" textAnchor="end">
                        {maxLoss.toFixed(2)}
                      </text>
                      <text x={PAD.left - 8} y={H - PAD.bottom + 4} className="axis-label" textAnchor="end">
                        0
                      </text>
                      <text x={W - PAD.right + 8} y={yEce(maxEce) + 10} className="axis-label">
                        {maxEce.toFixed(2)}
                      </text>
                      <text x={W - PAD.right + 8} y={H - PAD.bottom + 4} className="axis-label">
                        0
                      </text>
                      <path
                        d={curve.map((row, i) => `${i === 0 ? 'M' : 'L'}${x(row.epoch).toFixed(1)},${yLoss(row.train_loss).toFixed(1)}`).join(' ')}
                        className="sensitivity-curve"
                      />
                      <path
                        d={curve.map((row, i) => `${i === 0 ? 'M' : 'L'}${x(row.epoch).toFixed(1)},${yEce(row.val_ece).toFixed(1)}`).join(' ')}
                        className="training-ece"
                      />
                      {sampled.map((row) => (
                        <circle key={row.epoch} cx={x(row.epoch)} cy={yLoss(row.train_loss)} r={8} className="hit-target">
                          <title>{`epoch ${row.epoch}: train loss ${row.train_loss.toFixed(4)}, val ECE ${row.val_ece.toFixed(4)}`}</title>
                        </circle>
                      ))}
                      <text x={PAD.left} y={H - 8} className="axis-label">
                        epoch 1
                      </text>
                      <text x={W - PAD.right} y={H - 8} className="axis-label" textAnchor="end">
                        epoch {curve.length}
                      </text>
                      <text x={(PAD.left + W - PAD.right) / 2} y={H - 8} className="axis-title" textAnchor="middle">
                        training epoch (seed 13)
                      </text>
                    </>
                  );
                })()}
              </svg>
              <div className="chart-legend">
                <span><i className="legend-line legend-blue" /> Train loss (left axis)</span>
                <span><i className="legend-line legend-teal" /> Validation ECE (right axis)</span>
              </div>
            </>
          ) : null}
          <table className="corpus-table">
            <thead>
              <tr>
                <th>comparison</th>
                <th>test</th>
                <th>detail</th>
                <th>q-value</th>
                <th>sig @0.05</th>
              </tr>
            </thead>
            <tbody>
              {data.comparison_battery.comparisons.map((row, index) => (
                <tr key={`${row.pair}-${row.test}-${index}`}>
                  <td>{row.pair}</td>
                  <td>{row.test}</td>
                  <td>
                    {row.test === 'mcnemar'
                      ? `b=${row.b} c=${row.c}`
                      : `ΔAUROC ${((row.auc_a ?? 0) - (row.auc_b ?? 0)).toFixed(3)}`}
                  </td>
                  <td>{row.q_value.toFixed(3)}</td>
                  <td>{row.significant_at_0_05 ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="figure-caption">{data.comparison_battery.gbm_note}</p>
          <p className="figure-caption">
            Gate: {data.power_gate.passes ? 'passed' : 'not passed'} — {data.power_gate.rationale} · weights:{' '}
            {data.weights.local} · {data.weights.release}
          </p>
        </>
      )}
    </article>
  );
}
