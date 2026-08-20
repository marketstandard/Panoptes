import { InfoTip } from './InfoTip';
import { useArtifact } from '../hooks';
import { formatNumber } from '../format';

interface ScoreBlock {
  n?: number;
  n_tested?: number;
  'detection_rate_0.05'?: number | null;
  mean_z?: number | null;
  mean_green_rate?: number | null;
}

interface RadioactivityCard {
  schema: string;
  mode: string;
  tier: string;
  teacher?: {
    watermarked?: ScoreBlock;
    control?: ScoreBlock;
    delta?: number;
  };
  inheritance?: {
    student_on_watermarked?: ScoreBlock;
    student_on_control?: ScoreBlock;
    attenuation?: { z_ratio?: number | null; detection_ratio?: number | null };
  };
  removal?: {
    paraphrase_pre?: ScoreBlock | null;
    neutralize_post?: ScoreBlock | null;
  };
  knowledge_preservation?: Record<string, unknown>;
  limitations?: string[];
}

function fmt(score: ScoreBlock | null | undefined): string {
  if (!score || score.mean_z == null) return '—';
  const det = score['detection_rate_0.05'];
  return `z=${formatNumber(score.mean_z)} · det=${det == null ? '—' : formatNumber(det)}`;
}

export function RadioactivityPanel() {
  const { data } = useArtifact<RadioactivityCard>('radioactivity');

  if (!data) {
    return (
      <article className="glass-panel figure-card">
        <h3>Watermark radioactivity</h3>
        <p className="figure-empty">
          Radioactivity card not found. Run <code>python -m bench.run_radioactivity</code>.
        </p>
      </article>
    );
  }

  const inh = data.inheritance;
  const rem = data.removal;

  return (
    <article className="glass-panel figure-card">
      <h3>
        Watermark radioactivity (distillation inheritance)
        <InfoTip
          label="Radioactivity"
          text="Student models trained on watermarked teacher outputs can inherit a detectable green-list bias. This card reports teacher vs student detection, attenuation, and removal arms (pre-distillation paraphrase and post-distillation neutralization). Lineage evidence — not proof of unauthorized distillation. Demo key only."
        />
      </h3>
      <div className="stat-row">
        <div className="stat">
          <span>mode / tier</span>
          <strong>{data.mode} / {data.tier}</strong>
        </div>
        <div className="stat">
          <span>teacher watermarked</span>
          <strong>{fmt(data.teacher?.watermarked)}</strong>
        </div>
        <div className="stat">
          <span>teacher control</span>
          <strong>{fmt(data.teacher?.control)}</strong>
        </div>
      </div>
      <table className="corpus-table">
        <thead>
          <tr>
            <th>condition</th>
            <th>mean z</th>
            <th>detection @ 0.05</th>
            <th>green rate</th>
          </tr>
        </thead>
        <tbody>
          {(
            [
              ['student on watermarked', inh?.student_on_watermarked],
              ['student on control', inh?.student_on_control],
              ['removal: paraphrase-pre', rem?.paraphrase_pre ?? undefined],
              ['removal: neutralize-post', rem?.neutralize_post ?? undefined],
            ] as Array<[string, ScoreBlock | undefined]>
          ).map(([label, score]) => {
            return (
              <tr key={label}>
                <td>{label}</td>
                <td>{score?.mean_z == null ? '—' : formatNumber(score.mean_z)}</td>
                <td>
                  {score?.['detection_rate_0.05'] == null
                    ? '—'
                    : formatNumber(score['detection_rate_0.05'])}
                </td>
                <td>{score?.mean_green_rate == null ? '—' : formatNumber(score.mean_green_rate)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {inh?.attenuation ? (
        <p className="figure-caption">
          Attenuation z-ratio={formatNumber(inh.attenuation.z_ratio ?? null)} · detection-ratio=
          {formatNumber(inh.attenuation.detection_ratio ?? null)}
        </p>
      ) : null}
    </article>
  );
}
