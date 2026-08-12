import { HelpCircle } from 'lucide-react';
import type { AnalysisResponse } from '../types';
import { formatPercent } from '../format';

export function SourceFamilyPanel({ result }: { result: AnalysisResponse }) {
  const rows = result.source_families.conditional_on_ai;
  const unknown = result.source_families.unknown_score;
  return (
    <article className="family-panel glass-panel">
      <div className="panel-heading">
        <div>
          <p className="panel-kicker">Open-set attribution</p>
          <h3>Source-family similarity</h3>
        </div>
        <span>unknown {formatPercent(unknown)}</span>
      </div>
      <div className="unknown-meter" aria-label={`Unknown score ${formatPercent(unknown)}`}>
        <div style={{ width: `${unknown * 100}%` }} />
      </div>
      <div className="family-list">
        {rows.map((row) => (
          <div className="family-row" key={row.family}>
            <span>{row.family}</span>
            <div className="family-track">
              <div style={{ width: `${row.probability * 100}%` }} />
            </div>
            <strong>{formatPercent(row.probability)}</strong>
          </div>
        ))}
      </div>
      <p className="panel-note">
        <HelpCircle size={14} />
        Conditional similarity among calibrated candidates. A high unknown score means Panoptes should not
        force a model-family label.
      </p>
    </article>
  );
}
