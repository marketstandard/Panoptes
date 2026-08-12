import { FileCheck2 } from 'lucide-react';
import type { AnalysisResponse } from '../types';
import { statusTone } from '../format';

export function ProvenancePanel({ result }: { result: AnalysisResponse }) {
  const provenance = result.provenance;
  return (
    <article className="provenance-panel glass-panel">
      <div className="panel-heading">
        <div>
          <p className="panel-kicker">Cryptographic provenance</p>
          <h3>Signed file record</h3>
        </div>
        <span className={`tone-${statusTone(provenance.status)}`}>{provenance.status.replaceAll('_', ' ')}</span>
      </div>
      <p>{provenance.summary}</p>
      <dl className="definition-list">
        <div><dt>Issuer</dt><dd>{provenance.issuer ?? 'not available'}</dd></div>
        <div><dt>Timestamp</dt><dd>{provenance.timestamp ?? 'not available'}</dd></div>
        <div><dt>Actions</dt><dd>{provenance.actions.length ? provenance.actions.join(', ') : 'none recorded'}</dd></div>
      </dl>
      <p className="panel-note">
        <FileCheck2 size={14} />
        Provenance is a signed processing record. It is not authorship evidence and does not replace text-level
        statistical analysis.
      </p>
    </article>
  );
}
