import { FileCheck2 } from 'lucide-react';
import type { AnalysisResponse } from '../types';
import { statusTone } from '../format';

const LEVEL_MEANING: Record<'P0' | 'P1' | 'P2' | 'P3' | 'P4', string> = {
  P0: 'no provenance',
  P1: 'self-declared identity',
  P2: 'authenticated provider metadata',
  P3: 'cryptographically signed generation receipt',
  P4: 'independently verifiable execution'
};

export function ProvenancePanel({ result }: { result: AnalysisResponse }) {
  const provenance = result.provenance;
  const level = provenance.level ?? 'P0';
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
        <div>
          <dt>Level</dt>
          <dd>
            {level} — {LEVEL_MEANING[level]}
          </dd>
        </div>
        <div><dt>Issuer</dt><dd>{provenance.issuer ?? 'not available'}</dd></div>
        <div><dt>Timestamp</dt><dd>{provenance.timestamp ?? 'not available'}</dd></div>
        <div><dt>Actions</dt><dd>{provenance.actions.length ? provenance.actions.join(', ') : 'none recorded'}</dd></div>
      </dl>
      <p className="panel-note">
        <FileCheck2 size={14} />
        Provenance is a signed processing record. It is not authorship evidence and does not replace text-level
        statistical analysis. Self-declared model identity is P1, not independently verified provenance.
      </p>
    </article>
  );
}
