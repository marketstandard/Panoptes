import { Layers, Scale, ScrollText, Signature } from 'lucide-react';
import type { AnalysisResponse, EvidenceEntry, EvidenceValidity } from '../types';
import { formatPercent } from '../format';

const VALIDITY_TONE: Record<EvidenceValidity, 'good' | 'warn' | 'bad' | 'muted'> = {
  valid: 'good',
  weakened: 'warn',
  invalid: 'bad',
  not_applicable: 'muted',
  unknown: 'muted'
};

const CHANNEL_META = {
  statistical: { label: 'Statistical', icon: Scale },
  watermark: { label: 'Watermark', icon: ScrollText },
  provenance: { label: 'Provenance', icon: Signature }
} as const;

function EntryCard({ entry }: { entry: EvidenceEntry }) {
  return (
    <div className="ledger-entry">
      <div className="ledger-entry-head">
        <strong>{entry.source_identity}</strong>
        <span className={`tone-${VALIDITY_TONE[entry.validity]}`}>{entry.validity.replaceAll('_', ' ')}</span>
      </div>
      <p className="ledger-claim">
        Target claim: <em>{entry.target_claim.replaceAll('_', ' ')}</em>
      </p>
      <dl className="definition-list">
        <div>
          <dt>Applicability</dt>
          <dd>{entry.applicability_scope}</dd>
        </div>
        {entry.strength !== null ? (
          <div>
            <dt>Strength</dt>
            <dd>{formatPercent(entry.strength)}</dd>
          </div>
        ) : null}
        {entry.uncertainty ? (
          <div>
            <dt>Uncertainty</dt>
            <dd>{entry.uncertainty}</dd>
          </div>
        ) : null}
      </dl>
      {entry.limitations.length ? (
        <ul className="ledger-limitations">
          {entry.limitations.map((limitation) => (
            <li key={limitation}>{limitation}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

export function EvidenceLedgerPanel({ result }: { result: AnalysisResponse }) {
  const ledger = result.evidence_ledger;
  if (!ledger) return null;
  const channels = [
    { key: 'statistical' as const, entries: ledger.statistical },
    { key: 'watermark' as const, entries: ledger.watermark },
    { key: 'provenance' as const, entries: ledger.provenance }
  ];
  return (
    <section className="evidence-ledger glass-panel" aria-label="Evidence ledger">
      <div className="panel-heading">
        <div>
          <p className="panel-kicker">Evidence ledger</p>
          <h3>Three channels, kept separate</h3>
        </div>
        <Layers size={18} />
      </div>
      <div className="ledger-channels">
        {channels.map(({ key, entries }) => {
          const Meta = CHANNEL_META[key];
          const Icon = Meta.icon;
          return (
            <article key={key} className="ledger-channel" data-channel={key}>
              <div className="ledger-channel-head">
                <Icon size={15} />
                <h4>{Meta.label}</h4>
              </div>
              {ledger.channel_summaries[key] ? (
                <p className="ledger-summary">{ledger.channel_summaries[key]}</p>
              ) : null}
              {entries.map((entry, index) => (
                <EntryCard key={`${entry.source_identity}-${index}`} entry={entry} />
              ))}
            </article>
          );
        })}
      </div>
      <p className="panel-note ledger-fusion-note">{ledger.fusion_note}</p>
    </section>
  );
}
