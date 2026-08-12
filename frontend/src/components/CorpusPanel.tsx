import { InfoTip } from './InfoTip';
import { useArtifact } from '../hooks';

export interface CorpusCohort {
  family: string;
  kind: string;
  n: number;
  features: Record<string, { mean: number; min: number; max: number }>;
}

export interface CorpusSummary {
  n_records: number;
  n_runs: number;
  n_human: number;
  n_ai: number;
  families: string[];
  catalog_entries: number;
  cohorts: CorpusCohort[];
}

const FEATURE_LABELS: Array<[string, string]> = [
  ['connectors', 'connector rate'],
  ['unique_ratio', 'unique ratio'],
  ['long_words', 'long-word rate'],
  ['token_entropy', 'token entropy']
];

export function CorpusPanel() {
  const { data } = useArtifact<CorpusSummary>('corpus-summary');

  return (
    <article className="glass-panel figure-card corpus-panel">
      <h3>
        Verified reference corpus
        <InfoTip
          label="Corpus"
          text="Every record is SHA-256-verified against its run manifest before inclusion. Human controls plus per-model reference runs, grouped by prompt for leakage-safe evaluation. Community catalog entries contribute hashes, never raw text."
        />
      </h3>
      {!data ? (
        <p className="figure-empty">Corpus summary artifact not found. Run python research/baseline_corpus.py.</p>
      ) : (
        <>
          <div className="stat-row">
            <div className="stat">
              <span>records</span>
              <strong>{data.n_records}</strong>
            </div>
            <div className="stat">
              <span>human controls</span>
              <strong>{data.n_human}</strong>
            </div>
            <div className="stat">
              <span>AI outputs</span>
              <strong>{data.n_ai}</strong>
            </div>
            <div className="stat">
              <span>runs</span>
              <strong>{data.n_runs}</strong>
            </div>
            <div className="stat">
              <span>catalog entries</span>
              <strong>{data.catalog_entries}</strong>
            </div>
          </div>
          <table className="corpus-table">
            <thead>
              <tr>
                <th>family</th>
                <th>kind</th>
                <th>n</th>
                {FEATURE_LABELS.map(([, label]) => (
                  <th key={label}>{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.cohorts.map((cohort) => (
                <tr key={`${cohort.family}-${cohort.kind}`}>
                  <td>{cohort.family}</td>
                  <td>{cohort.kind}</td>
                  <td>{cohort.n}</td>
                  {FEATURE_LABELS.map(([key]) => (
                    <td key={key}>{cohort.features[key] ? cohort.features[key].mean.toFixed(3) : '—'}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </article>
  );
}
