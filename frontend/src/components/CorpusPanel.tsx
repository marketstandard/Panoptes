import { InfoTip } from './InfoTip';
import { useArtifact } from '../hooks';

export interface CorpusCohort {
  family: string;
  kind: string;
  n: number;
  watermark_status?: string;
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
  contaminated_cohorts?: Array<{
    family: string;
    kind: string;
    watermark_status: string;
    notes?: string;
  }>;
}

export interface DefactifySummary {
  n_records: number;
  n_human: number;
  n_ai: number;
  families: Record<string, number>;
  hygiene: {
    rows_raw: number;
    dropped_error_artifacts: number;
    dropped_exact_duplicates: number;
    dropped_under_50_tokens: number;
  };
  group_reconstruction: { n_groups: number; threshold: number };
  leakage_audit: { official_split_story_leakage_rate: number };
}

const FEATURE_LABELS: Array<[string, string]> = [
  ['connectors', 'connector rate'],
  ['unique_ratio', 'unique ratio'],
  ['long_words', 'long-word rate'],
  ['token_entropy', 'token entropy']
];

export function CorpusPanel() {
  const { data } = useArtifact<CorpusSummary>('corpus-summary');
  const { data: defactify } = useArtifact<DefactifySummary>('defactify-summary');

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
        <p className="figure-empty">Corpus summary artifact not found. Run python -m bench.baseline_corpus.</p>
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
          {data.contaminated_cohorts && data.contaminated_cohorts.length > 0 ? (
            <p className="error" role="status">
              Watermark-flagged cohorts: {data.contaminated_cohorts.map((c) => `${c.family}/${c.kind} (${c.watermark_status})`).join('; ')}.
              Calibration evidence may partly reflect model lineage rather than direct use.
            </p>
          ) : null}
          <table className="corpus-table">
            <thead>
              <tr>
                <th>family</th>
                <th>kind</th>
                <th>n</th>
                <th>watermark</th>
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
                  <td>{cohort.watermark_status ?? 'unknown'}</td>
                  {FEATURE_LABELS.map(([key]) => (
                    <td key={key}>{cohort.features[key] ? cohort.features[key].mean.toFixed(3) : '—'}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {defactify && (
        <div className="defactify-block">
          <h4>
            Defactify bench dataset
            <InfoTip
              label="Defactify"
              text="Roy et al. 2026 (arXiv:2510.22874): NYT human articles plus single-prompt rewrites from six LLM families. Fetched locally, hash-pinned, and hygiene-filtered; raw text never enters the repo. Story groups are reconstructed by TF-IDF near-duplicate clustering for leakage-safe cross-validation."
            />
          </h4>
          <div className="stat-row">
            <div className="stat">
              <span>records</span>
              <strong>{defactify.n_records.toLocaleString()}</strong>
            </div>
            <div className="stat">
              <span>human</span>
              <strong>{defactify.n_human.toLocaleString()}</strong>
            </div>
            <div className="stat">
              <span>AI</span>
              <strong>{defactify.n_ai.toLocaleString()}</strong>
            </div>
            <div className="stat">
              <span>families</span>
              <strong>{Object.keys(defactify.families).length}</strong>
            </div>
            <div className="stat">
              <span>story groups</span>
              <strong>{defactify.group_reconstruction.n_groups.toLocaleString()}</strong>
            </div>
            <div className="stat">
              <span>error artifacts dropped</span>
              <strong>{defactify.hygiene.dropped_error_artifacts.toLocaleString()}</strong>
            </div>
            <div className="stat">
              <span>official-split leakage</span>
              <strong>{(defactify.leakage_audit.official_split_story_leakage_rate * 100).toFixed(1)}%</strong>
            </div>
          </div>
        </div>
      )}
    </article>
  );
}
