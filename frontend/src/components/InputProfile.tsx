import { InfoTip } from './InfoTip';
import { useArtifact } from '../hooks';
import type { CorpusSummary } from './CorpusPanel';

interface Props {
  featureProfile: Record<string, number>;
}

const FEATURES: Array<[string, string]> = [
  ['long_words', 'long-word rate'],
  ['connectors', 'connector rate'],
  ['unique_ratio', 'unique ratio'],
  ['short_sentences', 'short-sentence rate'],
  ['structured', 'structure'],
  ['digits', 'digit rate'],
  ['balanced_lines', 'line balance']
];

interface Range {
  min: number;
  max: number;
}

function pooledRange(summary: CorpusSummary, feature: string, human: boolean): Range | null {
  const cohorts = summary.cohorts.filter((cohort) => (cohort.family === 'human') === human);
  const stats = cohorts.map((cohort) => cohort.features[feature]).filter(Boolean);
  if (stats.length === 0) {
    return null;
  }
  return {
    min: Math.min(...stats.map((stat) => stat.min)),
    max: Math.max(...stats.map((stat) => stat.max))
  };
}

export function InputProfile({ featureProfile }: Props) {
  const { data } = useArtifact<CorpusSummary>('corpus-summary');
  const entries = FEATURES.filter(([key]) => key in featureProfile);
  if (entries.length === 0) {
    return null;
  }

  return (
    <article className="glass-panel figure-card">
      <h3>
        Input profile vs corpus ranges
        <InfoTip
          label="Input profile"
          text="The submitted text's stylometric features against the observed ranges in the verified corpus: green band = human controls, blue band = AI reference outputs. The marker is your input."
        />
      </h3>
      <div className="input-profile">
        {entries.map(([key, label]) => {
          const value = featureProfile[key];
          const human = data ? pooledRange(data, key, true) : null;
          const ai = data ? pooledRange(data, key, false) : null;
          const lo = Math.min(0, human?.min ?? value, ai?.min ?? value);
          const hi = Math.max(value, human?.max ?? value, ai?.max ?? value, 1e-6);
          const span = hi - lo;
          const pct = (v: number) => `${(((v - lo) / span) * 100).toFixed(1)}%`;
          return (
            <div className="profile-row" key={key}>
              <span className="profile-label">{label}</span>
              <div
                className="profile-track"
                title={`${label}: input ${value.toFixed(3)}${human ? ` · human ${human.min.toFixed(3)}–${human.max.toFixed(3)}` : ''}${ai ? ` · AI ${ai.min.toFixed(3)}–${ai.max.toFixed(3)}` : ''}`}
              >
                {human ? (
                  <div
                    className="profile-band human"
                    style={{ left: pct(human.min), width: `${(((human.max - human.min) / span) * 100).toFixed(1)}%` }}
                  />
                ) : null}
                {ai ? (
                  <div
                    className="profile-band ai"
                    style={{ left: pct(ai.min), width: `${(((ai.max - ai.min) / span) * 100).toFixed(1)}%` }}
                  />
                ) : null}
                <div className="profile-marker" style={{ left: pct(value) }} />
              </div>
              <span className="profile-value">{value.toFixed(3)}</span>
            </div>
          );
        })}
      </div>
      <div className="chart-legend">
        <span><i className="legend-band legend-teal" /> Human corpus range</span>
        <span><i className="legend-band legend-blue" /> AI corpus range</span>
        <span><i className="legend-dot legend-amber" /> Your input</span>
      </div>
      {!data ? <p className="figure-empty">Corpus ranges unavailable; showing input values only.</p> : null}
    </article>
  );
}
