import { useMemo, useState } from 'react';
import { Aperture, BookOpen, ExternalLink, Play, ScanEye, Upload } from 'lucide-react';
import { analyze } from './api';
import { AnswerObservatory } from './components/AnswerObservatory';
import { CorpusPanel } from './components/CorpusPanel';
import { CoverageCurve } from './components/CoverageCurve';
import { EvidenceMatrix } from './components/EvidenceMatrix';
import { InputProfile } from './components/InputProfile';
import { ModelCardPanel } from './components/ModelCardPanel';
import { PosteriorSensitivity } from './components/PosteriorSensitivity';
import { PowerCurve } from './components/PowerCurve';
import { ProvenancePanel } from './components/ProvenancePanel';
import { ReliabilityDiagram } from './components/ReliabilityDiagram';
import { SourceFamilyPanel } from './components/SourceFamilyPanel';
import { TechnicalDrilldown } from './components/TechnicalDrilldown';
import { TokenEvidenceOverlay } from './components/TokenEvidenceOverlay';
import { useArtifact } from './hooks';
import type { DefactifySummary } from './components/CorpusPanel';
import type { AnalysisResponse } from './types';
import 'katex/dist/katex.min.css';
import './styles.css';

const samples = [
  { id: 'human-prose', label: 'Human prose' },
  { id: 'ai-prose', label: 'AI prose' },
  { id: 'code', label: 'Code' }
];

export default function App() {
  const [text, setText] = useState('');
  const [priorOdds, setPriorOdds] = useState(1);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [selectedSegment, setSelectedSegment] = useState<string | null>(null);
  const [showTechnical, setShowTechnical] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { data: defactifySummary } = useArtifact<DefactifySummary>('defactify-summary');

  const selected = useMemo(
    () => result?.segments.find((segment) => segment.id === selectedSegment) ?? result?.segments[0] ?? null,
    [result, selectedSegment]
  );

  async function runAnalysis(fixture?: string) {
    setLoading(true);
    setError(null);
    try {
      const response = await analyze({
        text: fixture ? undefined : text,
        fixture,
        prior_odds: priorOdds,
        include_text: true
      });
      setResult(response);
      setSelectedSegment(response.segments[0]?.id ?? null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="observatory-shell">
      <nav className="topbar" aria-label="Panoptes navigation">
        <div className="brand">
          <div className="brand-mark"><Aperture size={22} /></div>
          <div>
            <strong>Panoptes</strong>
            <span>evidence observatory</span>
          </div>
        </div>
        <div className="topbar-meta">
          <span>{result ? result.runtime.profile : 'local-first'}</span>
          <span>{result ? result.runtime.device : 'gpu-ready'}</span>
          <a href="/paper.html" target="_blank" rel="noreferrer"><BookOpen size={16} /> Research paper</a>
          <a href="https://github.com/Encryptic1/Panoptes" target="_blank" rel="noreferrer"><ExternalLink size={16} /> Source</a>
        </div>
      </nav>

      <header className="hero-observatory">
        <div className="hero-copy-block">
          <div className="eyebrow"><ScanEye size={15} /> Statistical provenance and participation</div>
          <h1>See the evidence. Keep the uncertainty visible.</h1>
          <p>
            Panoptes combines calibrated AI-participation probabilities, public watermark hypothesis tests,
            source-family geometry, and signed provenance into one local-first research interface.
          </p>
        </div>
        <div className="hero-instrument glass-panel">
          <span>Observation state</span>
          <strong>{result ? result.summary.evidence_state.replaceAll('_', ' ') : 'standing by'}</strong>
          <p>{result ? `${result.input.segment_count} synchronized segments` : 'Fixture mode available offline'}</p>
        </div>
      </header>

      <section className="workbench glass-panel" aria-labelledby="input-heading">
        <div className="panel-heading">
          <div>
            <p className="panel-kicker">Input workbench</p>
            <h2 id="input-heading">Analyze prose, code, or a fixture</h2>
          </div>
          <div className="sample-row">
            {samples.map((sample) => (
              <button key={sample.id} type="button" onClick={() => runAnalysis(sample.id)}>
                <BookOpen size={14} /> {sample.label}
              </button>
            ))}
          </div>
        </div>
        <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Paste prose or source code. Token-level watermark overlays are returned when the input supports them."
          aria-label="Text or code to analyze"
        />
        <div className="command-row">
          <label>
            Prior odds
            <input
              type="number"
              min="0.01"
              step="0.1"
              value={priorOdds}
              onChange={(event) => setPriorOdds(Number(event.target.value))}
            />
          </label>
          <button type="button" className="secondary" disabled>
            <Upload size={15} /> File provenance soon
          </button>
          <button className="primary" type="button" disabled={loading || !text.trim()} onClick={() => runAnalysis()}>
            <Play size={15} /> {loading ? 'Analyzing…' : 'Run analysis'}
          </button>
        </div>
        {error ? <p className="error">{error}</p> : null}
      </section>

      {result ? (
        <>
          <AnswerObservatory result={result} />
          <section className="evidence-grid">
            <TokenEvidenceOverlay result={result} selectedSegment={selected} onSelectSegment={setSelectedSegment} />
            <SourceFamilyPanel result={result} />
            <ProvenancePanel result={result} />
          </section>
          <section className="matrix-grid-section">
            <EvidenceMatrix
              title="Source family by segment"
              matrix={result.matrices.source_by_segment}
              selectedSegment={selected?.id ?? null}
              onSelectSegment={setSelectedSegment}
              tone="blue"
              legendTip="Conditional stylometric similarity among supported source families. When the signed calibration artifact is present, distances use Mahalanobis geometry fitted on the verified reference corpus; otherwise a hand-tuned heuristic is used and the report says so."
            />
            <EvidenceMatrix
              title="Watermark evidence by segment"
              matrix={result.matrices.watermark_evidence_by_segment}
              selectedSegment={selected?.id ?? null}
              onSelectSegment={setSelectedSegment}
              tone="green"
            />
          </section>
          <section className="figures-grid" aria-label="Statistical figures">
            <PosteriorSensitivity priorOdds={priorOdds} likelihoodRatio={result.posterior.likelihood_ratio} />
            <ReliabilityDiagram calibration={result.calibration} />
            <PowerCurve currentN={result.calibration?.n_records ?? 0} defactifyN={defactifySummary?.n_records} />
            <CoverageCurve />
            <InputProfile featureProfile={result.input.feature_profile} />
          </section>
          <section className="technical-toggle-panel glass-panel">
            <div>
              <p className="panel-kicker">Equation lab</p>
              <h2>Technical drilldown and assumptions</h2>
            </div>
            <button type="button" onClick={() => setShowTechnical(!showTechnical)}>
              {showTechnical ? 'Collapse' : 'Expand'}
            </button>
          </section>
          {showTechnical ? <TechnicalDrilldown result={result} /> : null}
        </>
      ) : (
        <section className="empty-observatory glass-panel">
          <div className="empty-orbit"><ScanEye size={42} /></div>
          <h2>Run a fixture or paste evidence.</h2>
          <p>
            The first view is intentionally plain-language. Expand the laboratory to inspect formulas,
            confidence intervals, q-values, calibration cohorts, and segment-level evidence.
          </p>
        </section>
      )}
      <section className="corpus-section" aria-label="Corpus and models">
        <CorpusPanel />
        <ModelCardPanel />
      </section>
    </main>
  );
}
