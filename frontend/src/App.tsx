import { useMemo, useState } from 'react';
import {
  AlertTriangle,
  Beaker,
  Braces,
  ChevronDown,
  ChevronRight,
  FileCheck2,
  FlaskConical,
  Info,
  Layers,
  ShieldCheck
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import { analyze } from './api';
import type { AnalysisResponse, Matrix } from './types';
import './styles.css';

const samples = [
  { id: 'human-prose', label: 'Human prose' },
  { id: 'ai-prose', label: 'AI prose' },
  { id: 'code', label: 'Code' }
];

function App() {
  const [text, setText] = useState('');
  const [priorOdds, setPriorOdds] = useState(1);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [selectedSegment, setSelectedSegment] = useState<string | null>(null);
  const [technical, setTechnical] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selected = useMemo(
    () => result?.segments.find((segment) => segment.id === selectedSegment) ?? null,
    [result, selectedSegment]
  );

  async function runAnalysis(fixture?: string) {
    setLoading(true);
    setError(null);
    try {
      const response = await analyze({
        text: fixture ? undefined : text,
        fixture,
        prior_odds: priorOdds
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
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Panoptes evidence workbench</p>
          <h1>Analyze AI participation without hiding uncertainty.</h1>
          <p className="hero-copy">
            Calibrated probabilities, known watermark tests, source-family similarity, signed
            provenance, and segment-level mathematical drilldown in one local-first report.
          </p>
        </div>
        <div className="runtime-card" aria-label="Runtime status">
          <ShieldCheck size={22} />
          <span>Privacy first</span>
          <strong>{result ? result.runtime.profile : 'ready'}</strong>
        </div>
      </header>

      <section className="input-panel" aria-labelledby="input-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Input</p>
            <h2 id="input-heading">Paste text or code</h2>
          </div>
          <div className="sample-row" aria-label="Sample inputs">
            {samples.map((sample) => (
              <button key={sample.id} type="button" onClick={() => runAnalysis(sample.id)}>
                {sample.label}
              </button>
            ))}
          </div>
        </div>
        <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Paste prose or source code here. For fixture/demo output, use a sample button."
          aria-label="Text or code to analyze"
        />
        <div className="input-actions">
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
          <button className="primary" type="button" disabled={loading || !text.trim()} onClick={() => runAnalysis()}>
            {loading ? 'Analyzing…' : 'Analyze'}
          </button>
        </div>
        {error ? <p className="error">{error}</p> : null}
      </section>

      {result ? (
        <>
          <AnswerPanel result={result} />
          <Explanation result={result} />
          <EvidenceExplorer
            result={result}
            selectedSegment={selectedSegment}
            onSelectSegment={setSelectedSegment}
            selected={selected}
          />
          <section className="technical-toggle">
            <button type="button" onClick={() => setTechnical(!technical)} aria-expanded={technical}>
              {technical ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
              Technical drilldown
            </button>
          </section>
          {technical ? <TechnicalDrilldown result={result} /> : null}
        </>
      ) : (
        <section className="empty-state">
          <FlaskConical size={32} />
          <h2>Start with a sample or paste content.</h2>
          <p>
            Panoptes will show a plain-language answer first, then let you inspect the evidence,
            formulas, calibration context, and limitations.
          </p>
        </section>
      )}
    </main>
  );
}

function AnswerPanel({ result }: { result: AnalysisResponse }) {
  const data = [
    { name: 'Human', value: result.summary.overall.human },
    { name: 'AI-generated', value: result.summary.overall.ai_generated },
    { name: 'AI-refined / mixed', value: result.summary.overall.ai_refined_or_mixed }
  ];
  return (
    <section className="answer-grid" aria-labelledby="answer-heading">
      <article className="answer-card primary-card">
        <p className="eyebrow">Answer</p>
        <h2 id="answer-heading">{result.summary.plain_language}</h2>
        <div className="evidence-meta">
          <span>{result.summary.evidence_state.replaceAll('_', ' ')}</span>
          <span>{result.summary.confidence_label} confidence</span>
          <span>{result.input.token_count} tokens</span>
        </div>
      </article>
      <article className="answer-card">
        <div className="card-title">
          <Layers size={18} />
          <h3>Calibrated participation</h3>
        </div>
        <ResponsiveContainer width="100%" height={190}>
          <BarChart data={data} layout="vertical" margin={{ left: 24, right: 16 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" domain={[0, 1]} tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} />
            <YAxis type="category" dataKey="name" width={120} />
            <Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`} />
            <Bar dataKey="value" fill="#2563eb" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </article>
    </section>
  );
}

function Explanation({ result }: { result: AnalysisResponse }) {
  return (
    <section className="explanation" aria-labelledby="meaning-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">What this means</p>
          <h2 id="meaning-heading">Evidence, separated</h2>
        </div>
      </div>
      <div className="explanation-grid">
        <InfoCard icon={<Info />} title="Probability">
          This is calibrated statistical evidence, not proof of who wrote the content.
        </InfoCard>
        <InfoCard icon={<Beaker />} title="Watermark">
          {result.watermarks.some((item) => item.status === 'tested')
            ? 'At least one public watermark scheme could be tested. Absence does not clear the content.'
            : 'No supported watermark detector could run for this input.'}
        </InfoCard>
        <InfoCard icon={<Braces />} title="Source similarity">
          {result.source_families.interpretation}
        </InfoCard>
        <InfoCard icon={<FileCheck2 />} title="Provenance">
          {result.provenance.summary}
        </InfoCard>
      </div>
    </section>
  );
}

function InfoCard({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <article className="info-card">
      <div className="card-title">
        {icon}
        <h3>{title}</h3>
      </div>
      <p>{children}</p>
    </article>
  );
}

function EvidenceExplorer({
  result,
  selectedSegment,
  onSelectSegment,
  selected
}: {
  result: AnalysisResponse;
  selectedSegment: string | null;
  onSelectSegment: (id: string) => void;
  selected: AnalysisResponse['segments'][number] | null;
}) {
  return (
    <section className="evidence" aria-labelledby="evidence-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Evidence explorer</p>
          <h2 id="evidence-heading">Where the signal appears</h2>
        </div>
      </div>
      <div className="evidence-layout">
        <div className="source-viewer" aria-label="Source segment viewer">
          {result.segments.map((segment) => (
            <button
              key={segment.id}
              type="button"
              className={segment.id === selectedSegment ? 'segment selected' : 'segment'}
              onClick={() => onSelectSegment(segment.id)}
            >
              <span>{segment.id}</span>
              <span>{Math.round(segment.posterior.ai_generated * 100)}% AI</span>
              <small>
                chars {segment.start}–{segment.end}
              </small>
            </button>
          ))}
        </div>
        <div className="matrix-stack">
          <Heatmap title="Source family by segment" matrix={result.matrices.source_by_segment} kind="probability" />
          <Heatmap
            title="Watermark evidence by segment"
            matrix={result.matrices.watermark_evidence_by_segment}
            kind="evidence"
          />
        </div>
      </div>
      {selected ? (
        <aside className="segment-detail">
          <h3>{selected.id}</h3>
          <dl>
            <div>
              <dt>Posterior</dt>
              <dd>{Math.round(selected.posterior.ai_generated * 100)}% AI-generated</dd>
            </div>
            <div>
              <dt>Anomaly percentile</dt>
              <dd>{selected.anomaly_percentile === null ? 'n/a' : selected.anomaly_percentile.toFixed(2)}</dd>
            </div>
            <div>
              <dt>Source offsets</dt>
              <dd>
                {selected.start}–{selected.end}
              </dd>
            </div>
          </dl>
        </aside>
      ) : null}
    </section>
  );
}

function Heatmap({ title, matrix, kind }: { title: string; matrix: Matrix; kind: 'probability' | 'evidence' }) {
  const max = kind === 'probability' ? 1 : Math.max(1, ...matrix.values.flat().filter((v): v is number => v !== null));
  return (
    <article className="matrix-card">
      <div className="card-title">
        <h3>{title}</h3>
        <span>{matrix.legend}</span>
      </div>
      <div className="matrix" role="img" aria-label={`${title}: ${matrix.legend}`}>
        <div />
        {matrix.columns.map((column) => (
          <strong key={column}>{column.replace('segment-', 'S')}</strong>
        ))}
        {matrix.rows.map((row, rowIndex) => (
          <FragmentRow
            key={row}
            row={row}
            values={matrix.values[rowIndex]}
            max={max}
            kind={kind}
          />
        ))}
      </div>
    </article>
  );
}

function FragmentRow({
  row,
  values,
  max,
  kind
}: {
  row: string;
  values: Array<number | null>;
  max: number;
  kind: 'probability' | 'evidence';
}) {
  return (
    <>
      <strong>{row}</strong>
      {values.map((value, index) => (
        <span
          key={`${row}-${index}`}
          className="matrix-cell"
          title={value === null ? 'insufficient evidence' : `${row}: ${value.toFixed(3)}`}
          style={{
            backgroundColor:
              value === null
                ? '#e5e7eb'
                : kind === 'probability'
                  ? `rgba(37, 99, 235, ${0.12 + 0.78 * (value / max)})`
                  : `rgba(5, 150, 105, ${0.12 + 0.78 * (value / max)})`
          }}
        >
          {value === null ? '—' : value >= 10 ? value.toFixed(0) : value.toFixed(2)}
        </span>
      ))}
    </>
  );
}

function TechnicalDrilldown({ result }: { result: AnalysisResponse }) {
  return (
    <section className="technical" aria-labelledby="technical-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Technical</p>
          <h2 id="technical-heading">Math, calibration, and limitations</h2>
        </div>
      </div>
      <div className="technical-grid">
        <article className="technical-card">
          <h3>Posterior</h3>
          <dl>
            <div><dt>Prior odds</dt><dd>{result.posterior.prior_odds.toFixed(3)}</dd></div>
            <div><dt>Likelihood ratio</dt><dd>{formatNullable(result.posterior.likelihood_ratio)}</dd></div>
            <div><dt>Posterior odds</dt><dd>{formatNullable(result.posterior.posterior_odds)}</dd></div>
            <div><dt>Calibration</dt><dd>{result.posterior.calibration_bundle}</dd></div>
            <div><dt>Reliability error</dt><dd>{formatNullable(result.posterior.reliability_error)}</dd></div>
          </dl>
        </article>
        <article className="technical-card">
          <h3>Watermark tests</h3>
          {result.watermarks.map((watermark) => (
            <dl key={watermark.scheme}>
              <div><dt>{watermark.scheme}</dt><dd>{watermark.status}</dd></div>
              <div><dt>z</dt><dd>{formatNullable(watermark.z)}</dd></div>
              <div><dt>p</dt><dd>{formatNullable(watermark.p_value)}</dd></div>
              <div><dt>q</dt><dd>{formatNullable(watermark.q_value)}</dd></div>
              <div><dt>Power</dt><dd>{formatNullable(watermark.power)}</dd></div>
            </dl>
          ))}
        </article>
        <article className="technical-card wide">
          <h3>Contribution waterfall</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={result.matrices.contribution_waterfall}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#7c3aed" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </article>
        {result.math.map((item) => (
          <article className="technical-card" key={item.name}>
            <h3>{item.name}</h3>
            <p>{item.meaning}</p>
            <code>{item.formula}</code>
            <p><strong>Units:</strong> {item.units}</p>
            <p><strong>Limits:</strong> {item.limitations.join(' ')}</p>
          </article>
        ))}
        <article className="technical-card warning-card">
          <div className="card-title">
            <AlertTriangle size={18} />
            <h3>Limitations</h3>
          </div>
          <ul>
            {result.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}
          </ul>
        </article>
      </div>
    </section>
  );
}

function formatNullable(value: number | null): string {
  if (value === null) return 'n/a';
  if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toFixed(4);
}

export default App;
