import { useMemo } from 'react';
import type { AnalysisResponse, Segment, WatermarkTokenSpan } from '../types';

export function TokenEvidenceOverlay({
  result,
  selectedSegment,
  onSelectSegment
}: {
  result: AnalysisResponse;
  selectedSegment: Segment | null;
  onSelectSegment: (id: string) => void;
}) {
  const watermark = result.watermarks.find((item) => (item.tokens?.length ?? 0) > 0) ?? null;
  const text = result.submitted_text ?? '';

  const rendered = useMemo(() => {
    if (!text || !watermark?.tokens) return null;
    return renderTextWithTokens(text, watermark.tokens, selectedSegment);
  }, [text, watermark, selectedSegment]);

  return (
    <article className="source-panel glass-panel">
      <div className="panel-heading">
        <div>
          <p className="panel-kicker">Token evidence</p>
          <h3>Source overlay</h3>
        </div>
        <span>{watermark ? watermark.scheme : 'not available'}</span>
      </div>
      <div className="segment-strip" aria-label="Segments">
        {result.segments.map((segment) => (
          <button
            type="button"
            key={segment.id}
            className={segment.id === selectedSegment?.id ? 'segment-chip selected' : 'segment-chip'}
            onClick={() => onSelectSegment(segment.id)}
          >
            <span>{segment.id.replace('segment-', 'S')}</span>
            <i style={{ width: `${Math.round(segment.posterior.ai_generated * 100)}%` }} />
          </button>
        ))}
      </div>
      <div className="source-document" aria-label="Analyzed source text with watermark token overlay">
        {rendered ?? (
          <div className="source-empty">
            Token-level watermark spans are unavailable for this input. This may mean the scheme was not
            tested, the input was too short, or token overlay was disabled.
          </div>
        )}
      </div>
      <div className="legend-row">
        <span><i className="legend-token green" /> green-list compatible</span>
        <span><i className="legend-token red" /> non-green</span>
        <span><i className="legend-token segment" /> selected segment</span>
      </div>
    </article>
  );
}

function renderTextWithTokens(
  text: string,
  tokens: WatermarkTokenSpan[],
  selectedSegment: Segment | null
) {
  const pieces: React.ReactNode[] = [];
  let cursor = 0;
  tokens.forEach((token, index) => {
    if (token.start > cursor) {
      pieces.push(<span key={`plain-${index}`}>{text.slice(cursor, token.start)}</span>);
    }
    const inSelected = selectedSegment
      ? token.start >= selectedSegment.start && token.end <= selectedSegment.end
      : false;
    pieces.push(
      <mark
        key={`token-${index}`}
        className={`token-mark ${token.green ? 'green' : 'red'} ${inSelected ? 'in-segment' : ''}`}
        title={token.green ? 'Green-list compatible token' : 'Non-green token'}
      >
        {text.slice(token.start, token.end)}
      </mark>
    );
    cursor = Math.max(cursor, token.end);
  });
  if (cursor < text.length) {
    pieces.push(<span key="tail">{text.slice(cursor)}</span>);
  }
  return pieces;
}
