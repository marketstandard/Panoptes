import type { Matrix } from '../types';

export function EvidenceMatrix({
  title,
  matrix,
  selectedSegment,
  onSelectSegment,
  tone = 'blue'
}: {
  title: string;
  matrix: Matrix;
  selectedSegment: string | null;
  onSelectSegment: (id: string) => void;
  tone?: 'blue' | 'green';
}) {
  const numeric = matrix.values.flat().filter((value): value is number => value !== null);
  const max = Math.max(1, ...numeric.map(Math.abs));
  return (
    <article className="matrix-panel glass-panel">
      <div className="panel-heading">
        <div>
          <p className="panel-kicker">Evidence matrix</p>
          <h3>{title}</h3>
        </div>
        <span>{matrix.scale}</span>
      </div>
      <div className="matrix-scroll">
        <div
          className="matrix-grid"
          style={{ gridTemplateColumns: `minmax(150px, 0.9fr) repeat(${matrix.columns.length}, minmax(44px, 1fr))` }}
          role="grid"
          aria-label={matrix.legend}
        >
          <span className="matrix-corner" aria-hidden="true" />
          {matrix.columns.map((column) => (
            <button
              type="button"
              key={column}
              className={column === selectedSegment ? 'matrix-col selected' : 'matrix-col'}
              onClick={() => onSelectSegment(column)}
            >
              {column.replace('segment-', 'S')}
            </button>
          ))}
          {matrix.rows.map((row, rowIndex) => (
            <div className="matrix-row" key={row} style={{ display: 'contents' }}>
              <strong>{row}</strong>
              {matrix.values[rowIndex].map((value, columnIndex) => {
                const column = matrix.columns[columnIndex];
                const intensity = value === null ? 0 : Math.min(Math.abs(value) / max, 1);
                return (
                  <button
                    type="button"
                    key={`${row}-${column}`}
                    className={[
                      'matrix-cell',
                      tone,
                      column === selectedSegment ? 'selected' : '',
                      value === null ? 'null' : ''
                    ].join(' ')}
                    onClick={() => onSelectSegment(column)}
                    title={value === null ? `${row} / ${column}: insufficient evidence` : `${row} / ${column}: ${value.toFixed(4)}`}
                    style={{
                      background:
                        value === null
                          ? undefined
                          : tone === 'green'
                            ? `linear-gradient(180deg, rgba(45, 212, 191, ${0.14 + intensity * 0.78}), rgba(16, 185, 129, ${0.08 + intensity * 0.42}))`
                            : `linear-gradient(180deg, rgba(96, 165, 250, ${0.14 + intensity * 0.78}), rgba(59, 130, 246, ${0.08 + intensity * 0.42}))`
                    }}
                  >
                    {value === null ? '—' : value >= 10 ? value.toFixed(0) : value.toFixed(2)}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      <p className="matrix-legend">{matrix.legend}</p>
    </article>
  );
}
