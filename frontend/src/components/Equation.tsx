import katex from 'katex';
import { useMemo } from 'react';

export function Equation({ formula, display = true }: { formula: string; display?: boolean }) {
  const html = useMemo(
    () =>
      katex.renderToString(formula, {
        displayMode: display,
        throwOnError: false,
        strict: false
      }),
    [formula, display]
  );
  return <span className={display ? 'equation equation-display' : 'equation'} dangerouslySetInnerHTML={{ __html: html }} />;
}
