import { Info } from 'lucide-react';

export function InfoTip({ label, text }: { label: string; text: string }) {
  return (
    <button type="button" className="info-tip" aria-label={`What this means: ${label}`}>
      <Info size={13} aria-hidden="true" />
      <span className="info-tip-bubble" role="tooltip">
        {text}
      </span>
    </button>
  );
}
