export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`;
}

export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'n/a';
  if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toFixed(digits);
}

export function formatSigned(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'n/a';
  const sign = value > 0 ? '+' : '';
  return `${sign}${formatNumber(value, digits)}`;
}

export function statusTone(status: string): 'good' | 'warn' | 'bad' | 'muted' {
  if (['verified', 'tested', 'supported', 'high'].includes(status)) return 'good';
  if (['insufficient_data', 'not_present', 'medium', 'low'].includes(status)) return 'warn';
  if (['tampered', 'error'].includes(status)) return 'bad';
  return 'muted';
}
