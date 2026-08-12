import type { AnalysisRequest, AnalysisResponse } from './types';

export async function analyze(request: AnalysisRequest): Promise<AnalysisResponse> {
  const response = await fetch('/api/v1/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
  if (!response.ok) {
    throw new Error(`Analysis failed: ${response.status}`);
  }
  return response.json() as Promise<AnalysisResponse>;
}

export async function capabilities(): Promise<Record<string, unknown>> {
  const response = await fetch('/api/v1/capabilities');
  if (!response.ok) {
    throw new Error(`Capabilities failed: ${response.status}`);
  }
  return response.json() as Promise<Record<string, unknown>>;
}
