import type { AnalysisRequest, AnalysisResponse } from './types';

export async function analyze(request: AnalysisRequest): Promise<AnalysisResponse> {
  const response = await fetch('/api/v1/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ include_text: true, ...request })
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

export type ArtifactName =
  | 'baseline-calibration'
  | 'defactify-calibration'
  | 'corpus-summary'
  | 'defactify-summary'
  | 'methodology-report'
  | 'panoptes-v0-card'
  | 'logistic-tier0-card'
  | 'gbm-tier1-card'
  | 'logistic-defactify-card'
  | 'gbm-defactify-card'
  | 'attribution-defactify-card'
  | 'defactify-external-validation';

export async function fetchArtifact<T = unknown>(name: ArtifactName): Promise<T | null> {
  const response = await fetch(`/api/v1/artifacts/${name}`);
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Artifact fetch failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}
