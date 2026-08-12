# ADR 0001: Python FastAPI backend with React frontend

## Status

Accepted

## Context

Panoptes needs statistical computing, model integration, a stable API, and an interactive UI. TypeScript is preferred for interface work, but Python has the stronger scientific and ML ecosystem.

## Decision

Use Python/FastAPI for analysis and TypeScript/React/Vite for the UI. FastAPI serves the built frontend so normal users run one process.

## Consequences

- Scientific code can use NumPy, SciPy, scikit-learn, PyTorch, and C2PA tooling.
- The UI remains type-safe and independently testable.
- The API response schema is the compatibility boundary.
- The repository must keep Python and TypeScript tooling easy to install.
