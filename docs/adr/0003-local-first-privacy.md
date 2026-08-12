# ADR 0003: Local-first privacy posture

## Status

Accepted

## Context

Users may analyze sensitive unpublished text or proprietary code. Requiring a hosted API would create unnecessary privacy risk.

## Decision

The default launcher binds to localhost, uses an in-memory pipeline, and stores no submitted text. Cloud deployment is optional and uses the same no-retention default.

## Consequences

- Local model caches are explicit and user-controlled.
- Report export is an explicit user action.
- Logging must not contain raw text.
- Fixture mode supports demos and CI without external calls.
