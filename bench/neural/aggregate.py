"""Document-level aggregation of window scores.

Two preregistered aggregation rules:

  - ``overlap_corrected_logit_mean``: combine per-window log-odds so each
    *token* contributes once. A token covered by two overlapping windows gets
    the mean of those windows' log-odds rather than being double-counted, and
    the document score is the mean over covered tokens. This is the simple,
    parameter-free aggregator.
  - ``hierarchical_summary_head``: a learned head (see
    :mod:`bench.neural.model`) that pools window embeddings; applied per
    document from cached window embeddings.

Window scores are combined in log-odds space (the AI-vs-human evidence), then
squashed with a logistic sigmoid to a probability.
"""

from __future__ import annotations

import numpy as np


def log_odds(window_logits: np.ndarray) -> np.ndarray:
    """AI-vs-human log-odds from two-class window logits ``[B, 2]``."""
    window_logits = np.asarray(window_logits, dtype=np.float64)
    return window_logits[:, 1] - window_logits[:, 0]


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def overlap_corrected_logodds(
    window_logodds: np.ndarray,
    spans: list[tuple[int, int]],
    n_tokens: int,
) -> float:
    """Token-coverage-corrected mean of window log-odds for one document.

    ``spans`` are the half-open token spans ``(token_start, token_end)`` of each
    window in the document's token sequence; ``n_tokens`` is the document's
    token count. Tokens covered by multiple windows contribute the average of
    their covering windows; every covered token has equal weight.
    """
    window_logodds = np.asarray(window_logodds, dtype=np.float64)
    if len(window_logodds) == 0:
        return 0.0
    if n_tokens <= 0:
        return float(window_logodds.mean())
    accum = np.zeros(n_tokens, dtype=np.float64)
    cov = np.zeros(n_tokens, dtype=np.float64)
    for lo, (s, e) in zip(window_logodds, spans, strict=True):
        s = max(0, min(int(s), n_tokens))
        e = max(s, min(int(e), n_tokens))
        if e > s:
            accum[s:e] += lo
            cov[s:e] += 1.0
    covered = cov > 0
    if not covered.any():
        return float(window_logodds.mean())
    per_token = accum[covered] / cov[covered]
    return float(per_token.mean())


def aggregate_documents(
    doc_window_logits: list[np.ndarray],
    doc_spans: list[list[tuple[int, int]]],
    doc_n_tokens: list[int],
) -> np.ndarray:
    """Overlap-corrected logit-mean document probabilities for a corpus.

    ``doc_window_logits[d]`` is ``[n_windows_d, 2]``; returns P(AI) per document.
    """
    probs = np.zeros(len(doc_window_logits), dtype=np.float64)
    for d, (logits, spans) in enumerate(zip(doc_window_logits, doc_spans, strict=True)):
        lo = log_odds(logits)
        doc_lo = overlap_corrected_logodds(lo, spans, doc_n_tokens[d])
        probs[d] = sigmoid(doc_lo)
    return probs
