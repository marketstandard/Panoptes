"""Regression test for the reproduction metric-path recomputation.

The plan requires reproduction to recompute at least one full metric path, not
merely re-hash existing JSON. ``recompute_metric_path`` reloads the committed
fixture, re-runs detector -> probabilities -> ``binary_metrics``, and compares
against the committed expected values. This test locks that behavior: if the
detector or metric definitions drift, the recomputation (and this test) fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.reproduce import recompute_metric_path  # noqa: E402


def test_metric_path_recomputes_cleanly():
    result, errors = recompute_metric_path()
    assert errors == []
    assert result["n_documents"] == 16
    metrics = result["metrics"]
    assert metrics, "expected per-metric comparison entries"
    assert all(entry["match"] for entry in metrics.values())
    # The fixture separates cleanly; guard against a degenerate constant score.
    assert result["recomputed_auroc"] > 0.9
