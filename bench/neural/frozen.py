"""Frozen neural ensemble as a bench ``Detector`` for transport experiments.

The Phase 5 ensemble is trained once and frozen. For calibration-transfer and
pooled-generalization analyses the same frozen scorer must be evaluated across
cohorts without retraining; this adapter exposes it through the bench
``fit(dataset, idx)`` / ``predict_proba(dataset, idx)`` protocol so it runs
through ``bench.transport`` (and ``bench.measure``) exactly like the heuristic,
logistic, and GBM tiers.

``fit`` is a no-op (the ensemble is already trained); ``predict_proba`` returns
the ensemble's *raw* participation probability so the transport framework fits
its own per-cohort calibrator under the data firewall. Leave-one-cohort-out
neural runs retrain the encoder per split and are launched separately
(GPU-bound); this adapter is the frozen-scorer path only.
"""

from __future__ import annotations

import numpy as np

from bench.datasets import Dataset


class FrozenNeuralDetector:
    """Bench ``Detector`` adapter over the frozen, hash-verified neural ensemble."""

    name = "neural-ensemble-v1"

    def __init__(
        self,
        manager=None,
        *,
        artifact_dir: str | None = None,
        device: str | None = None,
        single_seed: int | None = None,
    ):
        # Lazy import: the backend runtime pulls in torch only when an ensemble
        # is actually loaded, so importing this module stays cheap.
        from panoptes.analysis.neural_runtime import NeuralModelManager

        if manager is not None:
            self.manager = manager
        else:
            kwargs: dict = {}
            if artifact_dir is not None:
                kwargs["artifact_dir"] = artifact_dir
            if device is not None:
                kwargs["device"] = device
            if single_seed is not None:
                kwargs["single_seed"] = single_seed
            self.manager = NeuralModelManager(**kwargs)

    def available(self) -> bool:
        return self.manager.available()

    def fit(self, dataset: Dataset, train_idx: np.ndarray) -> FrozenNeuralDetector:
        # Frozen ensemble: no fitting. The index is accepted for interface
        # compatibility and intentionally unused.
        return self

    def predict_proba(self, dataset: Dataset, idx: np.ndarray) -> np.ndarray:
        """Raw (uncalibrated) participation probability per document."""
        scores = [self.manager.score_text(dataset.texts[int(i)])["raw_participation"] for i in idx]
        return np.array(scores, dtype=float)
