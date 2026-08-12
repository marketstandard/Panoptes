"""Tiered model zoo with an explicit statistical power gate.

Tier 0 (penalized logistic regression) is always admissible: it is the
strongest small-sample baseline in the literature and trains in
milliseconds. Tier 1 (gradient boosting) requires n >= 300. Tier 2
(neural, including Panoptes-v0) is only *admitted into the comparison
zoo* when a power calculation says the corpus could detect a meaningful
improvement; below that, neural results are reported as exploratory.
The gate decision and its rationale are written onto every model card.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Protocol

import numpy as np

TIER1_MIN_N = 300


class BenchModel(Protocol):
    name: str
    tier: int

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BenchModel": ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(AI-generated) for each row."""


def power_gate(n: int, mde: float = 0.05, alpha: float = 0.05, power: float = 0.8) -> dict:
    """Can a corpus of size n detect an mde accuracy gain between two models?

    Worst-case (p = 0.5) two-proportion calculation: each class needs
    2 (z_{1-a/2} + z_power)^2 p(1-p) / mde^2 examples.
    """
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    per_class = math.ceil(2 * (z_alpha + z_power) ** 2 * 0.25 / mde**2)
    required = 2 * per_class
    passes = n >= required
    rationale = (
        f"n={n} >= {required} required (two-proportion worst case, mde={mde}, "
        f"alpha={alpha}, power={power}): gate {'passes' if passes else 'FAILS'}"
    )
    return {
        "passes": passes,
        "n": n,
        "required_n": required,
        "mde": mde,
        "alpha": alpha,
        "power": power,
        "rationale": rationale,
    }


@dataclass
class LogisticTier0:
    name: str = "logistic-tier0"
    tier: int = 0
    C: float = 1.0
    seed: int = 13
    _model: Any = field(default=None, repr=False)
    _mean: np.ndarray | None = field(default=None, repr=False)
    _scale: np.ndarray | None = field(default=None, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticTier0":
        from sklearn.linear_model import LogisticRegression

        self._mean = X.mean(axis=0)
        self._scale = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
        scaled = (X - self._mean) / self._scale
        self._model = LogisticRegression(C=self.C, max_iter=5000, random_state=self.seed)
        self._model.fit(scaled, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scaled = (X - self._mean) / self._scale
        return self._model.predict_proba(scaled)[:, 1]


@dataclass
class GbmTier1:
    name: str = "gbm-tier1"
    tier: int = 1
    seed: int = 13
    _model: Any = field(default=None, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GbmTier1":
        from sklearn.ensemble import HistGradientBoostingClassifier

        self._model = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.06, max_iter=300,
            early_stopping=True, validation_fraction=0.15, random_state=self.seed,
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]


def zoo(n_records: int, include_neural: bool = True) -> list[dict]:
    """Admissible models for a dataset of size n_records, with gate rationale."""
    entries: list[dict] = [
        {
            "name": "logistic-tier0",
            "tier": 0,
            "factory": LogisticTier0,
            "admitted": True,
            "rationale": "Penalized logistic regression is always admissible (small-sample baseline).",
        },
        {
            "name": "gbm-tier1",
            "tier": 1,
            "factory": GbmTier1,
            "admitted": n_records >= TIER1_MIN_N,
            "rationale": (
                f"Gradient boosting requires n >= {TIER1_MIN_N}; n={n_records} "
                f"{'meets' if n_records >= TIER1_MIN_N else 'does not meet'} this."
            ),
        },
    ]
    if include_neural:
        gate = power_gate(n_records)
        entries.append(
            {
                "name": "panoptes-v0",
                "tier": 2,
                "factory": None,  # constructed lazily; requires torch
                "admitted": gate["passes"],
                "rationale": gate["rationale"]
                + " — neural results below the gate are exploratory, not comparative.",
                "gate": gate,
            }
        )
    return entries
