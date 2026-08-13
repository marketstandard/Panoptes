"""Detector abstraction for protocol-compliant evaluation.

Every compared detector exposes the same fit/predict surface so that
heuristic, logistic, and boosting models are scored on identical splits.
Feature-based models see the shared stylometric vector; the heuristic
scores raw text and ignores X.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from bench.datasets import Dataset
from bench.features import heuristic_raw_score
from bench.models import GbmTier1, LogisticTier0


class Detector(Protocol):
    name: str

    def fit(self, dataset: Dataset, train_idx: np.ndarray) -> "Detector": ...

    def predict_proba(self, dataset: Dataset, idx: np.ndarray) -> np.ndarray: ...


@dataclass
class HeuristicDetector:
    name: str = "panoptes-heuristic"
    fitted: bool = False

    def fit(self, dataset: Dataset, train_idx: np.ndarray) -> "HeuristicDetector":
        self.fitted = True
        return self

    def predict_proba(self, dataset: Dataset, idx: np.ndarray) -> np.ndarray:
        return np.array(
            [heuristic_raw_score(dataset.texts[int(i)], dataset.kinds[int(i)]) for i in idx],
            dtype=float,
        )


@dataclass
class FeatureDetector:
    factory: Any
    name: str = "feature-model"
    _model: Any = field(default=None, repr=False)
    _constant: float | None = field(default=None, repr=False)

    def fit(self, dataset: Dataset, train_idx: np.ndarray) -> "FeatureDetector":
        X = dataset.features()[train_idx]
        y = dataset.labels[train_idx]
        self._constant = None
        if len(set(int(v) for v in y.tolist())) < 2:
            self._model = None
            self._constant = float(np.mean(y))
            return self
        self._model = self.factory()
        self._model.fit(X, y)
        self.name = getattr(self._model, "name", self.name)
        return self

    def predict_proba(self, dataset: Dataset, idx: np.ndarray) -> np.ndarray:
        if self._constant is not None:
            return np.full(len(idx), self._constant, dtype=float)
        if self._model is None:
            raise RuntimeError("detector must be fit before predict_proba")
        return self._model.predict_proba(dataset.features()[idx])


def catalog() -> dict[str, callable]:
    """Named constructors for the protocol detector zoo."""
    return {
        "heuristic": HeuristicDetector,
        "logistic": lambda: FeatureDetector(LogisticTier0, name="logistic-tier0"),
        "gbm": lambda: FeatureDetector(GbmTier1, name="gbm-tier1"),
    }


def make_detector(name: str) -> Detector:
    zoo = catalog()
    if name not in zoo:
        raise ValueError(f"unknown detector {name!r}; choose from {sorted(zoo)}")
    return zoo[name]()
