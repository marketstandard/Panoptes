"""Evaluation: grouped cross-validation, calibration metrics, reliability
bins, coverage-vs-abstention curves, conformal prediction sets, and
fairness slices. All evaluation is out-of-fold; nothing is scored on
its own training data."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from bench.datasets import Dataset, grouped_splits

SEED = 13


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if not np.any(mask):
            continue
        error += mask.mean() * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
    return float(error)


def tpr_at_fpr(labels: np.ndarray, probabilities: np.ndarray, target_fpr: float) -> float:
    negative_scores = probabilities[labels == 0]
    if len(negative_scores) == 0:
        return 0.0
    threshold = np.quantile(negative_scores, 1 - target_fpr, method="higher")
    positives = labels == 1
    return float(np.mean(probabilities[positives] >= threshold)) if np.any(positives) else 0.0


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    result = {
        "auroc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) > 1 else float("nan"),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
        "accuracy": float(np.mean((probabilities >= 0.5) == labels)),
    }
    for fpr in (0.01, 0.05):
        result[f"tpr_at_{int(fpr * 100)}fpr"] = tpr_at_fpr(labels, probabilities, fpr)
    return result


def auroc_ci(labels: np.ndarray, probabilities: np.ndarray, bootstrap: int = 1000) -> list[float]:
    rng = np.random.default_rng(SEED)
    stats_: list[float] = []
    n = len(labels)
    for _ in range(bootstrap):
        idx = rng.integers(0, n, n)
        if len(set(labels[idx])) < 2:
            continue
        stats_.append(float(roc_auc_score(labels[idx], probabilities[idx])))
    if not stats_:
        return [float("nan"), float("nan")]
    return [float(np.percentile(stats_, 2.5)), float(np.percentile(stats_, 97.5))]


def reliability_bins(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_lo": float(edges[index]),
                "bin_hi": float(edges[index + 1]),
                "n": int(mask.sum()),
                "mean_predicted": float(probabilities[mask].mean()),
                "observed": float(labels[mask].mean()),
            }
        )
    return rows


def coverage_curve(labels: np.ndarray, probabilities: np.ndarray, steps: int = 10) -> list[dict]:
    """Coverage vs accuracy as the abstention threshold on confidence rises."""
    confidence = np.maximum(probabilities, 1 - probabilities)
    rows = []
    for t in np.linspace(0.5, 0.99, steps):
        keep = confidence >= t
        if not np.any(keep):
            continue
        rows.append(
            {
                "min_confidence": float(t),
                "coverage": float(keep.mean()),
                "accuracy": float(np.mean((probabilities[keep] >= 0.5) == labels[keep])),
                "n_kept": int(keep.sum()),
            }
        )
    return rows


def conformal_sets(labels: np.ndarray, probabilities: np.ndarray, alpha: float = 0.1) -> dict:
    """Split-conformal over out-of-fold probabilities (exchangeability holds
    because every point is scored by a model that never saw it)."""
    nonconformity = 1.0 - np.where(labels == 1, probabilities, 1 - probabilities)
    quantile = math.ceil((len(nonconformity) + 1) * (1 - alpha)) / len(nonconformity)
    quantile = min(max(quantile, 0.0), 1.0)
    threshold = float(np.quantile(nonconformity, quantile, method="higher"))
    set_sizes = []
    covered = []
    for p, y in zip(probabilities, labels, strict=True):
        included = {c for c in (0, 1) if (1 - (p if c == 1 else 1 - p)) <= threshold}
        set_sizes.append(len(included))
        covered.append(y in included)
    return {
        "alpha": alpha,
        "threshold": threshold,
        "mean_set_size": float(np.mean(set_sizes)),
        "empirical_coverage": float(np.mean(covered)),
        "singleton_rate": float(np.mean([size == 1 for size in set_sizes])),
        "abstention_rate": float(np.mean([size == 2 for size in set_sizes])),
    }


def fairness_slices(dataset: Dataset, probabilities: np.ndarray) -> dict[str, list[dict]]:
    labels = dataset.labels
    slices: dict[str, list[dict]] = {"length_bucket": [], "kind": [], "family": []}
    for dimension, values in (
        ("length_bucket", dataset.buckets),
        ("kind", dataset.kinds),
        ("family", dataset.families),
    ):
        for value in sorted(set(values)):
            mask = np.array([v == value for v in values])
            if mask.sum() < 4:
                continue
            slice_labels = labels[mask]
            row: dict[str, Any] = {"value": value, "n": int(mask.sum())}
            if len(set(slice_labels)) > 1:
                row["auroc"] = float(roc_auc_score(slice_labels, probabilities[mask]))
            row["brier"] = float(brier_score_loss(slice_labels, probabilities[mask]))
            row["mean_predicted"] = float(probabilities[mask].mean())
            slices[dimension].append(row)
    return slices


def cross_validate(model_factory, dataset: Dataset, n_splits: int = 5) -> dict:
    """Out-of-fold probabilities for every document, then all metrics."""
    X = dataset.features()
    y = dataset.labels
    oof = np.zeros(len(dataset))
    fold_metrics = []
    for fold, (train, test) in enumerate(grouped_splits(dataset, n_splits)):
        model = model_factory()
        model.fit(X[train], y[train])
        oof[test] = model.predict_proba(X[test])
        fold_metrics.append({"fold": fold, "n_test": len(test), **binary_metrics(y[test], oof[test])})
    return {
        "oof_probabilities": oof,
        "metrics": binary_metrics(y, oof),
        "auroc_ci95": auroc_ci(y, oof),
        "reliability_bins": reliability_bins(y, oof),
        "coverage_curve": coverage_curve(y, oof),
        "conformal": conformal_sets(y, oof),
        "fairness_slices": fairness_slices(dataset, oof),
        "folds": fold_metrics,
        "n_splits": len(fold_metrics),
    }
