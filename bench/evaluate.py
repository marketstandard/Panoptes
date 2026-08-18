"""Evaluation: grouped cross-validation, calibration metrics, reliability
bins, coverage-vs-abstention curves, conformal prediction sets, and
fairness slices. Protocol-compliant runs also separate train, calibration,
and untouched test groups. Nothing is scored on its own training data."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from bench.datasets import Dataset, grouped_splits
from research.protocol import COVERAGE_LEVELS

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


def auprc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, probabilities))


def calibration_slope_intercept(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Logistic calibration slope and intercept: logit(y) ~ a + b logit(p).

    Perfect calibration has slope 1 and intercept 0. Slope < 1 is
    overconfidence; intercept ≠ 0 is a systematic shift.
    """
    from research.methodology import logistic_irls

    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(labels, dtype=float)
    if len(set(y.astype(int))) < 2:
        return {"calibration_slope": float("nan"), "calibration_intercept": float("nan")}
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    fit = logistic_irls(logit, y)
    return {
        "calibration_slope": float(fit["beta"][1]),
        "calibration_intercept": float(fit["beta"][0]),
    }


def selective_risk_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    coverages: tuple[float, ...] = COVERAGE_LEVELS,
) -> list[dict]:
    """Error on the highest-confidence subset at each protocol coverage level."""
    confidence = np.maximum(probabilities, 1 - probabilities)
    n = len(labels)
    rows = []
    order = np.argsort(-confidence)
    for coverage in coverages:
        k = max(1, int(math.ceil(coverage * n))) if coverage < 1 else n
        kept = order[:k]
        pred = probabilities[kept] >= 0.5
        risk = float(np.mean(pred != labels[kept]))
        rows.append(
            {
                "coverage": float(coverage),
                "n_kept": int(len(kept)),
                "empirical_coverage": float(len(kept) / n),
                "selective_risk": risk,
                "accuracy": 1.0 - risk,
                "brier": float(brier_score_loss(labels[kept], probabilities[kept])),
            }
        )
    return rows


def worst_group_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: list[str],
    min_n: int = 4,
) -> dict[str, Any]:
    rows = []
    for value in sorted(set(groups)):
        mask = np.array([g == value for g in groups])
        if int(mask.sum()) < min_n or len(set(labels[mask])) < 2:
            continue
        metrics = binary_metrics(labels[mask], probabilities[mask])
        rows.append({"group": value, "n": int(mask.sum()), **metrics})
    if not rows:
        return {"groups": [], "worst_auroc": float("nan"), "worst_brier": float("nan")}
    return {
        "groups": rows,
        "worst_auroc": float(min(row["auroc"] for row in rows)),
        "worst_brier": float(max(row["brier"] for row in rows)),
        "worst_ece": float(max(row["ece"] for row in rows)),
    }


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    result = {
        "auroc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) > 1 else float("nan"),
        "auprc": auprc(labels, probabilities),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
        "accuracy": float(np.mean((probabilities >= 0.5) == labels)),
    }
    result.update(calibration_slope_intercept(labels, probabilities))
    result["tpr_at_0.1fpr"] = tpr_at_fpr(labels, probabilities, 0.001)
    result["tpr_at_1fpr"] = tpr_at_fpr(labels, probabilities, 0.01)
    result["tpr_at_5fpr"] = tpr_at_fpr(labels, probabilities, 0.05)
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


def sliced_conformal_coverage(
    labels: np.ndarray,
    probabilities: np.ndarray,
    slices: dict[str, list[str]],
    alpha: float = 0.1,
    min_n: int = 30,
) -> dict[str, list[dict]]:
    """Conditional conformal coverage: does the pooled threshold hold per slice?

    The split-conformal guarantee is marginal over the evaluation cohort. This
    measures the empirical coverage of the *pooled* threshold within each
    slice (length bucket, generator family, human/AI class), which is where
    coverage gaps actually hide.
    """
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    nonconformity = 1.0 - np.where(y == 1, p, 1 - p)
    quantile = math.ceil((len(nonconformity) + 1) * (1 - alpha)) / len(nonconformity)
    quantile = min(max(quantile, 0.0), 1.0)
    threshold = float(np.quantile(nonconformity, quantile, method="higher"))
    covered = nonconformity <= threshold
    out: dict[str, list[dict]] = {}
    for dimension, values in slices.items():
        rows = []
        for value in sorted(set(values)):
            mask = np.array([v == value for v in values])
            n = int(mask.sum())
            if n < min_n:
                continue
            rows.append(
                {
                    "value": value,
                    "n": n,
                    "nominal_coverage": float(1 - alpha),
                    "empirical_coverage": float(covered[mask].mean()),
                    "coverage_gap": float(covered[mask].mean() - (1 - alpha)),
                }
            )
        out[dimension] = rows
    out["pooled"] = {
        "alpha": alpha,
        "threshold": threshold,
        "empirical_coverage": float(covered.mean()),
        "n": int(len(y)),
    }
    return out


def cross_dataset_transport(
    train_dataset: Dataset,
    test_dataset: Dataset,
    detector_factory,
    seed: int = SEED,
) -> dict:
    """Fit + calibrate on one cohort, score the other cohort untouched.

    The within-dataset transport matrix only sees domains that share a
    feature distribution. Cross-dataset transport is the harder question the
    paper cares about: does a detector fitted on cohort A stay calibrated on
    cohort B? Calibration is fit on the source cohort's calibration partition
    only; the target cohort is never used for fitting or tuning.
    """
    from bench.splits import protocol_splits

    split = protocol_splits(train_dataset, seed=seed)[0]
    detector = detector_factory()
    detector.fit(train_dataset, split.train)
    raw_cal = detector.predict_proba(train_dataset, split.calibration)
    calibrator = fit_isotonic(raw_cal, train_dataset.labels[split.calibration])
    all_idx = np.arange(len(test_dataset))
    raw_test = detector.predict_proba(test_dataset, all_idx)
    if calibrator is None:
        calibrated = np.clip(raw_test, 1e-6, 1 - 1e-6)
        calibration_applied = False
    else:
        calibrated = np.clip(calibrator.predict(raw_test), 1e-6, 1 - 1e-6)
        calibration_applied = True
    y = test_dataset.labels
    metrics = binary_metrics(y, calibrated)
    return {
        "train_dataset": train_dataset.provenance,
        "test_dataset": test_dataset.provenance,
        "n_train": int(len(split.train)),
        "n_calibration": int(len(split.calibration)),
        "n_test": int(len(all_idx)),
        "calibration_applied": calibration_applied,
        "detector": getattr(detector, "name", type(detector).__name__),
        **metrics,
        "selective_risk": selective_risk_curve(y, calibrated),
        "conformal": conformal_sets(y, calibrated),
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
        "selective_risk": selective_risk_curve(y, oof),
        "conformal": conformal_sets(y, oof),
        "fairness_slices": fairness_slices(dataset, oof),
        "worst_group": worst_group_metrics(y, oof, dataset.families),
        "folds": fold_metrics,
        "n_splits": len(fold_metrics),
    }


def fit_isotonic(raw: np.ndarray, labels: np.ndarray) -> IsotonicRegression | None:
    """Fit isotonic regression, or return None when calibration is unidentified.

    A single-class calibration fold cannot identify a monotone map. Constant
    raw scores are likewise unidentified. Callers must then pass raw scores
    through rather than collapsing every test probability to the class mean.
    """
    x = np.asarray(raw, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(x) < 2 or len(set(y.astype(int).tolist())) < 2 or len(np.unique(x)) < 2:
        return None
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(x, y)
    return model


def evaluate_protocol_split(detector, dataset: Dataset, split) -> dict:
    """Fit on train, calibrate on calibration, score untouched test."""
    from bench.evidence import likelihood_ratio, prior_sensitivity

    detector.fit(dataset, split.train)
    raw_cal = detector.predict_proba(dataset, split.calibration)
    calibrator = fit_isotonic(raw_cal, dataset.labels[split.calibration])
    raw_test = detector.predict_proba(dataset, split.test)
    if calibrator is None:
        calibrated = np.clip(raw_test, 1e-6, 1 - 1e-6)
        calibration_applied = False
    else:
        calibrated = np.clip(calibrator.predict(raw_test), 1e-6, 1 - 1e-6)
        calibration_applied = True
    labels = dataset.labels[split.test]
    prevalence = float(dataset.labels[split.calibration].mean())
    metrics = binary_metrics(labels, calibrated)
    mean_lr = float(np.mean(likelihood_ratio(calibrated, prevalence)))
    return {
        "method": split.method,
        "n_train": int(len(split.train)),
        "n_calibration": int(len(split.calibration)),
        "n_test": int(len(split.test)),
        "cohort_prevalence": prevalence,
        "calibration_applied": calibration_applied,
        "metrics": metrics,
        "selective_risk": selective_risk_curve(labels, calibrated),
        "reliability_bins": reliability_bins(labels, calibrated),
        "conformal": conformal_sets(labels, calibrated),
        "worst_group": worst_group_metrics(
            labels, calibrated, [dataset.families[int(i)] for i in split.test]
        ),
        "mean_likelihood_ratio": mean_lr,
        "prior_sensitivity": prior_sensitivity(mean_lr),
        "raw_metrics": binary_metrics(labels, np.clip(raw_test, 1e-6, 1 - 1e-6)),
        "probabilities": calibrated,
        "labels": labels,
        "test_idx": split.test,
    }


def evaluate_protocol(detector_factory, dataset: Dataset, seed: int = SEED) -> dict:
    """Run the frozen protocol: nested grouped CV or holdout, then pool test scores."""
    from bench.splits import protocol_splits

    splits = protocol_splits(dataset, seed=seed)
    fold_rows = []
    pooled_p: list[np.ndarray] = []
    pooled_y: list[np.ndarray] = []
    pooled_idx: list[np.ndarray] = []
    for split in splits:
        detector = detector_factory()
        row = evaluate_protocol_split(detector, dataset, split)
        pooled_p.append(row["probabilities"])
        pooled_y.append(row["labels"])
        pooled_idx.append(row["test_idx"])
        fold_rows.append({k: v for k, v in row.items() if k not in {"probabilities", "labels", "test_idx"}})
    y = np.concatenate(pooled_y)
    p = np.concatenate(pooled_p)
    idx = np.concatenate(pooled_idx)
    from bench.evidence import likelihood_ratio, prior_sensitivity

    prevalence = float(dataset.labels.mean())
    mean_lr = float(np.mean(likelihood_ratio(p, prevalence)))
    return {
        "n_splits": len(splits),
        "method": splits[0].method.split(":")[0],
        "n_groups": splits[0].n_groups,
        "metrics": binary_metrics(y, p),
        "selective_risk": selective_risk_curve(y, p),
        "reliability_bins": reliability_bins(y, p),
        "conformal": conformal_sets(y, p),
        "mean_likelihood_ratio": mean_lr,
        "prior_sensitivity": prior_sensitivity(mean_lr),
        "folds": fold_rows,
        "pooled_probabilities": p,
        "pooled_labels": y,
        "pooled_test_idx": idx,
    }


def transport_matrix(detector_factory, dataset: Dataset, axis: str = "domains") -> dict:
    """Train-domain × test-domain metrics for evidence transportability."""
    values = dataset.domains if axis == "domains" else dataset.families
    unique = sorted(set(values))
    X = dataset.features()
    y = dataset.labels
    matrix = []
    for train_value in unique:
        train_mask = np.array([v == train_value for v in values])
        if int(train_mask.sum()) < 8 or len(set(y[train_mask])) < 2:
            continue
        model = detector_factory()
        train_idx = np.where(train_mask)[0]
        try:
            model.fit(dataset, train_idx)

            def predict(idx: np.ndarray, _model=model) -> np.ndarray:
                return _model.predict_proba(dataset, idx)

        except TypeError:
            model.fit(X[train_mask], y[train_mask])

            def predict(idx: np.ndarray, _model=model) -> np.ndarray:
                return _model.predict_proba(X[idx])

        for test_value in unique:
            test_mask = np.array([v == test_value for v in values])
            if int(test_mask.sum()) < 4 or len(set(y[test_mask])) < 2:
                continue
            idx = np.where(test_mask)[0]
            probabilities = np.clip(predict(idx), 1e-6, 1 - 1e-6)
            metrics = binary_metrics(y[test_mask], probabilities)
            matrix.append(
                {
                    "train": train_value,
                    "test": test_value,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "in_domain": train_value == test_value,
                    **metrics,
                }
            )
    return {"axis": axis, "cells": matrix}


def leave_one_family_out(dataset: Dataset) -> dict:
    """Train source-family geometry on known families; test unseen families.

    Unknown-source rejection uses Mahalanobis distance to the nearest known
    centroid. A held-out family should look more unknown than a seen family.
    """
    from sklearn.covariance import LedoitWolf

    X = dataset.features()
    families = np.array(dataset.families)
    unique = [f for f in sorted(set(dataset.families)) if f != "human"]
    rows = []
    for held in unique:
        known = [f for f in unique if f != held]
        known_mask = np.isin(families, known)
        if int(known_mask.sum()) < 8:
            continue
        centroids = {}
        cov = LedoitWolf().fit(X[known_mask])
        precision = cov.precision_
        for family in known:
            mask = families == family
            centroids[family] = X[mask].mean(axis=0)

        def min_mahal(row: np.ndarray, _centroids=centroids, _precision=precision) -> float:
            return min(float((row - mu) @ _precision @ (row - mu)) for mu in _centroids.values())

        seen_scores = np.array([min_mahal(X[i]) for i in np.where(known_mask)[0]])
        unseen_scores = np.array([min_mahal(X[i]) for i in np.where(families == held)[0]])
        labels = np.concatenate([np.zeros(len(seen_scores)), np.ones(len(unseen_scores))])
        scores = np.concatenate([seen_scores, unseen_scores])
        auroc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
        rows.append(
            {
                "held_out_family": held,
                "n_seen": int(len(seen_scores)),
                "n_unseen": int(len(unseen_scores)),
                "mean_distance_seen": float(seen_scores.mean()),
                "mean_distance_unseen": float(unseen_scores.mean()),
                "unknown_rejection_auroc": auroc,
            }
        )
    if not rows:
        return {"families": [], "mean_unknown_rejection_auroc": float("nan")}
    return {
        "families": rows,
        "mean_unknown_rejection_auroc": float(
            np.nanmean([r["unknown_rejection_auroc"] for r in rows])
        ),
    }
