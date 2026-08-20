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

from bench import validity
from bench.datasets import Dataset, grouped_splits
from bench.protocol import COVERAGE_LEVELS

SEED = 13


def _valid_operating_block(
    cal_labels: np.ndarray,
    cal_probs: np.ndarray,
    test_labels: np.ndarray,
    test_probs: np.ndarray,
    test_groups,
    alpha: float = 0.1,
) -> dict:
    """Calibration-fit operating quantities applied, frozen, to untouched test.

    Every threshold and the conformal level are derived from the calibration
    partition only; the test partition is never used to fit them. This is the
    v2.1 repair over the old in-sample (test-ranked) conformal/selective/FPR
    computations, which are retained separately and labeled descriptive.
    """
    cal_labels = np.asarray(cal_labels, dtype=int)
    test_labels = np.asarray(test_labels, dtype=int)
    conformal_fit = validity.fit_conformal(cal_labels, cal_probs, alpha=alpha)
    fpr_thresholds = validity.fit_fpr_thresholds(cal_labels, cal_probs)
    selective_thresholds = validity.fit_selective_thresholds(cal_probs)
    auroc_boot = validity.group_bootstrap_ci(
        test_labels,
        test_probs,
        np.asarray(list(test_groups)),
        lambda y, p: float(roc_auc_score(y, p)),
        n_boot=1000,
        seed=SEED,
    )
    return {
        "conformal": validity.apply_conformal(test_probs, test_labels, conformal_fit),
        "operating_points": validity.tpr_at_fixed_thresholds(test_labels, test_probs, fpr_thresholds),
        "operating_points_fit_on": "calibration",
        "selective_risk": validity.apply_selective_thresholds(test_labels, test_probs, selective_thresholds),
        "auroc_group_bootstrap": auroc_boot,
        "adaptive_ece": validity.adaptive_ece(
            test_labels, test_probs, groups=np.asarray(list(test_groups)), n_boot=1000, seed=SEED
        ),
        "prevalence": validity.standardized_prevalence_metrics(test_labels, test_probs),
    }


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
    from bench.methodology import logistic_irls

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

    This is REPRESENTATION transport: the scorer AND the calibrator are fit on
    the source cohort (calibration on the source calibration partition only)
    and then applied, frozen, to the untouched target cohort. Source-calibration
    and target prevalences are reported separately rather than silently mixed.
    Operating thresholds and conformal levels are fit on the source calibration
    partition and applied to the target; coverage is not guaranteed after shift.
    """
    from bench.splits import protocol_splits

    split = protocol_splits(train_dataset, seed=seed)[0]
    detector = detector_factory()
    detector.fit(train_dataset, split.train)
    raw_cal = detector.predict_proba(train_dataset, split.calibration)
    cal_labels = train_dataset.labels[split.calibration]
    calibrator = fit_isotonic(raw_cal, cal_labels)
    all_idx = np.arange(len(test_dataset))
    raw_test = detector.predict_proba(test_dataset, all_idx)
    if calibrator is None:
        calibrated = np.clip(raw_test, 1e-6, 1 - 1e-6)
        calibrated_cal = np.clip(raw_cal, 1e-6, 1 - 1e-6)
        calibration_applied = False
    else:
        calibrated = np.clip(calibrator.predict(raw_test), 1e-6, 1 - 1e-6)
        calibrated_cal = np.clip(calibrator.predict(raw_cal), 1e-6, 1 - 1e-6)
        calibration_applied = True
    y = test_dataset.labels
    metrics = binary_metrics(y, calibrated)
    valid = _valid_operating_block(cal_labels, calibrated_cal, y, calibrated, test_dataset.groups)
    return {
        "kind": "representation_transport",
        "train_dataset": train_dataset.provenance,
        "test_dataset": test_dataset.provenance,
        "n_train": int(len(split.train)),
        "n_calibration": int(len(split.calibration)),
        "n_test": int(len(all_idx)),
        "calibration_applied": calibration_applied,
        "calibration_fit_on": "source_calibration",
        "source_calibration_prevalence": float(cal_labels.mean()),
        "target_prevalence": float(y.mean()),
        "detector": getattr(detector, "name", type(detector).__name__),
        **metrics,
        "selective_risk": valid["selective_risk"],
        "operating_points": valid["operating_points"],
        "conformal": valid["conformal"],
        "auroc_group_bootstrap": valid["auroc_group_bootstrap"],
        "prevalence_views": valid["prevalence"],
    }


def calibration_transfer(
    train_dataset: Dataset,
    target_dataset: Dataset,
    detector_factory,
    seed: int = SEED,
) -> dict:
    """CALIBRATION transfer: freeze the scorer, re-fit only the calibrator on the target.

    The scorer is fit on the source cohort's train partition and frozen. Only
    the isotonic calibrator is fit — on the TARGET cohort's calibration
    partition — then applied to the target's untouched test. Comparing this to
    representation transport (source-fit calibrator) isolates how much of a
    transport gap is calibration shift versus representation shift. Source and
    target prevalences are reported separately.
    """
    from bench.splits import protocol_splits

    src_split = protocol_splits(train_dataset, seed=seed)[0]
    detector = detector_factory()
    detector.fit(train_dataset, src_split.train)
    src_cal_prev = float(train_dataset.labels[src_split.calibration].mean())

    tgt_split = protocol_splits(target_dataset, seed=seed)[0]
    raw_cal = detector.predict_proba(target_dataset, tgt_split.calibration)
    cal_labels = target_dataset.labels[tgt_split.calibration]
    calibrator = fit_isotonic(raw_cal, cal_labels)
    raw_test = detector.predict_proba(target_dataset, tgt_split.test)
    y = target_dataset.labels[tgt_split.test]
    if calibrator is None:
        calibrated = np.clip(raw_test, 1e-6, 1 - 1e-6)
        calibrated_cal = np.clip(raw_cal, 1e-6, 1 - 1e-6)
        calibration_applied = False
    else:
        calibrated = np.clip(calibrator.predict(raw_test), 1e-6, 1 - 1e-6)
        calibrated_cal = np.clip(calibrator.predict(raw_cal), 1e-6, 1 - 1e-6)
        calibration_applied = True
    test_groups = [target_dataset.groups[int(i)] for i in tgt_split.test]
    metrics = binary_metrics(y, calibrated)
    valid = _valid_operating_block(cal_labels, calibrated_cal, y, calibrated, test_groups)
    return {
        "kind": "calibration_transfer",
        "train_dataset": train_dataset.provenance,
        "target_dataset": target_dataset.provenance,
        "n_source_train": int(len(src_split.train)),
        "n_target_calibration": int(len(tgt_split.calibration)),
        "n_target_test": int(len(tgt_split.test)),
        "calibration_applied": calibration_applied,
        "calibration_fit_on": "target_calibration",
        "source_calibration_prevalence": src_cal_prev,
        "target_calibration_prevalence": float(cal_labels.mean()),
        "target_test_prevalence": float(y.mean()),
        "detector": getattr(detector, "name", type(detector).__name__),
        **metrics,
        "selective_risk": valid["selective_risk"],
        "operating_points": valid["operating_points"],
        "conformal": valid["conformal"],
        "auroc_group_bootstrap": valid["auroc_group_bootstrap"],
        "prevalence_views": valid["prevalence"],
    }


def fit_calibrate_score(
    detector,
    train_dataset: Dataset,
    calibration_dataset: Dataset,
    test_dataset: Dataset,
) -> dict:
    """Fit on an explicit train set, calibrate on an explicit calibration set,
    and score an untouched test set.

    This is the primitive for cohorts with publisher-defined partitions (MAGE,
    CoAuthor): instead of re-splitting one dataset, the three partitions are
    supplied directly. Calibration is fit on the calibration set only; the test
    set is never used for fitting or tuning.
    """
    detector.fit(train_dataset, np.arange(len(train_dataset)))
    raw_cal = detector.predict_proba(calibration_dataset, np.arange(len(calibration_dataset)))
    calibrator = fit_isotonic(raw_cal, calibration_dataset.labels)
    raw_test = detector.predict_proba(test_dataset, np.arange(len(test_dataset)))
    if calibrator is None:
        calibrated = np.clip(raw_test, 1e-6, 1 - 1e-6)
        calibrated_cal = np.clip(raw_cal, 1e-6, 1 - 1e-6)
        calibration_applied = False
    else:
        calibrated = np.clip(calibrator.predict(raw_test), 1e-6, 1 - 1e-6)
        calibrated_cal = np.clip(calibrator.predict(raw_cal), 1e-6, 1 - 1e-6)
        calibration_applied = True
    y = test_dataset.labels
    valid = _valid_operating_block(
        calibration_dataset.labels, calibrated_cal, y, calibrated, test_dataset.groups
    )
    return {
        "train_dataset": train_dataset.provenance,
        "calibration_dataset": calibration_dataset.provenance,
        "test_dataset": test_dataset.provenance,
        "n_train": int(len(train_dataset)),
        "n_calibration": int(len(calibration_dataset)),
        "n_test": int(len(test_dataset)),
        "calibration_applied": calibration_applied,
        "detector": getattr(detector, "name", type(detector).__name__),
        **binary_metrics(y, calibrated),
        "selective_risk": valid["selective_risk"],
        "selective_risk_descriptive": selective_risk_curve(y, calibrated),
        "operating_points": valid["operating_points"],
        "operating_points_fit_on": "calibration",
        "conformal": valid["conformal"],
        "auroc_group_bootstrap": valid["auroc_group_bootstrap"],
        "adaptive_ece": valid["adaptive_ece"],
        "prevalence_views": valid["prevalence"],
        "raw_metrics": binary_metrics(y, np.clip(raw_test, 1e-6, 1 - 1e-6)),
        "probabilities": calibrated,
        "labels": y,
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
    """Fit on train, calibrate on calibration, score untouched test.

    v2.1: conformal levels, low-FPR operating points, and selective-prediction
    thresholds are fit on the CALIBRATION partition and applied frozen to test.
    The test-ranked ``metrics``/``raw_metrics`` blocks are retained but labeled
    descriptive; the headline operating points come from the calibration-fit
    block. Uncertainty uses the group bootstrap, not i.i.d. document resampling.
    """
    from bench.evidence import likelihood_ratio, prior_sensitivity

    detector.fit(dataset, split.train)
    raw_cal = detector.predict_proba(dataset, split.calibration)
    cal_labels = dataset.labels[split.calibration]
    calibrator = fit_isotonic(raw_cal, cal_labels)
    raw_test = detector.predict_proba(dataset, split.test)
    if calibrator is None:
        calibrated = np.clip(raw_test, 1e-6, 1 - 1e-6)
        calibrated_cal = np.clip(raw_cal, 1e-6, 1 - 1e-6)
        calibration_applied = False
    else:
        calibrated = np.clip(calibrator.predict(raw_test), 1e-6, 1 - 1e-6)
        calibrated_cal = np.clip(calibrator.predict(raw_cal), 1e-6, 1 - 1e-6)
        calibration_applied = True
    labels = dataset.labels[split.test]
    prevalence = float(cal_labels.mean())
    metrics = binary_metrics(labels, calibrated)
    mean_lr = float(np.mean(likelihood_ratio(calibrated, prevalence)))
    test_groups = [dataset.groups[int(i)] for i in split.test]
    valid = _valid_operating_block(cal_labels, calibrated_cal, labels, calibrated, test_groups)
    return {
        "method": split.method,
        "n_train": int(len(split.train)),
        "n_calibration": int(len(split.calibration)),
        "n_test": int(len(split.test)),
        "cohort_prevalence": prevalence,
        "calibration_applied": calibration_applied,
        "metrics": metrics,
        "metrics_note": "test-ranked and descriptive; headline operating points are calibration-fit",
        "selective_risk": valid["selective_risk"],
        "selective_risk_descriptive": selective_risk_curve(labels, calibrated),
        "operating_points": valid["operating_points"],
        "operating_points_fit_on": "calibration",
        "reliability_bins": reliability_bins(labels, calibrated),
        "conformal": valid["conformal"],
        "auroc_group_bootstrap": valid["auroc_group_bootstrap"],
        "adaptive_ece": valid["adaptive_ece"],
        "prevalence_views": valid["prevalence"],
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
    pooled_groups = np.array([dataset.groups[int(i)] for i in idx])
    # Pooled conformal is fit on the pooled OUT-OF-FOLD test scores (each point
    # scored by a model that never trained on it), so it is marginally valid
    # (CV+-style) but not a per-cohort guarantee; per-fold calibration-fit
    # conformal is in each fold's row. Uncertainty uses the group bootstrap.
    pooled_conformal = validity.apply_conformal(p, y, validity.fit_conformal(y, p, alpha=0.1))
    pooled_conformal["method"] = "split_conformal_mondrian_pooled_out_of_fold"
    return {
        "n_splits": len(splits),
        "method": splits[0].method.split(":")[0],
        "n_groups": splits[0].n_groups,
        "metrics": binary_metrics(y, p),
        "selective_risk": selective_risk_curve(y, p),
        "selective_risk_note": "test-ranked over pooled out-of-fold scores; descriptive",
        "reliability_bins": reliability_bins(y, p),
        "conformal": pooled_conformal,
        "auroc_group_bootstrap": validity.group_bootstrap_ci(
            y, p, pooled_groups, lambda yy, pp: float(roc_auc_score(yy, pp)), n_boot=1000, seed=seed
        ),
        "adaptive_ece": validity.adaptive_ece(y, p, groups=pooled_groups, n_boot=1000, seed=seed),
        "prevalence_views": validity.standardized_prevalence_metrics(y, p),
        "mean_likelihood_ratio": mean_lr,
        "prior_sensitivity": prior_sensitivity(mean_lr),
        "folds": fold_rows,
        "pooled_probabilities": p,
        "pooled_labels": y,
        "pooled_test_idx": idx,
    }


def _group_holdout_indices(
    groups: np.ndarray, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic group-disjoint local train/test split (positions, not labels)."""
    unique = sorted(set(groups.tolist()))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    n_test = max(1, int(round(len(unique) * test_fraction)))
    test_groups = {unique[i] for i in order[:n_test]}
    test_pos = np.array([i for i, g in enumerate(groups) if g in test_groups], dtype=int)
    train_pos = np.array([i for i, g in enumerate(groups) if g not in test_groups], dtype=int)
    return train_pos, test_pos


def transport_matrix(
    detector_factory, dataset: Dataset, axis: str = "domains", seed: int = SEED
) -> dict:
    """Train-domain × test-domain metrics for evidence transportability.

    v2.1 repairs the diagonal: in-domain (train == test) cells no longer use
    train-on-all/in-sample scores. Each diagonal cell does a within-domain
    group-disjoint holdout — fit on one set of groups, score a held-out set.
    Off-diagonal cells exclude from training any leakage-control group that
    appears in the target cell, so cross-domain transport is not contaminated
    by shared stories/prompts/clusters.
    """
    values = np.array(dataset.domains if axis == "domains" else dataset.families)
    groups = np.array(dataset.groups)
    unique = sorted(set(values.tolist()))
    y = dataset.labels
    matrix = []
    for train_value in unique:
        for test_value in unique:
            test_mask = values == test_value
            if int(test_mask.sum()) < 4 or len(set(y[test_mask].tolist())) < 2:
                continue
            if train_value == test_value:
                domain_idx = np.where(test_mask)[0]
                train_pos, test_pos = _group_holdout_indices(
                    groups[domain_idx], test_fraction=0.4, seed=seed
                )
                train_idx = domain_idx[train_pos]
                test_idx = domain_idx[test_pos]
                if len(train_idx) < 8 or len(test_idx) < 4:
                    continue
            else:
                target_groups = set(groups[test_mask].tolist())
                train_mask = (values == train_value) & ~np.isin(groups, list(target_groups))
                if int(train_mask.sum()) < 8:
                    continue
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
            if len(set(y[train_idx].tolist())) < 2:
                continue
            model = detector_factory()
            model.fit(dataset, train_idx)
            probabilities = np.clip(model.predict_proba(dataset, test_idx), 1e-6, 1 - 1e-6)
            metrics = binary_metrics(y[test_idx], probabilities)
            matrix.append(
                {
                    "train": train_value,
                    "test": test_value,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "in_domain": train_value == test_value,
                    "diagonal_held_out": train_value == test_value,
                    **metrics,
                }
            )
    return {"axis": axis, "cells": matrix, "diagonal": "held_out_group_disjoint"}


def leave_one_family_out(dataset: Dataset) -> dict:
    """Train source-family geometry on known families; test unseen families.

    Unknown-source rejection uses Mahalanobis distance to the nearest known
    centroid. A held-out family should look more unknown than a seen family.

    v2.1 strictness: when a family is held out, every training row that shares
    a leakage-control group (story / prompt / near-duplicate cluster) with the
    held-out family is also excluded, so the open-set test is not contaminated
    by shared sources between train and the held-out generator.
    """
    from sklearn.covariance import LedoitWolf

    X = dataset.features()
    families = np.array(dataset.families)
    groups = np.array(dataset.groups)
    unique = [f for f in sorted(set(dataset.families)) if f != "human"]
    rows = []
    for held in unique:
        known = [f for f in unique if f != held]
        held_groups = set(groups[families == held].tolist())
        # Exclude training rows whose group also appears in the held-out family.
        known_mask = np.isin(families, known) & ~np.isin(groups, list(held_groups))
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
