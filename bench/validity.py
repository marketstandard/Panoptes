"""Validity-first inference primitives for the v2.1 evaluation contract.

These functions repair the evaluation contract before any outcome-bearing
neural run. The single hard rule: **nothing that sets a threshold, a
calibrator, or a nonconformity level may touch final-test labels.** Every
operating quantity is fit on the calibration partition and then *applied*,
frozen, to the untouched test partition.

What lives here:

  * True split (inductive) conformal with class-conditional (Mondrian)
    thresholds fit on calibration only — replaces the old in-sample
    ``conformal_sets(labels, test_probabilities)``.
  * Calibration-only operating thresholds: low-FPR points and
    selective-prediction (abstention) thresholds are derived from calibration
    scores and applied, fixed, to test/transport cells. Test-ranked curves
    remain available elsewhere but are descriptive only.
  * Group (cluster) bootstrap confidence intervals and paired group-bootstrap
    comparisons — resample leakage-control groups / authors / stories, never
    i.i.d. documents.
  * Natural- and standardized-prevalence metric views, and an adaptive
    (equal-mass) ECE with a group-bootstrap interval.

Coverage and calibration guarantees are population-conditional: they hold
under exchangeability with the calibration cohort and are NOT guaranteed after
distribution shift. That caveat is part of every result these functions emit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

SEED = 13
COVERAGE_LEVELS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
FPR_TARGETS = (0.001, 0.01, 0.05)

SHIFT_CAVEAT = (
    "Split-conformal coverage and calibration are guaranteed only under "
    "exchangeability with the calibration cohort; they are not guaranteed "
    "after distribution shift."
)


# --- split (inductive) conformal, calibration-fit ------------------------------


@dataclass(frozen=True)
class ConformalFit:
    """Frozen class-conditional nonconformity thresholds fit on calibration."""

    alpha: float
    threshold_by_class: dict[int, float]
    n_calibration: int
    prevalence: float
    mondrian: bool


def _conformal_quantile(nonconformity: np.ndarray, alpha: float) -> float:
    n = len(nonconformity)
    if n == 0:
        return 1.0
    level = math.ceil((n + 1) * (1 - alpha)) / n
    level = min(max(level, 0.0), 1.0)
    return float(np.quantile(nonconformity, level, method="higher"))


def fit_conformal(
    cal_labels: np.ndarray, cal_probs: np.ndarray, alpha: float = 0.1, mondrian: bool = True
) -> ConformalFit:
    """Fit split-conformal thresholds on the CALIBRATION partition only.

    Nonconformity of a candidate class c for a point scored p=P(AI) is
    ``1 - p`` for c=1 and ``p`` for c=0. With ``mondrian=True`` a separate
    threshold is fit per true class, giving class-conditional (within-class)
    coverage under exchangeability.
    """
    y = np.asarray(cal_labels, dtype=int)
    p = np.asarray(cal_probs, dtype=float)
    threshold_by_class: dict[int, float] = {}
    for cls in (0, 1):
        if mondrian:
            nc = np.where(y == cls, np.where(cls == 1, 1 - p, p), np.nan)
            nc = nc[~np.isnan(nc)]
        else:
            nc = np.where(y == 1, 1 - p, p)
        threshold_by_class[cls] = _conformal_quantile(nc, alpha)
    return ConformalFit(
        alpha=float(alpha),
        threshold_by_class=threshold_by_class,
        n_calibration=int(len(y)),
        prevalence=float(y.mean()) if len(y) else float("nan"),
        mondrian=bool(mondrian),
    )


def apply_conformal(
    test_probs: np.ndarray, test_labels: np.ndarray, fit: ConformalFit
) -> dict:
    """Apply frozen conformal thresholds to untouched test labels."""
    p = np.asarray(test_probs, dtype=float)
    y = np.asarray(test_labels, dtype=int)
    t1 = fit.threshold_by_class[1]
    t0 = fit.threshold_by_class[0]
    set_sizes = []
    covered = []
    per_class_covered: dict[int, list[bool]] = {0: [], 1: []}
    for prob, label in zip(p, y, strict=True):
        included = set()
        if (1 - prob) <= t1:
            included.add(1)
        if prob <= t0:
            included.add(0)
        set_sizes.append(len(included))
        covered.append(label in included)
        per_class_covered[int(label)].append(label in included)
    return {
        "method": "split_conformal_mondrian" if fit.mondrian else "split_conformal_pooled",
        "fit_on": "calibration",
        "alpha": fit.alpha,
        "n_calibration": fit.n_calibration,
        "calibration_prevalence": fit.prevalence,
        "threshold_class_0": t0,
        "threshold_class_1": t1,
        "n_test": int(len(y)),
        "empirical_coverage": float(np.mean(covered)) if covered else float("nan"),
        "mean_set_size": float(np.mean(set_sizes)) if set_sizes else float("nan"),
        "singleton_rate": float(np.mean([s == 1 for s in set_sizes])) if set_sizes else float("nan"),
        "abstention_rate": float(np.mean([s == 2 for s in set_sizes])) if set_sizes else float("nan"),
        "empty_rate": float(np.mean([s == 0 for s in set_sizes])) if set_sizes else float("nan"),
        "coverage_by_class": {
            str(cls): (float(np.mean(vals)) if vals else float("nan"))
            for cls, vals in per_class_covered.items()
        },
        "caveat": SHIFT_CAVEAT,
    }


# --- calibration-only operating thresholds ------------------------------------


def fit_fpr_thresholds(
    cal_labels: np.ndarray, cal_probs: np.ndarray, target_fprs: tuple[float, ...] = FPR_TARGETS
) -> dict[float, float]:
    """Low-FPR operating thresholds derived from CALIBRATION negatives only."""
    y = np.asarray(cal_labels, dtype=int)
    p = np.asarray(cal_probs, dtype=float)
    negatives = p[y == 0]
    thresholds: dict[float, float] = {}
    for fpr in target_fprs:
        if len(negatives) == 0:
            thresholds[float(fpr)] = 1.0
        else:
            thresholds[float(fpr)] = float(np.quantile(negatives, 1 - fpr, method="higher"))
    return thresholds


def tpr_at_fixed_thresholds(
    test_labels: np.ndarray, test_probs: np.ndarray, thresholds: dict[float, float]
) -> dict[str, float]:
    """TPR at thresholds frozen on calibration — applied, never re-derived, on test."""
    y = np.asarray(test_labels, dtype=int)
    p = np.asarray(test_probs, dtype=float)
    positives = p[y == 1]
    out: dict[str, float] = {}
    for fpr, threshold in thresholds.items():
        out[f"tpr_at_{_fpr_label(fpr)}fpr"] = (
            float(np.mean(positives >= threshold)) if len(positives) else float("nan")
        )
    return out


def _fpr_label(fpr: float) -> str:
    # 0.001 -> "0.1", 0.01 -> "1", 0.05 -> "5" (percent)
    percent = fpr * 100
    return f"{percent:g}"


def fit_selective_thresholds(
    cal_probs: np.ndarray, coverages: tuple[float, ...] = COVERAGE_LEVELS
) -> dict[float, float]:
    """Confidence thresholds achieving each target coverage on CALIBRATION.

    Confidence is max(p, 1-p). The threshold for target coverage c is the
    (1-c) quantile of calibration confidence, so ~c of calibration points are
    kept. Applied frozen to test, the realized coverage is reported.
    """
    p = np.asarray(cal_probs, dtype=float)
    confidence = np.maximum(p, 1 - p)
    thresholds: dict[float, float] = {}
    for cov in coverages:
        if cov >= 1.0:
            thresholds[float(cov)] = 0.0
        else:
            thresholds[float(cov)] = float(np.quantile(confidence, 1 - cov, method="higher"))
    return thresholds


def apply_selective_thresholds(
    test_labels: np.ndarray, test_probs: np.ndarray, thresholds: dict[float, float]
) -> list[dict]:
    """Selective risk at coverage thresholds frozen on calibration."""
    y = np.asarray(test_labels, dtype=int)
    p = np.asarray(test_probs, dtype=float)
    confidence = np.maximum(p, 1 - p)
    n = len(y)
    rows = []
    for cov, threshold in thresholds.items():
        keep = confidence >= threshold if cov < 1.0 else np.ones(n, dtype=bool)
        if not np.any(keep):
            continue
        pred = p[keep] >= 0.5
        risk = float(np.mean(pred != y[keep]))
        rows.append(
            {
                "coverage_target": float(cov),
                "confidence_threshold": float(threshold),
                "empirical_coverage": float(np.mean(keep)),
                "n_kept": int(keep.sum()),
                "selective_risk": risk,
                "accuracy": 1.0 - risk,
                "brier": float(brier_score_loss(y[keep], p[keep])),
                "threshold_fit_on": "calibration",
            }
        )
    return rows


# --- group (cluster) bootstrap -------------------------------------------------


def _group_row_index(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    unique, inverse = np.unique(groups, return_inverse=True)
    rows = [np.where(inverse == i)[0] for i in range(len(unique))]
    return unique, rows


def group_bootstrap_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    groups: np.ndarray | list[str],
    metric_fn,
    n_boot: int = 1000,
    seed: int = SEED,
    level: float = 0.95,
) -> dict:
    """Cluster-bootstrap CI for a metric, resampling whole groups.

    Groups (near-duplicate clusters, authors, stories) are resampled with
    replacement; every row in a resampled group enters the replicate. This is
    the honest interval when documents are correlated within a group — i.i.d.
    document bootstrap understates variance and is not used here.
    """
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probs, dtype=float)
    groups = np.asarray(groups)
    unique, rows = _group_row_index(groups)
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(unique), len(unique))
        idx = np.concatenate([rows[c] for c in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        value = float(metric_fn(y[idx], p[idx]))
        if value == value and abs(value) != float("inf"):
            stats.append(value)
    if not stats:
        return {"point": float("nan"), "ci": [float("nan"), float("nan")], "n_boot": 0}
    alpha = 1 - level
    point = float(metric_fn(y, p))
    return {
        "point": point,
        "ci": [float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))],
        "n_boot": len(stats),
        "n_groups": int(len(unique)),
        "resample_unit": "group",
    }


def paired_group_bootstrap(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    groups: np.ndarray | list[str],
    metric_fn,
    n_boot: int = 1000,
    seed: int = SEED,
    level: float = 0.95,
) -> dict:
    """Paired cluster-bootstrap comparison of two scorers on the same groups.

    Each replicate resamples whole groups and computes metric(a) - metric(b),
    so the interval and sign-based p-value reflect the paired, group-correlated
    design rather than an i.i.d. approximation.
    """
    y = np.asarray(labels, dtype=int)
    pa = np.asarray(probs_a, dtype=float)
    pb = np.asarray(probs_b, dtype=float)
    groups = np.asarray(groups)
    unique, rows = _group_row_index(groups)
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(unique), len(unique))
        idx = np.concatenate([rows[c] for c in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        da = float(metric_fn(y[idx], pa[idx]))
        db = float(metric_fn(y[idx], pb[idx]))
        diff = da - db
        if diff == diff and abs(diff) != float("inf"):
            diffs.append(diff)
    if not diffs:
        return {"diff": float("nan"), "ci": [float("nan"), float("nan")], "p_value": float("nan"), "n_boot": 0}
    diffs_arr = np.asarray(diffs)
    alpha = 1 - level
    point = float(metric_fn(y, pa) - metric_fn(y, pb))
    p_le = float(np.mean(diffs_arr <= 0))
    p_ge = float(np.mean(diffs_arr >= 0))
    return {
        "diff": point,
        "ci": [float(np.percentile(diffs_arr, 100 * alpha / 2)), float(np.percentile(diffs_arr, 100 * (1 - alpha / 2)))],
        "p_value": float(min(1.0, 2 * min(p_le, p_ge))),
        "n_boot": len(diffs),
        "n_groups": int(len(unique)),
        "resample_unit": "group",
    }


# --- prevalence views and adaptive ECE ----------------------------------------


def standardized_prevalence_metrics(
    labels: np.ndarray, probs: np.ndarray, target_prevalence: float = 0.5
) -> dict:
    """Natural- and standardized-prevalence views of prevalence-dependent metrics.

    AUROC is rank-based and prevalence-invariant; Brier, log loss, and accuracy
    are not. Class reweighting to a standard prevalence (default 0.5) gives a
    view that is comparable across cohorts with different base rates.
    """
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probs, dtype=float)
    natural = float(y.mean()) if len(y) else float("nan")
    w = np.where(y == 1, target_prevalence / max(natural, 1e-9), (1 - target_prevalence) / max(1 - natural, 1e-9))

    def _views(sample_weight=None) -> dict:
        pred = p >= 0.5
        return {
            "brier": float(brier_score_loss(y, p, sample_weight=sample_weight)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), sample_weight=sample_weight, labels=[0, 1])),
            "accuracy": float(np.average((pred == y).astype(float), weights=sample_weight)),
        }

    return {
        "natural_prevalence": natural,
        "target_prevalence": float(target_prevalence),
        "auroc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan"),
        "natural": _views(),
        "standardized": _views(sample_weight=w),
    }


def adaptive_ece(
    labels: np.ndarray,
    probs: np.ndarray,
    groups: np.ndarray | list[str] | None = None,
    bins: int = 10,
    n_boot: int = 0,
    seed: int = SEED,
) -> dict:
    """Adaptive (equal-mass) ECE, optionally with a group-bootstrap interval.

    Equal-mass binning sizes each bin to hold ~n/bins points, which is more
    stable than fixed-width binning when scores are concentrated. When
    ``groups`` and ``n_boot`` are supplied, a cluster-bootstrap 95% interval is
    attached so the ECE is reported with its uncertainty rather than as a bare
    point estimate.
    """
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probs, dtype=float)
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    sorted_y = y[order]
    n = len(p)
    ece = 0.0
    for b in range(bins):
        lo = b * n // bins
        hi = (b + 1) * n // bins
        if hi <= lo:
            continue
        ece += (hi - lo) / n * abs(float(sorted_p[lo:hi].mean()) - float(sorted_y[lo:hi].mean()))
    result: dict = {"adaptive_ece": float(ece), "bins": bins, "binning": "equal_mass"}
    if groups is not None and n_boot > 0:
        def _ece(ylabels, yprobs):
            return adaptive_ece(ylabels, yprobs, bins=bins)["adaptive_ece"]

        boot = group_bootstrap_ci(y, p, np.asarray(groups), _ece, n_boot=n_boot, seed=seed)
        result["ci"] = boot["ci"]
        result["resample_unit"] = "group"
    return result
