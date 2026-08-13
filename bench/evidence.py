"""Likelihood ratios, priors, and segment-dependence models.

Calibration produces a cohort-conditional probability p. The likelihood
ratio that can be transported to a different prevalence is:

    LR = (p / (1 - p)) * ((1 - π) / π)

where π is the calibration-cohort prevalence. ECE is a diagnostic of p;
it is not applied as a discount to the posterior.
"""

from __future__ import annotations

import math

import numpy as np

from research.protocol import PREVALENCE_GRID

EPS = 1e-6


def clamp_probability(p: float | np.ndarray, eps: float = EPS) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def likelihood_ratio(p_calibrated: float | np.ndarray, prevalence: float = 0.5) -> np.ndarray:
    """Prior-independent LR from a calibrated probability and cohort prevalence."""
    p = clamp_probability(p_calibrated)
    pi = min(max(float(prevalence), EPS), 1.0 - EPS)
    return (p / (1.0 - p)) * ((1.0 - pi) / pi)


def posterior_odds(prior_odds: float, lr: float | np.ndarray) -> np.ndarray:
    return np.asarray(prior_odds, dtype=float) * np.asarray(lr, dtype=float)


def posterior_probability(prior_odds: float, lr: float | np.ndarray) -> np.ndarray:
    odds = posterior_odds(prior_odds, lr)
    return odds / (1.0 + odds)


def prior_odds_from_prevalence(prevalence: float) -> float:
    pi = min(max(float(prevalence), EPS), 1.0 - EPS)
    return pi / (1.0 - pi)


def prior_sensitivity(lr: float, prevalences: list[float] | None = None) -> list[dict]:
    grid = prevalences if prevalences is not None else list(PREVALENCE_GRID)
    rows = []
    for pi in grid:
        p = float(posterior_probability(prior_odds_from_prevalence(pi), lr))
        rows.append({"prevalence": pi, "prior_odds": prior_odds_from_prevalence(pi), "posterior": p})
    return rows


def log_likelihood_ratio(p_calibrated: float | np.ndarray, prevalence: float = 0.5) -> np.ndarray:
    return np.log(np.clip(likelihood_ratio(p_calibrated, prevalence), EPS, 1.0 / EPS))


def naive_accumulate(segment_llrs: np.ndarray) -> float:
    """Sum of segment log-likelihood ratios (independence assumption)."""
    return float(np.sum(segment_llrs))


def correlated_shrinkage(segment_llrs: np.ndarray, rho: float | None = None) -> dict:
    """Shrink the summed LLR by the effective sample size under exchangeable correlation.

    If rho is omitted it is estimated from the segment LLRs as the lag-1
    autocorrelation, floored at 0. Independent segments (rho=0) recover the
    naive sum; perfectly correlated segments (rho=1) recover the mean LLR.
    """
    values = np.asarray(segment_llrs, dtype=float)
    n = max(len(values), 1)
    if rho is None:
        if n >= 3:
            centered = values - values.mean()
            denom = float(np.dot(centered, centered))
            rho_hat = float(np.dot(centered[1:], centered[:-1]) / denom) if denom > 0 else 0.0
        else:
            rho_hat = 0.0
        rho_hat = min(max(rho_hat, 0.0), 1.0)
    else:
        rho_hat = min(max(float(rho), 0.0), 1.0)
    n_eff = n / (1.0 + (n - 1.0) * rho_hat)
    total = float(np.sum(values))
    shrunk = total * (n_eff / n)
    return {
        "model": "correlated_shrinkage",
        "rho": rho_hat,
        "n_segments": n,
        "n_effective": n_eff,
        "llr": shrunk,
        "lr": math.exp(shrunk),
    }


def document_level(p_calibrated: float, prevalence: float = 0.5) -> dict:
    lr = float(likelihood_ratio(p_calibrated, prevalence))
    return {
        "model": "document_level",
        "llr": math.log(max(lr, EPS)),
        "lr": lr,
    }


def compare_dependence(
    segment_p: np.ndarray,
    document_p: float,
    prevalence: float = 0.5,
) -> dict:
    """Score the three protocol dependence models on one document."""
    llrs = log_likelihood_ratio(segment_p, prevalence)
    naive = naive_accumulate(llrs)
    correlated = correlated_shrinkage(llrs)
    document = document_level(document_p, prevalence)
    return {
        "naive_sum": {"model": "naive_sum", "llr": naive, "lr": math.exp(naive)},
        "correlated_shrinkage": correlated,
        "document_level": document,
    }
