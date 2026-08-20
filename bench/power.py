"""Analytical power and required-n for calibration metrics.

The frozen protocol gates neural-tier comparisons on an accuracy MDE
(two-proportion test, required n = 3140). Calibration claims need the same
treatment: how large must the evaluation cohort be to detect a Brier or ECE
difference of a given size at alpha = 0.05 with 80% power?

Both estimators are means of bounded per-observation contributions, so a
normal approximation gives a defensible required-n:

    n = (z_{1-alpha/2} + z_{power})^2 * v / delta^2

where v is the variance of the per-observation contribution, estimated from
the scored evaluation cohort itself.

- Brier: contribution c_i = (p_i - y_i)^2; v = Var(c). Exact for the
  one-sample mean; conservative for two-detector comparisons.
- ECE: with fixed bins, ECE = sum_b w_b |acc_b - conf_b|. Each bin deviation
  is a difference of two means whose variance is approximately
  (conf_b (1 - conf_b) + acc_b (1 - acc_b)) / n_b. Treating the per-bin
  deviations as independent and propagating through the absolute value with
  the half-normal variance factor (1 - 2/pi) gives a per-observation
  variance proxy v = sum_b w_b * s_b * (1 - 2/pi) with
  s_b = conf_b (1 - conf_b) + acc_b (1 - acc_b), so Var(ECE) ~ v / n.
  This ignores bin-boundary estimation and cross-bin covariance; it is an
  approximation, reported as such.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

ALPHA = 0.05
TARGET_POWER = 0.8
MDE_GRID = (0.005, 0.01, 0.02, 0.05)


def _z(value: float) -> float:
    return float(NormalDist().inv_cdf(value))


def required_n(
    per_observation_variance: float, mde: float, alpha: float = ALPHA, power: float = TARGET_POWER
) -> int:
    """Normal-approximation required n to detect `mde` in a mean-type metric."""
    if per_observation_variance <= 0 or mde <= 0:
        return 0
    z = _z(1 - alpha / 2) + _z(power)
    return int(np.ceil(z * z * per_observation_variance / (mde * mde)))


def brier_contribution_variance(labels: np.ndarray, probabilities: np.ndarray) -> float:
    contributions = (np.asarray(probabilities, dtype=float) - np.asarray(labels, dtype=float)) ** 2
    return float(contributions.var(ddof=1)) if len(contributions) > 1 else 0.0


def ece_variance_proxy(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    """Per-observation variance proxy for the fixed-bin ECE estimator."""
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    proxy = 0.0
    n = len(y)
    for index in range(bins):
        mask = (p >= edges[index]) & (p < edges[index + 1])
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        w_b = n_b / n
        conf_b = float(p[mask].mean())
        acc_b = float(y[mask].mean())
        s_b = conf_b * (1 - conf_b) + acc_b * (1 - acc_b)
        proxy += w_b * s_b
    return float(proxy * (1 - 2 / np.pi))


def calibration_power(
    labels: np.ndarray,
    probabilities: np.ndarray,
    mdes: tuple[float, ...] = MDE_GRID,
    alpha: float = ALPHA,
    power: float = TARGET_POWER,
) -> dict:
    """Required-n grid for Brier and ECE on an already-scored cohort."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    n = len(y)
    brier_v = brier_contribution_variance(y, p)
    ece_v = ece_variance_proxy(y, p)
    return {
        "n_evaluated": int(n),
        "alpha": alpha,
        "target_power": power,
        "brier": {
            "observed": float(np.mean((p - y) ** 2)),
            "per_observation_variance": brier_v,
            "required_n": {str(mde): required_n(brier_v, mde, alpha, power) for mde in mdes},
        },
        "ece": {
            "per_observation_variance_proxy": ece_v,
            "required_n": {str(mde): required_n(ece_v, mde, alpha, power) for mde in mdes},
            "approximation": (
                "Fixed-bin ECE variance propagated from per-bin binomial "
                "variances with the half-normal factor; ignores bin-boundary "
                "estimation and cross-bin covariance."
            ),
        },
        "method": (
            "Normal approximation n = (z_{1-alpha/2} + z_{power})^2 v / delta^2 "
            "with v estimated from the scored cohort; complements the protocol's "
            "accuracy gate (required n = 3140 at MDE 0.05)."
        ),
    }
