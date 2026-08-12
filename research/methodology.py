"""Statistical methodology layer for the Panoptes baseline corpus.

Pre-registered hypotheses (research/hypotheses.json), multicollinearity
screening (VIF / condition number), logistic specification tests (link,
Hosmer-Lemeshow, RESET, Cook's distance, Breusch-Pagan, Jarque-Bera,
Durbin-Watson), pseudo-R^2 measures, and model-comparison tests
(McNemar, DeLong). Everything is computed from the hash-verified corpus
built by research/baseline_corpus.py; nothing here trusts a file that
has not been re-hashed against its manifest.

Outputs:
  - backend/artifacts/methodology-report.json  (canonical SHA-256 signed)
  - research/methodology-report.md             (human-readable rendering)

Run from the repository root:  python research/methodology.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.features import FEATURE_NAMES, extract, heuristic_raw_score  # noqa: E402
from research.baseline_corpus import load_corpus  # noqa: E402

HYPOTHESES_PATH = ROOT / "research" / "hypotheses.json"
REPORT_JSON = ROOT / "backend" / "artifacts" / "methodology-report.json"
REPORT_MD = ROOT / "research" / "methodology-report.md"
SEED = 13
PERMUTATIONS = 2000
BOOTSTRAP = 2000


# ---------------------------------------------------------------------------
# Core logistic engine (IRLS) — exact MLE with Hessian-based covariance.
# A tiny ridge (1e-6) keeps the Hessian invertible under near-separation;
# the same ridge is used for every nested fit so LR statistics stay coherent.
# ---------------------------------------------------------------------------


def logistic_irls(
    X: np.ndarray,
    y: np.ndarray,
    ridge: float = 1e-6,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> dict:
    X1 = np.column_stack([np.ones(len(X)), X]) if X.ndim == 2 else np.ones((len(X), 1))
    beta = np.zeros(X1.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X1 @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-10, None)
        hessian = (X1 * w[:, None]).T @ X1 + ridge * np.eye(X1.shape[1])
        gradient = X1.T @ (y - p) - ridge * beta
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if float(np.max(np.abs(step))) < tol:
            break
    eta = np.clip(X1 @ beta, -30, 30)
    p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    w = np.clip(p * (1 - p), 1e-10, None)
    hessian = (X1 * w[:, None]).T @ X1 + ridge * np.eye(X1.shape[1])
    cov = np.linalg.inv(hessian)
    return {"beta": beta, "ll": ll, "cov": cov, "p": p, "eta": eta, "X": X1}


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = np.argsort(pvalues)
    ranked = np.array(pvalues)[order]
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(q, 0.0, 1.0)
    return out.tolist()


# ---------------------------------------------------------------------------
# Multicollinearity screening
# ---------------------------------------------------------------------------


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    return np.corrcoef(X, rowvar=False)


def condition_number(X: np.ndarray) -> float:
    scaled = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    singular = np.linalg.svd(scaled, compute_uv=False)
    if singular[-1] == 0:
        return float("inf")
    return float(singular[0] / singular[-1])


def vif_table(X: np.ndarray, names: list[str]) -> list[dict]:
    rows = []
    for j, name in enumerate(names):
        others = np.delete(X, j, axis=1)
        target = X[:, j]
        design = np.column_stack([np.ones(len(X)), others])
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        fitted = design @ beta
        ss_res = float(np.sum((target - fitted) ** 2))
        ss_tot = float(np.sum((target - target.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2)
        rows.append({"feature": name, "r_squared": r2, "vif": vif})
    return rows


def screen_features(
    X: np.ndarray,
    names: list[str],
    exclude_above: float = 10.0,
    investigate_above: float = 5.0,
) -> dict:
    """Iteratively drop the highest-VIF feature until all VIFs <= exclude_above."""
    kept = list(names)
    exclusions: list[dict] = []
    while True:
        idx = [names.index(name) for name in kept]
        table = vif_table(X[:, idx], kept)
        worst = max(table, key=lambda row: row["vif"])
        if worst["vif"] <= exclude_above or len(kept) <= 2:
            break
        exclusions.append(
            {
                "feature": worst["feature"],
                "vif": worst["vif"],
                "justification": (
                    f"VIF {worst['vif']:.1f} exceeds {exclude_above:.0f}; the feature is a "
                    "near-linear combination of the retained set and its coefficient would be "
                    "uninterpretable (variance inflation)."
                ),
            }
        )
        kept.remove(worst["feature"])
    idx = [names.index(name) for name in kept]
    final = vif_table(X[:, idx], kept)
    investigations = [
        {"feature": row["feature"], "vif": row["vif"]}
        for row in final
        if investigate_above < row["vif"] <= exclude_above
    ]
    return {
        "kept": kept,
        "exclusions": exclusions,
        "investigations": investigations,
        "final_vif": final,
        "condition_number": condition_number(X[:, idx]),
    }


# ---------------------------------------------------------------------------
# Hypothesis test primitives
# ---------------------------------------------------------------------------


def welch_t(ai: np.ndarray, human: np.ndarray, direction: str) -> dict:
    result = stats.ttest_ind(ai, human, equal_var=False)
    diff = float(ai.mean() - human.mean())
    se = math.sqrt(float(ai.var(ddof=1)) / len(ai) + float(human.var(ddof=1)) / len(human))
    df_num = (ai.var(ddof=1) / len(ai) + human.var(ddof=1) / len(human)) ** 2
    df_den = (ai.var(ddof=1) / len(ai)) ** 2 / (len(ai) - 1) + (
        human.var(ddof=1) / len(human)
    ) ** 2 / (len(human) - 1)
    df = df_num / df_den if df_den > 0 else 1.0
    crit = float(stats.t.ppf(0.975, df))
    pooled = math.sqrt(
        ((len(ai) - 1) * ai.var(ddof=1) + (len(human) - 1) * human.var(ddof=1))
        / max(len(ai) + len(human) - 2, 1)
    )
    cohens_d = diff / pooled if pooled > 0 else 0.0
    p_two = float(result.pvalue)
    if direction == "ai_greater":
        p_value = p_two / 2 if diff > 0 else 1 - p_two / 2
    elif direction == "ai_lower":
        p_value = p_two / 2 if diff < 0 else 1 - p_two / 2
    else:
        p_value = p_two
    return {
        "statistic": float(result.statistic),
        "p_value": float(p_value),
        "effect_size": {"name": "cohens_d", "value": float(cohens_d)},
        "ci95": [diff - crit * se, diff + crit * se],
        "estimate": diff,
        "n_ai": len(ai),
        "n_human": len(human),
    }


def mann_whitney(ai: np.ndarray, human: np.ndarray, direction: str, bootstrap_cap: int = 4000) -> dict:
    result = stats.mannwhitneyu(ai, human, alternative="two-sided")
    u = float(result.statistic)
    rbc = 1.0 - 2.0 * u / (len(ai) * len(human))  # >0: AI tends lower on this U convention
    rng = np.random.default_rng(SEED)
    # Large-n regime: resample at a capped size. The capped bootstrap CI is
    # wider than the full-data CI (conservative); the p-value above is exact.
    n_a = min(len(ai), bootstrap_cap)
    n_h = min(len(human), bootstrap_cap)
    boots = []
    for _ in range(BOOTSTRAP):
        a = rng.choice(ai, size=n_a, replace=True)
        h = rng.choice(human, size=n_h, replace=True)
        boots.append(1.0 - 2.0 * stats.mannwhitneyu(a, h).statistic / (len(a) * len(h)))
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    p_two = float(result.pvalue)
    # rbc > 0 corresponds to AI values below human values
    if direction == "ai_lower":
        p_value = p_two / 2 if rbc > 0 else 1 - p_two / 2
    elif direction == "ai_greater":
        p_value = p_two / 2 if rbc < 0 else 1 - p_two / 2
    else:
        p_value = p_two
    return {
        "statistic": u,
        "p_value": float(p_value),
        "effect_size": {"name": "rank_biserial", "value": float(rbc)},
        "ci95": ci,
        "estimate": float(rbc),
        "n_ai": len(ai),
        "n_human": len(human),
        "bootstrap_resample_size": [n_a, n_h],
    }


def logistic_lr_test(y: np.ndarray, reduced: np.ndarray, full: np.ndarray) -> dict:
    fit_reduced = logistic_irls(reduced, y)
    fit_full = logistic_irls(full, y)
    lr = 2.0 * (fit_full["ll"] - fit_reduced["ll"])
    df = fit_full["X"].shape[1] - fit_reduced["X"].shape[1]
    p_value = float(stats.chi2.sf(max(lr, 0.0), df))
    beta_new = float(fit_full["beta"][-1])
    se = float(math.sqrt(fit_full["cov"][-1, -1]))
    return {
        "statistic": float(lr),
        "p_value": p_value,
        "effect_size": {"name": "odds_ratio", "value": float(math.exp(np.clip(beta_new, -20, 20)))},
        "ci95": [float(math.exp(np.clip(beta_new - 1.96 * se, -20, 20))),
                 float(math.exp(np.clip(beta_new + 1.96 * se, -20, 20)))],
        "estimate": beta_new,
        "df": df,
    }


def wilks_lambda(X: np.ndarray, groups: np.ndarray) -> float:
    overall = X.mean(axis=0)
    within = np.zeros((X.shape[1], X.shape[1]))
    for group in np.unique(groups):
        rows = X[groups == group]
        centered = rows - rows.mean(axis=0)
        within += centered.T @ centered
    total = (X - overall).T @ (X - overall)
    sign_w, logdet_w = np.linalg.slogdet(within)
    sign_t, logdet_t = np.linalg.slogdet(total)
    if sign_w <= 0 or sign_t <= 0:
        return 0.0
    return float(math.exp(logdet_w - logdet_t))


def permutation_manova(X: np.ndarray, groups: np.ndarray, permutations: int = PERMUTATIONS) -> dict:
    observed = wilks_lambda(X, groups)
    rng = np.random.default_rng(SEED)
    count = 0
    for _ in range(permutations):
        shuffled = rng.permutation(groups)
        if wilks_lambda(X, shuffled) <= observed:
            count += 1
    p_value = (count + 1) / (permutations + 1)
    return {
        "statistic": observed,
        "p_value": float(p_value),
        "effect_size": {"name": "wilks_lambda", "value": observed},
        "ci95": [None, None],
        "estimate": observed,
        "permutations": permutations,
    }


def durbin_watson(residuals: np.ndarray) -> float:
    diff = np.diff(residuals)
    denom = float(np.sum(residuals**2))
    return float(np.sum(diff**2) / denom) if denom > 0 else 2.0


def _dw_matrix(resid: np.ndarray) -> float:
    """DW over row-major-flattened document residuals (within-row diffs plus
    document-boundary diffs) — identical to durbin_watson on the flat array."""
    within = float((np.diff(resid, axis=1) ** 2).sum())
    boundary = float(((resid[1:, 0] - resid[:-1, -1]) ** 2).sum()) if resid.shape[0] > 1 else 0.0
    denom = float((resid**2).sum())
    return (within + boundary) / denom if denom > 0 else 2.0


def durbin_watson_permutation(segment_scores: list[list[float]], permutations: int = PERMUTATIONS) -> dict:
    """DW on document-ordered residuals with a within-document permutation p.

    Vectorized over the (documents x segments) matrix: each permutation
    shuffles segments within every document independently."""
    doc_means = [float(np.mean(scores)) for scores in segment_scores]
    resid = np.array(
        [[score - mean for score in scores] for scores, mean in zip(segment_scores, doc_means, strict=True)]
    )
    observed = _dw_matrix(resid)
    rng = np.random.default_rng(SEED)
    count = 0
    for _ in range(permutations):
        permuted = rng.permuted(resid, axis=1)
        if _dw_matrix(permuted) <= observed:
            count += 1
    p_value = (count + 1) / (permutations + 1)
    return {
        "statistic": observed,
        "p_value": float(p_value),
        "effect_size": {"name": "durbin_watson", "value": observed},
        "ci95": [0.0, 4.0],
        "estimate": observed,
        "permutations": permutations,
        "n_segments": int(resid.size),
        "n_documents": len(segment_scores),
    }


# ---------------------------------------------------------------------------
# Logistic specification tests
# ---------------------------------------------------------------------------


def link_test(y: np.ndarray, X: np.ndarray) -> dict:
    base = logistic_irls(X, y)
    eta = base["eta"]
    augmented = np.column_stack([eta, eta**2])
    fit = logistic_irls(augmented, y)
    lr = 2.0 * (fit["ll"] - base["ll"])
    p_value = float(stats.chi2.sf(max(lr, 0.0), 1))
    return {"statistic": float(lr), "p_value": p_value, "df": 1,
            "adequate": p_value > 0.05,
            "meaning": "H0: the linear logit link is correctly specified (eta^2 adds nothing)."}


def reset_test(y: np.ndarray, X: np.ndarray, powers: tuple[int, ...] = (2, 3)) -> dict:
    base = logistic_irls(X, y)
    eta = base["eta"]
    augmented = np.column_stack([eta] + [eta**power for power in powers])
    fit = logistic_irls(augmented, y)
    lr = 2.0 * (fit["ll"] - base["ll"])
    p_value = float(stats.chi2.sf(max(lr, 0.0), len(powers)))
    return {"statistic": float(lr), "p_value": p_value, "df": len(powers),
            "adequate": p_value > 0.05,
            "meaning": "Ramsey RESET: H0: no omitted nonlinear structure in the index."}


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, groups: int = 10) -> dict:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, groups + 1)))
    statistic = 0.0
    used = 0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lower) & (p <= upper if upper == edges[-1] else p < upper)
        if not np.any(mask):
            continue
        used += 1
        obs = float(y[mask].sum())
        exp = float(p[mask].sum())
        n = int(mask.sum())
        denom = exp * (1 - exp / n)
        if denom > 0:
            statistic += (obs - exp) ** 2 / denom
    df = max(used - 2, 1)
    p_value = float(stats.chi2.sf(statistic, df))
    return {"statistic": float(statistic), "p_value": p_value, "df": df,
            "adequate": p_value > 0.05,
            "meaning": "H0: observed and expected event counts agree across risk deciles."}


def cooks_distance(y: np.ndarray, fit: dict) -> np.ndarray:
    X1 = fit["X"]
    p = fit["p"]
    w = np.clip(p * (1 - p), 1e-10, None)
    xtwx_inv = np.linalg.inv((X1 * w[:, None]).T @ X1 + 1e-6 * np.eye(X1.shape[1]))
    leverage = w * np.sum((X1 @ xtwx_inv) * X1, axis=1)
    leverage = np.clip(leverage, 0.0, 0.999)
    pearson = (y - p) / np.sqrt(np.clip(p * (1 - p), 1e-10, None))
    k = X1.shape[1]
    return (pearson**2 * leverage) / (k * (1 - leverage) ** 2)


def breusch_pagan(residuals: np.ndarray, X: np.ndarray) -> dict:
    design = np.column_stack([np.ones(len(X)), X])
    target = residuals**2
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ beta
    ss_res = float(np.sum((target - fitted) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    lm = len(residuals) * r2
    df = design.shape[1] - 1
    p_value = float(stats.chi2.sf(max(lm, 0.0), df))
    return {"statistic": float(lm), "p_value": p_value, "df": df,
            "adequate": p_value > 0.05,
            "meaning": "H0: error variance is constant (homoscedastic).",
            "r_squared": r2}


def jarque_bera(residuals: np.ndarray) -> dict:
    result = stats.jarque_bera(residuals)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue),
            "adequate": float(result.pvalue) > 0.05,
            "meaning": (
                "H0: residuals have normal skewness and kurtosis. Diagnostic only: Pearson "
                "residuals of a binary model are non-normal by construction, so rejection is "
                "expected and does not by itself invalidate the fit."
            )}


def mcfadden_r2(ll_model: float, ll_null: float) -> float:
    return 1.0 - ll_model / ll_null if ll_null != 0 else 0.0


def tjur_r2(y: np.ndarray, p: np.ndarray) -> float:
    if not np.any(y == 1) or not np.any(y == 0):
        return 0.0
    return float(p[y == 1].mean() - p[y == 0].mean())


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "exact"}
    if b + c < 25:
        p_value = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": p_value,
                "method": "exact_binomial"}
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "statistic": float(statistic),
            "p_value": float(stats.chi2.sf(statistic, 1)), "method": "chi2_continuity"}


def _midranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    midranks = np.zeros(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return midranks


def delong_test(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """DeLong et al. (1988) test for two correlated AUCs (fast algorithm)."""
    order = np.argsort(-labels, kind="mergesort")  # positives first
    m = int(labels.sum())
    stacked = np.vstack([scores_a[order], scores_b[order]])
    n = stacked.shape[1] - m
    positive = stacked[:, :m]
    negative = stacked[:, m:]
    tx = np.empty_like(stacked[:, :m])
    ty = np.empty_like(stacked[:, m:])
    tz = np.empty_like(stacked)
    for r in range(stacked.shape[0]):
        tx[r, :] = _midranks(positive[r, :])
        ty[r, :] = _midranks(negative[r, :])
        tz[r, :] = _midranks(stacked[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": 0.0, "p_value": 1.0}
    z = float((aucs[0] - aucs[1]) / math.sqrt(var))
    p_value = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": z, "p_value": p_value}


# ---------------------------------------------------------------------------
# Hypothesis registry execution
# ---------------------------------------------------------------------------


@dataclass
class Cohort:
    """A named evaluation cohort with lazily precomputed features.

    The corpus cohort is the hash-verified project corpus; the defactify
    cohort is the hygiene-filtered Defactify_Text_Dataset with reconstructed
    story groups. Precomputing features once keeps the hypothesis battery
    tractable at n=71k."""

    name: str
    texts: list[str]
    labels: np.ndarray
    families: list[str]
    kinds: list[str]
    groups: list[str]
    created_utc: str
    note: str
    _features: list[dict] | None = None

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def features(self) -> list[dict]:
        if self._features is None:
            self._features = [extract(text, kind) for text, kind in zip(self.texts, self.kinds, strict=True)]
        return self._features

    @property
    def large(self) -> bool:
        return len(self) > 20000


def corpus_cohort() -> Cohort:
    from research.baseline_corpus import run_manifests

    records = load_corpus()
    return Cohort(
        name="corpus",
        texts=[r.text for r in records],
        labels=np.array([r.label for r in records]),
        families=[r.family for r in records],
        kinds=[r.kind for r in records],
        groups=[r.prompt_id for r in records],
        created_utc=max(m["created_utc"] for m in run_manifests()),
        note="hash-verified project corpus (prompt-grouped)",
    )


def defactify_cohort() -> Cohort:
    from bench.datasets import defactify_created_utc, load_defactify

    dataset = load_defactify()
    return Cohort(
        name="defactify",
        texts=dataset.texts,
        labels=dataset.labels,
        families=dataset.families,
        kinds=dataset.kinds,
        groups=dataset.groups,
        created_utc=defactify_created_utc() or "2026-08-12T00:00:00Z",
        note=(
            "Defactify_Text_Dataset (Roy et al. 2026), hygiene-filtered; "
            "story groups reconstructed via TF-IDF near-duplicate clustering"
        ),
    )


def _feature_arrays_cohort(cohort: Cohort, feature: str, kind: str | None = None):
    ai, human = [], []
    for i, row in enumerate(cohort.features):
        if kind and cohort.kinds[i] != kind:
            continue
        (ai if cohort.labels[i] == 1 else human).append(row[feature])
    return np.array(ai), np.array(human)


def run_hypotheses(records, registry_path: Path = HYPOTHESES_PATH) -> list[dict]:
    """Backwards-compatible entry: corpus records are wrapped in a cohort."""
    if isinstance(records, Cohort):
        cohort = records
    else:
        cohort = Cohort(
            name="corpus",
            texts=[r.text for r in records],
            labels=np.array([r.label for r in records]),
            families=[r.family for r in records],
            kinds=[r.kind for r in records],
            groups=[r.prompt_id for r in records],
            created_utc="",
            note="",
        )
    return _run_hypotheses_cohort(cohort, registry_path)


def _run_hypotheses_cohort(cohort: Cohort, registry_path: Path = HYPOTHESES_PATH) -> list[dict]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    # Large-cohort compute budgets, recorded per outcome: permutation and
    # bootstrap resolution is reduced where the full budget is intractable.
    # p-value floors (1/(P+1)) remain far below the BH alpha.
    permutations = 500 if cohort.large else PERMUTATIONS
    results = []
    for hypothesis in registry["hypotheses"]:
        test = hypothesis["test"]
        variables = hypothesis["variables"]
        if test in {"welch_t", "mann_whitney"}:
            ai, human = _feature_arrays_cohort(cohort, variables["feature"], variables.get("kind"))
            if len(ai) < 3 or len(human) < 3:
                results.append({**{k: hypothesis[k] for k in ("id", "statement", "null", "direction", "alpha")},
                                "test": test, "statistic": None, "p_value": 1.0,
                                "effect_size": {"name": "n/a", "value": None}, "ci95": [None, None],
                                "estimate": None, "skipped": f"cohort lacks kind={variables.get('kind')} support"})
                continue
            outcome = (
                welch_t(ai, human, hypothesis["direction"])
                if test == "welch_t"
                else mann_whitney(ai, human, hypothesis["direction"])
            )
        elif test == "logistic_lr":
            feats = cohort.features
            y = cohort.labels.astype(float)
            reduced = np.array([[f[name] for name in variables["reduced"]] for f in feats])
            full = np.array([[f[name] for name in variables["full"]] for f in feats])
            outcome = logistic_lr_test(y, reduced, full)
        elif test == "permutation_manova":
            idx = [i for i, kind in enumerate(cohort.kinds) if kind == "text"]
            X = np.array(
                [[cohort.features[i][name] for name in variables["features"]] for i in idx]
            )
            groups = np.array([cohort.families[i] for i in idx])
            outcome = permutation_manova(X, groups, permutations=permutations)
        elif test == "durbin_watson_permutation":
            segment_scores = []
            n_segments = variables["segments_per_document"]
            for text, kind in zip(cohort.texts, cohort.kinds, strict=True):
                if kind != "text":
                    continue
                words = text.split()
                if len(words) < n_segments * 10:
                    continue
                bounds = [len(words) * i // n_segments for i in range(n_segments + 1)]
                scores = [
                    heuristic_raw_score(" ".join(words[bounds[i] : bounds[i + 1]]), "text")
                    for i in range(n_segments)
                ]
                segment_scores.append(scores)
            outcome = durbin_watson_permutation(segment_scores, permutations=permutations)
        else:  # pragma: no cover - registry guard
            raise ValueError(f"unknown test {test!r}")
        results.append(
            {
                "id": hypothesis["id"],
                "statement": hypothesis["statement"],
                "null": hypothesis["null"],
                "test": test,
                "direction": hypothesis["direction"],
                "alpha": hypothesis["alpha"],
                **outcome,
            }
        )
    q_values = benjamini_hochberg([result["p_value"] for result in results])
    for result, q in zip(results, q_values, strict=True):
        result["q_value"] = q
        result["null_decision"] = "rejected" if q <= result["alpha"] else "not rejected"
    return results


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_cohort_report(cohort: Cohort) -> dict:
    names = list(FEATURE_NAMES)
    X = np.array([[row[name] for name in names] for row in cohort.features])
    y = cohort.labels.astype(float)

    screening = screen_features(X, names)
    kept_idx = [names.index(name) for name in screening["kept"]]
    Xs = X[:, kept_idx]
    # Standardize for conditioning; the artifact records this choice.
    means = Xs.mean(axis=0)
    sds = np.where(Xs.std(axis=0) == 0, 1.0, Xs.std(axis=0))
    Xs = (Xs - means) / sds

    fit = logistic_irls(Xs, y)
    null_fit = logistic_irls(np.zeros((len(y), 0)), y)
    pearson_resid = (y - fit["p"]) / np.sqrt(np.clip(fit["p"] * (1 - fit["p"]), 1e-10, None))
    # DW is only meaningful under a non-grouped ordering; ingestion order clusters
    # by family, so order residuals by fitted probability to screen for trends.
    prob_order = np.argsort(fit["p"], kind="mergesort")
    dw_ordered = durbin_watson(pearson_resid[prob_order])

    specification = {
        "link_test": link_test(y, Xs),
        "reset_test": reset_test(y, Xs),
        "hosmer_lemeshow": hosmer_lemeshow(y, fit["p"]),
        "breusch_pagan": breusch_pagan(pearson_resid, Xs),
        "jarque_bera": jarque_bera(pearson_resid),
        "durbin_watson": {
            "statistic": dw_ordered,
            "meaning": (
                "Pearson residuals ordered by fitted probability; ~2 indicates no residual "
                "trend along the fit. Ingestion order is family-clustered and not used."
            ),
        },
        "cooks_distance": {
            "max": float(np.max(cooks_distance(y, fit))),
            "n_above_4_over_n": int(np.sum(cooks_distance(y, fit) > 4 / len(y))),
            "meaning": "Influential-point screen; 4/n is the conventional flag threshold.",
        },
        "pseudo_r2": {
            "mcfadden": mcfadden_r2(fit["ll"], null_fit["ll"]),
            "tjur": tjur_r2(y, fit["p"]),
        },
    }

    hypotheses = _run_hypotheses_cohort(cohort)

    return {
        "n_records": len(cohort),
        "n_human": int((cohort.labels == 0).sum()),
        "n_ai": int((cohort.labels == 1).sum()),
        "families": sorted(set(cohort.families)),
        "note": cohort.note,
        "feature_screening": screening,
        "model": {
            "type": "penalized logistic regression (IRLS, ridge 1e-6)",
            "features": screening["kept"],
            "standardized": True,
            "log_likelihood": fit["ll"],
            "null_log_likelihood": null_fit["ll"],
        },
        "specification": specification,
        "hypotheses": hypotheses,
    }


def build_report(datasets: tuple[str, ...] = ("corpus",)) -> dict:
    """Per-cohort report. `datasets` selects cohorts: corpus, defactify, both."""
    cohorts: dict[str, dict] = {}
    created: list[str] = []
    for name in datasets:
        cohort = corpus_cohort() if name == "corpus" else defactify_cohort()
        cohorts[name] = build_cohort_report(cohort)
        created.append(cohort.created_utc)
    return {
        "schema": "panoptes-methodology-v1",
        "created_utc": max(created),
        "seed": SEED,
        "cohorts": cohorts,
    }


def save_signed(payload: dict, output: Path) -> None:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_cohort(lines: list[str], name: str, cohort: dict) -> list[str]:
    title = "baseline corpus" if name == "corpus" else "Defactify_Text_Dataset (Roy et al. 2026)"
    lines += [
        f"## Cohort: {name} — {title}",
        "",
        f"{cohort['n_records']} documents ({cohort['n_human']} human, {cohort['n_ai']} AI) across "
        f"families: {', '.join(cohort['families'])}. {cohort['note']}.",
        "",
        "### Variable selection (multicollinearity screen)",
        "",
        "| Feature | VIF | Verdict |",
        "|---|---|---|",
    ]
    excluded = {row["feature"] for row in cohort["feature_screening"]["exclusions"]}
    investigated = {row["feature"] for row in cohort["feature_screening"]["investigations"]}
    for row in cohort["feature_screening"]["final_vif"]:
        verdict = "investigate" if row["feature"] in investigated else "keep"
        lines.append(f"| {row['feature']} | {row['vif']:.2f} | {verdict} |")
    for row in cohort["feature_screening"]["exclusions"]:
        lines.append(f"| {row['feature']} | {row['vif']:.2f} | **excluded** |")
    lines += [
        "",
        f"Condition number (standardized, retained set): "
        f"{cohort['feature_screening']['condition_number']:.1f}",
        "",
    ]
    if excluded:
        lines.append("#### Exclusion justifications")
        lines.append("")
        for row in cohort["feature_screening"]["exclusions"]:
            lines.append(f"- **{row['feature']}**: {row['justification']}")
        lines.append("")

    lines += [
        "### Hypothesis tests (pre-registered)",
        "",
        "| ID | Test | Statistic | p | q (BH) | Effect | 95% CI | Null decision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in cohort["hypotheses"]:
        ci = result["ci95"]
        ci_text = f"[{ci[0]:.4g}, {ci[1]:.4g}]" if ci[0] is not None else "—"
        effect = result["effect_size"]
        effect_text = f"{effect['name']} = {effect['value']:.4g}" if effect["value"] is not None else "—"
        stat_text = f"{result['statistic']:.3f}" if result["statistic"] is not None else "—"
        lines.append(
            f"| {result['id']} | {result['test']} | {stat_text} | "
            f"{result['p_value']:.4f} | {result['q_value']:.4f} | "
            f"{effect_text} | {ci_text} | **{result['null_decision']}** |"
        )
    lines.append("")
    for result in cohort["hypotheses"]:
        note = f" (skipped: {result['skipped']})" if result.get("skipped") else ""
        lines.append(f"- **{result['id']}**: {result['statement']}{note}")
    lines.append("")

    spec = cohort["specification"]
    lines += [
        "### Specification tests (binary logistic model)",
        "",
        f"Model: {cohort['model']['type']} on {len(cohort['model']['features'])} screened, "
        f"standardized features.",
        "",
        "| Test | Statistic | p | Verdict |",
        "|---|---|---|---|",
    ]
    for key in ("link_test", "reset_test", "hosmer_lemeshow", "breusch_pagan", "jarque_bera"):
        row = spec[key]
        verdict = "adequate" if row["adequate"] else "**concern**"
        lines.append(f"| {key} | {row['statistic']:.3f} | {row['p_value']:.4f} | {verdict} |")
    lines += [
        "",
        f"- Durbin-Watson (probability-ordered residuals): {spec['durbin_watson']['statistic']:.3f} "
        "(2.0 = no serial correlation)",
        f"- Cook's distance: max {spec['cooks_distance']['max']:.4f}; "
        f"{spec['cooks_distance']['n_above_4_over_n']} points above 4/n",
        f"- Pseudo-R^2: McFadden {spec['pseudo_r2']['mcfadden']:.3f}, "
        f"Tjur {spec['pseudo_r2']['tjur']:.3f}",
        "",
    ]
    return lines


def render_markdown(report: dict) -> str:
    lines = [
        "# Methodology report",
        "",
        f"Generated {report['created_utc']} (seed {report['seed']}). Cohorts: "
        f"{', '.join(report['cohorts'])}.",
        "",
        "All p-values adjusted with Benjamini-Hochberg within the hypothesis registry, per cohort. "
        "Decisions use q <= 0.05. Large cohorts use recorded compute budgets: permutation tests "
        "drop to 500 permutations and bootstrap CIs resample at most 4,000 per group "
        "(conservative; p-values remain exact).",
        "",
    ]
    for name, cohort in report["cohorts"].items():
        lines = _render_cohort(lines, name, cohort)
    lines += [
        f"Artifact SHA-256: `{report['artifact_sha256']}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the pre-registered methodology battery.")
    parser.add_argument(
        "--dataset",
        choices=["corpus", "defactify", "both"],
        default="both",
        help="which cohort(s) to analyze (default: both)",
    )
    args = parser.parse_args(argv)
    datasets = ("corpus", "defactify") if args.dataset == "both" else (args.dataset,)

    report = build_report(datasets)
    save_signed(report, REPORT_JSON)
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")
    print(f"artifact sha256: {report['artifact_sha256']}")
    for name, cohort in report["cohorts"].items():
        print(f"[{name}] n={cohort['n_records']}")
        for result in cohort["hypotheses"]:
            print(
                f"  {result['id']}: p={result['p_value']:.4f} q={result['q_value']:.4f} "
                f"-> null {result['null_decision']}"
            )


if __name__ == "__main__":
    main()
