"""Tests for research/methodology.py.

Run from the repository root: python -m pytest research/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import methodology  # noqa: E402
from research.baseline_corpus import load_corpus  # noqa: E402


def test_vif_excludes_constructed_collinear_feature():
    rng = np.random.default_rng(7)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    c = 2.0 * a + b + rng.normal(scale=1e-6, size=200)  # near-exact linear combination
    d = rng.normal(size=200)
    X = np.column_stack([a, b, c, d])
    result = methodology.screen_features(X, ["a", "b", "c", "d"])
    assert "c" in {row["feature"] for row in result["exclusions"]}
    assert set(result["kept"]) == {"a", "b", "d"}
    for row in result["final_vif"]:
        assert row["vif"] <= 10.0
    assert all(row["justification"] for row in result["exclusions"])


def test_vif_keeps_independent_features():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(150, 4))
    result = methodology.screen_features(X, ["w", "x", "y", "z"])
    assert result["kept"] == ["w", "x", "y", "z"]
    assert result["exclusions"] == []


def test_durbin_watson_bounds_and_autocorrelation_detection():
    rng = np.random.default_rng(5)
    white = rng.normal(size=300)
    dw_white = methodology.durbin_watson(white)
    assert 0.0 <= dw_white <= 4.0
    assert abs(dw_white - 2.0) < 0.3  # white noise sits near 2

    trending = np.cumsum(rng.normal(size=300))  # strongly autocorrelated
    dw_trend = methodology.durbin_watson(trending)
    assert 0.0 <= dw_trend <= 4.0
    assert dw_trend < 1.0  # flags positive autocorrelation


def test_durbin_watson_permutation_flags_constructed_autocorrelation():
    rng = np.random.default_rng(3)
    # Each "document" has a strong within-document level shift: residuals
    # built against the grand mean are positively autocorrelated.
    segment_scores = []
    for _ in range(30):
        level = rng.normal(scale=2.0)
        segment_scores.append([level + rng.normal(scale=0.01) for _ in range(3)])
    result = methodology.durbin_watson_permutation(segment_scores)
    assert 0.0 <= result["statistic"] <= 4.0
    assert result["p_value"] < 0.05  # permutation test detects the clustering


def test_mcnemar_matches_hand_computed_exact():
    # b=10, c=2 discordant pairs -> exact two-sided binomial p = 158/4096.
    correct_a = np.array([True] * 10 + [False] * 2 + [True] * 5)
    correct_b = np.array([False] * 10 + [True] * 2 + [True] * 5)
    result = methodology.mcnemar(correct_a, correct_b)
    assert result["b"] == 10 and result["c"] == 2
    assert result["method"] == "exact_binomial"
    assert result["p_value"] == pytest.approx(158 / 4096, rel=1e-9)


def test_mcnemar_chi2_continuity_matches_hand_computed():
    # b=30, c=10 -> (|30-10|-1)^2/40 = 361/40 = 9.025
    correct_a = np.array([True] * 30 + [False] * 10)
    correct_b = np.array([False] * 30 + [True] * 10)
    result = methodology.mcnemar(correct_a, correct_b)
    assert result["method"] == "chi2_continuity"
    assert result["statistic"] == pytest.approx(9.025, rel=1e-9)


def test_delong_known_aucs():
    labels = np.array([1, 1, 0, 0])
    perfect = np.array([0.9, 0.8, 0.4, 0.3])
    useless = np.array([0.3, 0.4, 0.8, 0.9])
    result = methodology.delong_test(labels, perfect, useless)
    assert result["auc_a"] == pytest.approx(1.0)
    assert result["auc_b"] == pytest.approx(0.0)
    identical = methodology.delong_test(labels, perfect, perfect)
    assert identical["p_value"] == pytest.approx(1.0)


def test_delong_reproduces_classic_example():
    # DeLong et al. 1988 style: moderate separation, correlated scores.
    rng = np.random.default_rng(21)
    labels = np.array([1] * 40 + [0] * 40)
    base = np.concatenate([rng.normal(0.8, 1.0, 40), rng.normal(0.0, 1.0, 40)])
    scores_a = base + rng.normal(0, 0.3, 80)
    scores_b = base * 0.8 + rng.normal(0, 0.5, 80)
    result = methodology.delong_test(labels, scores_a, scores_b)
    assert 0.5 < result["auc_a"] <= 1.0
    assert 0.0 <= result["p_value"] <= 1.0
    # Same scores -> zero variance path
    same = methodology.delong_test(labels, scores_a, scores_a)
    assert same["p_value"] == 1.0


def test_hypotheses_record_decisions_and_qvalues():
    records = load_corpus()
    results = methodology.run_hypotheses(records)
    assert [r["id"] for r in results] == ["H1", "H2", "H3", "H4", "H5", "H6"]
    for result in results:
        assert 0.0 <= result["p_value"] <= 1.0
        assert 0.0 <= result["q_value"] <= 1.0
        assert result["q_value"] >= result["p_value"] - 1e-12  # BH never shrinks p
        assert result["null_decision"] in {"rejected", "not rejected"}
        assert (result["q_value"] <= result["alpha"]) == (result["null_decision"] == "rejected")


def test_specification_tests_run_on_real_corpus():
    report = methodology.build_report()
    spec = report["specification"]
    for key in ("link_test", "reset_test", "hosmer_lemeshow", "breusch_pagan", "jarque_bera"):
        assert 0.0 <= spec[key]["p_value"] <= 1.0
    assert 0.0 <= spec["durbin_watson"]["statistic"] <= 4.0
    assert -1.0 <= spec["pseudo_r2"]["tjur"] <= 1.0
    assert spec["pseudo_r2"]["mcfadden"] > 0.0
    assert report["feature_screening"]["kept"]  # non-empty model
