"""Tests for the frozen measurement protocol: splits, evidence, mixtures, metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import detectors, evaluate, evidence, mixtures, splits  # noqa: E402
from bench.datasets import Dataset  # noqa: E402
from bench.tests.test_bench import tiny_dataset  # noqa: E402
from bench.protocol import validate_protocol  # noqa: E402


def test_protocol_document_validates():
    payload = validate_protocol()
    assert payload["schema"] == "panoptes-research-protocol-v1"
    assert [rq["id"] for rq in payload["research_questions"]] == ["RQ1", "RQ2", "RQ3", "RQ4"]
    assert payload["split_rules"]["hard_rule"]


def test_holdout_and_nested_splits_are_group_disjoint():
    dataset = tiny_dataset(48)
    holdout = splits.holdout_split(dataset)
    holdout.assert_disjoint(dataset)
    assert len(holdout.train) and len(holdout.calibration) and len(holdout.test)
    nested = splits.nested_grouped_splits(dataset, n_outer=3)
    assert len(nested) >= 2
    for split in nested:
        split.assert_disjoint(dataset)


def test_protocol_splits_use_nested_cv_when_groups_are_few():
    dataset = tiny_dataset(48)
    chosen = splits.protocol_splits(dataset)
    assert chosen[0].method.startswith("nested_grouped_cv")


def test_likelihood_ratio_is_prior_independent():
    lr = float(evidence.likelihood_ratio(0.8, prevalence=0.8))
    assert lr == pytest.approx(1.0, rel=1e-6)
    lr_half = float(evidence.likelihood_ratio(0.8, prevalence=0.5))
    assert lr_half == pytest.approx(4.0, rel=1e-6)
    posterior = float(evidence.posterior_probability(1.0, lr_half))
    assert posterior == pytest.approx(0.8, rel=1e-6)


def test_prior_sensitivity_grid_is_monotonic():
    rows = evidence.prior_sensitivity(lr=4.0)
    posteriors = [row["posterior"] for row in rows]
    assert posteriors == sorted(posteriors)
    assert rows[0]["prevalence"] == 0.001
    assert rows[-1]["prevalence"] == 0.75


def test_correlated_shrinkage_recovers_naive_when_independent():
    llrs = np.array([0.2, -0.1, 0.3, 0.0])
    shrunk = evidence.correlated_shrinkage(llrs, rho=0.0)
    assert shrunk["llr"] == pytest.approx(evidence.naive_accumulate(llrs))
    collapsed = evidence.correlated_shrinkage(llrs, rho=1.0)
    assert collapsed["n_effective"] == pytest.approx(1.0)
    assert collapsed["llr"] == pytest.approx(float(llrs.mean()))


def test_binary_metrics_include_protocol_fields():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    metrics = evaluate.binary_metrics(labels, probabilities)
    for key in (
        "auroc",
        "auprc",
        "brier",
        "ece",
        "tpr_at_0.1fpr",
        "tpr_at_1fpr",
        "calibration_slope",
        "calibration_intercept",
    ):
        assert key in metrics
        assert math_isfinite(metrics[key])
    risk = evaluate.selective_risk_curve(labels, probabilities)
    coverages = [row["coverage"] for row in risk]
    assert 1.0 in coverages and 0.5 in coverages
    assert risk[0]["selective_risk"] >= risk[-1]["selective_risk"] - 1e-9


def math_isfinite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def test_protocol_evaluation_never_fits_calibration_on_test():
    dataset = tiny_dataset(48)
    result = evaluate.evaluate_protocol(detectors.HeuristicDetector, dataset)
    assert result["method"] == "nested_grouped_cv"
    assert result["metrics"]["auroc"] > 0.8
    assert result["prior_sensitivity"]
    for fold in result["folds"]:
        assert fold["n_calibration"] >= 1
        assert fold["n_test"] >= 1
        assert fold["n_train"] >= 1


def test_isotonic_skips_unidentified_calibration_folds():
    raw = np.array([0.2, 0.8, 0.3, 0.9])
    assert evaluate.fit_isotonic(raw, np.array([1, 1, 1, 1])) is None
    assert evaluate.fit_isotonic(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 0, 1, 1])) is None
    assert evaluate.fit_isotonic(raw, np.array([0, 0, 1, 1])) is not None


def test_nested_calibration_covers_both_classes_when_possible():
    dataset = tiny_dataset(48)
    for split in splits.nested_grouped_splits(dataset, n_outer=3):
        assert set(int(v) for v in dataset.labels[split.calibration]) == {0, 1}
        assert set(int(v) for v in dataset.labels[split.train]) == {0, 1}


def test_heuristic_and_logistic_share_detector_surface():
    dataset = tiny_dataset(36)
    split = splits.holdout_split(dataset)
    for factory in (detectors.HeuristicDetector, lambda: detectors.make_detector("logistic")):
        detector = factory()
        row = evaluate.evaluate_protocol_split(detector, dataset, split)
        assert 0.0 <= row["metrics"]["brier"] <= 1.0
        assert len(row["probabilities"]) == len(split.test)


def test_mixture_curve_tracks_controlled_participation():
    texts, labels, groups, families = [], [], [], []
    for i in range(12):
        groups.append(f"prompt-{i}")
        texts.append(
            f"i fixed the thing on my machine after a few tries. note {i}: the logs "
            "were messy but the fix worked. ask me if it breaks again. extra words here "
            "so the detector has enough tokens instead of abstaining on a short note. "
            "the neighbor still wants the ladder moved after dinner."
        )
        labels.append(0)
        families.append("human")
        texts.append(
            "Furthermore, the systematic approach improves overall reliability. "
            f"Moreover, iteration {i} additionally reinforces consistent verification. "
            "Therefore the process is robust and comprehensive. Additionally, overall "
            "documentation furthermore supports systematic and consistent verification."
        )
        labels.append(1)
        families.append("ai-x")
        groups.append(f"prompt-{i}")
    dataset = Dataset(
        texts=texts,
        labels=np.array(labels),
        families=families,
        kinds=["text"] * len(texts),
        groups=groups,
        buckets=["50-149"] * len(texts),
        provenance="synthetic-mix",
        sha256="0" * 64,
    )
    curve = mixtures.mixture_curve(dataset, detectors.HeuristicDetector())
    assert curve["n_pairs"] > 0
    rates = [row["actual_ai_rate"] for row in curve["rates"]]
    assert rates[0] == 0.0 and rates[-1] == 1.0
    estimated = [row["mean_estimated"] for row in curve["rates"]]
    assert estimated[-1] > estimated[0]
    assert curve["correlation"] > 0.5


def test_leave_one_family_out_runs():
    dataset = tiny_dataset(48)
    result = evaluate.leave_one_family_out(dataset)
    assert "mean_unknown_rejection_auroc" in result


def test_sliced_conformal_coverage_math():
    # Perfectly calibrated scores: coverage should be near nominal overall,
    # and the pooled threshold must be applied unchanged within slices.
    rng = np.random.default_rng(7)
    n = 400
    labels = rng.integers(0, 2, n)
    probabilities = np.clip(labels * 0.7 + (1 - labels) * 0.3 + rng.normal(0, 0.1, n), 1e-4, 1 - 1e-4)
    slices = {
        "length_bucket": ["150-499" if i % 2 == 0 else "500plus" for i in range(n)],
        "family": ["human" if int(v) == 0 else "ai-x" for v in labels],
        "class": ["ai" if int(v) == 1 else "human" for v in labels],
    }
    out = evaluate.sliced_conformal_coverage(labels, probabilities, slices, alpha=0.1)
    assert out["pooled"]["empirical_coverage"] >= 0.9
    for dimension in ("length_bucket", "family", "class"):
        assert len(out[dimension]) == 2
        for row in out[dimension]:
            assert 0.0 <= row["empirical_coverage"] <= 1.0
            assert row["nominal_coverage"] == pytest.approx(0.9)
            assert row["coverage_gap"] == pytest.approx(row["empirical_coverage"] - 0.9)
    # Tiny slices are skipped rather than reported on noise.
    slices["family"] = ["rare"] * 5 + ["human" if int(v) == 0 else "ai-x" for v in labels[5:]]
    out = evaluate.sliced_conformal_coverage(labels, probabilities, slices, alpha=0.1)
    assert "rare" not in [row["value"] for row in out["family"]]


def test_cross_dataset_transport_never_fits_on_target():
    source = tiny_dataset(48)
    target = tiny_dataset(36)
    row = evaluate.cross_dataset_transport(source, target, detectors.HeuristicDetector)
    assert row["n_test"] == len(target)
    assert row["n_train"] + row["n_calibration"] <= len(source)
    assert 0.0 <= row["brier"] <= 1.0
    assert row["selective_risk"] and row["conformal"]["empirical_coverage"] >= 0.0


def test_calibration_power_required_n_scales_inverse_square():
    from bench.power import calibration_power, required_n

    labels = np.array([0, 1] * 200)
    rng = np.random.default_rng(3)
    probabilities = np.clip(labels * 0.65 + (1 - labels) * 0.35 + rng.normal(0, 0.12, 400), 1e-4, 1 - 1e-4)
    block = calibration_power(labels, probabilities)
    assert block["n_evaluated"] == 400
    for metric in ("brier", "ece"):
        grid = block[metric]["required_n"]
        # Halving the MDE quadruples the required n.
        assert grid["0.01"] == pytest.approx(4 * grid["0.02"], rel=0.02)
        assert grid["0.02"] > 0
    assert required_n(0.0, 0.01) == 0
    assert required_n(0.25, 0.05) == int(np.ceil((1.959964 + 0.841621) ** 2 * 0.25 / 0.05**2))


def test_explicit_authors_are_kept_disjoint():
    texts = [f"hello world words {i} " * 20 for i in range(12)]
    dataset = Dataset(
        texts=texts,
        labels=np.array([i % 2 for i in range(12)]),
        families=["human" if i % 2 == 0 else "ai" for i in range(12)],
        kinds=["text"] * 12,
        groups=[f"g-{i}" for i in range(12)],
        buckets=["50-149"] * 12,
        provenance="authors",
        sha256="0" * 64,
        authors=[f"author-{i // 4}" for i in range(12)],
    )
    assert dataset.enforce_author_disjoint
    split = splits.holdout_split(dataset)
    split.assert_disjoint(dataset)
    dataset = Dataset(
        texts=["hello world " * 20] * 4,
        labels=np.array([0, 1, 0, 1]),
        families=["human", "ai", "human", "ai"],
        kinds=["text"] * 4,
        groups=["g"] * 4,
        buckets=["50-149"] * 4,
        provenance="tiny",
        sha256="0" * 64,
    )
    with pytest.raises(splits.SplitError):
        splits.holdout_split(dataset)
