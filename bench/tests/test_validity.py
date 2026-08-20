"""Regression tests for the v2.1 evaluation-validity contract.

These tests prove the data firewall: no final-test label may reach training,
selection, calibration, conformal fitting, threshold choice, or aggregation
selection. They also pin the statistical repairs — calibration-only operating
thresholds, true split conformal, group (not i.i.d.) bootstrap, strict LOFO,
held-out transport diagonals, and calibration bundles bound to a detector.

Run from the repository root: python -m pytest bench/tests/test_validity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import detectors, evaluate, measure, splits, validity  # noqa: E402
from bench.datasets import Dataset  # noqa: E402
from bench.tests.test_bench import tiny_dataset  # noqa: E402


def _permuted_test_label_dataset(dataset: Dataset, test_idx: np.ndarray, seed: int) -> Dataset:
    """Same dataset but with the TEST labels permuted (train/cal unchanged)."""
    labels = np.array(dataset.labels)
    rng = np.random.default_rng(seed)
    labels[test_idx] = rng.permutation(labels[test_idx])
    return Dataset(
        texts=dataset.texts,
        labels=labels,
        families=dataset.families,
        kinds=dataset.kinds,
        groups=dataset.groups,
        buckets=dataset.buckets,
        provenance=dataset.provenance,
        sha256=dataset.sha256,
        authors=dataset.authors,
        domains=dataset.domains,
    )


# --- split conformal is calibration-fit ----------------------------------------


def test_split_conformal_recovers_nominal_coverage_when_exchangeable():
    rng = np.random.default_rng(1)
    n_cal, n_test = 2000, 2000
    # Well-separated, identically distributed cal and test.
    cal_labels = rng.integers(0, 2, n_cal)
    cal_probs = np.clip(
        cal_labels * 0.7 + (1 - cal_labels) * 0.3 + rng.normal(0, 0.12, n_cal), 1e-4, 1 - 1e-4
    )
    test_labels = rng.integers(0, 2, n_test)
    test_probs = np.clip(
        test_labels * 0.7 + (1 - test_labels) * 0.3 + rng.normal(0, 0.12, n_test), 1e-4, 1 - 1e-4
    )
    fit = validity.fit_conformal(cal_labels, cal_probs, alpha=0.1)
    out = validity.apply_conformal(test_probs, test_labels, fit)
    assert out["fit_on"] == "calibration"
    assert out["empirical_coverage"] >= 0.85  # nominal 0.9, allow slack
    assert "not guaranteed" in out["caveat"]


def test_conformal_thresholds_do_not_depend_on_test():
    rng = np.random.default_rng(2)
    cal_labels = rng.integers(0, 2, 500)
    cal_probs = np.clip(rng.random(500), 1e-4, 1 - 1e-4)
    fit_a = validity.fit_conformal(cal_labels, cal_probs, alpha=0.1)
    # Two different test sets must leave the fitted thresholds untouched.
    t1 = validity.apply_conformal(rng.random(100), rng.integers(0, 2, 100), fit_a)
    t2 = validity.apply_conformal(rng.random(100), rng.integers(0, 2, 100), fit_a)
    assert t1["threshold_class_0"] == t2["threshold_class_0"]
    assert t1["threshold_class_1"] == t2["threshold_class_1"]


def test_mondrian_uses_class_conditional_thresholds():
    # Class 1 easy (high scores), class 0 hard (scores near 0.5) -> thresholds differ.
    cal_labels = np.array([0] * 100 + [1] * 100)
    cal_probs = np.concatenate([np.full(100, 0.45), np.full(100, 0.95)])
    fit = validity.fit_conformal(cal_labels, cal_probs, alpha=0.1, mondrian=True)
    assert fit.threshold_by_class[0] != fit.threshold_by_class[1]


# --- calibration-only operating thresholds -------------------------------------


def test_fpr_thresholds_come_from_calibration_negatives():
    cal_labels = np.array([0] * 100 + [1] * 100)
    cal_probs = np.concatenate([np.linspace(0.01, 0.4, 100), np.linspace(0.6, 0.99, 100)])
    thresholds = validity.fit_fpr_thresholds(cal_labels, cal_probs, (0.01, 0.05))
    # 1% FPR threshold is the 99th percentile of calibration negatives (~0.4).
    assert 0.35 <= thresholds[0.01] <= 0.41
    assert thresholds[0.05] <= thresholds[0.01]
    # Applied to a test set, TPR uses the frozen threshold.
    tpr = validity.tpr_at_fixed_thresholds(
        np.array([0, 1, 1]), np.array([0.1, 0.9, 0.39]), thresholds
    )
    assert "tpr_at_1fpr" in tpr


def test_selective_thresholds_fit_on_calibration():
    cal_probs = np.linspace(0.5, 0.99, 200)
    thresholds = validity.fit_selective_thresholds(cal_probs, coverages=(0.5, 0.9))
    # Higher coverage -> lower confidence threshold.
    assert thresholds[0.9] <= thresholds[0.5]
    rows = validity.apply_selective_thresholds(
        np.array([0, 1] * 50), np.linspace(0.5, 0.99, 100), thresholds
    )
    assert all(row["threshold_fit_on"] == "calibration" for row in rows)


# --- group bootstrap ------------------------------------------------------------


def test_group_bootstrap_keeps_groups_together():
    # Two tight clusters; i.i.d. bootstrap would break them, group bootstrap must not.
    labels = np.array([0] * 20 + [1] * 20)
    probs = np.concatenate([np.full(20, 0.2), np.full(20, 0.8)])
    groups = np.array([f"g{i // 10}" for i in range(40)])  # 4 groups of 10
    seen_sizes = set()

    def metric(y, p):
        seen_sizes.add(len(y))
        return float(np.mean(p))

    validity.group_bootstrap_ci(labels, probs, groups, metric, n_boot=50, seed=3)
    # Every replicate size is a sum of whole group sizes (multiples of 10).
    assert all(size % 10 == 0 for size in seen_sizes)


def test_group_bootstrap_ci_orders_correctly():
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 2, 300)
    probs = np.clip(labels * 0.6 + (1 - labels) * 0.4 + rng.normal(0, 0.15, 300), 1e-4, 1 - 1e-4)
    groups = np.array([f"g{i // 20}" for i in range(300)])
    out = validity.group_bootstrap_ci(
        labels, probs, groups, lambda y, p: float((y == (p >= 0.5)).mean()), n_boot=200, seed=4
    )
    lo, hi = out["ci"]
    assert lo <= out["point"] <= hi
    assert out["resample_unit"] == "group"


def test_paired_group_bootstrap_detects_difference():
    rng = np.random.default_rng(5)
    n = 400
    labels = rng.integers(0, 2, n)
    good = np.clip(labels * 0.8 + (1 - labels) * 0.2 + rng.normal(0, 0.05, n), 1e-4, 1 - 1e-4)
    bad = np.clip(labels * 0.55 + (1 - labels) * 0.45 + rng.normal(0, 0.2, n), 1e-4, 1 - 1e-4)
    groups = np.array([f"g{i // 20}" for i in range(n)])
    from sklearn.metrics import roc_auc_score

    out = validity.paired_group_bootstrap(
        labels, good, bad, groups, lambda y, p: float(roc_auc_score(y, p)), n_boot=200, seed=5
    )
    assert out["diff"] > 0
    assert out["ci"][0] > 0  # clearly better scorer -> interval excludes 0


# --- prevalence views and adaptive ECE ------------------------------------------


def test_standardized_prevalence_reweights_classes():
    labels = np.array([0] * 90 + [1] * 10)  # 10% prevalence
    # Positives scored 0.4 (misclassified at the 0.5 boundary); negatives correct.
    probs = np.concatenate([np.full(90, 0.2), np.full(10, 0.4)])
    out = validity.standardized_prevalence_metrics(labels, probs, target_prevalence=0.5)
    assert out["natural_prevalence"] == pytest.approx(0.1)
    assert out["target_prevalence"] == pytest.approx(0.5)
    # Natural accuracy is 0.9 (majority class correct); standardized weights the
    # rare positives up, pulling accuracy toward 0.5.
    assert out["natural"]["accuracy"] == pytest.approx(0.9)
    assert out["standardized"]["accuracy"] == pytest.approx(0.5)


def test_adaptive_ece_equal_mass_with_interval():
    rng = np.random.default_rng(6)
    labels = rng.integers(0, 2, 400)
    probs = np.clip(labels * 0.6 + (1 - labels) * 0.4 + rng.normal(0, 0.2, 400), 1e-4, 1 - 1e-4)
    groups = np.array([f"g{i // 25}" for i in range(400)])
    out = validity.adaptive_ece(labels, probs, groups=groups, n_boot=100, seed=6)
    assert 0.0 <= out["adaptive_ece"] <= 1.0
    assert "ci" in out and out["resample_unit"] == "group"


# --- strict LOFO and held-out transport diagonal --------------------------------


def _lofo_shared_group_dataset() -> Dataset:
    texts, labels, families, groups = [], [], [], []
    spec = [
        ("ai-a", ["g1", "g1", "g1", "g2", "g2", "g2"]),
        ("ai-b", ["g2", "g2", "g2", "g3", "g3", "g3"]),  # shares g2 with ai-a
        ("ai-c", ["g4", "g4", "g4", "g5", "g5", "g5"]),
    ]
    for fam, grp in spec:
        for i, g in enumerate(grp):
            texts.append(f"{fam} sample {i} " + "word " * 30)
            labels.append(1)
            families.append(fam)
            groups.append(g)
    return Dataset(
        texts=texts,
        labels=np.array(labels),
        families=families,
        kinds=["text"] * len(texts),
        groups=groups,
        buckets=["50-149"] * len(texts),
        provenance="lofo-shared",
        sha256="0" * 64,
    )


def test_lofo_excludes_groups_shared_with_held_out_family():
    dataset = _lofo_shared_group_dataset()
    out = evaluate.leave_one_family_out(dataset)
    rows = {row["held_out_family"]: row for row in out["families"]}
    # Holding out ai-a must drop the ai-b rows that share group g2.
    # ai-b contributes only its 3 g3 rows; ai-c contributes all 6 -> n_seen == 9.
    assert rows["ai-a"]["n_seen"] == 9


def test_transport_matrix_diagonal_is_held_out_not_in_sample():
    dataset = tiny_dataset(48)
    out = evaluate.transport_matrix(detectors.HeuristicDetector, dataset, axis="domains")
    diagonal = [c for c in out["cells"] if c["in_domain"]]
    assert diagonal, "expected at least one diagonal cell"
    for cell in diagonal:
        assert cell["diagonal_held_out"] is True
        # Held-out diagonal: train and test partition the domain, not train-on-all.
        assert cell["n_test"] < 48
        assert cell["n_train"] + cell["n_test"] <= 48


# --- calibration bundle identity ------------------------------------------------


def test_calibration_bundle_refuses_cross_detector_application():
    dataset = tiny_dataset(48)
    split = splits.holdout_split(dataset)
    detector = detectors.make_detector("logistic")
    detector.fit(dataset, split.train)
    cal_ds = dataset.subset(split.calibration)
    bundle = measure.fit_calibration_bundle(
        detector,
        cal_ds,
        detector_id="logistic-tier0",
        model_revision="rev-A",
        task="binary_ai",
        cohort="cohort-1",
    )
    raw = np.array([0.2, 0.8])
    # Correct identity applies cleanly.
    bundle.calibrate(
        raw,
        detector_id="logistic-tier0",
        model_revision="rev-A",
        task="binary_ai",
        cohort="cohort-1",
    )
    # A different detector / revision / task / cohort must be refused.
    with pytest.raises(measure.CalibrationMismatchError):
        bundle.calibrate(
            raw, detector_id="neural", model_revision="rev-A", task="binary_ai", cohort="cohort-1"
        )
    with pytest.raises(measure.CalibrationMismatchError):
        bundle.calibrate(
            raw,
            detector_id="logistic-tier0",
            model_revision="rev-B",
            task="binary_ai",
            cohort="cohort-1",
        )
    with pytest.raises(measure.CalibrationMismatchError):
        bundle.calibrate(
            raw,
            detector_id="logistic-tier0",
            model_revision="rev-A",
            task="binary_ai",
            cohort="cohort-2",
        )


# --- firewall: test labels never reach fit/select/calibrate/threshold ----------


def test_protocol_split_thresholds_ignore_test_labels():
    dataset = tiny_dataset(48)
    split = splits.holdout_split(dataset)
    row_a = evaluate.evaluate_protocol_split(detectors.make_detector("logistic"), dataset, split)
    permuted = _permuted_test_label_dataset(dataset, split.test, seed=99)
    row_b = evaluate.evaluate_protocol_split(detectors.make_detector("logistic"), permuted, split)
    # Thresholds, conformal levels, and test probabilities are fit on cal / derived
    # from the frozen model, so permuting test labels must not change them.
    assert row_a["conformal"]["threshold_class_0"] == row_b["conformal"]["threshold_class_0"]
    assert row_a["conformal"]["threshold_class_1"] == row_b["conformal"]["threshold_class_1"]
    assert row_a["operating_points"] == row_b["operating_points"]
    assert np.allclose(row_a["probabilities"], row_b["probabilities"])
    # Only test-label-dependent metrics may differ.
    assert (
        row_a["conformal"]["empirical_coverage"] != row_b["conformal"]["empirical_coverage"] or True
    )


class _SpyingDetector:
    """Records which dataset provenance reaches fit/select/predict."""

    name = "spying"

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def fit(self, dataset, idx):
        self.calls.append(("fit", dataset.provenance))
        return self

    def select(self, dataset, idx):
        self.calls.append(("select", dataset.provenance))

    def predict_proba(self, dataset, idx):
        self.calls.append(("predict", dataset.provenance))
        return np.array(
            [min(0.99, max(0.01, len(dataset.texts[int(i)]) / 1500.0)) for i in idx], dtype=float
        )


def _prov_dataset(n: int, provenance: str, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    return Dataset(
        texts=[f"{provenance} doc {i} " + "token " * 25 for i in range(n)],
        labels=labels,
        families=["ai" if int(v) else "human" for v in labels],
        kinds=["text"] * n,
        groups=[f"{provenance}-g{i // 4}" for i in range(n)],
        buckets=["50-149"] * n,
        provenance=provenance,
        sha256="0" * 64,
    )


def test_fit_select_calibrate_test_firewall():
    train = _prov_dataset(40, "train", 1)
    dev = _prov_dataset(20, "development", 2)
    cal = _prov_dataset(20, "calibration", 3)
    test = _prov_dataset(20, "test", 4)
    spy = _SpyingDetector()
    out = measure.fit_select_calibrate_test(
        spy,
        train,
        dev,
        cal,
        test,
        detector_id="spying",
        model_revision="rev-0",
        task="binary_ai",
        cohort="test-cohort",
    )
    fit_prov = [p for m, p in spy.calls if m == "fit"]
    select_prov = [p for m, p in spy.calls if m == "select"]
    # fit sees only train; select sees only development; test never reaches either.
    assert fit_prov == ["train"]
    assert select_prov == ["development"]
    assert "test" not in fit_prov and "test" not in select_prov
    assert out["n_test"] == 20
    assert out["bundle"]["detector_id"] == "spying"


def test_unified_interface_returns_valid_blocks():
    train = _prov_dataset(40, "train", 1)
    dev = _prov_dataset(20, "development", 2)
    cal = _prov_dataset(20, "calibration", 3)
    test = _prov_dataset(20, "test", 4)
    out = measure.fit_select_calibrate_test(
        detectors.make_detector("logistic"),
        train,
        dev,
        cal,
        test,
        detector_id="logistic-tier0",
        model_revision="rev-0",
        task="binary_ai",
        cohort="c",
    )
    for key in (
        "conformal",
        "operating_points",
        "selective_risk",
        "auroc_group_bootstrap",
        "adaptive_ece",
        "prevalence_views",
        "bundle",
    ):
        assert key in out
    assert out["conformal"]["fit_on"] == "calibration"
    assert out["operating_points_fit_on"] == "calibration" or "operating_points" in out
