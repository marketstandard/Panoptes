"""Tests for the bench package.

Run from the repository root: python -m pytest bench/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import cards, datasets, evaluate, features, models  # noqa: E402


def tiny_dataset(n: int = 48) -> datasets.Dataset:
    texts, labels, groups = [], [], []
    # HeuristicProseDetector abstains below 40 tokens (raw_score=0.5).
    ai_pad = (
        " Furthermore, the overall process additionally remains systematic, comprehensive, "
        "and consistent across verification steps that moreover reinforce reliability."
    )
    human_pad = (
        " the neighbor asked about the ladder and i said i would move it after dinner "
        "once the rain stopped. extra words keep the detector from abstaining on short notes."
    )
    for i in range(n):
        if i % 2 == 0:
            texts.append(
                "Furthermore, the systematic approach improves overall reliability. "
                f"Moreover, iteration {i} additionally reinforces consistent verification. "
                "Therefore the process is robust and comprehensive." + ai_pad
            )
            labels.append(1)
        else:
            texts.append(
                f"i fixed the thing on my machine after a few tries. note {i}: the logs "
                "were messy but the fix worked. ask me if it breaks again." + human_pad
            )
            labels.append(0)
        groups.append(f"prompt-{i % 6}")
    return datasets.Dataset(
        texts=texts,
        labels=np.array(labels),
        families=["ai-x" if label else "human" for label in labels],
        kinds=["text"] * n,
        groups=groups,
        buckets=["50-149"] * n,
        provenance="synthetic-test",
        sha256="0" * 64,
    )


# --- tier gate ---------------------------------------------------------------


def test_power_gate_boundaries():
    assert not models.power_gate(100)["passes"]
    assert not models.power_gate(3139)["passes"]
    assert models.power_gate(3140)["passes"]
    gate = models.power_gate(104)
    assert gate["required_n"] == 3140
    assert "FAILS" in gate["rationale"]


def test_zoo_tier_admission():
    small = {entry["name"]: entry for entry in models.zoo(104)}
    assert small["logistic-tier0"]["admitted"]
    assert not small["gbm-tier1"]["admitted"]
    assert not small["panoptes-v0"]["admitted"]
    large = {entry["name"]: entry for entry in models.zoo(4000)}
    assert large["gbm-tier1"]["admitted"]
    assert large["panoptes-v0"]["admitted"]


# --- end-to-end logistic on synthetic data -----------------------------------


def test_logistic_end_to_end_separable():
    dataset = tiny_dataset()
    result = evaluate.cross_validate(models.LogisticTier0, dataset, n_splits=3)
    assert result["oof_probabilities"].shape == (len(dataset),)
    assert result["metrics"]["auroc"] > 0.9  # constructed to be separable
    assert result["auroc_ci95"][0] <= result["metrics"]["auroc"] <= result["auroc_ci95"][1]
    assert result["reliability_bins"]
    assert result["coverage_curve"]
    assert 0.0 <= result["conformal"]["empirical_coverage"] <= 1.0
    assert result["fairness_slices"]["family"]


def test_logistic_fit_predict_shapes():
    dataset = tiny_dataset()
    X = dataset.features()
    model = models.LogisticTier0().fit(X, dataset.labels)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(dataset),)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


# --- user dataset validation ---------------------------------------------------


def test_user_dataset_csv_roundtrip(tmp_path):
    path = tmp_path / "mine.csv"
    rows = ["text,label,kind"]
    for i in range(10):
        label = "ai" if i % 2 else "human"
        rows.append(f"some words here for row {i} to read,{label},text")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    dataset = datasets.load_user_dataset(path)
    assert len(dataset) == 10
    assert set(dataset.labels) == {0, 1}
    assert dataset.kinds == ["text"] * 10


def test_user_dataset_rejects_bad_label(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        "\n".join(f'{{"text": "row {i} text", "label": "maybe"}}' for i in range(10)),
        encoding="utf-8",
    )
    with pytest.raises(datasets.DatasetError):
        datasets.load_user_dataset(path)


def test_user_dataset_requires_minimum_rows(tmp_path):
    path = tmp_path / "tiny.jsonl"
    path.write_text('{"text": "short", "label": 1}\n' * 3, encoding="utf-8")
    with pytest.raises(datasets.DatasetError, match="at least 8"):
        datasets.load_user_dataset(path)


# --- cards ---------------------------------------------------------------------


def test_card_signing_roundtrip():
    dataset = tiny_dataset()
    result = evaluate.cross_validate(models.LogisticTier0, dataset, n_splits=3)
    card = cards.model_card(
        model_name="logistic-tier0",
        tier=0,
        dataset=dataset,
        evaluation=result,
        gate=models.power_gate(len(dataset)),
        config={"model": "logistic"},
    )
    assert cards.verify_card(card)
    card["evaluation"]["metrics"]["auroc"] = 0.9999  # tamper
    assert not cards.verify_card(card)


# --- feature parity with the runtime -------------------------------------------


def test_attribution_feature_parity_with_backend():
    from panoptes.analysis.attribution import _features as backend_features
    from panoptes.schemas import ContentType

    samples = [
        ("A short note with a few words.", "text"),
        (
            "Furthermore, the systematic evaluation of evidence improves reliability. "
            "Moreover, consistent documentation supports verification; therefore, teams "
            "should maintain detailed records of every decision and its rationale.",
            "text",
        ),
        ("def f(x):\n    return x * 2  # double it\n", "code"),
    ]
    for text, kind in samples:
        bench = features.extract(text, kind)
        backend = backend_features(text, ContentType.CODE if kind == "code" else ContentType.PROSE)
        for name in features.ATTRIBUTION_FEATURES:
            assert bench[name] == pytest.approx(backend[name], rel=1e-9, abs=1e-12), name
