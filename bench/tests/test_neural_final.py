"""Tests for the Phase 5 final-training runner's pure functions and card schema.

These cover the data-firewall partitioning helpers (group-disjoint inner-dev
split, calibration disjoint from the pilot's development subsample), the winner
loader, the isotonic calibrator table, and the neural-detector card schema. They
run without torch/transformers or a GPU; the heavy training path is exercised
separately by a smoke run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from research.run_neural_final import (
    _group_disjoint_split,
    _isotonic_thresholds,
    _calibration_indices,
    load_winner,
)


def _pilot_card() -> dict:
    return {
        "schema": "panoptes-neural-pilot-v1",
        "artifact_sha256": "abc123",
        "winner": {
            "encoder": "deberta-v3-base",
            "objective": "group_dro",
            "aggregation": "hierarchical_summary_head",
            "window": {"max_length": 512, "overlap": 128},
            "dev": {"worst_cohort_auroc": 0.9},
            "latency": {"ms_per_doc": 12.0},
        },
    }


def test_load_winner_extracts_frozen_config(tmp_path):
    path = tmp_path / "card.json"
    path.write_text(json.dumps(_pilot_card()), encoding="utf-8")
    winner = load_winner(path)
    assert winner["encoder"] == "deberta-v3-base"
    assert winner["hf"] == "microsoft/deberta-v3-base"
    assert winner["objective"] == "group_dro"
    assert winner["aggregation"] == "hierarchical_summary_head"
    assert winner["max_length"] == 512
    assert winner["overlap"] == 128
    assert winner["pilot_card_sha256"] == "abc123"


def test_load_winner_rejects_wrong_schema(tmp_path):
    card = _pilot_card()
    card["schema"] = "panoptes-model-card-v1"
    path = tmp_path / "card.json"
    path.write_text(json.dumps(card), encoding="utf-8")
    with pytest.raises(ValueError, match="panoptes-neural-pilot-v1"):
        load_winner(path)


def test_load_winner_rejects_unknown_encoder(tmp_path):
    card = _pilot_card()
    card["winner"]["encoder"] = "longformer-base-4096"
    path = tmp_path / "card.json"
    path.write_text(json.dumps(card), encoding="utf-8")
    with pytest.raises(ValueError, match="preregistered"):
        load_winner(path)


def test_group_disjoint_split_never_shares_groups():
    groups = [f"g{i // 4}" for i in range(400)]  # 100 groups of 4 rows
    train_idx, dev_idx = _group_disjoint_split(groups, dev_frac=0.2, seed=13)
    train_groups = {groups[i] for i in train_idx}
    dev_groups = {groups[i] for i in dev_idx}
    assert train_groups.isdisjoint(dev_groups)
    assert len(train_idx) + len(dev_idx) == 400
    assert 0 < len(dev_idx) < 400


def test_group_disjoint_split_is_deterministic():
    groups = [f"g{i}" for i in range(200)]
    a = _group_disjoint_split(groups, 0.15, seed=42)
    b = _group_disjoint_split(groups, 0.15, seed=42)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_isotonic_thresholds_monotone_and_bounded():
    rng = np.random.default_rng(0)
    scores = rng.random(500)
    labels = (scores + rng.normal(0, 0.15, 500) > 0.5).astype(int)
    table = _isotonic_thresholds(scores, labels)
    xs = np.array(table["x_thresholds"])
    ys = np.array(table["y_thresholds"])
    assert len(xs) == len(ys) and len(xs) >= 2
    assert np.all(np.diff(ys) >= -1e-9)  # monotone non-decreasing
    assert xs.min() >= 0.0 and xs.max() <= 1.0
    assert ys.min() >= 0.0 and ys.max() <= 1.0
    # The table must be usable by np.interp (the runtime calibrate path).
    probe = np.interp([0.2, 0.8], xs, ys)
    assert probe[0] <= probe[1]


def _mage_like(n: int, seed: int = 0) -> "object":
    from bench.datasets import Dataset

    rng = np.random.default_rng(seed)
    texts = [f"document number {i} with some text" for i in range(n)]
    labels = (rng.random(n) > 0.5).astype(int)
    domains = np.where(labels == 1, "cmv", "xsum").tolist()
    families = np.where(labels == 1, "gpt-j", "human").tolist()
    groups = [f"cluster-{i // 3}" for i in range(n)]
    return Dataset(
        texts=texts,
        labels=labels,
        families=families,
        kinds=["text"] * n,
        groups=groups,
        buckets=["short"] * n,
        provenance="synthetic",
        sha256="deadbeef",
        domains=domains,
    )


def test_calibration_indices_disjoint_from_pilot_dev():
    from bench.neural.data import stratified_group_subsample

    mage_valid = _mage_like(600)
    pilot_dev_rows = 100
    cal_idx, info = _calibration_indices(mage_valid, pilot_dev_rows, max_rows=200, seed=13)
    # Recompute the pilot's development groups and confirm no overlap.
    pilot_dev_idx, _ = stratified_group_subsample(mage_valid, pilot_dev_rows, 13)
    pilot_dev_groups = {str(mage_valid.groups[int(i)]) for i in pilot_dev_idx}
    cal_groups = {str(mage_valid.groups[int(i)]) for i in cal_idx}
    assert cal_groups.isdisjoint(pilot_dev_groups)
    assert info["disjoint_from_pilot_development"] is True
    assert len(cal_idx) > 0


def test_calibration_indices_deterministic():
    mage_valid = _mage_like(400)
    a, _ = _calibration_indices(mage_valid, 80, max_rows=150, seed=13)
    b, _ = _calibration_indices(mage_valid, 80, max_rows=150, seed=13)
    assert np.array_equal(a, b)


def _minimal_detector_card() -> dict:
    return {
        "schema": "panoptes-neural-detector-v1",
        "created_utc": "2026-08-19T00:00:00Z",
        "phase": "Phase 5",
        "winner": {
            "encoder": "deberta-v3-base",
            "hf": "microsoft/deberta-v3-base",
            "objective": "erm",
            "aggregation": "overlap_corrected_logit_mean",
            "max_length": 512,
            "overlap": 128,
            "pilot_dev": None,
            "pilot_card_sha256": None,
        },
        "seeds": [13, 42, 87],
        "data_firewall": "train -> inner-dev -> calibration; test sealed",
        "calibration_partition": {
            "source": "mage valid",
            "excluded_pilot_dev_groups": 10,
            "n_rows_selected": 15000,
            "n_groups_selected": 9000,
            "disjoint_from_pilot_development": True,
        },
        "environment": {"torch": "x"},
        "config": {"lr": 2e-5},
        "seed_runs": [
            {
                "seed": 13,
                "encoder_sha256": "a" * 64,
                "summary_head_sha256": None,
                "train_sec": 100.0,
                "best_epoch": 2,
                "inner_dev": {"auroc": 0.9},
                "calibration_metrics_raw": {"auroc": 0.9},
            }
        ],
        "ensemble": {
            "aggregation": "mean of per-seed document probabilities",
            "calibration_metrics_raw": {"auroc": 0.9},
            "calibration_metrics_calibrated": {"auroc": 0.9},
        },
        "calibration": {
            "method": "isotonic_regression_fit_on_calibration",
            "binary_calibrator": {"x_thresholds": [0.0, 1.0], "y_thresholds": [0.0, 1.0]},
            "conformal": {"alpha": 0.1},
            "fpr_thresholds": {"0.01": 0.9},
            "selective": [],
        },
        "limitations": ["trained on MAGE only"],
    }


def test_detector_card_schema_validates(tmp_path):
    from bench.cards import sign
    from research.validate_submission import validate_file

    path = tmp_path / "card.json"
    path.write_text(json.dumps(sign(_minimal_detector_card())), encoding="utf-8")
    assert validate_file(path) == []


def test_detector_card_requires_disjoint_calibration(tmp_path):
    from bench.cards import sign
    from research.validate_submission import validate_file

    card = _minimal_detector_card()
    card["calibration_partition"]["disjoint_from_pilot_development"] = False
    path = tmp_path / "card.json"
    path.write_text(json.dumps(sign(card)), encoding="utf-8")
    assert validate_file(path) != []
