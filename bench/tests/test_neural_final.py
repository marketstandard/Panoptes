"""Schema-only tests for the neural detector card (no private training runner)."""

from __future__ import annotations

import json

from bench.cards import sign
from bench.validate_submission import validate_file


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
        "training_pool": {
            "cohorts": {"mage": 15000, "raid": 15000, "defactify": 5856},
            "n_rows": 35856,
            "provenance": (
                "pooled public-weight training pool (MAGE train + RAID clean + DeFactify train)"
            ),
            "sha256": "b" * 64,
        },
        "calibration_partition": {
            "source": "pooled calibration (mage valid + raid clean holdout + defactify)",
            "cohorts": {"mage": 6000, "raid": 6037, "defactify": 2535},
            "n_rows": 14572,
            "excluded_pilot_dev_groups": 10,
            "disjoint_from_pilot_development": True,
            "disjoint_from_pooled_train": True,
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
    path = tmp_path / "card.json"
    path.write_text(json.dumps(sign(_minimal_detector_card())), encoding="utf-8")
    assert validate_file(path) == []


def test_detector_card_requires_disjoint_calibration(tmp_path):
    card = _minimal_detector_card()
    card["calibration_partition"]["disjoint_from_pilot_development"] = False
    path = tmp_path / "card.json"
    path.write_text(json.dumps(sign(card)), encoding="utf-8")
    assert validate_file(path) != []
