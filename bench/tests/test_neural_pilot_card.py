"""Schema validation for the neural pilot selection card.

Runs in every environment (no torch/transformers needed): a synthetic card must
validate, a card missing a required field must fail, and the real signed card
(when the pilot has been run) must validate and carry a frozen winner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _minimal_card() -> dict:
    run = {
        "stage": "A_encoder",
        "encoder": "deberta-v3-base",
        "objective": "erm",
        "aggregation": "overlap_corrected_logit_mean",
        "window": {"max_length": 512, "overlap": 128},
        "status": "ok",
        "dev": {
            "worst_cohort_auroc": 0.9,
            "auroc": 0.93,
            "brier": 0.08,
            "worst_cohort_brier": 0.12,
        },
        "latency": {"ms_per_doc": 12.5},
    }
    return {
        "schema": "panoptes-neural-pilot-v1",
        "created_utc": "2026-08-19T00:00:00Z",
        "seed": 13,
        "phase": "Phase 4",
        "data_firewall": "train/development only",
        "cohort": {
            "dataset": "mage",
            "train_rows_selected": 100,
            "dev_rows_selected": 50,
            "train_subsample": {},
            "dev_subsample": {},
        },
        "environment": {"torch": "x"},
        "config": {"lr": 2e-5},
        "window_conditions": {"deberta-v3-base": {"max_length": 512, "overlap": 128}},
        "selection_rule": "lexicographic",
        "runs": [run],
        "winner": {
            "encoder": "deberta-v3-base",
            "objective": "erm",
            "aggregation": "overlap_corrected_logit_mean",
            "window": {"max_length": 512, "overlap": 128},
            "dev": run["dev"],
            "latency": {"ms_per_doc": 12.5},
        },
        "limitations": ["pilot is train/dev only"],
    }


def _signed_card() -> dict:
    from bench.cards import sign

    return sign(_minimal_card())


def test_synthetic_pilot_card_validates(tmp_path):
    from bench.validate_submission import validate_file

    path = tmp_path / "card.json"
    path.write_text(json.dumps(_signed_card()), encoding="utf-8")
    assert validate_file(path) == []


def test_pilot_card_missing_required_field_fails(tmp_path):
    from bench.validate_submission import validate_file

    card = _signed_card()
    del card["winner"]
    path = tmp_path / "card.json"
    path.write_text(json.dumps(card), encoding="utf-8")
    assert validate_file(path) != []


def test_pilot_card_bad_objective_fails(tmp_path):
    from bench.validate_submission import validate_file

    card = _signed_card()
    card["runs"][0]["objective"] = "not_an_objective"
    path = tmp_path / "card.json"
    path.write_text(json.dumps(card), encoding="utf-8")
    assert validate_file(path) != []


def test_real_pilot_card_validates_and_freezes_winner():
    from bench.validate_submission import validate_file

    card_path = ROOT / "backend" / "artifacts" / "cards" / "neural-pilot.json"
    if not card_path.exists():
        pytest.skip(
            "pilot card not present; private training runner not shipped in the public repo"
        )
    assert validate_file(card_path) == []
    card = json.loads(card_path.read_text(encoding="utf-8"))
    # The winner must be one of the evaluated ok runs, and every run recorded.
    assert card["winner"]["encoder"] in {r["encoder"] for r in card["runs"]}
    assert len(card["runs"]) >= 1
    assert all(r["status"] in {"ok", "failed"} for r in card["runs"])
