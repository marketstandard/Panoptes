"""Calibration pipeline tests against the signed public artifact (no private runner)."""

from __future__ import annotations

import json
from pathlib import Path

from bench.validate_submission import validate_file

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "backend" / "artifacts" / "baseline-calibration.json"
DEFACTIFY = ROOT / "backend" / "artifacts" / "defactify-calibration.json"


def test_baseline_calibration_artifact_validates() -> None:
    assert ARTIFACT.exists()
    assert validate_file(ARTIFACT) == []
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["schema"] == "panoptes-calibration-v1"
    assert payload["bundle_id"]
    assert "binary_calibrator" in payload


def test_defactify_calibration_artifact_validates() -> None:
    assert DEFACTIFY.exists()
    assert validate_file(DEFACTIFY) == []
    payload = json.loads(DEFACTIFY.read_text(encoding="utf-8"))
    assert payload["schema"] == "panoptes-calibration-v1"
    assert payload["bundle_id"]
    assert "binary_calibrator" in payload
