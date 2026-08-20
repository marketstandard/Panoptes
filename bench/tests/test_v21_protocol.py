"""Tests for the v2.1 preregistered protocol addendum and its schemas."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.protocol import (  # noqa: E402
    PROTOCOL_V21_SEEDS,
    canonical_hash,
    load_protocol_v21,
    validate_protocol_v21,
)


def test_v21_protocol_validates_and_self_hash_matches():
    payload = validate_protocol_v21()
    assert payload["schema"] == "panoptes-research-protocol-v2-1"
    assert payload["artifact_sha256"] == canonical_hash(payload)


def test_v21_protocol_seeds_and_candidate_grid():
    payload = load_protocol_v21()
    assert tuple(payload["seeds"]) == PROTOCOL_V21_SEEDS == (13, 42, 87)
    encoders = {e["id"] for e in payload["candidate_grid"]["encoders"]}
    assert {"deberta-v3-base", "modernbert-base", "deberta-v3-small"} <= encoders
    # Longformer is excluded by default; the windowing problem is already solved.
    assert any("longformer" in x.lower() for x in payload["candidate_grid"]["excluded_by_default"])


def test_v21_protocol_label_ontology_never_blends_channels():
    ontology = load_protocol_v21()["label_ontology"]
    for key in (
        "ai_participation",
        "ai_majority_generation",
        "ai_contribution_fraction",
        "source_family",
        "watermark",
        "provenance",
    ):
        assert key in ontology
    assert "never" in ontology["hard_rule"].lower()


def test_v21_protocol_primary_endpoints_have_directions():
    payload = load_protocol_v21()
    ids = {e["id"] for e in payload["primary_endpoints"]}
    assert "worst_cohort_auroc" in ids
    assert "calibration_transfer_delta" in ids
    assert "selective_risk_calfixed" in ids
    for endpoint in payload["primary_endpoints"]:
        assert endpoint["direction"]


def test_v21_release_gate_requires_license_and_reproducibility():
    gates = " ".join(load_protocol_v21()["release_gate"]["gates"]).lower()
    assert "license" in gates
    assert "reproducibility" in gates
    assert "calibration" in gates


@pytest.mark.parametrize(
    "schema_name",
    ["dataset-registry-v2-1.schema.json", "split-manifest-v2-1.schema.json"],
)
def test_v21_registry_and_split_schemas_are_valid_json_schema(schema_name):
    import jsonschema

    path = ROOT / "schemas" / schema_name
    assert path.exists(), schema_name
    schema = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)


def test_v21_split_manifest_requires_four_partitions():
    schema = json.loads(
        (ROOT / "schemas" / "split-manifest-v2-1.schema.json").read_text(encoding="utf-8")
    )
    required = schema["properties"]["cohorts"]["items"]["properties"]["partitions"]["required"]
    assert required == ["train", "development", "calibration", "test"]
