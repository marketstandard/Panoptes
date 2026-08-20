"""Load and validate the frozen Panoptes research protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "bench" / "protocol.json"
SCHEMA_PATH = ROOT / "schemas" / "research-protocol.schema.json"
PROTOCOL_V21_PATH = ROOT / "bench" / "protocol-v2.1.json"
SCHEMA_V21_PATH = ROOT / "schemas" / "research-protocol-v2-1.schema.json"

PREVALENCE_GRID = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75)
COVERAGE_LEVELS = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)
MIXTURE_RATES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
NESTED_CV_GROUP_THRESHOLD = 30
PROTOCOL_V21_SEEDS = (13, 42, 87)


def canonical_hash(payload: dict) -> str:
    """SHA-256 over the canonical JSON, excluding any existing self-hash."""
    clone = dict(payload)
    clone.pop("artifact_sha256", None)
    canonical = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol(path: Path = PROTOCOL_PATH) -> dict:
    import jsonschema

    payload = load_protocol(path)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)
    return payload


def load_protocol_v21(path: Path = PROTOCOL_V21_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol_v21(path: Path = PROTOCOL_V21_PATH) -> dict:
    """Schema-validate the v2.1 addendum and verify its self-hash."""
    import jsonschema

    payload = load_protocol_v21(path)
    schema = json.loads(SCHEMA_V21_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)
    recorded = payload.get("artifact_sha256", "")
    if recorded != canonical_hash(payload):
        raise ValueError("protocol-v2.1.json self-hash mismatch; re-sign after editing")
    return payload


def sign_protocol_v21(path: Path = PROTOCOL_V21_PATH) -> str:
    """Recompute and store the v2.1 addendum self-hash. Returns the hash."""
    payload = load_protocol_v21(path)
    payload["artifact_sha256"] = canonical_hash(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload["artifact_sha256"]
