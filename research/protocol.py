"""Load and validate the frozen Panoptes research protocol."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "research" / "protocol.json"
SCHEMA_PATH = ROOT / "schemas" / "research-protocol.schema.json"

PREVALENCE_GRID = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75)
COVERAGE_LEVELS = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)
MIXTURE_RATES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
NESTED_CV_GROUP_THRESHOLD = 30


def load_protocol(path: Path = PROTOCOL_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol(path: Path = PROTOCOL_PATH) -> dict:
    import jsonschema

    payload = load_protocol(path)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)
    return payload
