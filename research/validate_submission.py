from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"

FILES = {
    "panoptes-dataset-manifest-v1": "dataset-manifest.schema.json",
    "panoptes-calibration-v1": "calibration-artifact.schema.json",
    "panoptes-benchmark-card-v1": "benchmark-card.schema.json",
    "panoptes-watermark-eval-card-v1": "watermark-eval-card.schema.json",
}


def canonical_hash(payload: dict) -> str:
    clone = dict(payload)
    clone.pop("artifact_sha256", None)
    canonical = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_file(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_name = payload.get("schema")
    errors: list[str] = []
    if schema_name not in FILES:
        return [f"{path}: unknown schema {schema_name!r}"]
    schema_path = SCHEMAS / FILES[schema_name]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError as error:
        errors.append(f"{path}: {error.message}")
    expected = payload.get("artifact_sha256")
    if expected:
        actual = canonical_hash(payload)
        if actual != expected:
            errors.append(f"{path}: artifact_sha256 mismatch ({actual})")
    if schema_name == "panoptes-dataset-manifest-v1" and payload["privacy"]["raw_text_in_repo"]:
        errors.append(f"{path}: raw_text_in_repo must be false")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    for path in args.paths:
        errors.extend(validate_file(path))
    for error in errors:
        print(error)
    if errors:
        return 1
    print(f"Validated {len(args.paths)} artifact(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
