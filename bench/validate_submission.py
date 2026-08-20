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
    "panoptes-baseline-prompts-v1": "baseline-prompts.schema.json",
    "panoptes-baseline-run-v1": "baseline-run.schema.json",
    "panoptes-corpus-summary-v1": "corpus-summary.schema.json",
    "panoptes-methodology-v1": "methodology-report.schema.json",
    "panoptes-model-card-v1": "model-card.schema.json",
    "panoptes-bench-dataset-v1": "bench-dataset.schema.json",
    "panoptes-defactify-summary-v1": "defactify-summary.schema.json",
    "panoptes-research-protocol-v1": "research-protocol.schema.json",
    "panoptes-measurement-card-v1": "measurement-card.schema.json",
    "panoptes-hypotheses-v1": "hypotheses.schema.json",
    "panoptes-external-baselines-v1": "external-baselines.schema.json",
    "panoptes-reproduction-selfcheck-v1": "reproduction-selfcheck.schema.json",
    "panoptes-transport-matrix-v1": "transport-matrix.schema.json",
    "panoptes-watermarked-generations-v1": "watermarked-generations.schema.json",
    "panoptes-watermark-removal-eval-card-v1": "watermark-removal-eval-card.schema.json",
    "panoptes-watermark-temperature-card-v1": "watermark-temperature-card.schema.json",
    "panoptes-radioactivity-card-v1": "radioactivity-card.schema.json",
    "panoptes-external-repo-eval-v1": "external-repo-eval.schema.json",
    "panoptes-research-protocol-v2-1": "research-protocol-v2-1.schema.json",
    "panoptes-dataset-registry-v2-1": "dataset-registry-v2-1.schema.json",
    "panoptes-split-manifest-v2-1": "split-manifest-v2-1.schema.json",
    "panoptes-mage-eval-v1": "mage-eval-card.schema.json",
    "panoptes-coauthor-eval-v1": "coauthor-eval-card.schema.json",
    "panoptes-neural-pilot-v1": "neural-pilot-card.schema.json",
    "panoptes-neural-detector-v1": "neural-detector-card.schema.json",
    "panoptes-representation-transport-v1": "representation-transport-card.schema.json",
    "panoptes-calibration-transfer-v1": "calibration-transfer-card.schema.json",
}

BASELINE_OUTPUT_KEYS = {"prompt_id", "file", "sha256", "bytes"}


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
    if schema_name == "panoptes-baseline-run-v1":
        for output in payload.get("outputs", []):
            if set(output) - BASELINE_OUTPUT_KEYS:
                errors.append(
                    f"{path}: baseline outputs carry hash metadata only; "
                    "raw model text must never be embedded in a manifest"
                )
                break
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
