"""Run every v1 protocol experiment that this environment can complete.

  python research/run_v1_experiments.py

Writes signed cards under backend/artifacts/ and signs research/protocol.json.
Does not download RAID/M4GT, train neural external detectors, or claim an
independent reproduction.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.cards import sign  # noqa: E402
from bench.datasets import load_verified_corpus  # noqa: E402
from bench.external_baselines import status_card  # noqa: E402
from bench.measure import _jsonable, run_measurement  # noqa: E402
from bench.mixtures import mixture_workflows  # noqa: E402
from bench.robustness import robustness_curve  # noqa: E402
from bench.watermarks_eval import watermark_degradation  # noqa: E402
from research.protocol import PROTOCOL_PATH, load_protocol  # noqa: E402
from research.reproduce import reproduce  # noqa: E402
from research.validate_submission import canonical_hash  # noqa: E402

CARDS = ROOT / "backend" / "artifacts" / "cards"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(name: str, payload: dict) -> Path:
    CARDS.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(payload)
    if "created_utc" not in payload:
        payload["created_utc"] = _now()
    sign(payload)
    path = CARDS / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}  sha256={payload['artifact_sha256'][:16]}…")
    return path


def sign_protocol() -> None:
    payload = load_protocol()
    payload.pop("artifact_sha256", None)
    payload["artifact_sha256"] = canonical_hash(payload)
    PROTOCOL_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"signed {PROTOCOL_PATH.relative_to(ROOT)}  sha256={payload['artifact_sha256'][:16]}…")


def sign_hypotheses() -> None:
    path = ROOT / "research" / "hypotheses.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("artifact_sha256", None)
    payload["artifact_sha256"] = canonical_hash(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"signed {path.relative_to(ROOT)}  sha256={payload['artifact_sha256'][:16]}…")


def main() -> int:
    sign_protocol()
    sign_hypotheses()
    dataset = load_verified_corpus()
    print(f"corpus n={len(dataset)} sha256={dataset.sha256[:16]}…")

    measurement = run_measurement(dataset)
    measurement["created_utc"] = _now()
    _write("measurement-protocol.json", measurement)

    _write(
        "mixture-workflows.json",
        {
            "schema": "panoptes-measurement-card-v1",
            "protocol_id": "mixture-workflows-pilot",
            "protocol_registered_utc": load_protocol()["registered_utc"],
            "dataset": dataset.provenance,
            "dataset_sha256": dataset.sha256,
            "n": len(dataset),
            "split": {"method": "paired-prompt", "n_groups": len(set(dataset.groups)), "n_splits": 1},
            "detectors": ["heuristic"],
            "metrics": {},
            "mixtures": mixture_workflows(dataset),
            "limitations": [
                "Token splices are proxies for coauthoring, not recorded human–AI editing sessions.",
                "Human controls are few; correlations are a pilot.",
            ],
        },
    )

    _write(
        "robustness-pilot.json",
        {
            "schema": "panoptes-measurement-card-v1",
            "protocol_id": "robustness-pilot",
            "protocol_registered_utc": load_protocol()["registered_utc"],
            "dataset": dataset.provenance,
            "dataset_sha256": dataset.sha256,
            "n": len(dataset),
            "split": {"method": "full-corpus-proxy", "n_groups": len(set(dataset.groups)), "n_splits": 1},
            "detectors": ["heuristic"],
            "metrics": robustness_curve(dataset),
            "limitations": [
                "Proxy edits only (truncate, drop, lowercase, strip punctuation, shuffle sentences).",
                "Not RAID, DIPPER, translation, or human adversarial rewriting.",
            ],
        },
    )

    watermark = watermark_degradation(dataset)
    sign(watermark)
    out = CARDS / "watermark-degradation.json"
    out.write_text(json.dumps(_jsonable(watermark), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}  sha256={watermark['artifact_sha256'][:16]}…")

    baselines = {**status_card(), "created_utc": _now(), "dataset": dataset.provenance}
    sign(baselines)
    path = ROOT / "backend" / "artifacts" / "external-baselines.json"
    path.write_text(json.dumps(_jsonable(baselines), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}  sha256={baselines['artifact_sha256'][:16]}…")
    repro = reproduce()
    repro_path = ROOT / "backend" / "artifacts" / "reproduction-selfcheck.json"
    repro_path.write_text(json.dumps(repro, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {repro_path.relative_to(ROOT)}  matches={repro['n_hash_matches']}/{repro['n_artifacts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
