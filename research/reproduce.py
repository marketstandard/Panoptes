"""Self-check reproduction of signed scientific artifacts.

This is not an independent reproduction by an outside researcher. It is
the project's own verifier: re-hash every committed artifact, re-verify
the baseline catalog, and record original versus recomputed hashes.
An independent reproduction should run the same commands on a clean
checkout and fill docs/v2-updates/independent-reproduction.md.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.cards import canonical_hash, sign  # noqa: E402
from research.validate_submission import FILES, validate_file  # noqa: E402

ARTIFACTS = [
    ROOT / "research" / "protocol.json",
    ROOT / "research" / "hypotheses.json",
    ROOT / "backend" / "artifacts" / "corpus-summary.json",
    ROOT / "backend" / "artifacts" / "methodology-report.json",
    ROOT / "backend" / "artifacts" / "baseline-calibration.json",
    ROOT / "backend" / "artifacts" / "defactify-summary.json",
    ROOT / "backend" / "artifacts" / "defactify-calibration.json",
    ROOT / "backend" / "artifacts" / "panoptes-v0-card.json",
    ROOT / "backend" / "artifacts" / "cards" / "logistic-tier0.json",
    ROOT / "backend" / "artifacts" / "cards" / "logistic-tier0-defactify.json",
    ROOT / "backend" / "artifacts" / "cards" / "gbm-tier1-defactify.json",
    ROOT / "backend" / "artifacts" / "cards" / "defactify-external-validation.json",
    ROOT / "backend" / "artifacts" / "cards" / "attribution-defactify.json",
    ROOT / "backend" / "artifacts" / "cards" / "measurement-protocol.json",
    ROOT / "backend" / "artifacts" / "cards" / "mixture-workflows.json",
    ROOT / "backend" / "artifacts" / "cards" / "robustness-pilot.json",
    ROOT / "backend" / "artifacts" / "cards" / "watermark-degradation.json",
    ROOT / "backend" / "artifacts" / "external-baselines.json",
]


def _schema_ok(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_name = payload.get("schema")
    if schema_name not in FILES:
        return [f"{path.name}: schema {schema_name!r} is not in the validator registry"]
    return validate_file(path)


def reproduce() -> dict:
    rows = []
    errors: list[str] = []
    for path in ARTIFACTS:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        original = payload.get("artifact_sha256")
        recomputed = canonical_hash(payload) if "schema" in payload else None
        schema_errors = _schema_ok(path) if payload.get("schema") in FILES else []
        match = original is not None and original == recomputed
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "schema": payload.get("schema"),
                "original_sha256": original,
                "recomputed_sha256": recomputed,
                "match": match,
                "schema_errors": schema_errors,
            }
        )
        if original and recomputed and not match:
            errors.append(f"hash mismatch: {path.relative_to(ROOT)}")
        errors.extend(schema_errors)

    catalog_ok = True
    try:
        from baselines import baseline as baseline_mod

        if baseline_mod.main(["verify-catalog"]) != 0:
            catalog_ok = False
            errors.append("catalog: verify-catalog exited non-zero")
    except Exception as exc:  # noqa: BLE001 — record the failure on the card
        catalog_ok = False
        errors.append(f"catalog: {exc}")

    card = {
        "schema": "panoptes-reproduction-selfcheck-v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "author_selfcheck",
        "independent": False,
        "catalog_verified": catalog_ok,
        "n_artifacts": len(rows),
        "n_hash_matches": sum(1 for row in rows if row["match"]),
        "n_unsigned": sum(1 for row in rows if not row["original_sha256"]),
        "artifacts": rows,
        "errors": errors,
        "commands": [
            "python research/reproduce.py",
            "python baselines/baseline.py verify-catalog",
            "python research/validate_submission.py research/protocol.json",
            "python -m pytest backend research bench baselines",
        ],
        "limitations": [
            "This is a first-party self-check, not an independent reproduction.",
            "Headline metric deltas versus an outside researcher are not yet available.",
        ],
    }
    return sign(card)


def main() -> int:
    card = reproduce()
    out = ROOT / "backend" / "artifacts" / "reproduction-selfcheck.json"
    out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({card['n_hash_matches']}/{card['n_artifacts']} signed hashes match)")
    for error in card["errors"]:
        print(f"error: {error}")
    failed = bool(card["errors"]) or card["n_hash_matches"] != card["n_artifacts"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
