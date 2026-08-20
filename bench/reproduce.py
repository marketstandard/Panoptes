"""Self-check reproduction of signed scientific artifacts.

This is not an independent reproduction by an outside researcher. It is
the project's own verifier: re-hash every committed artifact, re-verify
the baseline catalog, and record original versus recomputed hashes.
An independent reproduction should run the same commands on a clean
checkout and fill docs/independent-reproduction.md.
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
from bench.validate_submission import FILES, validate_file  # noqa: E402

RECOMPUTE_FIXTURE = ROOT / "bench" / "fixtures" / "recompute-corpus.json"
METRIC_TOLERANCE = 1e-4

ARTIFACTS = [
    ROOT / "bench" / "protocol.json",
    ROOT / "bench" / "hypotheses.json",
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
    ROOT / "backend" / "artifacts" / "cards" / "measurement-protocol-defactify.json",
    ROOT / "backend" / "artifacts" / "cards" / "measurement-protocol-raid.json",
    ROOT / "backend" / "artifacts" / "cards" / "measurement-protocol-m4gt.json",
    ROOT / "backend" / "artifacts" / "cards" / "measurement-protocol-m4gtml.json",
    ROOT / "backend" / "artifacts" / "cards" / "measurement-protocol-evobench.json",
    ROOT / "backend" / "artifacts" / "cards" / "mixture-workflows.json",
    ROOT / "backend" / "artifacts" / "cards" / "robustness-pilot.json",
    ROOT / "backend" / "artifacts" / "cards" / "watermark-degradation.json",
    ROOT / "backend" / "artifacts" / "cards" / "watermarked-generations.json",
    ROOT / "backend" / "artifacts" / "cards" / "watermarked-paraphrases.json",
    ROOT / "backend" / "artifacts" / "cards" / "watermark-removal.json",
    ROOT / "backend" / "artifacts" / "transport-matrix-external.json",
    ROOT / "backend" / "artifacts" / "external-baselines.json",
    # --- v2.1 artifacts (Phase 0-4) ---
    ROOT / "bench" / "protocol-v2.1.json",
    ROOT / "backend" / "artifacts" / "dataset-registry-v2.1.json",
    ROOT / "backend" / "artifacts" / "split-manifest-v2.1.json",
    ROOT / "backend" / "artifacts" / "cards" / "mage-eval.json",
    ROOT / "backend" / "artifacts" / "cards" / "coauthor-eval.json",
    ROOT / "backend" / "artifacts" / "cards" / "neural-pilot.json",
    # --- v2.1 Phase 5 (frozen neural detector) ---
    ROOT / "backend" / "artifacts" / "cards" / "neural-detector.json",
    # --- v2.1 Phase 6 (evidence transportability), per detector tier ---
    ROOT / "backend" / "artifacts" / "cards" / "transport-logistic" / "representation-transport.json",
    ROOT / "backend" / "artifacts" / "cards" / "transport-logistic" / "calibration-transfer.json",
    ROOT / "backend" / "artifacts" / "cards" / "transport-gbm" / "representation-transport.json",
    ROOT / "backend" / "artifacts" / "cards" / "transport-gbm" / "calibration-transfer.json",
    ROOT / "backend" / "artifacts" / "cards" / "transport-heuristic" / "representation-transport.json",
    ROOT / "backend" / "artifacts" / "cards" / "transport-heuristic" / "calibration-transfer.json",
    ROOT / "backend" / "artifacts" / "cards" / "transport-neural" / "representation-transport.json",
    ROOT / "backend" / "artifacts" / "cards" / "transport-neural" / "calibration-transfer.json",
]


def _schema_ok(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_name = payload.get("schema")
    if schema_name not in FILES:
        return [f"{path.name}: schema {schema_name!r} is not in the validator registry"]
    return validate_file(path)


def recompute_metric_path() -> tuple[dict, list[str]]:
    """Recompute one full metric path (detector -> probabilities -> metrics).

    Loads the committed fixture, re-fits the ``logistic`` detector, re-scores the
    corpus, recomputes ``binary_metrics``, and compares against the committed
    expected values. This proves the metric code reproduces a committed result
    rather than merely re-hashing existing JSON.
    """
    errors: list[str] = []
    if not RECOMPUTE_FIXTURE.exists():
        return {}, [f"missing recompute fixture: {RECOMPUTE_FIXTURE.relative_to(ROOT)}"]
    try:
        import numpy as np

        from bench.datasets import Dataset
        from bench.detectors import make_detector
        from bench.evaluate import binary_metrics

        fixture = json.loads(RECOMPUTE_FIXTURE.read_text(encoding="utf-8"))
        texts = fixture["texts"]
        labels = np.asarray(fixture["labels"], dtype=int)
        dataset = Dataset(
            texts=texts,
            labels=labels,
            families=["human" if int(x) == 0 else "gpt" for x in labels],
            kinds=["text"] * len(texts),
            groups=[f"fx-{i}" for i in range(len(texts))],
            buckets=["short"] * len(texts),
            provenance="committed recompute fixture",
            sha256="0" * 64,
            domains=["fixture"] * len(texts),
        )
        idx = np.arange(len(dataset))
        detector = make_detector(fixture.get("detector", "logistic"))
        detector.fit(dataset, idx)
        probabilities = np.asarray(detector.predict_proba(dataset, idx), dtype=float)
        recomputed = binary_metrics(labels, probabilities)
        expected = fixture["expected_metrics"]
        compared = {}
        for key, exp in expected.items():
            got = recomputed.get(key)
            ok = got is not None and abs(float(got) - float(exp)) <= METRIC_TOLERANCE
            compared[key] = {"expected": exp, "recomputed": got, "match": bool(ok)}
            if not ok:
                errors.append(f"metric {key}: expected {exp}, recomputed {got}")
        result = {
            "fixture": str(RECOMPUTE_FIXTURE.relative_to(ROOT)),
            "detector": fixture.get("detector", "logistic"),
            "n_documents": len(texts),
            "metrics": compared,
            "recomputed_auroc": recomputed.get("auroc"),
            "recomputed_brier": recomputed.get("brier"),
        }
        return result, errors
    except Exception as exc:  # noqa: BLE001 — record the failure on the card
        return {}, [f"metric recomputation failed: {exc}"]


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

    metric_path, metric_errors = recompute_metric_path()
    errors.extend(metric_errors)

    card = {
        "schema": "panoptes-reproduction-selfcheck-v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "author_selfcheck",
        "independent": False,
        "catalog_verified": catalog_ok,
        "metric_path_recomputed": not metric_errors,
        "metric_path": metric_path,
        "n_artifacts": len(rows),
        "n_hash_matches": sum(1 for row in rows if row["match"]),
        "n_unsigned": sum(1 for row in rows if not row["original_sha256"]),
        "artifacts": rows,
        "errors": errors,
        "commands": [
            "python -m bench.reproduce",
            "python baselines/baseline.py verify-catalog",
            "python -m bench.validate_submission bench/protocol.json",
            "python -m pytest backend bench baselines",
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
