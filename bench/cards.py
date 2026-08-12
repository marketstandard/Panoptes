"""Model cards and datasheets, canonically signed (SHA-256).

Card schema follows Mitchell et al. 2019 (model cards) and Gebru et al.
2021 (datasheets): intended use, training/evaluation data provenance,
metrics with uncertainty, fairness slices, gate rationale, limitations.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from bench.datasets import Dataset

CARD_SCHEMA = "panoptes-model-card-v1"


def canonical_hash(payload: dict) -> str:
    clone = dict(payload)
    clone.pop("artifact_sha256", None)
    canonical = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def sign(payload: dict) -> dict:
    payload["artifact_sha256"] = canonical_hash(payload)
    return payload


def verify_card(card: dict) -> bool:
    return card.get("artifact_sha256") == canonical_hash(card)


def dataset_datasheet(dataset: Dataset) -> dict:
    return {
        "motivation": (
            "AI-text detection evaluation corpus. The project corpus is hash-verified "
            "from run manifests; community datasets are contributor-supplied."
        ),
        "composition": {
            "n": len(dataset),
            "n_human": int((dataset.labels == 0).sum()),
            "n_ai": int((dataset.labels == 1).sum()),
            "families": sorted(set(dataset.families)),
            "kinds": sorted(set(dataset.kinds)),
            "length_buckets": {b: dataset.buckets.count(b) for b in sorted(set(dataset.buckets))},
            "sha256": dataset.sha256,
        },
        "collection": (
            "Reference runs: single-turn responses to 8 pinned prompts per kind, product "
            "defaults. Human controls: original texts written by project authors. "
            "Community datasets: contributor-declared labels."
        ),
        "labeling": "Labels are declared at ingestion and verified structurally (manifest hashes); content labels are contributor claims.",
        "recommended_uses": "Model evaluation and calibration research. Not for training generators.",
        "distribution": "Hash-pointer manifests only for community data; raw text stays with the contributor.",
        "maintenance": "Maintained by the Panoptes project; corrections via pull request.",
    }


def model_card(
    *,
    model_name: str,
    tier: int,
    dataset: Dataset,
    evaluation: dict,
    gate: dict | None,
    config: dict,
    limitations: list[str] | None = None,
    extra: dict | None = None,
    created_utc: str | None = None,
) -> dict:
    card = {
        "schema": CARD_SCHEMA,
        "model": {"name": model_name, "tier": tier},
        "created_utc": created_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "intended_use": (
            "Research-grade AI-text detection support with visible uncertainty. "
            "Not for high-stakes automated decisions about individuals."
        ),
        "training_data": {
            "provenance": dataset.provenance,
            "sha256": dataset.sha256,
            "n": len(dataset),
            "datasheet": dataset_datasheet(dataset),
        },
        "config": config,
        "evaluation": {
            "protocol": (
                f"GroupKFold({evaluation.get('n_splits', 5)}) by prompt group; all metrics "
                "out-of-fold."
            ),
            "metrics": evaluation["metrics"],
            "auroc_ci95": evaluation["auroc_ci95"],
            "reliability_bins": evaluation["reliability_bins"],
            "coverage_curve": evaluation["coverage_curve"],
            "conformal": evaluation["conformal"],
            "fairness_slices": evaluation["fairness_slices"],
            "folds": evaluation["folds"],
        },
        "power_gate": gate,
        "limitations": limitations
        or [
            "Small corpus: absolute performance estimates carry wide intervals.",
            "Stylometric signals degrade under paraphrase and heavy editing.",
            "Family labels are contributor-declared.",
        ],
    }
    if extra:
        card.update(extra)
    return sign(card)


def write_card(card: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
