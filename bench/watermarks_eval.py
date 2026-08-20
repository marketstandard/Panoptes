"""Watermark subsystem evaluation, kept separate from passive attribution.

Runs the public KGW-style adapter on the hash-verified corpus under
controlled degradation: untouched text, truncation, token drop, and
sentence shuffle. Reports empirical false-positive rate on human controls
and mean z / power on AI documents. This does not claim the corpus is
watermarked; it measures how the *test* behaves as text is edited.
"""

from __future__ import annotations

from panoptes.analysis.watermarks import KGWReferenceAdapter
from panoptes.schemas import ContentType

from bench.datasets import Dataset
from bench.robustness import drop_tokens, shuffle_sentences, truncate


def _content_type(kind: str) -> ContentType:
    return ContentType.CODE if kind == "code" else ContentType.PROSE


def _detect(text: str, kind: str):
    return KGWReferenceAdapter().detect(text, _content_type(kind))[0]


def _summarize(results: list) -> dict:
    tested = [row for row in results if row.status == "tested" and row.p_value is not None]
    if not tested:
        return {
            "n": len(results),
            "n_tested": 0,
            "empirical_reject_0.05": None,
            "mean_z": None,
            "mean_power": None,
        }
    rejects = sum(1 for row in tested if row.p_value < 0.05)
    return {
        "n": len(results),
        "n_tested": len(tested),
        "empirical_reject_0.05": rejects / len(tested),
        "mean_z": sum(row.z or 0.0 for row in tested) / len(tested),
        "mean_power": sum(row.power or 0.0 for row in tested) / len(tested),
        "mean_green_rate": sum((row.green_rate or 0.0) for row in tested) / len(tested),
    }


def watermark_degradation(dataset: Dataset) -> dict:
    conditions = {
        "untouched": lambda text, i: text,
        "truncate_75": lambda text, i: truncate(text, 0.75),
        "truncate_50": lambda text, i: truncate(text, 0.50),
        "drop_20": lambda text, i: drop_tokens(text, 0.20, seed=7 + i),
        "shuffle_sentences": lambda text, i: shuffle_sentences(text, seed=7 + i),
    }
    human_idx = [i for i, label in enumerate(dataset.labels) if int(label) == 0]
    ai_idx = [i for i, label in enumerate(dataset.labels) if int(label) == 1]
    rows = []
    for name, transform in conditions.items():
        human = [_detect(transform(dataset.texts[i], i), dataset.kinds[i]) for i in human_idx]
        ai = [_detect(transform(dataset.texts[i], i), dataset.kinds[i]) for i in ai_idx]
        rows.append(
            {
                "condition": name,
                "human": _summarize(human),
                "ai": _summarize(ai),
            }
        )
    fpr = next(
        row["human"]["empirical_reject_0.05"] for row in rows if row["condition"] == "untouched"
    )
    tpr = {
        row["condition"]: (
            row["ai"]["empirical_reject_0.05"]
            if row["ai"]["empirical_reject_0.05"] is not None
            else 0.0
        )
        for row in rows
    }
    retention = {
        row["condition"]: row["ai"]["mean_z"] if row["ai"]["mean_z"] is not None else 0.0
        for row in rows
    }
    return {
        "schema": "panoptes-watermark-eval-card-v1",
        "scheme": "kgw-v1",
        "adapter_version": "kgw-v1",
        "dataset_manifest_id": dataset.provenance,
        "green_fraction": 0.5,
        "min_tokens": 50,
        "metrics": {
            "empirical_fpr_unwatermarked": 0.0 if fpr is None else float(fpr),
            "tpr_by_token_bucket": tpr,
            "retention_after_edit": retention,
            "segment_localization_error": None,
            "q_value_calibration": "untested",
        },
        "conditions": rows,
        "limitations": [
            "The project corpus is not a watermarked generation set; "
            "AI documents are ordinary model outputs.",
            "Empirical reject rates therefore measure Type I behavior "
            "of the public detector, not TPR of a known watermark.",
            "Truncation, token drop, and sentence shuffle are proxy edits, "
            "not paraphrase, translation, or adversarial rewriting.",
            "Watermark results must not be blended into the passive AI-attribution claim.",
        ],
    }
