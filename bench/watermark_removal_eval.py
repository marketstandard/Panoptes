"""Watermark-removal evaluation: how much detection survives each attack.

Tests both watermark families against the removal battery:

* **KGW statistical** (green-list; the SynthID-Text / Aaronson family) — retention
  of the z-test on the signed watermarked generations.
* **Unicode zero-width** — survival of the embedded signature, the family that
  "Unicode hygiene" removers target.

A passive-evasion section runs the same attacks through the shipped heuristic
detector on the corpus, answering: does stripping the watermark also evade
passive AI detection? Kept separate from the passive attribution claim.
"""

from __future__ import annotations

import numpy as np

from bench.datasets import Dataset
from bench.detectors import HeuristicDetector
from bench.evaluate import binary_metrics
from bench.watermark_attacks import ATTACKS
from bench.watermark_unicode import detect_unicode_watermark, embed_unicode_watermark
from panoptes.analysis.watermarks import KGWReferenceAdapter
from panoptes.schemas import ContentType


def _kgw_summary(det: KGWReferenceAdapter, texts: list[str]) -> dict:
    rows = [det.detect(t, ContentType.PROSE)[0] for t in texts]
    tested = [r for r in rows if r.status == "tested" and r.p_value is not None]
    if not tested:
        return {"n": len(texts), "n_tested": 0}
    return {
        "n": len(texts),
        "n_tested": len(tested),
        "detection_rate_0.05": sum(1 for r in tested if r.p_value < 0.05) / len(tested),
        "mean_z": sum(r.z or 0.0 for r in tested) / len(tested),
        "mean_green_rate": sum(r.green_rate or 0.0 for r in tested) / len(tested),
    }


def _kgw_retention(
    wm_texts: list[str], ctrl_texts: list[str], para_texts: list[str] | None = None
) -> dict:
    det = KGWReferenceAdapter()
    baseline = _kgw_summary(det, wm_texts)
    per_attack = []
    for name, transform in ATTACKS.items():
        if name == "identity":
            continue
        after = _kgw_summary(det, [transform(t, i) for i, t in enumerate(wm_texts)])
        ctrl_after = _kgw_summary(det, [transform(t, i) for i, t in enumerate(ctrl_texts)])
        per_attack.append(
            {
                "attack": name,
                "detection_rate_before": baseline.get("detection_rate_0.05"),
                "detection_rate_after": after.get("detection_rate_0.05"),
                "mean_z_before": baseline.get("mean_z"),
                "mean_z_after": after.get("mean_z"),
                "mean_green_rate_after": after.get("mean_green_rate"),
                "control_fpr_after": ctrl_after.get("detection_rate_0.05"),
            }
        )
    if para_texts:
        after = _kgw_summary(det, para_texts)
        per_attack.append(
            {
                "attack": "llm_paraphrase",
                "detection_rate_before": baseline.get("detection_rate_0.05"),
                "detection_rate_after": after.get("detection_rate_0.05"),
                "mean_z_before": baseline.get("mean_z"),
                "mean_z_after": after.get("mean_z"),
                "mean_green_rate_after": after.get("mean_green_rate"),
                "control_fpr_after": None,
            }
        )
    return {
        "scheme": "kgw-v1",
        "n_watermarked": len(wm_texts),
        "n_control": len(ctrl_texts),
        "baseline": baseline,
        "control_baseline": _kgw_summary(det, ctrl_texts),
        "per_attack": per_attack,
    }


def _unicode_retention(ctrl_texts: list[str]) -> dict:
    base = [embed_unicode_watermark(t) for t in ctrl_texts]

    def summarize(texts: list[str]) -> dict:
        rows = [detect_unicode_watermark(t) for t in texts]
        n = len(rows)
        return {
            "n": n,
            "present_rate": sum(1 for r in rows if r["present"]) / n if n else None,
            "mean_payload_match": sum(r["payload_match"] for r in rows) / n if n else None,
            "mean_marks": sum(r["n_marks"] for r in rows) / n if n else None,
        }

    baseline = summarize(base)
    per_attack = []
    for name, transform in ATTACKS.items():
        if name == "identity":
            continue
        after = summarize([transform(t, i) for i, t in enumerate(base)])
        per_attack.append(
            {
                "attack": name,
                "present_rate_before": baseline["present_rate"],
                "present_rate_after": after["present_rate"],
                "payload_match_after": after["mean_payload_match"],
                "mean_marks_after": after["mean_marks"],
            }
        )
    return {"scheme": "unicode-zerowidth-v1", "n": len(base), "baseline": baseline, "per_attack": per_attack}


def _passive_evasion(corpus: Dataset) -> dict:
    detector = HeuristicDetector()
    detector.fit(corpus, np.arange(len(corpus)))
    n = len(corpus)

    def auroc(texts: list[str]) -> float:
        mutated = Dataset(
            texts=texts,
            labels=corpus.labels,
            families=corpus.families,
            kinds=corpus.kinds,
            groups=corpus.groups,
            buckets=corpus.buckets,
            provenance=corpus.provenance,
            sha256="0" * 64,
        )
        scores = detector.predict_proba(mutated, np.arange(n))
        return float(binary_metrics(mutated.labels, np.clip(scores, 1e-6, 1 - 1e-6))["auroc"])

    before = auroc(corpus.texts)
    per_attack = []
    for name, transform in ATTACKS.items():
        if name == "identity":
            continue
        after = auroc([transform(t, i) for i, t in enumerate(corpus.texts)])
        per_attack.append(
            {
                "attack": name,
                "auroc_before": before,
                "auroc_after": after,
                "delta_auroc": (None if before != before or after != after else after - before),
            }
        )
    return {"detector": detector.name, "dataset": corpus.provenance, "n": n, "per_attack": per_attack}


def watermark_removal_eval(
    generations_card: dict, corpus: Dataset, paraphrases_card: dict | None = None
) -> dict:
    samples = generations_card["samples"]
    wm_texts = [s["text"] for s in samples if s["kind"] == "watermarked"]
    ctrl_texts = [s["text"] for s in samples if s["kind"] == "control"]
    para_texts = (
        [s["text"] for s in paraphrases_card["samples"]] if paraphrases_card else None
    )
    attacks = [name for name in ATTACKS if name != "identity"]
    if para_texts:
        attacks = attacks + ["llm_paraphrase"]
    return {
        "schema": "panoptes-watermark-removal-eval-card-v1",
        "generations_ref": f"watermarked-generations.json sha256:{generations_card.get('artifact_sha256')}",
        "schemes": ["kgw-v1", "unicode-zerowidth-v1"],
        "attacks": attacks,
        "kgw": _kgw_retention(wm_texts, ctrl_texts, para_texts),
        "unicode": _unicode_retention(ctrl_texts),
        "passive": _passive_evasion(corpus),
        "external_repos": [],
        "limitations": [
            "KGW retention uses the demo-key watermarked generations (Aaronson/SynthID-Text family), not a vendor's private production key.",
            "The Unicode scheme is a reference zero-width signature; real vendor Unicode watermarks differ in payload but share the same fragility to hygiene.",
            "synonym_substitute is a deterministic stand-in for LLM paraphrase; a full neural rewrite is the strongest statistical attack and is evaluated via the external repo harness.",
            "Passive-evasion AUROC is a full-corpus in-sample shift, consistent with the robustness pilot; it is not blended into the watermark claim.",
        ],
    }
