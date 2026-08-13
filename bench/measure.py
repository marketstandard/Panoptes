"""Run the frozen measurement protocol and write a signed card.

  python -m bench measure --data corpus
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from bench.datasets import Dataset
from bench.detectors import HeuristicDetector, make_detector
from bench.evaluate import evaluate_protocol, leave_one_family_out, transport_matrix
from bench.evidence import compare_dependence
from bench.features import heuristic_raw_score
from bench.mixtures import mixture_curve
from bench.splits import protocol_splits
from research.protocol import load_protocol


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _dependence_pilot(dataset: Dataset, n_segments: int = 3, limit: int = 24) -> dict:
    """Compare naive, correlated, and document-level evidence on text documents."""
    rows = []
    prevalence = float(dataset.labels.mean())
    count = 0
    for text, kind, label in zip(dataset.texts, dataset.kinds, dataset.labels, strict=True):
        if kind != "text":
            continue
        words = text.split()
        if len(words) < n_segments * 12:
            continue
        bounds = [len(words) * i // n_segments for i in range(n_segments + 1)]
        segment_p = np.array(
            [
                heuristic_raw_score(" ".join(words[bounds[i] : bounds[i + 1]]), "text")
                for i in range(n_segments)
            ]
        )
        document_p = heuristic_raw_score(text, "text")
        compared = compare_dependence(segment_p, document_p, prevalence)
        rows.append(
            {
                "label": int(label),
                "document_p": float(document_p),
                "naive_llr": compared["naive_sum"]["llr"],
                "correlated_llr": compared["correlated_shrinkage"]["llr"],
                "document_llr": compared["document_level"]["llr"],
                "rho": compared["correlated_shrinkage"]["rho"],
            }
        )
        count += 1
        if count >= limit:
            break
    if not rows:
        return {"n_documents": 0}
    naive = np.array([row["naive_llr"] for row in rows])
    correlated = np.array([row["correlated_llr"] for row in rows])
    document = np.array([row["document_llr"] for row in rows])
    return {
        "n_documents": len(rows),
        "mean_naive_llr": float(naive.mean()),
        "mean_correlated_llr": float(correlated.mean()),
        "mean_document_llr": float(document.mean()),
        "mean_rho": float(np.mean([row["rho"] for row in rows])),
        "naive_vs_document_mae": float(np.mean(np.abs(naive - document))),
        "note": (
            "Naive accumulation inflates evidence when segments are correlated. "
            "A non-rejected Durbin-Watson test is not treated as proof of independence."
        ),
    }


def run_measurement(dataset: Dataset, detectors: tuple[str, ...] = ("heuristic", "logistic")) -> dict:
    protocol = load_protocol()
    splits = protocol_splits(dataset)
    detector_metrics = {}
    for name in detectors:
        result = evaluate_protocol(lambda n=name: make_detector(n), dataset)
        detector_metrics[name] = {
            "method": result["method"],
            "n_splits": result["n_splits"],
            "n_groups": result["n_groups"],
            "metrics": result["metrics"],
            "selective_risk": result["selective_risk"],
            "conformal": result["conformal"],
            "prior_sensitivity": result["prior_sensitivity"],
            "mean_likelihood_ratio": result["mean_likelihood_ratio"],
        }
    return _jsonable(
        {
            "schema": "panoptes-measurement-card-v1",
            "protocol_id": protocol["title"],
            "protocol_registered_utc": protocol["registered_utc"],
            "dataset": dataset.provenance,
            "dataset_sha256": dataset.sha256,
            "n": len(dataset),
            "split": {
                "method": splits[0].method.split(":")[0],
                "n_groups": splits[0].n_groups,
                "n_splits": len(splits),
                "rule": protocol["split_rules"]["hard_rule"],
            },
            "detectors": list(detectors),
            "metrics": detector_metrics,
            "transport": transport_matrix(HeuristicDetector, dataset, axis="domains"),
            "mixtures": mixture_curve(dataset, HeuristicDetector()),
            "open_set": leave_one_family_out(dataset),
            "dependence": _dependence_pilot(dataset),
            "limitations": [
                "Train, calibration, and test groups are disjoint; calibration is never fit on test.",
                "Human controls in the project corpus are few; mixture and calibration estimates are a pilot.",
                "Transport cells are omitted when a domain lacks both classes.",
                "Watermark evaluation is a separate subsystem and is not included in this card.",
            ],
        }
    )
