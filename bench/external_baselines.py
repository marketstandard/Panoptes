"""Optional external detector adapters.

Binoculars, DetectGPT / Fast-DetectGPT, and large transformer classifiers
require model weights and (usually) a GPU. This module exposes a uniform
surface so they can be scored on the same protocol splits when the
operator installs the extras. When weights are absent the adapters report
`unavailable` rather than silently substituting a different detector.
"""

from __future__ import annotations

from typing import Any

UNAVAILABLE = {
    "binoculars": {
        "name": "Binoculars",
        "citation": "Hans et al. 2024, ICML",
        "requires": "transformers + two observer/performer LMs",
        "status": "unavailable",
    },
    "detectgpt": {
        "name": "DetectGPT",
        "citation": "Mitchell et al. 2023, ICML",
        "requires": "source LM + perturbation model",
        "status": "unavailable",
    },
    "fast_detectgpt": {
        "name": "Fast-DetectGPT",
        "citation": "Bao et al. 2024, ICLR",
        "requires": "source/reference LM pair",
        "status": "unavailable",
    },
    "transformer_classifier": {
        "name": "Transformer classifier (RoBERTa-style)",
        "citation": "standard fine-tuned detector",
        "requires": "fine-tuned classifier weights + GPU",
        "status": "unavailable",
    },
}


def catalog() -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in UNAVAILABLE.items()}


def status_card() -> dict:
    return {
        "schema": "panoptes-external-baselines-v1",
        "detectors": catalog(),
        "note": (
            "External zero-shot and neural detectors are registered here so that "
            "a later run can score them on the frozen protocol splits. This "
            "environment does not ship their weights. Do not substitute the "
            "Panoptes heuristic for an unavailable external detector."
        ),
    }
