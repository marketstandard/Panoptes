from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calibration import evaluate_binary, expected_calibration_error, tpr_at_fpr


def segment_change_points(values: list[float], threshold: float = 0.25) -> list[int]:
    points: list[int] = []
    for index in range(1, len(values)):
        if abs(values[index] - values[index - 1]) >= threshold:
            points.append(index)
    return points


def adversarial_variants(text: str) -> dict[str, str]:
    return {
        "copy": text,
        "synonym": text.replace("important", "significant").replace("repair", "fix"),
        "format": text.replace("\n", " "),
        "translation_proxy": text.replace("the ", "le ").replace("and ", "et "),
        "mixed": text[: len(text) // 2] + " I then rewrote the remainder in my own words.",
    }


def write_benchmark_card(output: Path) -> None:
    labels = np.array([0] * 100 + [1] * 100)
    probabilities = np.array([0.2] * 80 + [0.5] * 20 + [0.8] * 80 + [0.45] * 20)
    metrics = evaluate_binary(labels, probabilities)
    metrics["ece_20_bins"] = expected_calibration_error(labels, probabilities, bins=20)
    metrics["tpr_at_1fpr"] = tpr_at_fpr(labels, probabilities, 0.01)
    payload = {
        "schema": "panoptes-benchmark-card-v1",
        "dataset": "synthetic-development",
        "metrics": metrics,
        "limitations": [
            "Synthetic values are placeholders until RAID-derived calibration is downloaded and approved.",
            "Report separate prose/code metrics and fixed-FPR operating points in release cards.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_benchmark_card(Path(__file__).resolve().parents[1] / "docs" / "benchmark-card.json")
