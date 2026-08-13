"""Controlled human–AI mixture experiments.

Pair a human control with an AI response to the same prompt and splice
tokens at a declared AI-contribution rate. The scientific question is
whether estimated P(AI participation) tracks the controlled rate.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from bench.datasets import Dataset
from bench.detectors import Detector, HeuristicDetector
from bench.features import word_tokens
from research.protocol import MIXTURE_RATES


def mix_tokens(human: str, ai: str, rate: float) -> str:
    """Replace the first `rate` fraction of human tokens with AI tokens."""
    human_tokens = word_tokens(human)
    ai_tokens = word_tokens(ai)
    if not human_tokens:
        return ai if rate >= 1 else human
    n_ai = int(round(len(human_tokens) * min(max(rate, 0.0), 1.0)))
    n_ai = min(n_ai, len(ai_tokens), len(human_tokens))
    mixed = list(ai_tokens[:n_ai]) + list(human_tokens[n_ai:])
    return " ".join(mixed) if mixed else human


def prompt_pairs(dataset: Dataset) -> list[tuple[int, int]]:
    """(human_index, ai_index) pairs that share a group/prompt."""
    by_group: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"human": [], "ai": []})
    for index, (group, label) in enumerate(zip(dataset.groups, dataset.labels, strict=True)):
        by_group[group]["human" if int(label) == 0 else "ai"].append(index)
    pairs: list[tuple[int, int]] = []
    for parts in by_group.values():
        for human_i in parts["human"]:
            for ai_i in parts["ai"]:
                if dataset.kinds[human_i] == dataset.kinds[ai_i]:
                    pairs.append((human_i, ai_i))
    return pairs


def mixture_curve(
    dataset: Dataset,
    detector: Detector | None = None,
    rates: tuple[float, ...] = MIXTURE_RATES,
) -> dict:
    """Score mixed documents and summarize tracking of controlled participation."""
    detector = detector or HeuristicDetector()
    pairs = prompt_pairs(dataset)
    if not pairs:
        return {"n_pairs": 0, "rates": [], "skipped": "no human/AI pairs share a group"}

    train_idx = np.arange(len(dataset))
    detector.fit(dataset, train_idx)

    rows = []
    for rate in rates:
        texts = [mix_tokens(dataset.texts[h], dataset.texts[a], rate) for h, a in pairs]
        kinds = [dataset.kinds[h] for h, _ in pairs]
        scores = np.array(
            [
                float(detector.predict_proba(_one_row(text, kind), np.array([0]))[0])
                for text, kind in zip(texts, kinds, strict=True)
            ]
        )
        rows.append(
            {
                "actual_ai_rate": rate,
                "mean_estimated": float(scores.mean()),
                "std_estimated": float(scores.std(ddof=0)),
                "n": int(len(scores)),
            }
        )

    actual = np.array([row["actual_ai_rate"] for row in rows])
    estimated = np.array([row["mean_estimated"] for row in rows])
    if len(actual) >= 2 and estimated.std() > 0 and actual.std() > 0:
        corr = float(np.corrcoef(actual, estimated)[0, 1])
        slope = float(np.polyfit(actual, estimated, 1)[0])
    else:
        corr, slope = float("nan"), float("nan")
    mae = float(np.mean(np.abs(actual - estimated)))
    return {
        "n_pairs": len(pairs),
        "rates": rows,
        "correlation": corr,
        "slope": slope,
        "mae": mae,
        "detector": detector.name,
    }


def _one_row(text: str, kind: str) -> Dataset:
    return Dataset(
        texts=[text],
        labels=np.array([1]),
        families=["mixed"],
        kinds=[kind],
        groups=["mixed"],
        buckets=["50-149"],
        provenance="mixture",
        sha256="0" * 64,
    )
