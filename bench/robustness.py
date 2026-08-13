"""Editing and truncation robustness pilots on a hash-verified corpus.

These are proxy attacks, not RAID-scale adversarial paraphrase. They answer:
does estimated AI participation degrade under truncation, token deletion,
and light surface edits that a human editor might apply?
"""

from __future__ import annotations

import re

import numpy as np

from bench.datasets import Dataset
from bench.detectors import HeuristicDetector
from bench.evaluate import binary_metrics
from bench.features import word_tokens


def truncate(text: str, keep: float) -> str:
    tokens = word_tokens(text)
    if not tokens:
        return text
    n = max(1, int(round(len(tokens) * min(max(keep, 0.0), 1.0))))
    return " ".join(tokens[:n])


def drop_tokens(text: str, drop: float, seed: int = 13) -> str:
    tokens = word_tokens(text)
    if len(tokens) < 4:
        return text
    rng = np.random.default_rng(seed)
    keep = rng.random(len(tokens)) >= min(max(drop, 0.0), 0.95)
    kept = [token for token, flag in zip(tokens, keep, strict=True) if flag]
    return " ".join(kept or tokens[:1])


def strip_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text)


def lowercase(text: str) -> str:
    return text.lower()


def shuffle_sentences(text: str, seed: int = 13) -> str:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(parts) < 2:
        return text
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(parts))
    return " ".join(parts[int(i)] for i in order)


TRANSFORMS = {
    "identity": lambda text, i: text,
    "truncate_75": lambda text, i: truncate(text, 0.75),
    "truncate_50": lambda text, i: truncate(text, 0.50),
    "truncate_25": lambda text, i: truncate(text, 0.25),
    "drop_20": lambda text, i: drop_tokens(text, 0.20, seed=13 + i),
    "lowercase": lambda text, i: lowercase(text),
    "strip_punctuation": lambda text, i: strip_punctuation(text),
    "shuffle_sentences": lambda text, i: shuffle_sentences(text, seed=13 + i),
}


def robustness_curve(dataset: Dataset) -> dict:
    detector = HeuristicDetector()
    detector.fit(dataset, np.arange(len(dataset)))
    rows = []
    for name, transform in TRANSFORMS.items():
        texts = [transform(text, i) for i, text in enumerate(dataset.texts)]
        mutated = Dataset(
            texts=texts,
            labels=dataset.labels,
            families=dataset.families,
            kinds=dataset.kinds,
            groups=dataset.groups,
            buckets=dataset.buckets,
            provenance=f"{dataset.provenance}+{name}",
            sha256="0" * 64,
        )
        scores = detector.predict_proba(mutated, np.arange(len(mutated)))
        metrics = binary_metrics(mutated.labels, np.clip(scores, 1e-6, 1 - 1e-6))
        rows.append({"transform": name, "n": len(mutated), **metrics})
    identity = next(row for row in rows if row["transform"] == "identity")
    for row in rows:
        row["delta_auroc"] = (
            None
            if row["auroc"] != row["auroc"] or identity["auroc"] != identity["auroc"]
            else float(row["auroc"] - identity["auroc"])
        )
    return {
        "detector": detector.name,
        "n": len(dataset),
        "transforms": rows,
        "note": (
            "Proxy edits on the project corpus. Not RAID adversarial paraphrase, "
            "DIPPER, or human rewriting. Truncation below the detector token floor "
            "forces abstention (raw_score=0.5)."
        ),
    }
