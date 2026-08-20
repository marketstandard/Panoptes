"""Generate the committed metric-recomputation fixture.

The fixture is a tiny, fully committed corpus (16 short documents, 8 human / 8
AI) together with the expected output of one full metric path: the ``logistic``
bench detector is fit and scored on the corpus, then ``binary_metrics`` is
computed on the resulting probabilities. ``bench/reproduce.py`` reloads this
fixture, re-runs the same detector -> probabilities -> metrics path, and asserts
the recomputed metrics match the committed values. This makes reproduction
recompute a real metric path instead of merely re-hashing existing JSON.

Regenerate (only if the detector or metric definitions intentionally change):
    .venv/Scripts/python.exe bench/fixtures/make_recompute_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.datasets import Dataset  # noqa: E402
from bench.detectors import make_detector  # noqa: E402
from bench.evaluate import binary_metrics  # noqa: E402

OUT = Path(__file__).resolve().parent / "recompute-corpus.json"
SEED = 13

HUMAN = [
    "the cat sat on the warm mat near the window",
    "i went to the store yesterday and bought milk",
    "my dog loves to play fetch in the park every morning",
    "the recipe calls for two cups of flour and sugar",
    "she walked her bicycle down the quiet street",
    "we had dinner together at a small restaurant",
    "the kids played outside until it got dark",
    "he fixed the leaky faucet in the bathroom",
]
AI = [
    "Furthermore, we utilize comprehensive methodologies "
    "to demonstrate substantial improvements across benchmarks.",
    "Moreover, the implementation leverages state-of-the-art techniques "
    "to optimize performance metrics.",
    "Consequently, the proposed framework facilitates robust generalization "
    "across diverse distribution shifts.",
    "Additionally, our approach demonstrates significant enhancements "
    "in computational efficiency and scalability.",
    "Subsequently, the empirical evaluation reveals considerable advancements "
    "over existing baseline methodologies.",
    "Nevertheless, the comprehensive analysis underscores "
    "the importance of rigorous experimental design.",
    "Therefore, we present a novel architecture "
    "that effectively addresses these fundamental challenges.",
    "Overall, the extensive experiments substantiate "
    "the effectiveness of our proposed methodological contributions.",
]


def build_dataset() -> Dataset:
    texts = HUMAN + AI
    labels = np.array([0] * len(HUMAN) + [1] * len(AI), dtype=int)
    return Dataset(
        texts=texts,
        labels=labels,
        families=["human"] * len(HUMAN) + ["gpt"] * len(AI),
        kinds=["text"] * len(texts),
        groups=[f"fx-{i}" for i in range(len(texts))],
        buckets=["short"] * len(texts),
        provenance="committed recompute fixture",
        sha256="0" * 64,
        domains=["fixture"] * len(texts),
    )


def run_metric_path() -> tuple[Dataset, np.ndarray, dict]:
    dataset = build_dataset()
    idx = np.arange(len(dataset))
    detector = make_detector("logistic")
    detector.fit(dataset, idx)
    probabilities = np.asarray(detector.predict_proba(dataset, idx), dtype=float)
    return dataset, probabilities, binary_metrics(dataset.labels, probabilities)


def main() -> int:
    dataset, probabilities, metrics = run_metric_path()
    fixture = {
        "schema": "panoptes-recompute-fixture-v1",
        "detector": "logistic",
        "seed": SEED,
        "texts": list(dataset.texts),
        "labels": [int(x) for x in dataset.labels],
        "probabilities": [round(float(p), 6) for p in probabilities],
        "expected_metrics": {k: round(float(v), 6) for k, v in metrics.items()},
    }
    OUT.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT} (auroc={metrics['auroc']:.4f} brier={metrics['brier']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
