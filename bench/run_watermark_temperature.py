"""Temperature sweep for KGW detection power, including the greedy dead zone.

  python -m bench.run_watermark_temperature

Sweeps temperature x delta, generates short passages with the demo KGW key,
scores them with KGWReferenceAdapter, and writes a signed card to
backend/artifacts/cards/watermark-temperature.json.

By default this runner builds a *synthetic* card from the pure-numpy sampling
math (no model download) so CI stays offline. Pass ``--with-model`` to run the
real gpt2-family generator (requires torch/transformers).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panoptes.analysis.watermarks import KGWReferenceAdapter  # noqa: E402
from panoptes.schemas import ContentType  # noqa: E402

from bench.cards import sign  # noqa: E402
from bench.watermark_gen import biased_pick  # noqa: E402

CARDS = ROOT / "backend" / "artifacts" / "cards"
DEFAULT_TEMPERATURES = [0.0, 0.3, 0.7, 1.0, 1.3]
DEFAULT_DELTAS = [0.0, 2.0]
VOCAB = [
    "the",
    "a",
    "and",
    "of",
    "to",
    "in",
    "is",
    "for",
    "on",
    "with",
    "that",
    "this",
    "from",
    "by",
    "as",
    "at",
    "are",
    "be",
    "or",
    "an",
    "city",
    "team",
    "study",
    "report",
    "new",
    "local",
    "public",
    "data",
    "result",
    "method",
    "system",
    "model",
    "group",
    "year",
    "day",
    "time",
    ".",
    ",",
    "!",
    "?",
    ";",
    ":",
]


def _summarize(det: KGWReferenceAdapter, texts: list[str]) -> dict:
    rows = [det.detect(t, ContentType.PROSE)[0] for t in texts]
    tested = [r for r in rows if r.status == "tested" and r.p_value is not None]
    if not tested:
        return {
            "n": len(texts),
            "n_tested": 0,
            "detection_rate_0.05": None,
            "mean_z": None,
            "mean_power": None,
            "mean_green_rate": None,
        }
    return {
        "n": len(texts),
        "n_tested": len(tested),
        "detection_rate_0.05": sum(1 for r in tested if r.p_value < 0.05) / len(tested),
        "mean_z": sum(r.z or 0.0 for r in tested) / len(tested),
        "mean_power": sum(r.power or 0.0 for r in tested) / len(tested),
        "mean_green_rate": sum(r.green_rate or 0.0 for r in tested) / len(tested),
    }


def synthesize_passage(
    *,
    n_tokens: int,
    delta: float,
    temperature: float,
    seed: int,
    prompt: str = "The researchers reported that",
) -> str:
    """Offline fluency-free generator: sample from a fixed vocab with KGW bias."""
    rng = np.random.default_rng(seed)
    logits = np.zeros(len(VOCAB))
    text = prompt
    previous = prompt.split()[-1] if prompt.split() else ""
    for _ in range(n_tokens):
        pick = biased_pick(logits, VOCAB, previous, delta, rng, temperature=temperature)
        token = VOCAB[pick]
        text = text + token if token in {".", ",", "!", "?", ";", ":"} else text + " " + token
        previous = token
    return text


def run_synthetic_sweep(
    temperatures: list[float],
    deltas: list[float],
    *,
    n_passages: int = 24,
    n_tokens: int = 80,
) -> list[dict]:
    det = KGWReferenceAdapter()
    cells = []
    for temperature in temperatures:
        for delta in deltas:
            texts = [
                synthesize_passage(
                    n_tokens=n_tokens,
                    delta=delta,
                    temperature=temperature,
                    seed=10_000 + int(temperature * 100) * 17 + int(delta * 10) * 31 + i,
                )
                for i in range(n_passages)
            ]
            summary = _summarize(det, texts)
            cells.append(
                {
                    "temperature": temperature,
                    "delta": delta,
                    "greedy": temperature <= 0,
                    **summary,
                }
            )
    return cells


def run_model_sweep(
    temperatures: list[float],
    deltas: list[float],
    *,
    model_name: str,
    n_passages: int,
    max_tokens: int,
) -> list[dict]:
    from bench.watermark_gen import generate_watermarked

    prompts = [
        "The city council met on Tuesday to discuss",
        "Scientists announced a new discovery about",
        "The local team secured a dramatic victory after",
        "Researchers developed a novel method for",
        "The stock market reacted sharply to news of",
        "A new restaurant opened downtown, offering",
    ][: max(1, n_passages)]
    det = KGWReferenceAdapter()
    cells = []
    for temperature in temperatures:
        for delta in deltas:
            texts = generate_watermarked(
                prompts,
                model_name=model_name,
                delta=delta,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=int(temperature * 1000) + int(delta * 100),
                progress=True,
            )
            summary = _summarize(det, texts)
            cells.append(
                {
                    "temperature": temperature,
                    "delta": delta,
                    "greedy": temperature <= 0,
                    **summary,
                }
            )
    return cells


def build_card(cells: list[dict], *, mode: str, generator: dict) -> dict:
    temperatures = sorted({c["temperature"] for c in cells})
    deltas = sorted({c["delta"] for c in cells})
    return {
        "schema": "panoptes-watermark-temperature-card-v1",
        "scheme": "kgw-v1",
        "adapter_version": "kgw-v1",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {**generator, "mode": mode},
        "temperatures": temperatures,
        "deltas": deltas,
        "cells": cells,
        "limitations": [
            "Uses the Panoptes demo KGW key (Aaronson/SynthID-Text family), "
            "not a vendor private key.",
            "temperature=0 is greedy (argmax after bias); sampling randomness is gone, "
            "so watermark embedding is deterministic or absent "
            "depending on whether bias flips the ranking.",
            "Synthetic mode samples a fixed vocab without an LM fluency prior; "
            "--with-model runs the real generator.",
            "Detection power is reported per cell; "
            "short passages and low-entropy domains remain low-power.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    parser.add_argument("--deltas", type=float, nargs="+", default=DEFAULT_DELTAS)
    parser.add_argument("--n-passages", type=int, default=24)
    parser.add_argument("--n-tokens", type=int, default=80)
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--out", type=Path, default=CARDS / "watermark-temperature.json")
    args = parser.parse_args()

    if args.with_model:
        cells = run_model_sweep(
            args.temperatures,
            args.deltas,
            model_name=args.model,
            n_passages=args.n_passages,
            max_tokens=args.n_tokens,
        )
        mode = "model"
        generator = {"model": args.model, "key_id": "panoptes-demo-key"}
    else:
        cells = run_synthetic_sweep(
            args.temperatures,
            args.deltas,
            n_passages=args.n_passages,
            n_tokens=args.n_tokens,
        )
        mode = "synthetic"
        generator = {"vocab_size": len(VOCAB), "key_id": "panoptes-demo-key"}

    card = build_card(cells, mode=mode, generator=generator)
    sign(card)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}  sha256={card['artifact_sha256'][:16]}…")
    for cell in cells:
        print(
            f"  T={cell['temperature']:.1f} delta={cell['delta']:.1f} "
            f"det@0.05={cell['detection_rate_0.05']} mean_z={cell['mean_z']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
