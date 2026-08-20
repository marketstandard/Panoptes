"""Generate ground-truth KGW-watermarked passages + matched controls, verify
the detector fires on one and not the other, and sign the result.

  python -m bench.run_watermark_generation

The watermarked set is produced by bench.watermark_gen.generate_watermarked
with the KGW logit bias (delta>0); controls use delta=0 (same model, no bias).
Both are synthetic gpt2-family outputs written as test fixtures for the
watermark-removal evaluation — no human or third-party text, so passages are
embedded directly in the signed card. Writes
backend/artifacts/cards/watermarked-generations.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.cards import sign  # noqa: E402
from bench.watermark_gen import DEFAULT_DELTA, DEFAULT_MODEL, generate_watermarked  # noqa: E402
from panoptes.analysis.watermarks import KGWReferenceAdapter  # noqa: E402
from panoptes.schemas import ContentType  # noqa: E402

CARDS = ROOT / "backend" / "artifacts" / "cards"

PROMPTS = [
    "The city council met on Tuesday to discuss",
    "Scientists announced a new discovery about",
    "The local team secured a dramatic victory after",
    "Researchers developed a novel method for",
    "The stock market reacted sharply to news of",
    "A new restaurant opened downtown, offering",
    "The museum unveiled an exhibition featuring",
    "Engineers completed construction of the bridge that",
    "The author published a memoir describing",
    "Voters headed to the polls to decide on",
    "The company reported quarterly earnings that",
    "Astronomers observed a rare alignment of",
    "The university announced a scholarship program for",
    "Farmers in the region adopted new techniques to",
    "The film festival opened with a screening of",
    "Doctors recommended a revised treatment for",
    "The startup raised funding to expand its",
    "Historians uncovered letters that reveal",
    "The orchestra performed a symphony by",
    "Officials introduced legislation aimed at",
    "The hiking trail winds through a forest where",
    "Teachers gathered to protest changes to",
    "The tech giant unveiled a device that",
    "Chefs from around the world competed to",
    "The report highlighted growing concerns about",
    "A community garden project brought together",
    "The athlete broke the world record in",
    "Investigators examined the causes of",
    "The library launched a digital archive of",
    "Climate researchers warned that rising temperatures",
    "The band released an album that",
    "City planners proposed a redesign of",
    "The charity organized a fundraiser to support",
    "Paleontologists excavated a fossil that",
    "The software update introduced features that",
    "Residents voiced opinions about the proposal to",
    "The documentary explored the lives of",
    "Economists predicted that inflation would",
    "The garden club hosted a workshop on",
    "Marine biologists tagged sharks to study",
    "The theater company staged a production of",
    "Officials declared a state of emergency after",
    "The conference brought together experts in",
    "A viral video showed a dog that",
    "The architect designed a building that",
    "Students collaborated on a project to",
    "The journal published findings suggesting that",
    "Firefighters contained the blaze that",
]


def _detect(det: KGWReferenceAdapter, text: str):
    return det.detect(text, ContentType.PROSE)[0]


def _summarize(rows: list) -> dict:
    tested = [r for r in rows if r.status == "tested" and r.p_value is not None]
    if not tested:
        return {"n": len(rows), "n_tested": 0}
    return {
        "n": len(rows),
        "n_tested": len(tested),
        "detection_rate_0.05": sum(1 for r in tested if r.p_value < 0.05) / len(tested),
        "mean_z": sum(r.z or 0.0 for r in tested) / len(tested),
        "mean_green_rate": sum(r.green_rate or 0.0 for r in tested) / len(tested),
    }


def _samples(texts: list[str], prompts: list[str], kind: str, det: KGWReferenceAdapter) -> list[dict]:
    out = []
    for i, (prompt, text) in enumerate(zip(prompts, texts, strict=True)):
        res = _detect(det, text)
        out.append(
            {
                "id": f"{kind}-{i:03d}",
                "kind": kind,
                "prompt": prompt,
                "text": text,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "n_tokens": res.eligible_tokens,
                "green_rate": res.green_rate,
                "z": res.z,
                "p_value": res.p_value,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--out", type=Path, default=CARDS / "watermarked-generations.json")
    args = parser.parse_args()

    prompts: list[str] = []
    for seed in args.seeds:
        prompts.extend(PROMPTS)
    # tag each prompt occurrence with its seed so generation varies
    watermarked_texts: list[str] = []
    control_texts: list[str] = []
    for seed in args.seeds:
        watermarked_texts.extend(
            generate_watermarked(
                PROMPTS, model_name=args.model, delta=args.delta,
                max_tokens=args.max_tokens, top_k=args.top_k, seed=seed, progress=True,
            )
        )
        control_texts.extend(
            generate_watermarked(
                PROMPTS, model_name=args.model, delta=0.0,
                max_tokens=args.max_tokens, top_k=args.top_k, seed=1000 + seed, progress=True,
            )
        )

    det = KGWReferenceAdapter()
    wm_rows = [_detect(det, t) for t in watermarked_texts]
    ctrl_rows = [_detect(det, t) for t in control_texts]
    samples = _samples(watermarked_texts, prompts, "watermarked", det) + _samples(
        control_texts, prompts, "control", det
    )

    card = {
        "schema": "panoptes-watermarked-generations-v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {
            "scheme": "kgw-green-list",
            "family": "Aaronson 2022 / KGW / SynthID-Text",
            "key_id": "panoptes-demo-key",
            "model": args.model,
            "gamma": 0.5,
            "delta": args.delta,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seeds": args.seeds,
        },
        "n_watermarked": len(watermarked_texts),
        "n_control": len(control_texts),
        "verification": {
            "watermarked": _summarize(wm_rows),
            "control": _summarize(ctrl_rows),
        },
        "samples": samples,
        "limitations": [
            "Passages are synthetic gpt2-family generations produced with the demo KGW key; they model the Aaronson/SynthID-Text family, not Anthropic's private production key.",
            "Text naturalness is limited by the small generator model; the watermark (green-list density) is the controlled variable, not fluency.",
            "Controls share prompts and model with delta=0, so detection-rate differences isolate the watermark bias.",
        ],
    }
    sign(card)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    wm = card["verification"]["watermarked"]
    ctrl = card["verification"]["control"]
    print(f"wrote {args.out.relative_to(ROOT)}  sha256={card['artifact_sha256'][:16]}…")
    print(
        f"  watermarked: n={wm['n_tested']} detection@0.05={wm.get('detection_rate_0.05'):.3f} "
        f"mean_z={wm.get('mean_z'):.2f} mean_green={wm.get('mean_green_rate'):.3f}"
    )
    print(
        f"  control:     n={ctrl['n_tested']} fpr@0.05={ctrl.get('detection_rate_0.05'):.3f} "
        f"mean_z={ctrl.get('mean_z'):.2f} mean_green={ctrl.get('mean_green_rate'):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
