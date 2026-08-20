"""Paraphrase the watermarked generations with an instruction-tuned LM.

  python -m bench.run_watermark_paraphrase

This is the strongest removal attack — a complete rewrite, which Anthropic notes
defeats a statistical watermark. The paraphrase model does NOT know the
watermark key, so its rewrite re-rolls word choices off the green list. It
mirrors the "statistical rewrite" hook in tools like watermarks-remover. Runs on
GPU; writes backend/artifacts/cards/watermarked-paraphrases.json so the removal
evaluation itself stays CPU/deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panoptes.analysis.watermarks import KGWReferenceAdapter  # noqa: E402
from panoptes.schemas import ContentType  # noqa: E402

from bench.cards import sign  # noqa: E402

CARDS = ROOT / "backend" / "artifacts" / "cards"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

INSTRUCTION = (
    "Rewrite the following passage in different words while preserving its meaning. "
    "Output only the rewritten passage, with no preamble or commentary."
)


def paraphrase_all(
    texts: list[str],
    *,
    model_name: str = DEFAULT_MODEL,
    max_new_tokens: int = 220,
    seed: int = 0,
    device: str | None = None,
    progress: bool = False,
) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    out: list[str] = []
    for i, text in enumerate(texts):
        messages = [{"role": "user", "content": f"{INSTRUCTION}\n\n{text}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        torch.manual_seed(seed + i)  # per-sample determinism across transformers versions
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = generated[0, input_ids.shape[1] :]
        out.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
        del input_ids, generated, new_ids
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if progress and (i % 10 == 0 or i == len(texts) - 1):
            print(f"    paraphrased {i + 1}/{len(texts)}", flush=True)
    del model
    import gc

    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, default=CARDS / "watermarked-generations.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=None, help="paraphrase only the first N (debug)"
    )
    parser.add_argument("--out", type=Path, default=CARDS / "watermarked-paraphrases.json")
    args = parser.parse_args()

    generations = json.loads(args.generations.read_text(encoding="utf-8"))
    wm = [s for s in generations["samples"] if s["kind"] == "watermarked"]
    if args.limit:
        wm = wm[: args.limit]
    texts = [s["text"] for s in wm]

    paraphrases = paraphrase_all(
        texts,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        progress=True,
    )

    det = KGWReferenceAdapter()
    samples = []
    for i, (src, para) in enumerate(zip(wm, paraphrases, strict=True)):
        res = det.detect(para, ContentType.PROSE)[0]
        samples.append(
            {
                "id": f"paraphrase-{i:03d}",
                "source_id": src["id"],
                "text": para,
                "sha256": hashlib.sha256(para.encode("utf-8")).hexdigest(),
                "n_tokens": res.eligible_tokens,
                "green_rate": res.green_rate,
                "z": res.z,
                "p_value": res.p_value,
            }
        )
    tested = [s for s in samples if s["p_value"] is not None]
    card = {
        "schema": "panoptes-watermarked-generations-v1",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {
            "scheme": "llm-paraphrase-rewrite",
            "family": "complete-rewrite attack (no watermark key)",
            "key_id": "none",
            "model": args.model,
            "gamma": 0.0,
            "delta": 0.0,
            "max_tokens": args.max_new_tokens,
            "seeds": [args.seed],
        },
        "n_watermarked": 0,
        "n_control": len(samples),
        "verification": {
            "watermarked": {"n": 0, "n_tested": 0},
            "control": {
                "n": len(samples),
                "n_tested": len(tested),
                "detection_rate_0.05": (
                    sum(1 for s in tested if s["p_value"] < 0.05) / len(tested) if tested else None
                ),
                "mean_z": sum(s["z"] or 0.0 for s in tested) / len(tested) if tested else None,
                "mean_green_rate": (
                    sum(s["green_rate"] or 0.0 for s in tested) / len(tested) if tested else None
                ),
            },
        },
        "samples": samples,
        "limitations": [
            "Paraphrases are a complete-rewrite attack by an instruction-tuned LM "
            "that does not know the watermark key.",
            "Paraphrase quality/fidelity is not separately scored; "
            "the card measures only whether the KGW watermark survives rewriting.",
        ],
    }
    # Reuse the generations schema shape but relabel kind for clarity.
    for s in card["samples"]:
        s["kind"] = "control"
        s["prompt"] = "llm-paraphrase"
    sign(card)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ctrl = card["verification"]["control"]
    print(f"wrote {args.out.relative_to(ROOT)}  sha256={card['artifact_sha256'][:16]}…")
    print(
        f"  paraphrased: n={ctrl['n_tested']} detection@0.05={ctrl.get('detection_rate_0.05'):.3f} "
        f"mean_z={ctrl.get('mean_z'):.2f} mean_green={ctrl.get('mean_green_rate'):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
