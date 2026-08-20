"""Run the watermark radioactivity eval and sign the card.

  python -m bench.run_radioactivity
  python -m bench.run_radioactivity --with-model --student-model HuggingFaceTB/SmolLM2-135M
  python -m bench.run_radioactivity --with-model --tier gpu
      --student-model <7B> --teacher-model <7B>

Default synthetic mode is CI-friendly (no torch). Writes
backend/artifacts/cards/radioactivity.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.cards import sign  # noqa: E402
from bench.radioactivity import (  # noqa: E402
    DEFAULT_CPU_STUDENT,
    DEFAULT_CPU_TEACHER,
    DEFAULT_GPU_STUDENT,
    DEFAULT_GPU_TEACHER,
    run_model_radioactivity,
    run_synthetic_radioactivity,
)

CARDS = ROOT / "backend" / "artifacts" / "cards"


def build_card(result: dict) -> dict:
    return {
        "schema": "panoptes-radioactivity-card-v1",
        "scheme": "kgw-v1",
        "adapter_version": "kgw-v1",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": result["mode"],
        "tier": result["tier"],
        "teacher_model": result.get("teacher_model"),
        "student_model": result.get("student_model"),
        "teacher": result["teacher"],
        "inheritance": result["inheritance"],
        "removal": result["removal"],
        "knowledge_preservation": result.get("knowledge_preservation") or {},
        "limitations": [
            "Demo KGW key models the Aaronson/SynthID-Text family, "
            "not Anthropic's private production key.",
            "Positive inheritance is lineage-compatible evidence, "
            "not proof of unauthorized distillation.",
            "Web-scale scraping can contaminate unrelated models with weak radioactivity.",
            "Synthetic mode uses a bigram student fitted on teacher text; "
            "--with-model runs real HF SFT.",
            "Neutralize_post applies inverse green-list bias at decode (known demo key); "
            "vendor keys remain private.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--tier", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--student-model", default=None)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--n-tokens", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--out", type=Path, default=CARDS / "radioactivity.json")
    args = parser.parse_args()

    if args.with_model:
        teacher = args.teacher_model or (
            DEFAULT_GPU_TEACHER if args.tier == "gpu" else DEFAULT_CPU_TEACHER
        )
        student = args.student_model or (
            DEFAULT_GPU_STUDENT if args.tier == "gpu" else DEFAULT_CPU_STUDENT
        )
        result = run_model_radioactivity(
            teacher_model=teacher,
            student_model=student,
            delta=args.delta,
            n_tokens=min(args.n_tokens, 64),
            epochs=args.epochs,
        )
    else:
        result = run_synthetic_radioactivity(
            n_tokens=args.n_tokens, probe_tokens=args.n_tokens, delta=args.delta
        )

    card = build_card(result)
    sign(card)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}  sha256={card['artifact_sha256'][:16]}…")
    inh = card["inheritance"]
    print(
        "  student_on_wm det@0.05="
        f"{inh['student_on_watermarked'].get('detection_rate_0.05')} "
        f"mean_z={inh['student_on_watermarked'].get('mean_z')}"
    )
    print(
        "  student_on_ctrl det@0.05="
        f"{inh['student_on_control'].get('detection_rate_0.05')} "
        f"mean_z={inh['student_on_control'].get('mean_z')}"
    )
    rem = card["removal"]
    print(f"  paraphrase_pre={rem.get('paraphrase_pre')}")
    print(f"  neutralize_post={rem.get('neutralize_post')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
