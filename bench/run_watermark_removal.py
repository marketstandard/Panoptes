"""Run the watermark-removal evaluation and sign the removal card.

  python -m bench.run_watermark_removal

Reads the signed watermarked generations (run_watermark_generation.py), applies
the removal attack battery to both watermark families, measures passive-detector
evasion on the corpus, and writes
backend/artifacts/cards/watermark-removal.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.cards import sign  # noqa: E402
from bench.datasets import load_verified_corpus  # noqa: E402
from bench.measure import _jsonable  # noqa: E402
from bench.watermark_removal_eval import watermark_removal_eval  # noqa: E402

CARDS = ROOT / "backend" / "artifacts" / "cards"


def _external_remover_results() -> list[dict]:
    """Summarize any signed external-repo remover cards into the removal card."""
    out = []
    for path in sorted(CARDS.glob("external-repo-*.json")):
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if card.get("kind") != "watermark-remover":
            continue
        res = card.get("result", {})
        out.append(
            {
                "repo": card.get("repo", {}).get("url"),
                "adapter": card.get("adapter", {}).get("name"),
                "card": path.name,
                "kgw_detection_before": res.get("kgw", {}).get("detection_rate_before"),
                "kgw_detection_after": res.get("kgw", {}).get("detection_rate_after"),
                "unicode_present_before": res.get("unicode", {}).get("present_rate_before"),
                "unicode_present_after": res.get("unicode", {}).get("present_rate_after"),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, default=CARDS / "watermarked-generations.json")
    parser.add_argument("--paraphrases", type=Path, default=CARDS / "watermarked-paraphrases.json")
    parser.add_argument("--out", type=Path, default=CARDS / "watermark-removal.json")
    args = parser.parse_args()

    generations = json.loads(args.generations.read_text(encoding="utf-8"))
    paraphrases = (
        json.loads(args.paraphrases.read_text(encoding="utf-8"))
        if args.paraphrases.exists()
        else None
    )
    corpus = load_verified_corpus()
    card = watermark_removal_eval(generations, corpus, paraphrases)
    card["external_repos"] = _external_remover_results()
    card["created_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    card = _jsonable(card)
    sign(card)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}  sha256={card['artifact_sha256'][:16]}…")

    print("\nKGW retention (detection@0.05 before -> after):")
    base = card["kgw"]["baseline"]["detection_rate_0.05"]
    for row in card["kgw"]["per_attack"]:
        print(f"  {row['attack']:18s} {base:.3f} -> {row['detection_rate_after']:.3f}")
    print("\nUnicode retention (present rate before -> after):")
    ubase = card["unicode"]["baseline"]["present_rate"]
    for row in card["unicode"]["per_attack"]:
        print(f"  {row['attack']:18s} {ubase:.3f} -> {row['present_rate_after']:.3f}")
    print("\nPassive evasion (heuristic AUROC before -> after):")
    for row in card["passive"]["per_attack"]:
        print(f"  {row['attack']:18s} {row['auroc_before']:.3f} -> {row['auroc_after']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
