"""Known-scheme watermark smoke screen over the verified corpus.

Runs public adapters (KGW demo key + Unicode zero-width) across corpus
records. This cannot detect private vendor keys (e.g. Anthropic SynthID-Text);
it only flags unexpected public-scheme hits — especially on human controls,
which would indicate a corpus integrity problem.
"""

from __future__ import annotations

from panoptes.analysis.watermarks import KGWReferenceAdapter
from panoptes.schemas import ContentType

from bench.baseline_corpus import CorpusRecord
from bench.watermark_unicode import detect_unicode_watermark


def _content_type(kind: str) -> ContentType:
    return ContentType.CODE if kind == "code" else ContentType.PROSE


def screen_corpus(records: list[CorpusRecord], *, alpha: float = 0.05) -> dict:
    """Return a screening summary suitable for embedding in corpus-summary.json."""
    det = KGWReferenceAdapter()
    kgw_rejects = 0
    kgw_tested = 0
    unicode_present = 0
    human_kgw_rejects = 0
    human_tested = 0
    flagged: list[dict] = []

    for record in records:
        result, _ = det.detect(record.text, _content_type(record.kind))
        if result.status == "tested" and result.p_value is not None:
            kgw_tested += 1
            if record.label == 0:
                human_tested += 1
            if result.p_value < alpha:
                kgw_rejects += 1
                if record.label == 0:
                    human_kgw_rejects += 1
                    flagged.append(
                        {
                            "run_id": record.run_id,
                            "prompt_id": record.prompt_id,
                            "family": record.family,
                            "scheme": "kgw-v1",
                            "p_value": result.p_value,
                            "z": result.z,
                        }
                    )
        unicode = detect_unicode_watermark(record.text)
        if unicode.get("present"):
            unicode_present += 1
            flagged.append(
                {
                    "run_id": record.run_id,
                    "prompt_id": record.prompt_id,
                    "family": record.family,
                    "scheme": "unicode-zerowidth-v1",
                    "payload_match": unicode.get("payload_match"),
                }
            )

    return {
        "alpha": alpha,
        "n_records": len(records),
        "kgw": {
            "n_tested": kgw_tested,
            "reject_rate": (kgw_rejects / kgw_tested) if kgw_tested else None,
            "human_reject_rate": (human_kgw_rejects / human_tested) if human_tested else None,
            "human_rejects": human_kgw_rejects,
        },
        "unicode": {
            "present_count": unicode_present,
            "present_rate": (unicode_present / len(records)) if records else None,
        },
        "flagged": flagged[:50],
        "limitations": [
            "Screening uses the public KGW demo key "
            "and a Unicode zero-width reference scheme only.",
            "Private vendor watermarks (e.g. Anthropic SynthID-Text) cannot be detected here.",
            "Unexpected rejects on human controls are integrity signals, not authorship claims.",
        ],
    }
