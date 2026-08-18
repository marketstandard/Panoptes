"""Panoptes adapter for guillaumemeyer/watermarks-remover (MIT).

Exposes the repo's deterministic "Layer A" Unicode-hygiene transform
(``service/scripts/text_unicode.clean_text``) as a Panoptes watermark-remover:

    watermark-remover contract:  transform(text) -> str

Layer A strips invisible/format Unicode (zero-width spaces, tag characters,
private-use carriers) and normalizes space homoglyphs — no API key or network
required. It targets *Unicode* watermarks; the repo's "Layer B" statistical
rewrite (LLM paraphrase) is a separate, network-dependent path not used here.

This file is injected into a fresh clone by ``bench evaluate-repo --adapter-path``
(see docs/testing-external-repos.md); it is not part of the upstream repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SCRIPTS = _REPO_ROOT / "service" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from text_unicode import clean_text  # noqa: E402


def transform(text: str) -> str:
    """Return the Unicode-hygiene-cleaned text (Layer A defaults)."""
    cleaned, _stats = clean_text(text)
    return cleaned
