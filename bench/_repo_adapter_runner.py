"""Subprocess entry point for the git-repo evaluation harness.

This runs INSIDE a subprocess spawned by bench.repo_harness. It puts the cloned
repository on sys.path, imports the declared adapter module, applies the
kind's conventional callable to every input text, and writes JSON results.

Input JSON (``--input``) is a list of strings. Output JSON (``--output``) is a
list with one result per input, in order:

* watermark-remover: ``transform(text) -> str``
* watermark-scheme:  ``detect(text) -> {"score": float, "p_value": float}``
* detector:          ``score(text) -> float``

Keeping this a separate, tiny script means the cloned code only ever runs in an
isolated process, never in the Panoptes process itself.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--callable", required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    module = importlib.import_module(args.module)
    fn = getattr(module, args.callable)

    texts = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = []
    for text in texts:
        out = fn(text)
        if args.kind == "detector":
            out = float(out)
        results.append(out)
    Path(args.output).write_text(json.dumps(results), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
