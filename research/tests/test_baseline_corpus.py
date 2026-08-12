"""Tests for research/baseline_corpus.py.

Run from the repository root: python -m pytest research/tests
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.baseline_corpus import (  # noqa: E402
    CorpusError,
    load_corpus,
    verify_run,
)

REFERENCE = ROOT / "baselines" / "reference"
CONTROLS = ROOT / "baselines" / "controls"


def test_real_corpus_verifies_end_to_end():
    records = load_corpus()
    run_dirs = [d for root in (REFERENCE, CONTROLS) for d in root.iterdir() if d.is_dir()]
    assert len(records) == 8 * len(run_dirs)
    assert {record.sha256 for record in records}  # every record carries its verified hash


def test_controls_are_human_reference_runs_are_ai():
    records = load_corpus()
    for record in records:
        if record.family == "human":
            assert record.label == 0
            assert "human" in record.run_id
        else:
            assert record.label == 1
            assert record.family in record.run_id


def test_kinds_and_buckets_are_well_formed():
    records = load_corpus()
    valid_buckets = {"lt50", "50-149", "150-499", "500plus"}
    for record in records:
        assert record.kind == ("code" if record.prompt_id.startswith("code-") else "text")
        assert record.length_bucket in valid_buckets
    kinds = {record.kind for record in records}
    assert kinds == {"text", "code"}


def test_tampered_output_file_is_rejected(tmp_path):
    source = sorted(REFERENCE.iterdir())[0]
    forged = tmp_path / source.name
    shutil.copytree(source, forged)
    manifest = json.loads((forged / "run.manifest.json").read_text(encoding="utf-8"))
    target = forged / manifest["outputs"][0]["file"]
    target.write_text(target.read_text(encoding="utf-8") + "\nTAMPERED\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="SHA-256 mismatch"):
        verify_run(forged)


def test_tampered_manifest_is_rejected(tmp_path):
    source = sorted(REFERENCE.iterdir())[0]
    forged = tmp_path / source.name
    shutil.copytree(source, forged)
    manifest_path = forged / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][0]["bytes"] += 1  # edit without re-signing
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="self-hash mismatch"):
        verify_run(forged)


def test_missing_output_is_rejected(tmp_path):
    source = sorted(REFERENCE.iterdir())[0]
    forged = tmp_path / source.name
    shutil.copytree(source, forged)
    manifest = json.loads((forged / "run.manifest.json").read_text(encoding="utf-8"))
    (forged / manifest["outputs"][1]["file"]).unlink()
    with pytest.raises(CorpusError, match="missing output"):
        verify_run(forged)
