"""Tests for bench/baseline_corpus.py.

Run from the repository root: python -m pytest bench/tests/test_baseline_corpus.py
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

from bench.baseline_corpus import (  # noqa: E402
    CorpusError,
    load_corpus,
    verify_run,
)

REFERENCE = ROOT / "baselines" / "reference"
CONTROLS = ROOT / "baselines" / "controls"


def _first_run_dir(root: Path) -> Path:
    for path in sorted(root.iterdir()):
        if path.is_dir():
            return path
    raise AssertionError(f"no run directories in {root}")


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
    source = _first_run_dir(REFERENCE)
    forged = tmp_path / source.name
    shutil.copytree(source, forged)
    manifest = json.loads((forged / "run.manifest.json").read_text(encoding="utf-8"))
    target = forged / manifest["outputs"][0]["file"]
    target.write_text(target.read_text(encoding="utf-8") + "\nTAMPERED\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="SHA-256 mismatch"):
        verify_run(forged)


def test_tampered_manifest_is_rejected(tmp_path):
    source = _first_run_dir(REFERENCE)
    forged = tmp_path / source.name
    shutil.copytree(source, forged)
    manifest_path = forged / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][0]["bytes"] += 1  # edit without re-signing
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(CorpusError, match="self-hash mismatch"):
        verify_run(forged)


def test_missing_output_is_rejected(tmp_path):
    source = _first_run_dir(REFERENCE)
    forged = tmp_path / source.name
    shutil.copytree(source, forged)
    manifest = json.loads((forged / "run.manifest.json").read_text(encoding="utf-8"))
    (forged / manifest["outputs"][1]["file"]).unlink()
    with pytest.raises(CorpusError, match="missing output"):
        verify_run(forged)


def test_watermark_status_defaults_to_unknown_for_legacy_manifests():
    records = load_corpus()
    assert all(
        record.watermark_status in {"declared-none", "declared-active", "suspected", "unknown"}
        for record in records
    )
    # Existing reference runs predate the watermark block.
    assert all(record.watermark_status == "unknown" for record in records)


def test_summarize_propagates_contaminated_cohorts(tmp_path):
    from bench.baseline_corpus import CorpusRecord, summarize

    records = [
        CorpusRecord(
            text="hello world " * 20,
            label=1,
            family="claude-test",
            kind="text",
            length_bucket="50-149",
            prompt_id="text-01",
            run_id="claude-test_text-1",
            sha256="a" * 64,
            watermark_status="suspected",
            watermark_notes="post-transition heuristic",
        ),
        CorpusRecord(
            text="human control " * 20,
            label=0,
            family="human",
            kind="text",
            length_bucket="50-149",
            prompt_id="text-01",
            run_id="human_text-1",
            sha256="b" * 64,
            watermark_status="declared-none",
        ),
    ]
    summary = summarize(records, catalog_entries=0)
    assert summary["contaminated_cohorts"] == [
        {
            "family": "claude-test",
            "kind": "text",
            "watermark_status": "suspected",
            "notes": "post-transition heuristic",
        }
    ]
    by_family = {c["family"]: c for c in summary["cohorts"]}
    assert by_family["claude-test"]["watermark_status"] == "suspected"
    assert by_family["human"]["watermark_status"] == "declared-none"
