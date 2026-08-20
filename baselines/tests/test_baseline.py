"""Tests for baselines/baseline.py.

Run from the repository root: python -m pytest baselines/tests
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("panoptes_baseline", ROOT / "baselines" / "baseline.py")
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)

sys.path.insert(0, str(ROOT / "bench"))
import validate_submission  # noqa: E402


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(baseline, "REFERENCE", tmp_path / "reference")
    monkeypatch.setattr(baseline, "CATALOG", tmp_path / "catalog")
    monkeypatch.setattr(baseline, "REGISTRY", tmp_path / "catalog" / "registry.jsonl")
    monkeypatch.setattr(baseline, "MANIFESTS", tmp_path / "catalog" / "manifests")
    return tmp_path


def parse(*argv: str):
    return baseline.build_parser().parse_args(list(argv))


def fill_outputs(run_dir: Path, kind: str) -> None:
    manifest, _ = baseline.load_prompt_manifest()
    for prompt in baseline.prompts_for_kind(manifest, kind):
        (run_dir / f"{prompt['id']}.md").write_text(
            f"output for {prompt['id']}\n", encoding="utf-8"
        )


def finalize_run(workspace: Path, contributor: str | None = None):
    run_dir = workspace / "runs" / "test-model_text"
    assert baseline.main(
        ["finalize", "--run", str(run_dir), "--interface", "chat-ui",
         "--reported-version", "test-model-2026-08-11"]
    ) == 0
    if contributor and baseline.main(
        ["submit", "--run", str(run_dir), "--contributor", contributor]
    ) != 0:
        raise AssertionError("submit failed")
    return run_dir


def test_canonical_hash_matches_validator():
    payload = {"b": 2, "a": [1, 2], "artifact_sha256": "0" * 64}
    assert baseline.canonical_hash(payload) == validate_submission.canonical_hash(payload)


def test_merkle_root_known_answers():
    empty = hashlib.sha256(b"").hexdigest()
    assert baseline.merkle_root([]) == empty
    single = "a" * 64
    assert baseline.merkle_root([single]) == single
    first, second = "1" * 64, "2" * 64
    expected = hashlib.sha256((first + second).encode("ascii")).hexdigest()
    assert baseline.merkle_root([second, first]) == expected
    # odd leaf duplicates itself
    third = "3" * 64
    left = hashlib.sha256((first + second).encode("ascii")).hexdigest()
    right = hashlib.sha256((third + third).encode("ascii")).hexdigest()
    expected = hashlib.sha256((left + right).encode("ascii")).hexdigest()
    assert baseline.merkle_root([third, first, second]) == expected


def test_prompt_docs_contain_every_prompt_verbatim():
    manifest, _ = baseline.load_prompt_manifest()
    docs = {
        "text": (ROOT / "baselines" / "prompts" / "text.md").read_text(encoding="utf-8"),
        "code": (ROOT / "baselines" / "prompts" / "code.md").read_text(encoding="utf-8"),
    }
    for prompt in manifest["prompts"]:
        assert prompt["prompt"] in docs[prompt["kind"]], prompt["id"]


def test_prompt_manifest_validates_against_schema():
    validate_submission.validate_file(ROOT / "baselines" / "prompts" / "prompts.manifest.json")


def test_scaffold_finalize_submit_verify_flow(workspace):
    assert baseline.main(["scaffold", "--model", "test-model", "--kind", "text"]) == 0
    run_dir = workspace / "runs" / "test-model_text"
    assert (run_dir / "_CHECKLIST.md").exists()
    assert (run_dir / "text-01.md").exists()

    # finalize must refuse an incomplete run
    assert baseline.main(
        ["finalize", "--run", str(run_dir), "--interface", "chat-ui"]
    ) == 2

    fill_outputs(run_dir, "text")
    finalize_run(workspace, contributor="tester")

    manifest = json.loads((run_dir / "run.manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "panoptes-baseline-run-v1"
    assert manifest["model"]["slug"] == "test-model"
    assert manifest["model"]["interface"] == "chat-ui"
    assert len(manifest["outputs"]) == 8
    assert manifest["artifact_sha256"] == baseline.canonical_hash(manifest)
    assert manifest["merkle_root"] == baseline.merkle_root(
        [o["sha256"] for o in manifest["outputs"]]
    )
    validate_submission.validate_file(run_dir / "run.manifest.json")

    registry = (workspace / "catalog" / "registry.jsonl").read_text(encoding="utf-8")
    line = json.loads(registry.strip())
    assert line["manifest_sha256"] == manifest["artifact_sha256"]
    stored = workspace / "catalog" / "manifests" / f"{manifest['artifact_sha256']}.json"
    assert stored.exists()
    # raw outputs never enter the catalog
    assert not (workspace / "catalog" / "manifests" / "text-01.md").exists()

    assert baseline.main(["verify-catalog"]) == 0


def test_pending_run_renamed_on_finalize(workspace):
    assert baseline.main(["scaffold", "--kind", "code"]) == 0
    pending = workspace / "runs" / "_pending_code"
    assert pending.exists()
    fill_outputs(pending, "code")
    assert baseline.main(
        ["finalize", "--run", str(pending), "--model", "agent-x",
         "--interface", "agent-chat"]
    ) == 0
    final_dir = workspace / "runs" / "agent-x_code"
    assert final_dir.exists()
    assert not pending.exists()
    manifest = json.loads((final_dir / "run.manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["interface"] == "agent-chat"


def test_finalize_requires_declared_model(workspace):
    assert baseline.main(["scaffold", "--kind", "text"]) == 0
    pending = workspace / "runs" / "_pending_text"
    fill_outputs(pending, "text")
    assert baseline.main(
        ["finalize", "--run", str(pending), "--interface", "agent-chat"]
    ) == 2


def test_submit_rejects_duplicate_run_id(workspace):
    assert baseline.main(["scaffold", "--model", "test-model", "--kind", "text"]) == 0
    run_dir = workspace / "runs" / "test-model_text"
    fill_outputs(run_dir, "text")
    finalize_run(workspace, contributor="tester")
    assert baseline.main(["submit", "--run", str(run_dir)]) == 2


def test_verify_catalog_detects_tampering(workspace):
    assert baseline.main(["scaffold", "--model", "test-model", "--kind", "text"]) == 0
    run_dir = workspace / "runs" / "test-model_text"
    fill_outputs(run_dir, "text")
    finalize_run(workspace, contributor="tester")

    registry_path = workspace / "catalog" / "registry.jsonl"
    line = json.loads(registry_path.read_text(encoding="utf-8").strip())
    line["merkle_root"] = "0" * 64
    registry_path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    assert baseline.main(["verify-catalog"]) == 1


def test_validator_rejects_raw_text_in_manifest(tmp_path):
    manifest, _ = baseline.load_prompt_manifest()
    payload = {
        "schema": "panoptes-baseline-run-v1",
        "run_id": "bad-run",
        "model": {
            "slug": "bad-model",
            "provider": "unspecified",
            "reported_version": "bad-model",
            "interface": "api",
        },
        "prompts": {"version": manifest["version"], "sha256": "0" * 64},
        "created_utc": "2026-08-11T00:00:00Z",
        "environment": {"os": "test", "python": "3.12", "tool": "test"},
        "outputs": [
            {
                "prompt_id": "text-01",
                "file": "text-01.md",
                "sha256": "0" * 64,
                "bytes": 10,
                "text": "raw model output must never appear here",
            }
        ],
        "merkle_root": "0" * 64,
    }
    path = tmp_path / "bad.manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    errors = validate_submission.validate_file(path)
    assert any("hash metadata only" in error for error in errors)
