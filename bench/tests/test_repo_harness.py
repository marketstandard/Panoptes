"""Tests for the git-repo evaluation harness, using a local mock repo (no network)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from bench import repo_harness
from bench.repo_harness import AdapterSpec

# A deterministic mock watermark-remover: strips the two zero-width carriers.
MOCK_ADAPTER = '''
def transform(text):
    return text.replace("\\u200b", "").replace("\\u200c", "")

def detect(text):
    return {"score": 0.5, "p_value": 0.5}

def score(text):
    return 0.5
'''

LONG = " ".join(["the quick brown fox jumps over the lazy dog"] * 12)


def _make_repo(tmp_path: Path, with_manifest: bool = True) -> Path:
    repo = tmp_path / "mockrepo"
    repo.mkdir()
    (repo / "panoptes_adapter.py").write_text(MOCK_ADAPTER, encoding="utf-8")
    if with_manifest:
        (repo / "panoptes.adapter.json").write_text(
            json.dumps(
                {
                    "kind": "watermark-remover",
                    "name": "mock-remover",
                    "version": "0.1",
                    "entry": {"type": "python-function", "module": "panoptes_adapter", "callable": "transform"},
                    "requires_network": False,
                }
            ),
            encoding="utf-8",
        )
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        check=True,
    )
    return repo


def _generations_card() -> dict:
    return {
        "samples": [
            {"kind": "watermarked", "text": LONG},
            {"kind": "watermarked", "text": LONG},
            {"kind": "control", "text": LONG},
        ]
    }


def test_find_adapter_manifest(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, with_manifest=True)
    spec = repo_harness.find_adapter(repo)
    assert spec is not None
    assert spec.kind == "watermark-remover"
    assert spec.module == "panoptes_adapter"
    assert spec.callable == "transform"
    assert spec.source == "manifest"


def test_find_adapter_autodetect(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, with_manifest=False)
    spec = repo_harness.find_adapter(repo)
    assert spec is not None
    assert spec.source == "auto"
    assert spec.module == "panoptes_adapter"


def test_find_adapter_missing(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert repo_harness.find_adapter(empty) is None


def test_clone_and_run_adapter_subprocess(tmp_path: Path) -> None:
    src = _make_repo(tmp_path)
    dest = repo_harness.clone_repo(str(src), dest_root=tmp_path / "repos")
    assert (dest / "panoptes_adapter.py").exists()
    spec = repo_harness.find_adapter(dest)
    marked = "​".join(["word"] * 3)  # zero-width spaces between words
    out = repo_harness.run_adapter(dest, spec, ["plain text", marked], timeout=60)
    assert out[0] == "plain text"
    assert out[1] == "wordwordword"  # zero-width chars stripped by the mock


def test_evaluate_repo_remover_strips_unicode(tmp_path: Path) -> None:
    src = _make_repo(tmp_path)
    result = repo_harness.evaluate_repo(
        str(src),
        "watermark-remover",
        generations_card=_generations_card(),
        dest_root=tmp_path / "repos",
        timeout=60,
    )
    assert result["schema"] == "panoptes-external-repo-eval-v1"
    assert result["kind"] == "watermark-remover"
    # Mock remover strips zero-width chars -> unicode watermark destroyed.
    assert result["result"]["unicode"]["present_rate_before"] == 1.0
    assert result["result"]["unicode"]["present_rate_after"] == 0.0


def test_inject_adapter(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, with_manifest=False)
    (repo / "panoptes_adapter.py").unlink()  # simulate a repo with no adapter
    ext = tmp_path / "myadapter" / "panoptes_adapter.py"
    ext.parent.mkdir()
    ext.write_text(MOCK_ADAPTER, encoding="utf-8")
    spec = repo_harness.inject_adapter(repo, ext, "watermark-remover")
    assert spec.source == "injected"
    assert (repo / "panoptes_adapter.py").exists()
    assert spec.callable == "transform"


def test_evaluate_repo_rejects_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        repo_harness.evaluate_repo("whatever", "not-a-kind", dest_root=tmp_path)
