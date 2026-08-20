"""Tests for CoAuthor reconstruction, author-disjoint splitting, and the loader.

The Quill-delta replay and the split hash are tested directly on synthetic
inputs; the loader runs on a synthetic parquet fixture (monkeypatched
COAUTHOR_DIR). The pointer manifest and signed card are validated only when the
real dataset has been fetched.

Run from the repository root: python -m pytest bench/tests/test_coauthor.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import datasets  # noqa: E402
from bench.fetch_coauthor import _assign_split, reconstruct_session  # noqa: E402

pd = pytest.importorskip("pandas", reason="CoAuthor loader requires pandas/pyarrow")


# --- Quill-delta reconstruction -------------------------------------------------


def _init(doc: str) -> str:
    return json.dumps({"eventName": "system-initialize", "eventSource": "api", "currentDoc": doc})


def _insert(pos: int, text: str, source: str) -> str:
    return json.dumps(
        {"eventName": "text-insert", "eventSource": source, "textDelta": {"ops": [{"retain": pos}, {"insert": text}]}}
    )


def _delete(pos: int, n: int, source: str) -> str:
    return json.dumps(
        {"eventName": "text-delete", "eventSource": source, "textDelta": {"ops": [{"retain": pos}, {"delete": n}]}}
    )


def test_reconstruct_insert_provenance():
    lines = [_init("AB"), _insert(2, "xy", "user"), _insert(4, "ZZ", "api")]
    text, mask = reconstruct_session(lines)
    assert text == "ABxyZZ"
    assert mask == "PPUUAA"  # prompt, then user, then api


def test_reconstruct_delete_removes_chars():
    lines = [_init("ABCD"), _delete(1, 2, "user")]
    text, mask = reconstruct_session(lines)
    assert text == "AD"
    assert mask == "PP"


def test_reconstruct_interleaved_edit_in_api_span():
    # User edits inside an AI-inserted span; the surrounding AI chars keep 'A'.
    lines = [_init(""), _insert(0, "aaaa", "api"), _insert(2, "U", "user")]
    text, mask = reconstruct_session(lines)
    assert text == "aaUaa"
    assert mask == "AAUAA"


def test_reconstruct_empty_init_then_typing():
    lines = [_init(""), _insert(0, "hi", "user"), _insert(2, "!", "api")]
    text, mask = reconstruct_session(lines)
    assert text == "hi!"
    assert mask == "UUA"


def test_reconstruct_returns_none_on_bad_ops():
    # A retain that runs past the end is tolerated, but malformed JSON is not.
    assert reconstruct_session(["not json"]) is None


# --- author-disjoint split hash -------------------------------------------------


def test_assign_split_deterministic_and_valid():
    valid = {"train", "development", "calibration", "test"}
    for worker in ["W1", "W2", "W3", "worker-xyz"]:
        first = _assign_split(worker)
        assert first in valid
        assert _assign_split(worker) == first  # stable across calls


def test_assign_split_covers_multiple_splits():
    splits = {_assign_split(f"worker-{i}") for i in range(200)}
    assert len(splits) > 1, "split hash should spread workers across partitions"


# --- load_coauthor --------------------------------------------------------------


def _coauthor_frame() -> "pd.DataFrame":
    rows = []
    # Two authors; each appears in only one split (author-disjoint by construction).
    for i in range(4):
        rows.append(
            {"id": f"s_train_{i}", "text": f"train doc {i} " * 30, "label": 1,
             "group": "workerA", "split": "train", "worker_id": "workerA",
             "prompt_code": "shapeshifter", "ai_contribution_fraction": 0.10 * (i + 1),
             "human_chars": 300, "ai_chars": 30 * (i + 1), "prompt_chars": 50,
             "num_query": "5", "num_selected": "4", "written_by_human_official": 1.0 - 0.10 * (i + 1)}
        )
    for i in range(3):
        rows.append(
            {"id": f"s_test_{i}", "text": f"test doc {i} " * 30, "label": 1,
             "group": "workerB", "split": "test", "worker_id": "workerB",
             "prompt_code": "obama", "ai_contribution_fraction": 0.50,
             "human_chars": 200, "ai_chars": 200, "prompt_chars": 40,
             "num_query": "9", "num_selected": "7", "written_by_human_official": 0.50}
        )
    return pd.DataFrame(rows)


def _coauthor_fixture(tmp_path: Path, monkeypatch) -> Path:
    local = tmp_path / "coauthor"
    local.mkdir()
    _coauthor_frame().to_parquet(local / "coauthor.parquet", index=False)
    (local / "fetch-manifest.json").write_text(
        json.dumps({"created_utc": "2026-01-01T00:00:00Z", "artifact_sha256": "0" * 64}),
        encoding="utf-8",
    )
    monkeypatch.setattr(datasets, "COAUTHOR_DIR", local)
    return local


def test_load_coauthor_split_and_provenance(tmp_path, monkeypatch):
    _coauthor_fixture(tmp_path, monkeypatch)
    test = datasets.load_coauthor(split="test")
    assert len(test) == 3
    assert set(test.labels.tolist()) == {1}, "every CoAuthor row is AI-assisted (never binary human)"
    # groups and authors are the worker id (author-disjoint unit)
    assert set(test.groups) == {"workerB"}
    assert test.enforce_author_disjoint is True
    # contribution-fraction ground truth aligns with rows
    frac = test.meta["ai_contribution_fraction"]
    assert len(frac) == len(test)
    assert all(abs(f - 0.50) < 1e-9 for f in frac)


def test_load_coauthor_train_split(tmp_path, monkeypatch):
    _coauthor_fixture(tmp_path, monkeypatch)
    train = datasets.load_coauthor(split="train")
    assert len(train) == 4
    assert set(train.groups) == {"workerA"}
    # fractions are monotone in the fixture
    assert train.meta["ai_contribution_fraction"] == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_load_coauthor_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "COAUTHOR_DIR", tmp_path / "empty")
    with pytest.raises(datasets.DatasetError, match="fetch_coauthor"):
        datasets.load_coauthor(split="test")


def test_load_coauthor_rejects_unknown_split(tmp_path, monkeypatch):
    _coauthor_fixture(tmp_path, monkeypatch)
    with pytest.raises(datasets.DatasetError, match="unknown CoAuthor split"):
        datasets.load_coauthor(split="bogus")


# --- real artifacts (skipped when not fetched) ----------------------------------


def test_coauthor_pointer_manifest_validates():
    pointer = ROOT / "datasets" / "manifests" / "coauthor.json"
    if not pointer.exists():
        pytest.skip("CoAuthor not fetched; run python -m bench.fetch_coauthor")
    from bench.validate_submission import validate_file

    assert validate_file(pointer) == []
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    assert payload["privacy"]["raw_text_in_repo"] is False
    assert payload["notice_entry_required"] is True


def test_coauthor_author_disjoint_firewall():
    parquet = ROOT / "datasets" / "local" / "coauthor" / "coauthor.parquet"
    if not parquet.exists():
        pytest.skip("CoAuthor not fetched; run python -m bench.fetch_coauthor")
    frame = pd.read_parquet(parquet)
    worker_splits = frame.groupby("worker_id")["split"].nunique()
    assert int((worker_splits > 1).sum()) == 0, "a writer must never cross the split firewall"


def test_coauthor_card_validates_and_reports_contribution():
    card = ROOT / "backend" / "artifacts" / "cards" / "coauthor-eval.json"
    if not card.exists():
        pytest.skip("CoAuthor card not generated; run python -m bench.run_coauthor_eval")
    from bench.validate_submission import validate_file

    assert validate_file(card) == []
    payload = json.loads(card.read_text(encoding="utf-8"))
    assert payload["never_binary_fully_ai"] is True
    assert payload["reconstruction_validation"]["pearson_r"] >= 0.95
    for name in ("heuristic", "logistic"):
        block = payload["detectors"][name]["contribution_estimation"]
        assert set(block) >= {"spearman", "pearson", "mae"}
