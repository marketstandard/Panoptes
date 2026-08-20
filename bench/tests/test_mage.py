"""Tests for the MAGE loader, fetch-script src parsing, and the near-dup index.

Loader tests run on synthetic parquet fixtures (monkeypatched MAGE_DIR); the
fetch-script parse and the near-duplicate index are tested directly.

Run from the repository root: python -m pytest bench/tests/test_mage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import datasets  # noqa: E402
from bench.fetch_mage import parse_src  # noqa: E402
from bench.near_dup import near_duplicate_clusters  # noqa: E402

pd = pytest.importorskip("pandas", reason="MAGE loader requires pandas/pyarrow")

BODY = (
    "the committee published its findings on monday after a lengthy review of "
    "the evidence and several witnesses described the sequence of events while "
    "officials declined to comment on the ongoing investigation into the matter "
    "and the report concluded without further action or recommendation "
)


def _long_text(tag: str) -> str:
    return f"{tag}: {BODY}{BODY}"


# --- parse_src -----------------------------------------------------------------


@pytest.mark.parametrize(
    "src,expected",
    [
        (
            "cmv_human",
            {"domain": "cmv", "family": "human", "prompt_mode": "human", "paraphrased": False},
        ),
        (
            "roct_machine_continuation_flan_t5_large",
            {
                "domain": "roct",
                "family": "flan_t5_large",
                "prompt_mode": "continuation",
                "paraphrased": False,
            },
        ),
        (
            "wp_machine_topical_gpt-3.5-trubo",
            {
                "domain": "wp",
                "family": "gpt-3.5-trubo",
                "prompt_mode": "topical",
                "paraphrased": False,
            },
        ),
        (
            "cnn_gpt4",
            {"domain": "cnn", "family": "gpt4", "prompt_mode": "ood", "paraphrased": False},
        ),
        (
            "cnn_gpt4_para",
            {"domain": "cnn", "family": "gpt4", "prompt_mode": "paraphrase", "paraphrased": True},
        ),
        (
            "imdb_human_para",
            {
                "domain": "imdb",
                "family": "paraphrased_human",
                "prompt_mode": "paraphrase",
                "paraphrased": True,
            },
        ),
    ],
)
def test_parse_src(src, expected):
    assert parse_src(src) == expected


def test_parse_src_sci_gen_human_maps_to_sci_domain():
    assert parse_src("sci_gen_human")["domain"] == "sci"
    assert parse_src("sci_gen_human")["family"] == "human"


# --- near-duplicate index -------------------------------------------------------


def test_near_duplicate_clusters_group_continuation_pairs():
    elections = "local elections council debate ballot measure voters turnout " * 12
    continuation = elections + "the campaign announced a recount after objections " * 6
    astro = "astrophysics observatory telescope supernova redshift galaxy spectrum " * 12
    labels, stats = near_duplicate_clusters([elections, continuation, astro])
    assert labels[0] == labels[1], "source and its continuation must share a group"
    assert labels[2] != labels[0], "an unrelated document must not join the cluster"
    assert stats["n_clusters"] == 2


def test_near_duplicate_clusters_deterministic():
    texts = [_long_text(f"document {i}") for i in range(20)]
    a, _ = near_duplicate_clusters(texts)
    b, _ = near_duplicate_clusters(texts)
    assert a.tolist() == b.tolist()


# --- load_mage ------------------------------------------------------------------


def _mage_frame() -> pd.DataFrame:
    rows = []
    for source in range(5):
        rows.append(
            {
                "text": _long_text(f"Human post {source}"),
                "label": 0,
                "family": "human",
                "domain": "cmv",
                "prompt_mode": "human",
                "paraphrased": False,
                "official_split": "test",
                "src": "cmv_human",
                "group": f"mage:{source}",
            }
        )
        rows.append(
            {
                "text": _long_text(f"Machine continuation {source}"),
                "label": 1,
                "family": "gpt_j",
                "domain": "cmv",
                "prompt_mode": "continuation",
                "paraphrased": False,
                "official_split": "test",
                "src": "cmv_machine_continuation_gpt_j",
                "group": f"mage:{source}",
            }
        )
    return pd.DataFrame(rows)


def _mage_fixture(tmp_path: Path, monkeypatch) -> Path:
    local = tmp_path / "mage"
    local.mkdir()
    _mage_frame().to_parquet(local / "test-clean.parquet", index=False)
    (local / "fetch-manifest.json").write_text(
        json.dumps({"created_utc": "2026-01-01T00:00:00Z", "artifact_sha256": "0" * 64}),
        encoding="utf-8",
    )
    monkeypatch.setattr(datasets, "MAGE_DIR", local)
    return local


def test_load_mage_shapes_and_groups(tmp_path, monkeypatch):
    _mage_fixture(tmp_path, monkeypatch)
    dataset = datasets.load_mage(split="test")
    assert len(dataset) == 10
    assert int((dataset.labels == 0).sum()) == 5
    assert int((dataset.labels == 1).sum()) == 5
    assert set(dataset.families) == {"human", "gpt_j"}
    # Source and its machine continuation share a near-duplicate group.
    assert dataset.groups[0] == dataset.groups[1]


def test_load_mage_domain_filter(tmp_path, monkeypatch):
    _mage_fixture(tmp_path, monkeypatch)
    dataset = datasets.load_mage(split="test", domains=("xsum",))
    assert len(dataset) == 0


def test_load_mage_subsample_preserves_groups(tmp_path, monkeypatch):
    _mage_fixture(tmp_path, monkeypatch)
    dataset = datasets.load_mage(split="test", max_rows=4)
    counts: dict[str, int] = {}
    for group in dataset.groups:
        counts[group] = counts.get(group, 0) + 1
    assert all(v == 2 for v in counts.values()), "no group may be split by subsampling"


def test_load_mage_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "MAGE_DIR", tmp_path / "empty")
    with pytest.raises(datasets.DatasetError, match="fetch_mage"):
        datasets.load_mage(split="test")


def test_load_mage_rejects_unknown_split(tmp_path, monkeypatch):
    _mage_fixture(tmp_path, monkeypatch)
    with pytest.raises(datasets.DatasetError, match="unknown MAGE split"):
        datasets.load_mage(split="bogus")


# --- pointer manifest (real data; skipped when not fetched) ---------------------


def test_mage_pointer_manifest_validates():
    pointer = ROOT / "datasets" / "manifests" / "mage.json"
    if not pointer.exists():
        pytest.skip("MAGE not fetched; run python -m bench.fetch_mage")
    from bench.validate_submission import validate_file

    assert validate_file(pointer) == []
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    assert payload["license"]["spdx"] == "Apache-2.0"
    assert payload["privacy"]["raw_text_in_repo"] is False
