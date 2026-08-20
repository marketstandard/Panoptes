"""Tests for the external-benchmark loaders (RAID, M4GT, EvoBench) and the
fetch-script hygiene filters, all on synthetic fixture data.

Run from the repository root: python -m pytest bench/tests
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import datasets  # noqa: E402

pd = pytest.importorskip("pandas", reason="benchmark loaders require pandas/pyarrow")

BODY = (
    "the committee published its findings on monday after a lengthy review of "
    "the evidence and several witnesses described the sequence of events while "
    "officials declined to comment on the ongoing investigation into the matter "
    "and the report concluded without further action or recommendation "
)


def _long_text(tag: str) -> str:
    return f"{tag}: {BODY}{BODY}"


def _fetch_manifest(directory: Path) -> None:
    (directory / "fetch-manifest.json").write_text(
        json.dumps({"created_utc": "2026-01-01T00:00:00Z", "artifact_sha256": "0" * 64}),
        encoding="utf-8",
    )


# --- load_raid ------------------------------------------------------------------


def _raid_frame() -> pd.DataFrame:
    rows = []
    for source in range(6):
        rows.append(
            {
                "text": _long_text(f"Human article {source}"),
                "label": 0,
                "family": "human",
                "domain": "news",
                "attack": "",
                "decoding": "",
                "group": f"src-{source}",
            }
        )
        for model in ("gpt4", "llama3"):
            rows.append(
                {
                    "text": _long_text(f"Clean {model} rewrite {source}"),
                    "label": 1,
                    "family": model,
                    "domain": "news",
                    "attack": "none",
                    "decoding": "greedy",
                    "group": f"src-{source}",
                }
            )
            rows.append(
                {
                    "text": _long_text(f"Paraphrased {model} rewrite {source}"),
                    "label": 1,
                    "family": model,
                    "domain": "news",
                    "attack": "paraphrase",
                    "decoding": "greedy",
                    "group": f"src-{source}",
                }
            )
    return pd.DataFrame(rows)


def _raid_fixture(tmp_path: Path, monkeypatch) -> Path:
    local = tmp_path / "raid"
    local.mkdir()
    _raid_frame().to_parquet(local / "train-clean.parquet", index=False)
    _fetch_manifest(local)
    monkeypatch.setattr(datasets, "RAID_DIR", local)
    return local


def test_load_raid_clean_includes_human_and_clean_ai_only(tmp_path, monkeypatch):
    _raid_fixture(tmp_path, monkeypatch)
    dataset = datasets.load_raid(attack="none")
    assert len(dataset) == 18  # 6 human + 6*2 clean AI
    assert set(dataset.families) == {"human", "gpt4", "llama3"}
    assert dataset.meta["attack"] == "none"
    assert dataset.domains == ["news"] * 18


def test_load_raid_attack_cell_keeps_human_rows(tmp_path, monkeypatch):
    _raid_fixture(tmp_path, monkeypatch)
    dataset = datasets.load_raid(attack="paraphrase")
    assert len(dataset) == 18
    assert int((dataset.labels == 0).sum()) == 6
    assert int((dataset.labels == 1).sum()) == 12


def test_load_raid_subsample_is_group_preserving_and_deterministic(tmp_path, monkeypatch):
    _raid_fixture(tmp_path, monkeypatch)
    first = datasets.load_raid(attack="none", max_rows=10)
    second = datasets.load_raid(attack="none", max_rows=10)
    assert first.groups == second.groups
    assert first.texts == second.texts
    # No group is split by the subsample.
    counts = {}
    for g in first.groups:
        counts[g] = counts.get(g, 0) + 1
    assert all(v == 3 for v in counts.values())  # each source group has 3 clean rows
    assert first.meta["subsample"]["n_rows_selected"] == len(first)


def test_raid_attacks_lists_non_none_attacks(tmp_path, monkeypatch):
    _raid_fixture(tmp_path, monkeypatch)
    assert datasets.raid_attacks() == ["paraphrase"]


def test_load_raid_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "RAID_DIR", tmp_path / "empty")
    with pytest.raises(datasets.DatasetError, match="fetch_raid"):
        datasets.load_raid()


# --- load_m4gt ------------------------------------------------------------------


def _m4gt_fixture(tmp_path: Path, monkeypatch) -> Path:
    local = tmp_path / "m4gt"
    local.mkdir()
    rows = []
    for i in range(4):
        rows.append(
            {
                "text": _long_text(f"Human {i}"),
                "label": 0,
                "family": "human",
                "domain": "arxiv",
                "group": f"g{i}",
            }
        )
        rows.append(
            {
                "text": _long_text(f"chatGPT {i}"),
                "label": 1,
                "family": "chatGPT",
                "domain": "arxiv",
                "group": f"h{i}",
            }
        )
    pd.DataFrame(rows).to_parquet(local / "subtask_a-clean.parquet", index=False)
    pd.DataFrame(rows).to_parquet(local / "subtask_a_multilingual-clean.parquet", index=False)
    _fetch_manifest(local)
    monkeypatch.setattr(datasets, "M4GT_DIR", local)
    return local


def test_load_m4gt_shapes(tmp_path, monkeypatch):
    _m4gt_fixture(tmp_path, monkeypatch)
    dataset = datasets.load_m4gt()
    assert len(dataset) == 8
    assert dataset.labels.tolist() == [0, 1] * 4
    assert set(dataset.families) == {"human", "chatGPT"}
    assert dataset.domains == ["arxiv"] * 8
    assert dataset.provenance.startswith("m4gt ")


def test_load_m4gtml_shapes(tmp_path, monkeypatch):
    _m4gt_fixture(tmp_path, monkeypatch)
    dataset = datasets.load_m4gtml()
    assert len(dataset) == 8
    assert dataset.provenance.startswith("m4gt-multilingual")


def test_load_m4gt_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "M4GT_DIR", tmp_path / "empty")
    with pytest.raises(datasets.DatasetError, match="fetch_m4gt"):
        datasets.load_m4gt()


# --- load_evobench ----------------------------------------------------------------


def _evobench_fixture(tmp_path: Path, monkeypatch) -> Path:
    local = tmp_path / "evobench"
    local.mkdir()
    rows = []
    for i in range(4):
        rows.append(
            {
                "text": _long_text(f"Original {i}"),
                "label": 0,
                "family": "human",
                "family_group": "human",
                "domain": "news",
                "group": f"news:g{i}",
            }
        )
        rows.append(
            {
                "text": _long_text(f"Version v1 {i}"),
                "label": 1,
                "family": "v1",
                "family_group": "gpt",
                "domain": "news",
                "group": f"news:g{i}",
            }
        )
    pd.DataFrame(rows).to_parquet(local / "clean.parquet", index=False)
    _fetch_manifest(local)
    monkeypatch.setattr(datasets, "EVOBENCH_DIR", local)
    return local


def test_load_evobench_families_are_versions(tmp_path, monkeypatch):
    _evobench_fixture(tmp_path, monkeypatch)
    dataset = datasets.load_evobench()
    assert len(dataset) == 8
    assert set(dataset.families) == {"human", "v1"}
    assert dataset.meta["family_groups"] == ["gpt", "human"]
    # Human original and its AI rewrite share a group (leakage control).
    assert dataset.groups[0] == dataset.groups[1]


def test_load_evobench_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "EVOBENCH_DIR", tmp_path / "empty")
    with pytest.raises(datasets.DatasetError, match="fetch_evobench"):
        datasets.load_evobench()


# --- fetch-script hygiene filters -------------------------------------------------


def test_fetch_m4gt_clean_split_filters(tmp_path):
    from bench import fetch_m4gt

    raw = tmp_path / "SubtaskA.jsonl"
    lines = [
        {"text": _long_text("keep human"), "label": 0, "model": "human", "source": "arxiv"},
        {"text": _long_text("keep ai"), "label": 1, "model": "chatGPT", "source": "arxiv"},
        {"text": _long_text("keep human"), "label": 0, "model": "human", "source": "arxiv"},  # dup
        {"text": "too short", "label": 1, "model": "chatGPT", "source": "arxiv"},
        {"text": "", "label": 1, "model": "chatGPT", "source": "arxiv"},
    ]
    raw.write_text("\n".join(json.dumps(row) for row in lines) + "\n", encoding="utf-8")
    counts = fetch_m4gt.clean_split(raw, tmp_path / "clean.parquet")
    assert counts["rows_raw"] == 5
    assert counts["dropped_exact_duplicates"] == 1
    assert counts["dropped_under_50_tokens"] == 1
    assert counts["dropped_empty"] == 1
    assert counts["rows_clean"] == 2
    frame = pd.read_parquet(tmp_path / "clean.parquet")
    assert frame["label"].tolist() == [0, 1]
    assert frame["family"].tolist() == ["human", "chatGPT"]
    # Group key is the content hash of the normalized (stripped) text.
    expected = hashlib.sha256(_long_text("keep human").strip().encode("utf-8")).hexdigest()[:16]
    assert frame["group"].iloc[0] == expected


def test_fetch_raid_clean_split_filters(tmp_path):
    from bench import fetch_raid

    raw = tmp_path / "train.csv"
    frame = pd.DataFrame(
        [
            {
                "id": 1,
                "source_id": "s1",
                "model": "human",
                "decoding": "",
                "attack": "",
                "domain": "news",
                "generation": _long_text("human one"),
            },
            {
                "id": 2,
                "source_id": "s1",
                "model": "gpt4",
                "decoding": "greedy",
                "attack": "none",
                "domain": "news",
                "generation": _long_text("ai one"),
            },
            {
                "id": 3,
                "source_id": "s2",
                "model": "gpt4",
                "decoding": "greedy",
                "attack": "none",
                "domain": "news",
                "generation": _long_text("ai one"),
            },  # dup text
            {
                "id": 4,
                "source_id": "s3",
                "model": "gpt4",
                "decoding": "greedy",
                "attack": "none",
                "domain": "news",
                "generation": "short",
            },
            {
                "id": 5,
                "source_id": "s4",
                "model": "gpt4",
                "decoding": "greedy",
                "attack": "none",
                "domain": "news",
                "generation": "",
            },
        ]
    )
    frame.to_csv(raw, index=False)
    counts = fetch_raid.clean_split(raw, tmp_path / "clean.parquet")
    assert counts["rows_raw"] == 5
    assert counts["dropped_empty"] == 1
    assert counts["dropped_exact_duplicates"] == 1
    assert counts["dropped_under_50_tokens"] == 1
    assert counts["rows_clean"] == 2
    clean = pd.read_parquet(tmp_path / "clean.parquet")
    assert clean["label"].tolist() == [0, 1]
    assert clean["group"].tolist() == ["s1", "s1"]  # source_id is the group key


def test_fetch_evobench_clean_groups_pairs_by_human_original(tmp_path):
    from bench import fetch_evobench

    payload_a = {
        "original": [_long_text("shared original"), _long_text("second original")],
        "sampled": [_long_text("v1 rewrite of shared"), _long_text("v1 rewrite of second")],
    }
    payload_b = {
        # Same human original appears again in a different version file.
        "original": [_long_text("shared original")],
        "sampled": [_long_text("v2 rewrite of shared")],
    }
    tarball = tmp_path / "evobench.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        for name, payload in (
            ("EvoBench-main/gpt/news_v1.raw_data.json", payload_a),
            ("EvoBench-main/gpt/news_v2.raw_data.json", payload_b),
            ("EvoBench-main/gpt/args/news_v1.raw_data.json", payload_b),  # skipped
        ):
            blob = json.dumps(payload).encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
    counts = fetch_evobench.clean(tarball, tmp_path / "clean.parquet")
    frame = pd.read_parquet(tmp_path / "clean.parquet")
    # The duplicate human original across files is deduped; both AI versions remain.
    assert counts["dropped_exact_duplicates"] == 1
    assert counts["rows_clean"] == 5
    shared = frame.loc[frame["text"].str.contains("shared")]
    assert shared["group"].nunique() == 1  # one group for original + both versions
    assert counts["n_source_files"] == 2  # args/ file skipped
