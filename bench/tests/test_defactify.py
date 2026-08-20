"""Tests for the Defactify bench loader and the attribution experiment.

Run from the repository root: python -m pytest bench/tests
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

from bench import attribution, datasets  # noqa: E402

pd = pytest.importorskip("pandas", reason="defactify loader requires pandas/pyarrow")


def _write_split(directory: Path, split: str, records: list[dict]) -> None:
    pd.DataFrame(records).to_parquet(directory / f"{split}-clean.parquet", index=False)


def _fixture(tmp_path: Path, monkeypatch, records_by_split: dict[str, list[dict]]) -> Path:
    local = tmp_path / "defactify"
    local.mkdir()
    for split, records in records_by_split.items():
        _write_split(local, split, records)
    (local / "fetch-manifest.json").write_text(
        json.dumps(
            {
                "created_utc": "2026-01-01T00:00:00Z",
                "splits": {
                    split: {"rows_clean": len(rows)} for split, rows in records_by_split.items()
                },
                "artifact_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(datasets, "DEFACTIFY_DIR", local)
    return local


def _row(text: str, label: int, family: str) -> dict:
    return {"text": text, "label": label, "family": family}


HUMAN_BODY = (
    "the city council met on tuesday to debate the budget and the school board "
    "presented its annual report while residents lined up to comment on zoning "
)


# --- loader mapping and hygiene shape ----------------------------------------


def test_load_defactify_maps_labels_and_families(tmp_path, monkeypatch):
    records = []
    families = list(datasets.DEFACTIFY_FAMILIES)
    for i, family in enumerate(families):
        text = f"Article {i}: " + HUMAN_BODY + "the meeting adjourned late."
        records.append(_row(text, 0 if family == "Human_Story" else 1, family))
    _fixture(tmp_path, monkeypatch, {"train": records})

    dataset = datasets.load_defactify(splits=("train",))
    assert len(dataset) == len(families)
    assert dataset.labels.tolist() == [0] + [1] * 6
    assert dataset.families == [
        "human",
        "gemma-2-9b",
        "mistral-7b",
        "qwen-2-72b",
        "llama-8b",
        "yi-large",
        "gpt-4o",
    ]
    assert dataset.kinds == ["text"] * len(families)
    assert dataset.provenance.startswith("defactify-text")
    assert len(dataset.sha256) == 64
    assert dataset.meta["official_split_counts"] == {"train": len(families)}


def test_load_defactify_rejects_unmapped_family(tmp_path, monkeypatch):
    _fixture(
        tmp_path,
        monkeypatch,
        {"train": [_row(HUMAN_BODY + "text here.", 1, "Mystery-Model-9")]},
    )
    with pytest.raises(datasets.DatasetError, match="unmapped Defactify families"):
        datasets.load_defactify(splits=("train",))


def test_load_defactify_missing_split_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "DEFACTIFY_DIR", tmp_path / "empty")
    with pytest.raises(datasets.DatasetError, match="fetch_defactify"):
        datasets.load_defactify(splits=("train",))


# --- story-group reconstruction -----------------------------------------------


STORY_VOCAB = [
    "council budget zoning schools residents vote meeting adjourned ordinance",
    "harbor fishing boats tide storm dock captain crew nets salmon",
    "stadium playoff quarterback touchdown coach fans season championship",
    "laboratory vaccine trial researchers dose antibody clinical patients",
    "vineyard harvest grapes winemaker barrel vintage drought irrigation",
    "museum exhibition painter canvas curator gallery century portrait",
]


def test_reconstruct_story_groups_finds_constructed_near_duplicates():
    texts = []
    for story, vocab in enumerate(STORY_VOCAB):
        base = f"Report {story}: the {vocab} dominated the discussion this week."
        texts.append(base)
        for variant in range(3):  # three near-rewrites of the same story
            texts.append(
                base + f" Furthermore, officials described the {vocab.split()[0]} outcome"
                f" as significant ({variant})."
            )
    loners = [
        "asteroid telescope orbit comet nasa astronomers tracked the object",
        "ballet dancer theater rehearsal symphony performers captivated audiences",
        "chess grandmaster tournament gambit checkmate players studied positions",
        "volcano eruption lava ash crater geologists monitored seismic activity",
    ]
    texts.extend(loners)  # unrelated stories stay singletons

    groups, stats = datasets.reconstruct_story_groups(texts, threshold=0.45, chunk=8)
    assert len(groups) == len(texts)
    # Each story block of 4 (original + 3 rewrites) must share one group.
    for story in range(6):
        block = groups[story * 4 : story * 4 + 4]
        assert len(set(block)) == 1, f"story {story} split across groups: {block}"
    # Distinct stories must not merge.
    assert len(set(groups[:24])) == 6
    assert stats["n_groups"] == len(set(groups))
    assert stats["group_size_max"] >= 4
    assert stats["singletons"] >= 1


def test_reconstruct_story_groups_stats_are_consistent():
    texts = [f"unique text number {i} " + HUMAN_BODY for i in range(10)]
    groups, stats = datasets.reconstruct_story_groups(texts, threshold=0.9, chunk=4)
    assert stats["n_groups"] == len(set(groups))
    assert stats["singletons"] <= stats["n_groups"]
    assert stats["group_size_mean"] == pytest.approx(len(texts) / stats["n_groups"])


# --- leakage audit --------------------------------------------------------------


def test_leakage_audit_math(tmp_path, monkeypatch):
    shared = "Shared story: " + HUMAN_BODY + "the decision was final."
    train = [
        _row(shared, 0, "Human_Story"),
        _row("Train-only story: " + HUMAN_BODY + "it ended quietly.", 0, "Human_Story"),
    ]
    test = [
        _row(
            shared + " Furthermore, the outcome was significant.", 1, "GPT-4o"
        ),  # leaks into train
        _row("Test-only story: " + HUMAN_BODY[::-1] + "nothing followed.", 1, "Llama-8B"),
    ]
    _fixture(tmp_path, monkeypatch, {"train": train, "test": test})

    dataset = datasets.load_defactify(splits=("train", "test"), threshold=0.45)
    audit = dataset.meta["leakage_audit"]
    assert audit["official_test_rows"] == 2
    assert audit["official_test_rows_with_train_near_duplicate"] == 1
    assert audit["official_split_story_leakage_rate"] == pytest.approx(0.5)


def test_story_group_cache_roundtrip(tmp_path, monkeypatch):
    records = [_row(f"Story {i}: " + HUMAN_BODY + "done.", 0, "Human_Story") for i in range(6)]
    local = _fixture(tmp_path, monkeypatch, {"train": records})
    first = datasets.load_defactify(splits=("train",))
    cache = local / "story-groups-t0.45.json"
    assert cache.exists()
    second = datasets.load_defactify(splits=("train",))
    assert second.groups == first.groups
    assert second.meta["group_reconstruction"] == first.meta["group_reconstruction"]


# --- attribution metric shapes ---------------------------------------------------


def _attribution_dataset(n_per_family: int = 8) -> datasets.Dataset:
    texts, labels, families, groups = [], [], [], []
    for story in range(n_per_family):
        base = f"Story {story}: " + HUMAN_BODY + f"the vote was {story + 4} to 1."
        texts.append(base)
        labels.append(0)
        families.append("human")
        groups.append(f"story-{story}")
        for family in attribution.ATTRIBUTION_CLASSES[1:]:
            texts.append(
                base + f" Furthermore, the {family} rewrite additionally"
                f" emphasized significance overall."
            )
            labels.append(1)
            families.append(family)
            groups.append(f"story-{story}")
    return datasets.Dataset(
        texts=texts,
        labels=np.array(labels),
        families=families,
        kinds=["text"] * len(texts),
        groups=groups,
        buckets=["50-149"] * len(texts),
        provenance="synthetic-attribution-test",
        sha256="0" * 64,
    )


def test_f1_report_shapes():
    y_true = np.array([0, 1, 2, 3, 4, 5, 6] * 4)
    y_pred = np.array([0, 1, 2, 3, 4, 5, 0] * 4)
    report = attribution._f1_report(y_true, y_pred)
    assert set(report["per_family_f1"]) == set(attribution.ATTRIBUTION_CLASSES)
    assert len(report["confusion_matrix"]) == 7
    assert all(len(row) == 7 for row in report["confusion_matrix"])
    assert 0.0 <= report["macro_f1"] <= 1.0
    assert report["accuracy"] == pytest.approx(6 / 7)


def test_logistic_attribution_cv_shapes():
    dataset = _attribution_dataset()
    report = attribution.logistic_attribution_cv(dataset)
    assert report["oof_probabilities"].shape == (len(dataset), 7)
    row_sums = report["oof_probabilities"].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    assert set(report["per_family_f1"]) == set(attribution.ATTRIBUTION_CLASSES)
    # Constructed rewrites carry a family marker phrase; the model should beat chance (1/7).
    assert report["accuracy"] > 1 / 7


def test_class_index_rejects_unknown_family():
    dataset = _attribution_dataset()
    dataset.families[0] = "mystery-model"
    with pytest.raises(attribution.AttributionError, match="7-class"):
        attribution._class_index(dataset)
