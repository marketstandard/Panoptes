"""Tests for v1 protocol extras: mixtures, robustness, watermarks, reproduction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import mixtures, robustness  # noqa: E402
from bench.tests.test_bench import tiny_dataset  # noqa: E402
from research.reproduce import reproduce  # noqa: E402


def test_paper_is_v1_with_named_authors():
    paper = (ROOT / "frontend" / "public" / "paper.html").read_text(encoding="utf-8")
    assert "Carrington Junior" in paper
    assert "Trey Huffine" in paper
    assert "Encryptic1" in paper
    assert "Version 1.0" in paper
    assert "Working paper, v2.1" not in paper
    assert "citation_author" in paper
    assert "Table 8" in paper


def test_citation_cff_lists_v1_authors():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "Carrington" in citation
    assert "Junior" in citation
    assert "Encryptic1" in citation
    assert "Huffine" in citation
    assert 'version: "1.0"' in citation


def test_v2_updates_document_blocked_work():
    docs = ROOT / "docs" / "v2-updates"
    readme = (docs / "README.md").read_text(encoding="utf-8")
    independent = (docs / "independent-reproduction.md").read_text(encoding="utf-8")
    assert "500–2,000" in readme or "500-2,000" in readme or "500–2000" in readme
    assert "RAID" in readme
    assert "independent: false" in independent
    assert "0.686" in independent


def test_mixture_workflows_all_run():
    texts, labels, groups, families = [], [], [], []
    pad_h = " extra words keep the detector from abstaining on a short human note after dinner."
    pad_a = " Furthermore the overall process additionally remains systematic and comprehensive."
    for i in range(8):
        groups.extend([f"prompt-{i}", f"prompt-{i}"])
        texts.append(f"i fixed the thing on my machine after a few tries. note {i}." + pad_h)
        labels.append(0)
        families.append("human")
        texts.append(
            "Furthermore, the systematic approach improves overall reliability. "
            f"Moreover, iteration {i} additionally reinforces consistent verification."
            + pad_a
        )
        labels.append(1)
        families.append("ai-x")
    from bench.datasets import Dataset

    dataset = Dataset(
        texts=texts,
        labels=np.array(labels),
        families=families,
        kinds=["text"] * len(texts),
        groups=groups,
        buckets=["50-149"] * len(texts),
        provenance="synthetic-mix",
        sha256="0" * 64,
    )
    result = mixtures.mixture_workflows(dataset)
    assert set(result["workflows"]) == {"ai_prefix", "ai_suffix", "interleave"}
    for curve in result["workflows"].values():
        assert curve["n_pairs"] > 0
        assert len(curve["rates"]) == 7


def test_robustness_includes_identity_and_truncation():
    dataset = tiny_dataset(24)
    result = robustness.robustness_curve(dataset)
    names = [row["transform"] for row in result["transforms"]]
    assert "identity" in names and "truncate_50" in names
    identity = next(row for row in result["transforms"] if row["transform"] == "identity")
    assert identity["delta_auroc"] == 0.0


def test_reproduce_selfcheck_is_labeled_first_party():
    card = reproduce()
    assert card["independent"] is False
    assert card["kind"] == "author_selfcheck"
    assert card["n_artifacts"] >= 10
    assert card["catalog_verified"] is True


def test_measurement_card_exists_and_validates():
    path = ROOT / "backend" / "artifacts" / "cards" / "measurement-protocol.json"
    assert path.exists()
    from research.validate_submission import validate_file

    errors = validate_file(path)
    assert errors == []
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n"] == 104
    assert "heuristic" in payload["metrics"]
    assert "logistic" in payload["metrics"]
