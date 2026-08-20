"""Tests for the Phase 6 evidence-transportability framework.

Uses a synthetic dataset with a known cohort structure and a separable feature
signal (AI texts use long/rare words, human texts short/common ones) so the
logistic tier learns a non-degenerate scorer. Assertions target the framework's
structure and the data firewall, not absolute accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench import transport
from bench.datasets import Dataset
from bench.detectors import make_detector

HUMAN_WORDS = ["the", "cat", "sat", "on", "a", "mat", "and", "ran", "big", "red", "dog", "sun"]
AI_WORDS = [
    "utilize", "furthermore", "consequently", "nevertheless", "comprehensive",
    "implementation", "optimization", "demonstrate", "substantial", "methodology",
]


def _text(label: int, rng: np.random.Generator, n_words: int = 24) -> str:
    vocab = AI_WORDS if label == 1 else HUMAN_WORDS
    return " ".join(rng.choice(vocab, size=n_words))


def _synthetic_dataset(
    domains: tuple[str, ...] = ("dA", "dB", "dC"),
    groups_per_domain: int = 8,
    rows_per_group: int = 8,
    seed: int = 0,
) -> Dataset:
    rng = np.random.default_rng(seed)
    texts, labels, families, kinds, groups, buckets, doms = [], [], [], [], [], [], []
    for domain in domains:
        for g in range(groups_per_domain):
            group = f"{domain}_g{g}"
            for r in range(rows_per_group):
                label = (g + r) % 2  # balanced within each group
                texts.append(_text(label, rng))
                labels.append(label)
                families.append(f"gen{label}" if label else "human")
                kinds.append("text")
                groups.append(group)
                buckets.append("short")
                doms.append(domain)
    return Dataset(
        texts=texts,
        labels=np.array(labels, dtype=int),
        families=families,
        kinds=kinds,
        groups=groups,
        buckets=buckets,
        provenance="synthetic-transport-test",
        sha256="0" * 64,
        domains=doms,
    )


def _logistic():
    return make_detector("logistic")


# --- cohort structure ----------------------------------------------------------


def test_cohort_keys_axes():
    ds = _synthetic_dataset()
    assert set(transport.cohort_keys(ds, "domain")) == {"dA", "dB", "dC"}
    assert set(transport.cohort_keys(ds, "generator")) == {"human", "gen1"}
    dg = transport.cohort_keys(ds, "domain_generator")
    assert all("|" in k for k in dg)
    with pytest.raises(ValueError):
        transport.cohort_keys(ds, "nope")


def test_cohort_index_partitions_rows():
    ds = _synthetic_dataset()
    index = transport.cohort_index(ds, "domain")
    assert set(index) == {"dA", "dB", "dC"}
    total = sum(len(v) for v in index.values())
    assert total == len(ds)
    # Disjoint.
    seen = set()
    for idx in index.values():
        assert not (seen & set(idx.tolist()))
        seen.update(idx.tolist())


# --- per-cell metrics ----------------------------------------------------------


def test_transport_cell_metrics_block():
    ds = _synthetic_dataset()
    n = len(ds)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    cal, test = idx[: n // 2], idx[n // 2 :]
    det = _logistic().fit(ds, cal)
    cal_p = det.predict_proba(ds, cal)
    test_p = det.predict_proba(ds, test)
    cell = transport.transport_cell_metrics(
        ds.labels[cal], cal_p, ds.labels[test], test_p,
        [ds.groups[int(i)] for i in test], n_boot=50,
    )
    assert cell["n_test"] == len(test)
    assert 0.0 <= cell["auroc"] <= 1.0
    assert "conformal" in cell and cell["conformal"]["fit_on"] == "calibration"
    assert "operating_points" in cell
    assert "selective_risk" in cell
    assert cell["auroc_group_bootstrap"]["resample_unit"] == "group"
    assert "prevalence_views" in cell
    assert cell["shift_caveat"]


def test_transport_cell_metrics_degenerate_single_class():
    labels = np.ones(10, dtype=int)
    probs = np.full(10, 0.9)
    cell = transport.transport_cell_metrics(
        labels, probs, labels, probs, ["g"] * 10, n_boot=10
    )
    assert cell.get("degenerate") is True


# --- LOCO representation transport ---------------------------------------------


def test_loco_holds_out_each_cohort_no_leakage():
    ds = _synthetic_dataset()
    result = transport.leave_one_cohort_out(ds, "domain", _logistic, min_cohort=20, n_boot=20)
    assert result["kind"] == "representation_transport_loco"
    assert result["n_cohorts"] == 3
    for cohort, cell in result["cells"].items():
        if cell.get("skipped"):
            continue
        assert cell["kind"] == "representation_transport_loco"
        assert cell["calibration_fit_on"] == "source_calibration"
        assert "diagonal" in cell and "target" in cell
        assert "delta_from_diagonal" in cell
        # Firewall: the held-out cohort's groups must not appear in source train/cal.
        held_groups = {ds.groups[int(i)] for i in transport.cohort_index(ds, "domain")[cohort]}
        # Recompute source split to confirm disjointness.
        source_mask = np.ones(len(ds), dtype=bool)
        source_mask[transport.cohort_index(ds, "domain")[cohort]] = False
        source = ds.subset(np.where(source_mask)[0])
        assert not (held_groups & set(source.groups))


def test_loco_skips_tiny_cohorts():
    ds = _synthetic_dataset()
    result = transport.leave_one_cohort_out(ds, "domain", _logistic, min_cohort=10**6, n_boot=10)
    assert all(c.get("skipped") for c in result["cells"].values())


# --- calibration transfer --------------------------------------------------------


def test_calibration_transfer_matrix_diagonal_and_offdiagonal():
    ds = _synthetic_dataset()
    result = transport.calibration_transfer(ds, "domain", _logistic, min_cell=4, n_boot=20)
    assert result["kind"] == "calibration_transfer"
    matrix = result["matrix"]
    assert matrix, "expected a non-empty source x target matrix"
    on_diag_seen = False
    off_diag_seen = False
    for src, row in matrix.items():
        for tgt, cell in row.items():
            assert cell["source_cohort"] == src
            assert cell["target_cohort"] == tgt
            if src == tgt:
                assert cell["on_diagonal"] is True
                on_diag_seen = True
            else:
                assert cell["on_diagonal"] is False
                off_diag_seen = True
    assert on_diag_seen and off_diag_seen
    summary = result["summary"]
    assert "calibration_transport_auroc_gap" in summary


# --- pooled generalization -------------------------------------------------------


def test_pooled_generalization_labels_seen_cohort():
    ds = _synthetic_dataset()
    result = transport.pooled_generalization(ds, "domain", _logistic, min_cell=4, n_boot=20)
    assert result["kind"] == "pooled_generalization"
    assert result["cohort_role"] == "seen-cohort"
    for cohort, cell in result["cells"].items():
        assert cell["cohort_role"] == "seen-cohort"
        assert "auroc" in cell


# --- dataset-origin probe ---------------------------------------------------------


def test_dataset_origin_probe_runs():
    ds = _synthetic_dataset()
    probe = transport.dataset_origin_probe(ds, "domain", n_splits=3)
    assert probe["n_cohorts"] == 3
    assert 0.0 <= probe["cv_accuracy"] <= 1.0
    assert "chance_accuracy" in probe
    assert "interpretation" in probe


# --- source-balanced sensitivity ----------------------------------------------------


def test_source_balanced_sensitivity_reports_both_views():
    ds = _synthetic_dataset()
    n = len(ds)
    idx = np.arange(n)
    det = _logistic().fit(ds, idx)
    probs = det.predict_proba(ds, idx)
    sens = transport.source_balanced_sensitivity(
        ds.labels, probs, ds.groups, n_boot=20
    )
    assert "natural" in sens and "source_balanced" in sens
    assert sens["target_prevalence"] == 0.5
    assert sens["auroc_group_bootstrap"]["resample_unit"] == "group"


# --- signed-card schema validation -------------------------------------------------


def _wrap_card(result: dict, schema: str, detector: str) -> dict:
    card = {
        "schema": schema,
        "detector": detector,
        "created_utc": "2026-08-19T00:00:00Z",
        "limitations": ["synthetic test card"],
        **{k: v for k, v in result.items() if k not in {"schema"}},
    }
    return card


def _wrap_transport_card_v21(loco_result: dict, detector: str) -> dict:
    """Wrap a LOCO result in the v2.1 representation-transport card structure."""
    return {
        "schema": "panoptes-representation-transport-v1",
        "detector": detector,
        "created_utc": "2026-08-19T00:00:00Z",
        "cross_dataset_loco": loco_result,
        "within_mage_loco": None,
        "pooled_seen_cohort": {},
        "external_targets": {},
        "limitations": ["synthetic test card"],
    }


def test_representation_transport_card_schema_validates(tmp_path):
    import json

    from bench.cards import sign
    from bench.validate_submission import validate_file

    ds = _synthetic_dataset()
    result = transport.leave_one_cohort_out(ds, "domain", _logistic, min_cohort=20, n_boot=20)
    card = _wrap_transport_card_v21(result, "logistic-tier0")
    path = tmp_path / "representation-transport.json"
    path.write_text(json.dumps(sign(card)), encoding="utf-8")
    assert validate_file(path) == []


def test_calibration_transfer_card_schema_validates(tmp_path):
    import json

    from bench.cards import sign
    from bench.validate_submission import validate_file

    ds = _synthetic_dataset()
    result = transport.calibration_transfer(ds, "domain", _logistic, min_cell=4, n_boot=20)
    card = _wrap_card(result, "panoptes-calibration-transfer-v1", "logistic-tier0")
    path = tmp_path / "calibration-transfer.json"
    path.write_text(json.dumps(sign(card)), encoding="utf-8")
    assert validate_file(path) == []


def test_representation_transport_card_rejects_bad_axis(tmp_path):
    import json

    from bench.cards import sign
    from bench.validate_submission import validate_file

    ds = _synthetic_dataset()
    result = transport.leave_one_cohort_out(ds, "domain", _logistic, min_cohort=20, n_boot=20)
    result["axis"] = "not-an-axis"
    card = _wrap_transport_card_v21(result, "logistic-tier0")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(sign(card)), encoding="utf-8")
    assert validate_file(path) != []
