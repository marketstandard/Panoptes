"""Calibration bundle loading, fallback, and corpus-fitted attribution."""

from __future__ import annotations

import json
from pathlib import Path

from panoptes.analysis import calibration_bundle
from panoptes.analysis.attribution import source_family_distribution
from panoptes.analysis.calibration_bundle import canonical_hash, load_bundle
from panoptes.analysis.pipeline import analyze
from panoptes.schemas import AnalysisRequest, ContentType, RuntimeProfile
from panoptes.settings import Settings

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "baseline-calibration.json"

PROSE = (
    "Furthermore, the systematic evaluation of evidence improves overall reliability. "
    "Moreover, consistent documentation supports verification; therefore teams should "
    "maintain detailed records of every decision and its rationale for future review. "
    "Additionally, transparent reporting enables independent replication of results."
)


def test_load_bundle_reads_committed_artifact():
    bundle = load_bundle("artifacts")
    assert bundle is not None
    assert bundle.payload["schema"] == "panoptes-calibration-v1"
    assert 0.0 <= bundle.metrics["ece"] <= 1.0
    assert bundle.n_records > 0
    assert bundle.reliability_bins


def test_load_bundle_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(
        calibration_bundle, "_candidate_paths", lambda _dir: [tmp_path / "nowhere.json"]
    )
    assert load_bundle("artifacts") is None


def test_load_bundle_tampered_returns_none(tmp_path, monkeypatch):
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    payload["metrics"]["auroc"] = 0.999  # tamper without re-signing
    forged = tmp_path / "baseline-calibration.json"
    forged.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    monkeypatch.setattr(calibration_bundle, "_candidate_paths", lambda _dir: [forged])
    assert load_bundle("artifacts") is None


def test_canonical_hash_ignores_only_signature_field():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert canonical_hash(payload) == payload["artifact_sha256"]


def test_attribution_basis_corpus_fitted_vs_heuristic():
    bundle = load_bundle("artifacts")
    assert bundle is not None
    fitted = source_family_distribution(PROSE, ContentType.PROSE, bundle=bundle)
    assert fitted.basis == "corpus-fitted"
    assert fitted.cohort_size == bundle.n_records
    assert fitted.conditional_on_ai  # non-empty distribution
    assert abs(sum(item.probability for item in fitted.conditional_on_ai) - 1.0) < 1e-6

    heuristic = source_family_distribution(PROSE, ContentType.PROSE)
    assert heuristic.basis == "heuristic"
    assert heuristic.cohort_size is None


def test_pipeline_includes_calibration_block_outside_fixture():
    settings = Settings(profile=RuntimeProfile.LOCAL_CPU)
    response = analyze(AnalysisRequest(text=PROSE), settings)
    assert response.calibration is not None
    assert response.calibration.bundle == "panoptes-reference-corpus-v1"
    assert response.calibration.n_records > 0
    assert response.posterior.reliability_error == response.calibration.ece
    assert response.source_families.basis == "corpus-fitted"


def test_pipeline_fixture_mode_has_no_calibration_block():
    settings = Settings(profile=RuntimeProfile.FIXTURE)
    response = analyze(AnalysisRequest(fixture="ai-prose"), settings)
    assert response.calibration is not None  # block describes the artifact even in fixture mode
    assert response.posterior.cohort == "fixture"
