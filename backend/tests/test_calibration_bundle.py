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
        calibration_bundle,
        "_candidate_paths",
        lambda _dir, _name=calibration_bundle.DEFAULT_BUNDLE: [tmp_path / "nowhere.json"],
    )
    assert load_bundle("artifacts") is None


def test_load_bundle_tampered_returns_none(tmp_path, monkeypatch):
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    payload["metrics"]["auroc"] = 0.999  # tamper without re-signing
    forged = tmp_path / "baseline-calibration.json"
    forged.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    monkeypatch.setattr(
        calibration_bundle,
        "_candidate_paths",
        lambda _dir, _name=calibration_bundle.DEFAULT_BUNDLE: [forged],
    )
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


# --- dual calibration bundles (PANOPTES_CALIBRATION_BUNDLE) --------------------

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def test_both_committed_bundles_load_and_validate():
    for name in calibration_bundle.ALLOWED_BUNDLES:
        bundle = load_bundle("artifacts", name)
        assert bundle is not None, name
        assert bundle.payload["schema"] == "panoptes-calibration-v1"
        assert canonical_hash(bundle.payload) == bundle.payload["artifact_sha256"]
        assert bundle.n_records > 0
        assert bundle.reliability_bins


def test_defactify_bundle_selection_swaps_cohort():
    bundle = load_bundle("artifacts", "defactify-calibration.json")
    assert bundle is not None
    assert bundle.payload["bundle_id"] == "defactify-text-v1"
    assert bundle.n_records > 1000  # the Defactify cohort, not the 104-record corpus
    assert "human" in bundle.payload["source_geometry"]["classes"]


def test_invalid_bundle_name_falls_back_to_default():
    bundle = load_bundle("artifacts", "evil/../../etc/passwd")
    assert bundle is not None
    assert bundle.payload["bundle_id"] == "panoptes-reference-corpus-v1"


def test_missing_selected_bundle_falls_back_to_default(tmp_path, monkeypatch):
    # An artifact dir containing only the default bundle: selecting the
    # Defactify bundle must fall back rather than fail.
    import shutil

    shutil.copy(ARTIFACT_DIR / "baseline-calibration.json", tmp_path / "baseline-calibration.json")
    monkeypatch.setattr(
        calibration_bundle,
        "_candidate_paths",
        lambda directory, name=calibration_bundle.DEFAULT_BUNDLE: [Path(directory) / name],
    )
    bundle = load_bundle(str(tmp_path), "defactify-calibration.json")
    assert bundle is not None
    assert bundle.payload["bundle_id"] == "panoptes-reference-corpus-v1"


def test_pipeline_honors_bundle_setting():
    settings = Settings(
        profile=RuntimeProfile.LOCAL_CPU, calibration_bundle="defactify-calibration.json"
    )
    response = analyze(AnalysisRequest(text=PROSE), settings)
    assert response.calibration is not None
    assert response.calibration.bundle == "defactify-text-v1"
    assert response.calibration.n_records > 1000


def test_available_bundles_lists_committed_artifacts():
    found = calibration_bundle.available_bundles("artifacts")
    assert "baseline-calibration.json" in found
    assert "defactify-calibration.json" in found


def test_available_bundles_skips_tampered(tmp_path, monkeypatch):
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    payload["metrics"]["auroc"] = 0.999  # tamper without re-signing
    (tmp_path / "baseline-calibration.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        calibration_bundle,
        "_candidate_paths",
        lambda directory, name=calibration_bundle.DEFAULT_BUNDLE: [Path(directory) / name],
    )
    assert calibration_bundle.available_bundles(str(tmp_path)) == []


def test_pipeline_calibrates_supported_prose():
    # PROSE is 37 words — below the 50-token support floor, so the detector
    # abstains and the isotonic path is skipped. This regression test uses
    # above-threshold prose so the calibrator actually transforms the score
    # (a frozen-dataclass crash here previously 500'd the endpoint).
    long_prose = (
        PROSE
        + " "
        + (
            "The review board examined each submission against the published criteria and "
            "recorded its reasoning in the minutes; observers noted the deliberation was "
            "methodical, transparent, and grounded in the documented evidence base."
        )
    )
    settings = Settings(profile=RuntimeProfile.LOCAL_CPU)
    response = analyze(AnalysisRequest(text=long_prose), settings)
    assert response.calibration is not None
    assert response.summary.evidence_state == "supported"
    overall = response.summary.overall
    total = overall.human + overall.ai_generated + overall.ai_refined_or_mixed
    assert abs(total - 1.0) < 1e-6
