import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research"))

from calibration import conformal_threshold, extract_features, main  # noqa: E402


def test_feature_vector_is_stable() -> None:
    first = extract_features("A repeatable feature vector for calibration.")
    second = extract_features("A repeatable feature vector for calibration.")
    assert first == second
    assert len(first) == 10


def test_conformal_threshold_requires_scores() -> None:
    try:
        conformal_threshold(__import__("numpy").array([]))
    except ValueError as error:
        assert "required" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_calibration_main_writes_artifact() -> None:
    main()
    artifact = Path(__file__).resolve().parents[2] / "backend" / "artifacts" / "baseline-calibration.json"
    assert artifact.exists()
    assert "artifact_sha256" in artifact.read_text(encoding="utf-8")
