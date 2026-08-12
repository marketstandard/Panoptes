"""Load and validate the signed calibration artifact produced from the
hash-verified baseline corpus (research/calibration.py).

The runtime treats the artifact as *calibration infrastructure*: it
supplies held-out reliability metrics, the isotonic calibrator, and the
corpus-fitted source-family geometry. If the artifact is absent,
invalid, or fails its self-hash check, every consumer falls back to the
documented heuristic behavior — the API never hard-fails on calibration
infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_KEYS = {
    "schema",
    "artifact_sha256",
    "metrics",
    "binary_calibrator",
    "source_geometry",
    "conformal",
    "corpus",
}
SCHEMA_ID = "panoptes-calibration-v1"


@dataclass(frozen=True)
class CalibrationBundle:
    payload: dict[str, Any]
    path: Path

    @property
    def metrics(self) -> dict[str, float]:
        return self.payload["metrics"]

    @property
    def cohort(self) -> str:
        return self.payload.get("cohort", "unknown")

    @property
    def n_records(self) -> int:
        return int(self.payload.get("corpus", {}).get("n_records", 0))

    @property
    def reliability_bins(self) -> list[dict]:
        return self.payload.get("reliability_bins", [])

    @property
    def conformal(self) -> dict:
        return self.payload.get("conformal", {})

    def calibrate(self, raw_score: float) -> float:
        calibrator = self.payload["binary_calibrator"]
        return float(
            np.interp(raw_score, calibrator["x_thresholds"], calibrator["y_thresholds"])
        )


def canonical_hash(payload: dict) -> str:
    clone = dict(payload)
    clone.pop("artifact_sha256", None)
    canonical = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


ALLOWED_BUNDLES = ("baseline-calibration.json", "defactify-calibration.json")
DEFAULT_BUNDLE = "baseline-calibration.json"


def _candidate_paths(artifact_dir: str, bundle_name: str = DEFAULT_BUNDLE) -> list[Path]:
    return [
        Path(artifact_dir) / bundle_name,
        Path(__file__).resolve().parents[2] / "artifacts" / bundle_name,
    ]


def available_bundles(artifact_dir: str) -> list[str]:
    """Allowlisted bundle names whose artifacts exist and self-validate."""
    found: list[str] = []
    for name in ALLOWED_BUNDLES:
        for path in _candidate_paths(artifact_dir, name):
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                break
            if (
                REQUIRED_KEYS.issubset(payload)
                and payload.get("schema") == SCHEMA_ID
                and payload.get("artifact_sha256") == canonical_hash(payload)
            ):
                found.append(name)
            break
    return found


def load_bundle(artifact_dir: str, bundle_name: str = DEFAULT_BUNDLE) -> CalibrationBundle | None:
    """Load the selected bundle if present and self-consistent; otherwise None.

    `bundle_name` is allowlist-validated (PANOPTES_CALIBRATION_BUNDLE): an
    unknown or missing selection falls back to the default corpus-fitted
    bundle, and an invalid artifact falls back to heuristic behavior.
    """
    if bundle_name not in ALLOWED_BUNDLES:
        bundle_name = DEFAULT_BUNDLE
    names = [bundle_name] if bundle_name == DEFAULT_BUNDLE else [bundle_name, DEFAULT_BUNDLE]
    for name in names:
        for path in _candidate_paths(artifact_dir, name):
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not REQUIRED_KEYS.issubset(payload):
                continue
            if payload.get("schema") != SCHEMA_ID:
                continue
            if payload.get("artifact_sha256") != canonical_hash(payload):
                continue
            return CalibrationBundle(payload=payload, path=path)
    return None


def geometry_membership(
    bundle: CalibrationBundle, features: dict[str, float]
) -> tuple[dict[str, float], float] | None:
    """Corpus-fitted source geometry: per-family softmax over Mahalanobis
    distance to family centroids, plus an open-set knownness score from the
    conformal threshold. Returns None when the geometry is unusable."""
    geometry = bundle.payload.get("source_geometry", {})
    names = bundle.payload.get("feature_names")
    if not names or "centroids" not in geometry or "classes" not in geometry:
        return None
    try:
        x = np.array([features[name] for name in names], dtype=float)
        mean = np.array(geometry["scaler_mean"], dtype=float)
        scale = np.array(geometry["scaler_scale"], dtype=float)
        components = np.array(geometry["pca_components"], dtype=float)
        pca_mean = np.array(geometry["pca_mean"], dtype=float)
        embedded = ((x - mean) / scale - pca_mean) @ components.T
        centroids = geometry["centroids"]
        distances: dict[str, float] = {}
        for family, centroid in centroids.items():
            diff = embedded - np.array(centroid, dtype=float)
            covariance = np.array(geometry["covariances"][family], dtype=float)
            precision = np.linalg.pinv(covariance + 1e-6 * np.eye(covariance.shape[0]))
            distances[family] = float(diff @ precision @ diff)
        if not distances:
            return None
        logits = {family: -0.5 * d for family, d in distances.items()}
        maximum = max(logits.values())
        exps = {family: math.exp(value - maximum) for family, value in logits.items()}
        total = sum(exps.values())
        probabilities = {family: value / total for family, value in exps.items()}
        nearest = min(distances.values())
        # Open-set unknown score: a monotone map of the nearest centroid
        # distance (squared Mahalanobis in the embedded space). Distances at
        # the scale of the training cohort map to low unknown; far points map
        # toward 1. Documented as distance-derived, not a coverage guarantee.
        unknown = 1.0 - float(math.exp(-0.5 * nearest))
        return probabilities, min(max(unknown, 0.0), 1.0)
    except (KeyError, ValueError, np.linalg.LinAlgError):
        return None
