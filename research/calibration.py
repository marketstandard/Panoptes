from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CalibrationRecord:
    text: str
    label: int
    group: str
    domain: str
    model_family: str
    length_bucket: str


def length_bucket(token_count: int) -> str:
    if token_count < 50:
        return "lt50"
    if token_count < 150:
        return "50-149"
    if token_count < 500:
        return "150-499"
    return "500plus"


def _token_entropy(tokens: list[str]) -> float:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = len(tokens)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def extract_features(text: str) -> list[float]:
    tokens = [token.lower() for token in text.split()]
    if not tokens:
        return [0.0] * 10
    lengths = [len(token.strip(".,;:!?()[]{}")) for token in tokens]
    unique = len(set(tokens)) / len(tokens)
    long_words = sum(length >= 7 for length in lengths) / len(lengths)
    digits = sum(char.isdigit() for char in text) / max(len(text), 1)
    punctuation = sum(char in ".,;:!?()[]{}" for char in text) / max(len(text), 1)
    connectors = sum(
        token in {"however", "therefore", "moreover", "furthermore", "additionally", "overall"}
        for token in tokens
    ) / len(tokens)
    mean_length = sum(lengths) / len(lengths)
    variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
    lines = [line for line in text.splitlines() if line.strip()]
    avg_line = sum(len(line) for line in lines) / max(len(lines), 1)
    line_sd = math.sqrt(
        sum((len(line) - avg_line) ** 2 for line in lines) / max(len(lines) - 1, 1)
    )
    entropy = _token_entropy(tokens)
    return [
        unique,
        long_words,
        digits,
        punctuation,
        connectors,
        mean_length,
        variance,
        avg_line,
        line_sd,
        entropy,
    ]


def fit_binary_calibrator(raw_scores: Iterable[float], labels: Iterable[int]) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(list(raw_scores), list(labels))
    return model


def fit_source_geometry(features: np.ndarray, families: np.ndarray) -> dict:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    components = min(16, scaled.shape[1], max(1, scaled.shape[0] - 1))
    pca = PCA(n_components=components, random_state=13)
    embedded = pca.fit_transform(scaled)

    centroids = {}
    covariances = {}
    for family in sorted(set(families)):
        family_rows = embedded[families == family]
        centroids[family] = family_rows.mean(axis=0).tolist()
        covariances[family] = LedoitWolf().fit(family_rows).covariance_.tolist()

    classifier = LogisticRegression(max_iter=1000, C=0.5)
    classifier.fit(embedded, families)
    return {
        "schema": "source-geometry-v1",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "pca_components": pca.components_.tolist(),
        "pca_mean": pca.mean_.tolist(),
        "centroids": centroids,
        "covariances": covariances,
        "classes": classifier.classes_.tolist(),
        "coef": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
    }


def conformal_threshold(nonconformity_scores: np.ndarray, alpha: float = 0.1) -> float:
    if len(nonconformity_scores) == 0:
        raise ValueError("nonconformity scores are required")
    quantile = math.ceil((len(nonconformity_scores) + 1) * (1 - alpha)) / len(nonconformity_scores)
    quantile = min(max(quantile, 0.0), 1.0)
    return float(np.quantile(nonconformity_scores, quantile, method="higher"))


def evaluate_binary(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    result = {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
    }
    for fpr_target in (0.01, 0.05):
        result[f"tpr_at_{int(fpr_target * 100)}fpr"] = tpr_at_fpr(labels, probabilities, fpr_target)
    return result


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if not np.any(mask):
            continue
        error += mask.mean() * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
    return float(error)


def tpr_at_fpr(labels: np.ndarray, probabilities: np.ndarray, target_fpr: float) -> float:
    negative_scores = probabilities[labels == 0]
    if len(negative_scores) == 0:
        return 0.0
    threshold = np.quantile(negative_scores, 1 - target_fpr, method="higher")
    positives = labels == 1
    return float(np.mean(probabilities[positives] >= threshold)) if np.any(positives) else 0.0


def save_signed_artifact(payload: dict, output: Path) -> None:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def synthetic_records() -> list[CalibrationRecord]:
    records: list[CalibrationRecord] = []
    human = (
        "I checked the issue on my machine and found the failing condition after a few attempts. "
        "The notes are uneven because I wrote them while testing, but they describe the actual fix."
    )
    ai = (
        "Furthermore, an effective troubleshooting process should systematically isolate variables, "
        "document observations, and validate the resolution. Overall, this approach improves "
        "reliability and supports maintainable software development."
    )
    for index in range(120):
        records.append(
            CalibrationRecord(
                text=human + f" Run {index} included a local detail and a short correction.",
                label=0,
                group=f"human-{index // 20}",
                domain="notes",
                model_family="human",
                length_bucket="50-149",
            )
        )
        records.append(
            CalibrationRecord(
                text=ai + f" Additionally, iteration {index} reinforces consistent verification.",
                label=1,
                group=f"ai-{index // 20}",
                domain="notes",
                model_family="generic",
                length_bucket="50-149",
            )
        )
    return records


def main() -> None:
    records = synthetic_records()
    features = np.array([extract_features(record.text) for record in records])
    labels = np.array([record.label for record in records])
    groups = np.array([record.group for record in records])

    raw_scores = 1 / (1 + np.exp(-(features[:, 0] * -2 + features[:, 1] * 3 + features[:, 4] * 8)))
    splitter = GroupKFold(n_splits=5)
    calibrated = np.zeros_like(raw_scores, dtype=float)
    for train, test in splitter.split(features, labels, groups=groups):
        calibrator = fit_binary_calibrator(raw_scores[train], labels[train])
        calibrated[test] = calibrator.predict(raw_scores[test])

    metrics = evaluate_binary(labels, calibrated)
    family_labels = np.array([record.model_family for record in records])
    geometry = fit_source_geometry(features, family_labels)
    save_signed_artifact(
        {
            "schema": "panoptes-calibration-v1",
            "cohort": "synthetic-development-cohort",
            "metrics": metrics,
            "binary_calibrator": {
                "x_thresholds": fit_binary_calibrator(raw_scores, labels).X_thresholds_.tolist(),
                "y_thresholds": fit_binary_calibrator(raw_scores, labels).y_thresholds_.tolist(),
            },
            "source_geometry": geometry,
        },
        Path(__file__).resolve().parents[1] / "backend" / "artifacts" / "baseline-calibration.json",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
