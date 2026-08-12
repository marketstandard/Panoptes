from __future__ import annotations

import hashlib
import json
import math
import sys
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def synthetic_main(output: Path) -> None:
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
            "bundle_id": "prose-en-baseline-v0",
            "cohort": "synthetic-development-cohort",
            "dataset_manifest_id": "synthetic-development-v1",
            "detector_id": "heuristic-prose-detector",
            "detector_version": "0.1.0",
            "feature_schema": "panoptes-features-v1",
            "created_utc": "2026-08-11T00:00:00Z",
            "repro": {
                "command": "python research/calibration.py --synthetic",
                "seed": 13,
                "code_commit": "workspace",
                "group_cv": "GroupKFold(n_splits=5)",
            },
            "metrics": metrics,
            "binary_calibrator": {
                "x_thresholds": fit_binary_calibrator(raw_scores, labels).X_thresholds_.tolist(),
                "y_thresholds": fit_binary_calibrator(raw_scores, labels).y_thresholds_.tolist(),
            },
            "source_geometry": geometry,
            "conformal": {"alpha": 0.1, "threshold": 0.0},
        },
        output,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def corpus_main(output: Path, min_family_n: int = 8) -> None:
    """Fit the production calibration artifact from the hash-verified corpus.

    Binary calibration: isotonic regression on the shipped heuristic prose
    detector's raw score, evaluated with GroupKFold by prompt so no prompt
    (and therefore no topic) appears in both train and test folds.
    Source geometry: corpus-fitted on the seven runtime attribution
    features; families below ``min_family_n`` records are excluded and
    the exclusion is recorded.
    """
    from bench.features import ATTRIBUTION_FEATURES, heuristic_raw_score
    from research.baseline_corpus import load_corpus, run_manifests

    records = load_corpus()
    manifests = run_manifests()
    created = max(manifest["created_utc"] for manifest in manifests)
    prompts_sha256 = manifests[0]["prompts"]["sha256"]

    text_records = [record for record in records if record.kind == "text"]
    labels = np.array([record.label for record in text_records])
    groups = np.array([record.prompt_id for record in text_records])
    raw = np.array([heuristic_raw_score(record.text, "text") for record in text_records])

    n_splits = min(5, len(set(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    calibrated = np.zeros_like(raw, dtype=float)
    for train, test in splitter.split(raw.reshape(-1, 1), labels, groups=groups):
        calibrator = fit_binary_calibrator(raw[train], labels[train])
        calibrated[test] = calibrator.predict(raw[test])
    metrics = evaluate_binary(labels, calibrated)
    bins = _reliability_bins(labels, calibrated)
    nonconformity = np.abs(labels - calibrated)
    threshold = conformal_threshold(nonconformity, alpha=0.1)

    ai_records = [record for record in records if record.label == 1]
    family_counts: dict[str, int] = {}
    for record in ai_records:
        family_counts[record.family] = family_counts.get(record.family, 0) + 1
    dropped = sorted(f for f, n in family_counts.items() if n < min_family_n)
    kept = [record for record in ai_records if family_counts[record.family] >= min_family_n]
    geometry_features = np.array(
        [
            [extract_attribution(record.text, record.kind)[name] for name in ATTRIBUTION_FEATURES]
            for record in kept
        ]
    )
    geometry = fit_source_geometry(
        geometry_features, np.array([record.family for record in kept])
    )

    full_calibrator = fit_binary_calibrator(raw, labels)
    save_signed_artifact(
        {
            "schema": "panoptes-calibration-v1",
            "bundle_id": "panoptes-reference-corpus-v1",
            "cohort": "panoptes-reference-corpus (6 model families + human controls)",
            "dataset_manifest_id": f"prompts@{prompts_sha256[:16]}",
            "detector_id": "heuristic-prose-detector",
            "detector_version": "0.1.0",
            "feature_schema": "panoptes-attribution-features-v1",
            "feature_names": list(ATTRIBUTION_FEATURES),
            "created_utc": created,
            "corpus": {
                "n_records": len(records),
                "n_text_records": len(text_records),
                "n_human": sum(1 for r in records if r.label == 0),
                "n_ai": sum(1 for r in records if r.label == 1),
                "families": sorted(family_counts),
                "geometry_dropped_families": dropped,
                "geometry_min_family_n": min_family_n,
            },
            "repro": {
                "command": "python research/calibration.py",
                "seed": 13,
                "code_commit": "workspace",
                "group_cv": f"GroupKFold(n_splits={n_splits}, groups=prompt_id)",
            },
            "metrics": metrics,
            "reliability_bins": bins,
            "binary_calibrator": {
                "x_thresholds": full_calibrator.X_thresholds_.tolist(),
                "y_thresholds": full_calibrator.y_thresholds_.tolist(),
            },
            "source_geometry": geometry,
            "conformal": {"alpha": 0.1, "threshold": threshold},
        },
        output,
    )
    print(json.dumps({"metrics": metrics, "conformal_threshold": threshold,
                      "dropped_families": dropped}, indent=2, sort_keys=True))


def defactify_main(output: Path, min_family_n: int = 8) -> None:
    """Fit a second calibration artifact on the Defactify_Text_Dataset.

    Same recipe as the corpus artifact — isotonic calibration of the shipped
    heuristic raw score under story-grouped GroupKFold, reliability bins,
    split-conformal threshold, and per-family Mahalanobis geometry — but
    fitted on 71,666 hygiene-filtered NYT-domain rows across 7 families.
    Selected at runtime with PANOPTES_CALIBRATION_BUNDLE=defactify-calibration.json.
    """
    from bench.datasets import defactify_created_utc, load_defactify
    from bench.features import ATTRIBUTION_FEATURES, heuristic_raw_score

    dataset = load_defactify()
    created = defactify_created_utc() or "2026-08-12T00:00:00Z"

    labels = dataset.labels
    groups = np.array(dataset.groups)
    raw = np.array([heuristic_raw_score(text, "text") for text in dataset.texts])

    n_splits = 5
    splitter = GroupKFold(n_splits=n_splits)
    calibrated = np.zeros_like(raw, dtype=float)
    for train, test in splitter.split(raw.reshape(-1, 1), labels, groups=groups):
        calibrator = fit_binary_calibrator(raw[train], labels[train])
        calibrated[test] = calibrator.predict(raw[test])
    metrics = evaluate_binary(labels, calibrated)
    bins = _reliability_bins(labels, calibrated)
    nonconformity = np.abs(labels - calibrated)
    threshold = conformal_threshold(nonconformity, alpha=0.1)

    family_counts: dict[str, int] = {}
    for family in dataset.families:
        family_counts[family] = family_counts.get(family, 0) + 1
    dropped = sorted(f for f, n in family_counts.items() if n < min_family_n)
    kept_idx = [i for i, f in enumerate(dataset.families) if family_counts[f] >= min_family_n]
    geometry_features = np.array(
        [
            [extract_attribution(dataset.texts[i], "text")[name] for name in ATTRIBUTION_FEATURES]
            for i in kept_idx
        ]
    )
    geometry = fit_source_geometry(
        geometry_features, np.array([dataset.families[i] for i in kept_idx])
    )

    full_calibrator = fit_binary_calibrator(raw, labels)
    save_signed_artifact(
        {
            "schema": "panoptes-calibration-v1",
            "bundle_id": "defactify-text-v1",
            "cohort": "defactify-text (Roy et al. 2026; NYT human + 6 LLM families; hygiene-filtered)",
            "dataset_manifest_id": "defactify-text",
            "detector_id": "heuristic-prose-detector",
            "detector_version": "0.1.0",
            "feature_schema": "panoptes-attribution-features-v1",
            "feature_names": list(ATTRIBUTION_FEATURES),
            "created_utc": created,
            "corpus": {
                "n_records": len(dataset),
                "n_text_records": len(dataset),
                "n_human": int((labels == 0).sum()),
                "n_ai": int((labels == 1).sum()),
                "families": sorted(family_counts),
                "geometry_dropped_families": dropped,
                "geometry_min_family_n": min_family_n,
            },
            "repro": {
                "command": "python research/calibration.py --dataset defactify",
                "seed": 13,
                "code_commit": "workspace",
                "group_cv": f"GroupKFold(n_splits={n_splits}, groups=reconstructed story id)",
            },
            "metrics": metrics,
            "reliability_bins": bins,
            "binary_calibrator": {
                "x_thresholds": full_calibrator.X_thresholds_.tolist(),
                "y_thresholds": full_calibrator.y_thresholds_.tolist(),
            },
            "source_geometry": geometry,
            "conformal": {"alpha": 0.1, "threshold": threshold},
        },
        output,
    )
    print(json.dumps({"metrics": metrics, "conformal_threshold": threshold,
                      "dropped_families": dropped}, indent=2, sort_keys=True))


def extract_attribution(text: str, kind: str) -> dict[str, float]:
    from bench.features import extract

    return extract(text, kind)


def _reliability_bins(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict]:
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_lo": float(edges[index]),
                "bin_hi": float(edges[index + 1]),
                "n": int(mask.sum()),
                "mean_predicted": float(probabilities[mask].mean()),
                "observed": float(labels[mask].mean()),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fit Panoptes calibration artifacts.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use the synthetic development cohort instead of the verified corpus",
    )
    parser.add_argument(
        "--dataset",
        choices=["corpus", "defactify"],
        default="corpus",
        help="corpus (default) fits the verified project corpus; defactify fits the local Defactify parquet",
    )
    parser.add_argument("--out", type=Path, default=None, help="override output path")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    if args.synthetic:
        output = args.out or root / "backend" / "artifacts" / "baseline-calibration.synthetic.json"
        synthetic_main(output)
    elif args.dataset == "defactify":
        output = args.out or root / "backend" / "artifacts" / "defactify-calibration.json"
        defactify_main(output)
    else:
        output = args.out or root / "backend" / "artifacts" / "baseline-calibration.json"
        corpus_main(output)


if __name__ == "__main__":
    main()
