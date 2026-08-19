"""Run the frozen measurement protocol and write a signed card.

  python -m bench measure --data corpus
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from bench import validity
from bench.datasets import Dataset
from bench.detectors import HeuristicDetector, make_detector
from bench.evaluate import (
    _valid_operating_block,
    binary_metrics,
    cross_dataset_transport,
    evaluate_protocol,
    fit_isotonic,
    leave_one_family_out,
    sliced_conformal_coverage,
    transport_matrix,
)
from bench.evidence import compare_dependence
from bench.features import heuristic_raw_score
from bench.mixtures import mixture_curve
from bench.power import calibration_power
from bench.splits import protocol_splits
from research.protocol import load_protocol


class CalibrationMismatchError(RuntimeError):
    """Raised when a calibration bundle is applied to the wrong detector/revision/task/cohort."""


@dataclass
class CalibrationBundle:
    """A frozen calibration + operating-threshold bundle bound to one detector.

    The bundle ties the isotonic calibrator, the split-conformal level, the
    low-FPR operating thresholds, and the selective-prediction thresholds to a
    specific (detector_id, model_revision, task, cohort). ``calibrate`` refuses
    to run unless the caller asserts the matching identity, so a heuristic-fit
    map can never be silently applied to neural logits (or any other detector).
    """

    detector_id: str
    model_revision: str
    task: str
    cohort: str
    prevalence: float
    conformal: validity.ConformalFit
    fpr_thresholds: dict[float, float]
    selective_thresholds: dict[float, float]
    iso_x: list[float] | None = None  # isotonic input thresholds (None = identity)
    iso_y: list[float] | None = None  # isotonic output values
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    def identity(self) -> dict[str, str]:
        return {
            "detector_id": self.detector_id,
            "model_revision": self.model_revision,
            "task": self.task,
            "cohort": self.cohort,
        }

    def bundle_sha256(self) -> str:
        payload = {
            **self.identity(),
            "prevalence": self.prevalence,
            "conformal": {
                "alpha": self.conformal.alpha,
                "threshold_by_class": self.conformal.threshold_by_class,
                "mondrian": self.conformal.mondrian,
            },
            "fpr_thresholds": {str(k): v for k, v in self.fpr_thresholds.items()},
            "selective_thresholds": {str(k): v for k, v in self.selective_thresholds.items()},
            "iso_x": self.iso_x,
            "iso_y": self.iso_y,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _assert_identity(self, detector_id: str, model_revision: str, task: str, cohort: str) -> None:
        expected = self.identity()
        given = {
            "detector_id": detector_id,
            "model_revision": model_revision,
            "task": task,
            "cohort": cohort,
        }
        if given != expected:
            raise CalibrationMismatchError(
                f"calibration bundle is bound to {expected}, not {given}; "
                "refusing to apply a calibration map across detectors/revisions/tasks/cohorts"
            )

    def calibrate(
        self, raw_scores: np.ndarray, *, detector_id: str, model_revision: str, task: str, cohort: str
    ) -> np.ndarray:
        """Apply the frozen isotonic map, enforcing the detector/revision/task/cohort."""
        self._assert_identity(detector_id, model_revision, task, cohort)
        raw = np.asarray(raw_scores, dtype=float)
        if self.iso_x is None or self.iso_y is None:
            return np.clip(raw, 1e-6, 1 - 1e-6)
        return np.clip(np.interp(raw, self.iso_x, self.iso_y), 1e-6, 1 - 1e-6)

    def to_dict(self) -> dict:
        return {
            **self.identity(),
            "prevalence": self.prevalence,
            "conformal": {
                "alpha": self.conformal.alpha,
                "threshold_by_class": {str(k): v for k, v in self.conformal.threshold_by_class.items()},
                "n_calibration": self.conformal.n_calibration,
                "mondrian": self.conformal.mondrian,
            },
            "fpr_thresholds": {str(k): v for k, v in self.fpr_thresholds.items()},
            "selective_thresholds": {str(k): v for k, v in self.selective_thresholds.items()},
            "iso_x": self.iso_x,
            "iso_y": self.iso_y,
            "created_utc": self.created_utc,
            "bundle_sha256": self.bundle_sha256(),
        }


def fit_calibration_bundle(
    detector,
    cal_dataset: Dataset,
    *,
    detector_id: str,
    model_revision: str,
    task: str,
    cohort: str,
    alpha: float = 0.1,
) -> CalibrationBundle:
    """Fit the calibrator and every operating threshold on the calibration cohort only."""
    raw_cal = detector.predict_proba(cal_dataset, np.arange(len(cal_dataset)))
    labels = cal_dataset.labels
    calibrator = fit_isotonic(raw_cal, labels)
    if calibrator is None:
        calibrated_cal = np.clip(raw_cal, 1e-6, 1 - 1e-6)
        iso_x = iso_y = None
    else:
        calibrated_cal = np.clip(calibrator.predict(raw_cal), 1e-6, 1 - 1e-6)
        iso_x = [float(v) for v in calibrator.X_thresholds_]
        iso_y = [float(v) for v in calibrator.y_thresholds_]
    return CalibrationBundle(
        detector_id=detector_id,
        model_revision=model_revision,
        task=task,
        cohort=cohort,
        prevalence=float(labels.mean()),
        conformal=validity.fit_conformal(labels, calibrated_cal, alpha=alpha),
        fpr_thresholds=validity.fit_fpr_thresholds(labels, calibrated_cal),
        selective_thresholds=validity.fit_selective_thresholds(calibrated_cal),
        iso_x=iso_x,
        iso_y=iso_y,
    )


def fit_select_calibrate_test(
    detector,
    train: Dataset,
    dev: Dataset,
    cal: Dataset,
    test: Dataset,
    *,
    detector_id: str,
    model_revision: str,
    task: str = "binary_ai",
    cohort: str = "unknown",
    alpha: float = 0.1,
) -> dict:
    """The single v2.1 measurement interface: fit(train) → select(dev) → calibrate(cal) → test(test).

    Every detector tier — heuristic, logistic, GBM, and the neural detector —
    runs through this one path so the data firewall is enforced uniformly:

      * ``fit`` sees only the train partition.
      * ``select`` (early stopping / hyperparameter or aggregation choice) sees
        only the development partition, via a ``select`` hook the detector may
        implement; tiers without selection simply no-op.
      * the calibrator, conformal level, and all operating thresholds are fit
        only on the calibration partition (a :class:`CalibrationBundle`).
      * the test partition is scored once, frozen, and never used for fitting,
        selection, calibration, or thresholding.
    """
    detector.fit(train, np.arange(len(train)))
    if hasattr(detector, "select"):
        detector.select(dev, np.arange(len(dev)))
    bundle = fit_calibration_bundle(
        detector, cal, detector_id=detector_id, model_revision=model_revision, task=task, cohort=cohort, alpha=alpha
    )
    raw_test = detector.predict_proba(test, np.arange(len(test)))
    calibrated = bundle.calibrate(
        raw_test, detector_id=detector_id, model_revision=model_revision, task=task, cohort=cohort
    )
    labels = test.labels
    valid = _valid_operating_block(cal.labels, bundle.calibrate(
        detector.predict_proba(cal, np.arange(len(cal))),
        detector_id=detector_id, model_revision=model_revision, task=task, cohort=cohort,
    ), labels, calibrated, test.groups, alpha=alpha)
    return {
        "detector_id": detector_id,
        "model_revision": model_revision,
        "task": task,
        "cohort": cohort,
        "n_train": int(len(train)),
        "n_development": int(len(dev)),
        "n_calibration": int(len(cal)),
        "n_test": int(len(test)),
        "bundle": bundle.to_dict(),
        "metrics": binary_metrics(labels, calibrated),
        "conformal": valid["conformal"],
        "operating_points": valid["operating_points"],
        "operating_points_fit_on": "calibration",
        "selective_risk": valid["selective_risk"],
        "auroc_group_bootstrap": valid["auroc_group_bootstrap"],
        "adaptive_ece": valid["adaptive_ece"],
        "prevalence_views": valid["prevalence"],
        "probabilities": calibrated,
        "labels": labels,
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _dependence_pilot(dataset: Dataset, n_segments: int = 3, limit: int = 24) -> dict:
    """Compare naive, correlated, and document-level evidence on text documents."""
    rows = []
    prevalence = float(dataset.labels.mean())
    count = 0
    for text, kind, label in zip(dataset.texts, dataset.kinds, dataset.labels, strict=True):
        if kind != "text":
            continue
        words = text.split()
        if len(words) < n_segments * 12:
            continue
        bounds = [len(words) * i // n_segments for i in range(n_segments + 1)]
        segment_p = np.array(
            [
                heuristic_raw_score(" ".join(words[bounds[i] : bounds[i + 1]]), "text")
                for i in range(n_segments)
            ]
        )
        document_p = heuristic_raw_score(text, "text")
        compared = compare_dependence(segment_p, document_p, prevalence)
        rows.append(
            {
                "label": int(label),
                "document_p": float(document_p),
                "naive_llr": compared["naive_sum"]["llr"],
                "correlated_llr": compared["correlated_shrinkage"]["llr"],
                "document_llr": compared["document_level"]["llr"],
                "rho": compared["correlated_shrinkage"]["rho"],
            }
        )
        count += 1
        if count >= limit:
            break
    if not rows:
        return {"n_documents": 0}
    naive = np.array([row["naive_llr"] for row in rows])
    correlated = np.array([row["correlated_llr"] for row in rows])
    document = np.array([row["document_llr"] for row in rows])
    return {
        "n_documents": len(rows),
        "mean_naive_llr": float(naive.mean()),
        "mean_correlated_llr": float(correlated.mean()),
        "mean_document_llr": float(document.mean()),
        "mean_rho": float(np.mean([row["rho"] for row in rows])),
        "naive_vs_document_mae": float(np.mean(np.abs(naive - document))),
        "note": (
            "Naive accumulation inflates evidence when segments are correlated. "
            "A non-rejected Durbin-Watson test is not treated as proof of independence."
        ),
    }


def run_measurement(
    dataset: Dataset,
    detectors: tuple[str, ...] = ("heuristic", "logistic"),
    cross_datasets: dict[str, Dataset] | None = None,
) -> dict:
    protocol = load_protocol()
    splits = protocol_splits(dataset)
    detector_metrics = {}
    power_blocks = {}
    for name in detectors:
        result = evaluate_protocol(lambda n=name: make_detector(n), dataset)
        pooled_idx = result["pooled_test_idx"]
        pooled_labels = result["pooled_labels"]
        pooled_p = result["pooled_probabilities"]
        coverage_slices = sliced_conformal_coverage(
            pooled_labels,
            pooled_p,
            {
                "length_bucket": [dataset.buckets[int(i)] for i in pooled_idx],
                "family": [dataset.families[int(i)] for i in pooled_idx],
                "class": ["ai" if int(v) == 1 else "human" for v in pooled_labels],
            },
        )
        detector_metrics[name] = {
            "method": result["method"],
            "n_splits": result["n_splits"],
            "n_groups": result["n_groups"],
            "metrics": result["metrics"],
            "selective_risk": result["selective_risk"],
            "conformal": result["conformal"],
            "conformal_coverage_slices": coverage_slices,
            "prior_sensitivity": result["prior_sensitivity"],
            "mean_likelihood_ratio": result["mean_likelihood_ratio"],
        }
        power_blocks[name] = calibration_power(pooled_labels, pooled_p)
    cross_transport = {}
    for label, other in (cross_datasets or {}).items():
        cross_transport[f"{label}_to_here"] = cross_dataset_transport(
            other, dataset, lambda: make_detector("logistic")
        )
        cross_transport[f"here_to_{label}"] = cross_dataset_transport(
            dataset, other, lambda: make_detector("logistic")
        )
    return _jsonable(
        {
            "schema": "panoptes-measurement-card-v1",
            "protocol_id": protocol["title"],
            "protocol_registered_utc": protocol["registered_utc"],
            "dataset": dataset.provenance,
            "dataset_sha256": dataset.sha256,
            "n": len(dataset),
            "split": {
                "method": splits[0].method.split(":")[0],
                "n_groups": splits[0].n_groups,
                "n_splits": len(splits),
                "rule": protocol["split_rules"]["hard_rule"],
            },
            "detectors": list(detectors),
            "metrics": detector_metrics,
            "transport": transport_matrix(HeuristicDetector, dataset, axis="domains"),
            "cross_dataset_transport": cross_transport,
            "power": power_blocks,
            "mixtures": mixture_curve(dataset, HeuristicDetector()),
            "open_set": leave_one_family_out(dataset),
            "dependence": _dependence_pilot(dataset),
            "limitations": [
                "Train, calibration, and test groups are disjoint; calibration is never fit on test.",
                "Human controls in the project corpus are few; mixture and calibration estimates are a pilot.",
                "Transport cells are omitted when a domain lacks both classes.",
                "Watermark evaluation is a separate subsystem and is not included in this card.",
                "Conformal coverage is guaranteed marginally; per-slice coverage is reported diagnostically.",
                "Calibration-metric power uses a normal approximation; ECE variance is a fixed-bin proxy.",
            ],
        }
    )
