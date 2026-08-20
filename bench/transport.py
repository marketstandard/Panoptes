"""Phase 6: evidence-transportability experiments.

Three analyses that are never conflated (protocol v2.1):

  * **Representation transport** — train on source cohort(s), calibrate on the
    source calibration groups, then evaluate untouched target cohorts. The
    leave-one-cohort-out (LOCO) form holds an entire cohort out of both
    training and calibration, giving a genuinely unseen population.
  * **Calibration transfer** — freeze one scorer, fit *only* the calibrator on
    each source cohort's calibration partition, then apply it to every target
    cohort. This isolates evidence portability (calibration shift) from
    representation learning.
  * **Pooled production generalization** — train on all allowed training
    partitions and evaluate each seen-cohort holdout. These cells are labeled
    ``seen-cohort``; a cohort represented in pooled training is never called
    external or unseen.

Every cell reports sample/group counts, prevalence, AUROC/AUPRC, Brier,
log-loss, calibration slope/intercept, adaptive ECE, calibration-fixed low-FPR
operating points, split-conformal coverage/set size, selective risk, and a
group-bootstrap 95% interval, plus the delta from the relevant diagonal. All
operating quantities (calibrator, conformal level, FPR/selective thresholds)
are fit on calibration partitions only and applied frozen to test cells; no
final-test label ever sets a threshold.

The module is detector-agnostic: any scorer implementing the
``fit(dataset, idx)`` / ``predict_proba(dataset, idx)`` protocol (heuristic,
logistic, GBM, or the frozen neural ensemble via an adapter) runs through the
same path so the data firewall is enforced uniformly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from bench import validity
from bench.datasets import Dataset
from bench.evaluate import _valid_operating_block, binary_metrics, fit_isotonic
from bench.splits import holdout_split

SEED = 13
COHORT_AXES = ("domain", "generator", "domain_generator", "dataset")


# --- cohort structure ----------------------------------------------------------


def cohort_keys(dataset: Dataset, axis: str) -> np.ndarray:
    """Per-row cohort label for a declared axis.

    ``domain`` and ``generator`` use the dataset's domain/family columns;
    ``domain_generator`` crosses them. ``dataset`` uses the per-row cohort label
    populated by :func:`bench.datasets.combine_datasets` for pooled multi-cohort
    pools, enabling leave-one-dataset-out transport. Dataset IDs are
    sampling/audit metadata and are never model inputs — they only define
    evaluation cohorts.
    """
    if axis not in COHORT_AXES:
        raise ValueError(f"unknown cohort axis {axis!r}; expected one of {COHORT_AXES}")
    if axis == "dataset":
        if dataset.datasets is None:
            raise ValueError(
                "axis='dataset' requires a pooled Dataset with per-row `datasets` labels"
            )
        return np.array([str(d) for d in dataset.datasets])
    if axis == "domain":
        return np.array([str(d) for d in dataset.domains])
    if axis == "generator":
        return np.array([str(f) for f in dataset.families])
    return np.array([f"{d}|{f}" for d, f in zip(dataset.domains, dataset.families, strict=True)])


def cohort_index(dataset: Dataset, axis: str) -> dict[str, np.ndarray]:
    """Map cohort label -> sorted row indices (deterministic order)."""
    keys = cohort_keys(dataset, axis)
    out: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        out.setdefault(str(key), []).append(i)
    return {k: np.array(sorted(v), dtype=int) for k, v in sorted(out.items())}


def _both_classes(labels: np.ndarray) -> bool:
    return len(set(int(v) for v in labels.tolist())) >= 2


# --- per-cell metrics ------------------------------------------------------------


def transport_cell_metrics(
    cal_labels: np.ndarray,
    cal_probs: np.ndarray,
    test_labels: np.ndarray,
    test_probs: np.ndarray,
    test_groups,
    *,
    alpha: float = 0.1,
    n_boot: int = 1000,
    seed: int = SEED,
) -> dict:
    """The full per-cell metric block; every operating quantity is calibration-fit.

    ``cal_*`` is the source calibration partition (fits the conformal level and
    the low-FPR / selective thresholds); ``test_*`` is the untouched target cell.
    Returns counts, prevalence, discrimination, calibration, conformal coverage,
    selective risk, and a group-bootstrap interval on AUROC.
    """
    cal_labels = np.asarray(cal_labels, dtype=int)
    test_labels = np.asarray(test_labels, dtype=int)
    cal_probs = np.asarray(cal_probs, dtype=float)
    test_probs = np.asarray(test_probs, dtype=float)
    groups = np.asarray(list(test_groups))

    out: dict[str, Any] = {
        "n_test": int(len(test_labels)),
        "n_groups": int(len(set(groups.tolist()))),
        "n_calibration": int(len(cal_labels)),
        "target_prevalence": float(test_labels.mean()) if len(test_labels) else float("nan"),
        "source_calibration_prevalence": float(cal_labels.mean())
        if len(cal_labels)
        else float("nan"),
    }
    if not len(test_labels) or not _both_classes(test_labels):
        out.update(
            {
                "degenerate": True,
                "note": "target cell is empty or single-class; discrimination metrics undefined",
            }
        )
        return out

    out.update(binary_metrics(test_labels, test_probs))
    valid = _valid_operating_block(
        cal_labels, cal_probs, test_labels, test_probs, groups, alpha=alpha
    )
    out["conformal"] = valid["conformal"]
    out["operating_points"] = valid["operating_points"]
    out["operating_points_fit_on"] = "calibration"
    out["selective_risk"] = valid["selective_risk"]
    out["auroc_group_bootstrap"] = valid["auroc_group_bootstrap"]
    out["adaptive_ece"] = valid["adaptive_ece"]
    out["prevalence_views"] = valid["prevalence"]
    out["shift_caveat"] = validity.SHIFT_CAVEAT
    return out


# --- frozen scorer + calibrator bundle -------------------------------------------


@dataclass
class FrozenScorer:
    """A fitted detector plus its source-calibration isotonic map."""

    detector: Any
    detector_id: str
    iso_x: list[float] | None
    iso_y: list[float] | None
    source_calibration_prevalence: float

    def raw(self, dataset: Dataset, idx: np.ndarray) -> np.ndarray:
        return np.asarray(self.detector.predict_proba(dataset, idx), dtype=float)

    def calibrated(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        if self.iso_x is None or self.iso_y is None:
            return np.clip(raw, 1e-6, 1 - 1e-6)
        return np.clip(np.interp(raw, self.iso_x, self.iso_y), 1e-6, 1 - 1e-6)


def fit_source_scorer(
    detector_factory: Callable[[], Any],
    dataset: Dataset,
    train_idx: np.ndarray,
    cal_idx: np.ndarray,
    *,
    detector_id: str,
) -> FrozenScorer:
    """Fit the detector on ``train_idx`` and the isotonic map on ``cal_idx`` only."""
    detector = detector_factory()
    detector.fit(dataset, np.asarray(train_idx, dtype=int))
    raw_cal = np.asarray(
        detector.predict_proba(dataset, np.asarray(cal_idx, dtype=int)), dtype=float
    )
    cal_labels = dataset.labels[np.asarray(cal_idx, dtype=int)]
    calibrator = fit_isotonic(raw_cal, cal_labels)
    if calibrator is None:
        iso_x = iso_y = None
    else:
        iso_x = [float(v) for v in calibrator.X_thresholds_]
        iso_y = [float(v) for v in calibrator.y_thresholds_]
    return FrozenScorer(
        detector=detector,
        detector_id=detector_id,
        iso_x=iso_x,
        iso_y=iso_y,
        source_calibration_prevalence=float(cal_labels.mean()) if len(cal_labels) else float("nan"),
    )


# --- analysis 1: leave-one-cohort-out representation transport --------------------


def leave_one_cohort_out(
    dataset: Dataset,
    axis: str,
    detector_factory: Callable[[], Any],
    *,
    min_cohort: int = 40,
    alpha: float = 0.1,
    n_boot: int = 1000,
    seed: int = SEED,
) -> dict:
    """LOCO representation transport: hold one cohort out of train AND calibration.

    For each target cohort the source is every other cohort; the source is split
    group-disjoint into train/calibration/diagonal-test. The detector is fit on
    source-train, calibrated on source-calibration, and applied frozen to the
    untouched target cohort. The source diagonal (source-test, seen) is the
    delta reference. Cohorts too small to form a clean target are skipped.
    """
    cells: dict[str, dict] = {}
    index = cohort_index(dataset, axis)
    for cohort, target_idx in index.items():
        if len(target_idx) < min_cohort or not _both_classes(dataset.labels[target_idx]):
            cells[cohort] = {
                "skipped": True,
                "reason": "too small or single-class",
                "n": int(len(target_idx)),
            }
            continue
        source_mask = np.ones(len(dataset), dtype=bool)
        source_mask[target_idx] = False
        source_idx = np.where(source_mask)[0]
        source = dataset.subset(source_idx)
        try:
            split = holdout_split(source, fractions=(0.7, 0.2, 0.1), seed=seed)
        except Exception as exc:
            cells[cohort] = {
                "skipped": True,
                "reason": f"source split failed: {exc}",
                "n": int(len(target_idx)),
            }
            continue
        scorer = fit_source_scorer(
            detector_factory,
            source,
            split.train,
            split.calibration,
            detector_id=f"loco-exclude-{cohort}",
        )
        # Source diagonal (seen) reference.
        diag_raw = scorer.raw(source, split.test)
        diag_probs = scorer.calibrated(diag_raw)
        diag_cal_probs = scorer.calibrated(scorer.raw(source, split.calibration))
        diagonal = transport_cell_metrics(
            source.labels[split.calibration],
            diag_cal_probs,
            source.labels[split.test],
            diag_probs,
            [source.groups[int(i)] for i in split.test],
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
        )
        # Untouched target cohort.
        tgt_raw = scorer.raw(dataset, target_idx)
        tgt_probs = scorer.calibrated(tgt_raw)
        target = transport_cell_metrics(
            source.labels[split.calibration],
            diag_cal_probs,
            dataset.labels[target_idx],
            tgt_probs,
            [dataset.groups[int(i)] for i in target_idx],
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
        )
        delta = _delta(diagonal, target)
        cells[cohort] = {
            "kind": "representation_transport_loco",
            "cohort": cohort,
            "axis": axis,
            "n_target": int(len(target_idx)),
            "n_source_train": int(len(split.train)),
            "n_source_calibration": int(len(split.calibration)),
            "calibration_fit_on": "source_calibration",
            "diagonal": diagonal,
            "target": target,
            "delta_from_diagonal": delta,
        }
    return {
        "kind": "representation_transport_loco",
        "axis": axis,
        "dataset": dataset.provenance,
        "dataset_sha256": dataset.sha256,
        "n_cohorts": len(index),
        "cells": cells,
        "summary": _transport_summary(cells),
    }


# --- analysis 2: calibration transfer ----------------------------------------------


def calibration_transfer(
    dataset: Dataset,
    axis: str,
    detector_factory: Callable[[], Any],
    *,
    min_cell: int = 30,
    alpha: float = 0.1,
    n_boot: int = 1000,
    seed: int = SEED,
) -> dict:
    """Freeze one scorer; fit only the calibrator per source cohort; apply to all targets.

    A single group-disjoint train/calibration/test split is taken. The scorer is
    fit on the global train partition and frozen. For each source cohort S an
    isotonic calibrator is fit on ``calibration ∩ S``; each source calibrator is
    then applied to ``test ∩ T`` for every target cohort T, producing an S x T
    matrix. The diagonal (S == T) is in-cohort calibration; off-diagonal cells
    isolate how calibration alone transports.
    """
    split = holdout_split(dataset, fractions=(0.6, 0.2, 0.2), seed=seed)
    scorer = fit_source_scorer(
        detector_factory,
        dataset,
        split.train,
        split.calibration,
        detector_id="calibration-transfer-frozen",
    )
    keys = cohort_keys(dataset, axis)
    cal_idx = split.calibration
    test_idx = split.test
    cal_cohorts = np.array([str(keys[int(i)]) for i in cal_idx])
    test_cohorts = np.array([str(keys[int(i)]) for i in test_idx])

    # Frozen raw scores on every cal/test row, computed once.
    raw_cal = scorer.raw(dataset, cal_idx)
    raw_test = scorer.raw(dataset, test_idx)

    cohorts = sorted(set(cal_cohorts.tolist()) | set(test_cohorts.tolist()))
    matrix: dict[str, dict[str, dict]] = {}
    for src in cohorts:
        s_mask = cal_cohorts == src
        if int(s_mask.sum()) < min_cell or not _both_classes(dataset.labels[cal_idx[s_mask]]):
            continue
        calibrator = fit_isotonic(raw_cal[s_mask], dataset.labels[cal_idx[s_mask]])
        row: dict[str, dict] = {}
        for tgt in cohorts:
            t_mask = test_cohorts == tgt
            if int(t_mask.sum()) < min_cell:
                continue
            t_labels = dataset.labels[test_idx[t_mask]]
            if not _both_classes(t_labels):
                continue
            if calibrator is None:
                t_probs = np.clip(raw_test[t_mask], 1e-6, 1 - 1e-6)
                s_cal_probs = np.clip(raw_cal[s_mask], 1e-6, 1 - 1e-6)
            else:
                t_probs = np.clip(calibrator.predict(raw_test[t_mask]), 1e-6, 1 - 1e-6)
                s_cal_probs = np.clip(calibrator.predict(raw_cal[s_mask]), 1e-6, 1 - 1e-6)
            t_groups = [dataset.groups[int(i)] for i in test_idx[t_mask]]
            cell = transport_cell_metrics(
                dataset.labels[cal_idx[s_mask]],
                s_cal_probs,
                t_labels,
                t_probs,
                t_groups,
                alpha=alpha,
                n_boot=n_boot,
                seed=seed,
            )
            cell["source_cohort"] = src
            cell["target_cohort"] = tgt
            cell["on_diagonal"] = bool(src == tgt)
            row[tgt] = cell
        if row:
            matrix[src] = row

    # Deltas: each off-diagonal cell vs the target's own diagonal (in-cohort) cell.
    deltas: dict[str, dict[str, dict]] = {}
    for src, row in matrix.items():
        deltas[src] = {}
        for tgt, cell in row.items():
            diag = matrix.get(tgt, {}).get(tgt)
            deltas[src][tgt] = _delta(diag, cell) if diag is not None else {}
    return {
        "kind": "calibration_transfer",
        "axis": axis,
        "dataset": dataset.provenance,
        "dataset_sha256": dataset.sha256,
        "detector": scorer.detector_id,
        "n_train": int(len(split.train)),
        "n_calibration": int(len(split.calibration)),
        "n_test": int(len(split.test)),
        "cohorts": cohorts,
        "matrix": matrix,
        "delta_from_target_diagonal": deltas,
        "summary": _calibration_transfer_summary(matrix),
    }


# --- analysis 3: pooled production generalization ----------------------------------


def pooled_generalization(
    dataset: Dataset,
    axis: str,
    detector_factory: Callable[[], Any],
    *,
    min_cell: int = 30,
    alpha: float = 0.1,
    n_boot: int = 1000,
    seed: int = SEED,
) -> dict:
    """Train on all training partitions; evaluate each seen-cohort holdout.

    Every cohort here is represented in pooled training, so cells are labeled
    ``seen-cohort`` — never external/unseen. This is the production-generalization
    reference against which LOCO and calibration-transfer deltas are computed.
    """
    split = holdout_split(dataset, fractions=(0.6, 0.2, 0.2), seed=seed)
    scorer = fit_source_scorer(
        detector_factory, dataset, split.train, split.calibration, detector_id="pooled"
    )
    keys = cohort_keys(dataset, axis)
    cal_labels = dataset.labels[split.calibration]
    cal_probs = scorer.calibrated(scorer.raw(dataset, split.calibration))
    test_cohorts = np.array([str(keys[int(i)]) for i in split.test])

    cells: dict[str, dict] = {}
    for cohort in sorted(set(test_cohorts.tolist())):
        mask = test_cohorts == cohort
        local = split.test[mask]
        if len(local) < min_cell or not _both_classes(dataset.labels[local]):
            continue
        probs = scorer.calibrated(scorer.raw(dataset, local))
        cell = transport_cell_metrics(
            cal_labels,
            cal_probs,
            dataset.labels[local],
            probs,
            [dataset.groups[int(i)] for i in local],
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
        )
        cell["cohort"] = cohort
        cell["cohort_role"] = "seen-cohort"
        cells[cohort] = cell
    return {
        "kind": "pooled_generalization",
        "axis": axis,
        "dataset": dataset.provenance,
        "dataset_sha256": dataset.sha256,
        "detector": scorer.detector_id,
        "n_train": int(len(split.train)),
        "n_calibration": int(len(split.calibration)),
        "cohort_role": "seen-cohort",
        "cells": cells,
        "summary": _transport_summary(cells, cell_key=None),
    }


# --- dataset-origin / shortcut probe -----------------------------------------------


def dataset_origin_probe(
    dataset: Dataset,
    axis: str,
    *,
    n_splits: int = 5,
    seed: int = SEED,
) -> dict:
    """How strongly do the shared features encode cohort (dataset-origin) identity?

    A multinomial logistic probe is fit to predict the cohort label from the same
    stylometric features the detector tiers use, under group-disjoint CV. High
    cross-validated accuracy means the representation carries a strong
    dataset-origin shortcut, which is reported as a limitation, not hidden.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    keys = cohort_keys(dataset, axis)
    cohorts = sorted(set(keys.tolist()))
    if len(cohorts) < 2:
        return {
            "axis": axis,
            "n_cohorts": len(cohorts),
            "note": "need >=2 cohorts for an origin probe",
        }
    X = dataset.features()
    # Standardize so the multinomial probe converges; features are on mixed scales.
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    X = (X - mu) / sd
    y = np.array([cohorts.index(k) for k in keys], dtype=int)
    groups = np.array([str(g) for g in dataset.groups])
    n_groups = len(set(groups.tolist()))
    splits = max(2, min(n_splits, n_groups))
    if n_groups < 2:
        return {"axis": axis, "n_cohorts": len(cohorts), "note": "need >=2 groups for CV"}
    gkf = GroupKFold(n_splits=splits)
    correct = 0
    total = 0
    per_cohort_acc: dict[str, list[float]] = {c: [] for c in cohorts}
    for tr, te in gkf.split(X, y, groups=groups):
        if len(set(y[tr].tolist())) < 2:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        correct += int((pred == y[te]).sum())
        total += len(te)
        for c_idx, c in enumerate(cohorts):
            m = y[te] == c_idx
            if m.any():
                per_cohort_acc[c].append(float((pred[m] == y[te][m]).mean()))
    if total == 0:
        return {"axis": axis, "n_cohorts": len(cohorts), "note": "CV produced no evaluable folds"}
    accuracy = correct / total
    chance = 1.0 / len(cohorts)
    return {
        "axis": axis,
        "n_cohorts": len(cohorts),
        "n_splits": splits,
        "cv_accuracy": float(accuracy),
        "chance_accuracy": float(chance),
        "accuracy_over_chance": float(accuracy - chance),
        "per_cohort_accuracy": {c: float(np.mean(v)) for c, v in per_cohort_acc.items() if v},
        "interpretation": (
            "strong dataset-origin signal; transport gaps may reflect shortcut features"
            if accuracy > 2 * chance
            else "weak dataset-origin signal"
        ),
    }


# --- source-balanced sensitivity -----------------------------------------------------


def source_balanced_sensitivity(
    labels: np.ndarray,
    probs: np.ndarray,
    groups,
    *,
    target_prevalence: float = 0.5,
    n_boot: int = 1000,
    seed: int = SEED,
) -> dict:
    """Recompute a cell's metrics after reweighting to a standard prevalence.

    If a transport gap vanishes under source balancing, it is a prevalence/base-rate
    artifact rather than a representation or calibration failure. Reports both the
    natural and the source-balanced views with a group-bootstrap interval on the
    balanced AUROC.
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    natural = float(labels.mean()) if len(labels) else float("nan")
    views = validity.standardized_prevalence_metrics(labels, probs, target_prevalence)
    boot = validity.group_bootstrap_ci(
        labels,
        probs,
        np.asarray(list(groups)),
        lambda y, p: float(roc_auc_score(y, p)),
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "natural_prevalence": natural,
        "target_prevalence": float(target_prevalence),
        "natural": views["natural"],
        "source_balanced": views["standardized"],
        "auroc": views["auroc"],
        "auroc_group_bootstrap": boot,
        "note": (
            "AUROC is prevalence-invariant; Brier/log-loss/accuracy are reweighted to the "
            "standard prevalence. A transport gap that disappears here is a base-rate artifact."
        ),
    }


# --- shared helpers ------------------------------------------------------------------


def _delta(reference: dict | None, cell: dict) -> dict:
    """Metric deltas of ``cell`` relative to a ``reference`` (diagonal) cell."""
    if not reference or reference.get("degenerate") or cell.get("degenerate"):
        return {}
    out: dict[str, float] = {}
    for key in ("auroc", "auprc", "brier", "ece", "calibration_slope"):
        a = cell.get(key)
        b = reference.get(key)
        if (
            isinstance(a, (int, float)) and isinstance(b, (int, float)) and a == a and b == b
        ):  # not NaN
            out[f"delta_{key}"] = float(a - b)
    return out


def _transport_summary(cells: dict[str, dict], cell_key: str | None = "target") -> dict:
    """Worst/mean cohort AUROC and Brier across evaluable cells."""
    metric_blocks = []
    for cell in cells.values():
        block = cell.get(cell_key) if cell_key else cell
        if isinstance(block, dict) and not block.get("degenerate") and "auroc" in block:
            metric_blocks.append(block)
    if not metric_blocks:
        return {"n_evaluable_cells": 0}
    aurocs = [b["auroc"] for b in metric_blocks if b.get("auroc") == b.get("auroc")]
    briers = [b["brier"] for b in metric_blocks if b.get("brier") == b.get("brier")]
    return {
        "n_evaluable_cells": len(metric_blocks),
        "worst_cohort_auroc": float(min(aurocs)) if aurocs else float("nan"),
        "mean_cohort_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "worst_cohort_brier": float(max(briers)) if briers else float("nan"),
        "mean_cohort_brier": float(np.mean(briers)) if briers else float("nan"),
    }


def _calibration_transfer_summary(matrix: dict[str, dict[str, dict]]) -> dict:
    """Summarize on- vs off-diagonal calibration-transfer performance."""
    on_auroc: list[float] = []
    off_auroc: list[float] = []
    on_brier: list[float] = []
    off_brier: list[float] = []
    for src, row in matrix.items():
        for tgt, cell in row.items():
            if cell.get("degenerate"):
                continue
            a = cell.get("auroc")
            b = cell.get("brier")
            if src == tgt:
                if a == a:
                    on_auroc.append(a)
                if b == b:
                    on_brier.append(b)
            else:
                if a == a:
                    off_auroc.append(a)
                if b == b:
                    off_brier.append(b)
    return {
        "n_source_cohorts": len(matrix),
        "on_diagonal_mean_auroc": float(np.mean(on_auroc)) if on_auroc else float("nan"),
        "off_diagonal_mean_auroc": float(np.mean(off_auroc)) if off_auroc else float("nan"),
        "on_diagonal_mean_brier": float(np.mean(on_brier)) if on_brier else float("nan"),
        "off_diagonal_mean_brier": float(np.mean(off_brier)) if off_brier else float("nan"),
        "calibration_transport_auroc_gap": (
            float(np.mean(on_auroc) - np.mean(off_auroc))
            if on_auroc and off_auroc
            else float("nan")
        ),
    }
