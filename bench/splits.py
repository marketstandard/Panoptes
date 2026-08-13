"""Group-disjoint train / calibration / test splits.

Hard protocol rule: never fit or tune the calibration layer on the
final test set. Splits are disjoint on the leakage-control group
(prompt or story) and, when present, on author and source identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bench.datasets import Dataset
from research.protocol import NESTED_CV_GROUP_THRESHOLD

SEED = 13


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class ProtocolSplit:
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    grouping: str
    method: str
    n_groups: int

    def assert_disjoint(self, dataset: Dataset) -> None:
        _assert_index_disjoint(self.train, self.calibration, self.test)
        _assert_group_disjoint(dataset.groups, self.train, self.calibration, self.test, "group")
        if dataset.enforce_author_disjoint and dataset.authors is not None:
            _assert_group_disjoint(dataset.authors, self.train, self.calibration, self.test, "author")
        if dataset.domains:
            # Domain overlap is allowed for in-domain evaluation; source/story
            # leakage is already covered by `groups` for Defactify.
            pass


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _indices_for_groups(values: list[str], chosen: set[str]) -> np.ndarray:
    return np.array([i for i, value in enumerate(values) if value in chosen], dtype=int)


def _assert_index_disjoint(*parts: np.ndarray) -> None:
    seen: set[int] = set()
    for part in parts:
        overlap = seen.intersection(int(i) for i in part)
        if overlap:
            raise SplitError(f"index leakage across partitions: {sorted(overlap)[:8]}")
        seen.update(int(i) for i in part)


def _assert_group_disjoint(
    values: list[str], train: np.ndarray, calibration: np.ndarray, test: np.ndarray, name: str
) -> None:
    if not values:
        return
    sets = []
    for part in (train, calibration, test):
        sets.append({values[int(i)] for i in part})
    for left, right, label in (
        (0, 1, "train/calibration"),
        (0, 2, "train/test"),
        (1, 2, "calibration/test"),
    ):
        leaked = sets[left] & sets[right]
        if leaked:
            raise SplitError(f"{name} leakage on {label}: {sorted(leaked)[:8]}")


def _assign_groups(
    groups: list[str],
    fractions: tuple[float, float, float],
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    unique = _unique_in_order(groups)
    if len(unique) < 3:
        raise SplitError(f"need at least 3 groups to form train/calibration/test, found {len(unique)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    shuffled = [unique[i] for i in order]
    n = len(shuffled)
    n_test = max(1, int(round(n * fractions[2])))
    n_cal = max(1, int(round(n * fractions[1])))
    if n_test + n_cal >= n:
        n_test = 1
        n_cal = 1
    test = set(shuffled[:n_test])
    calibration = set(shuffled[n_test : n_test + n_cal])
    train = set(shuffled[n_test + n_cal :])
    if not train:
        raise SplitError("train partition is empty after holdout assignment")
    return train, calibration, test


def _majority_label(dataset: Dataset, keys: list[str], group: str) -> int:
    labels = [int(dataset.labels[i]) for i, key in enumerate(keys) if key == group]
    if not labels:
        return 0
    return 1 if sum(labels) * 2 >= len(labels) else 0


def _train_cal_groups(
    unique_rest: list[str],
    dataset: Dataset,
    keys: list[str],
    seed: int,
) -> tuple[set[str], set[str]]:
    """Split remaining groups into train/cal, covering both classes when possible.

    Nested CV with label-pure groups can otherwise put a single class in
    calibration, which cannot identify an isotonic map.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[str]] = {0: [], 1: []}
    for group in unique_rest:
        by_label[_majority_label(dataset, keys, group)].append(group)
    for label in (0, 1):
        rng.shuffle(by_label[label])
    both = bool(by_label[0] and by_label[1])
    n_cal = max(1, len(unique_rest) // 4)
    if both and len(unique_rest) >= 3:
        n_cal = max(n_cal, 2)
    n_cal = min(n_cal, len(unique_rest) - 1)
    cal: list[str] = []
    if both and n_cal >= 2:
        cal.append(by_label[0].pop())
        cal.append(by_label[1].pop())
    leftover = by_label[0] + by_label[1]
    rng.shuffle(leftover)
    while len(cal) < n_cal and leftover:
        cal.append(leftover.pop())
    if not cal:
        cal.append(unique_rest[0])
    cal_g = set(cal)
    train_g = set(unique_rest) - cal_g
    if not train_g:
        moved = cal[-1]
        cal_g.remove(moved)
        train_g.add(moved)
    return train_g, cal_g


def _split_keys(dataset: Dataset) -> list[str]:
    """Author is the grouping unit when contributor-supplied authors exist."""
    if dataset.enforce_author_disjoint:
        return list(dataset.authors)
    return list(dataset.groups)


def holdout_split(
    dataset: Dataset,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = SEED,
) -> ProtocolSplit:
    """Single group-disjoint 60/20/20-style holdout."""
    keys = _split_keys(dataset)
    train_g, cal_g, test_g = _assign_groups(keys, fractions, seed)
    grouping = "author" if dataset.enforce_author_disjoint else "group"
    split = ProtocolSplit(
        train=_indices_for_groups(keys, train_g),
        calibration=_indices_for_groups(keys, cal_g),
        test=_indices_for_groups(keys, test_g),
        grouping=grouping,
        method="holdout",
        n_groups=len(set(keys)),
    )
    split.assert_disjoint(dataset)
    return split


def nested_grouped_splits(
    dataset: Dataset,
    n_outer: int = 5,
    seed: int = SEED,
) -> list[ProtocolSplit]:
    """Outer test groups stay untouched; remaining groups split train/cal.

    Used when the number of groups is too small for a single holdout to
    leave a stable calibration partition (protocol default: n_groups < 30).
    """
    from sklearn.model_selection import GroupKFold

    keys = _split_keys(dataset)
    groups = np.array(keys)
    n_groups = len(set(keys))
    splits = max(2, min(n_outer, n_groups))
    if n_groups < 3:
        raise SplitError(f"nested grouped CV needs at least 3 groups, found {n_groups}")
    outer = GroupKFold(n_splits=splits)
    X = np.zeros((len(dataset), 1))
    out: list[ProtocolSplit] = []
    grouping = "author" if dataset.enforce_author_disjoint else "group"
    for fold, (rest, test) in enumerate(outer.split(X, dataset.labels, groups=groups)):
        rest_groups = [keys[int(i)] for i in rest]
        unique_rest = _unique_in_order(rest_groups)
        if len(unique_rest) < 2:
            raise SplitError("not enough remaining groups to split train from calibration")
        train_g, cal_g = _train_cal_groups(unique_rest, dataset, keys, seed + fold)
        split = ProtocolSplit(
            train=_indices_for_groups(keys, train_g),
            calibration=_indices_for_groups(keys, cal_g),
            test=np.asarray(test, dtype=int),
            grouping=grouping,
            method=f"nested_grouped_cv:{fold}",
            n_groups=n_groups,
        )
        split.assert_disjoint(dataset)
        out.append(split)
    return out


def protocol_splits(dataset: Dataset, seed: int = SEED) -> list[ProtocolSplit]:
    """Choose holdout vs nested grouped CV from the frozen protocol rule."""
    n_groups = len(set(_split_keys(dataset)))
    if n_groups < NESTED_CV_GROUP_THRESHOLD:
        return nested_grouped_splits(dataset, seed=seed)
    return [holdout_split(dataset, seed=seed)]
