"""Windowed corpora for the neural pilot, with leakage-group metadata.

A :class:`bench.datasets.Dataset` is turned into a :class:`WindowedCorpus`:
every document is windowed (see :mod:`bench.neural.windowing`) and carries its
label, leakage-control group, domain, generator family, and the GroupDRO /
group-balanced audit key. The corpus exposes a flat window-level view for
training and a per-document view for aggregation.

Subsampling is group-disjoint and prevalence-preserving: whole near-duplicate
clusters are selected within each ``domain x label`` stratum in proportion to
the stratum's share, so the training pool keeps the source distribution and no
cluster crosses a partition.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from bench.datasets import Dataset
from bench.neural.windowing import Window, document_windows, pad_windows


def group_key(domain: str, family: str, label: int, dataset: str | None = None) -> str:
    """The audit/sampling group for balanced and GroupDRO objectives.

    Built from dataset-visible metadata only (dataset x domain x generator x
    label; the dataset component is present only for pooled multi-cohort
    training). It is never a model input; it only reweights or resamples the loss.
    """
    parts = [str(domain), str(family), str(int(label))]
    if dataset is not None:
        parts.insert(0, str(dataset))
    return "|".join(parts)


def stratified_group_subsample(
    dataset: Dataset,
    max_rows: int,
    seed: int,
    strata=("domain", "label"),
) -> tuple[np.ndarray, dict]:
    """Group-disjoint, prevalence-preserving subsample indices.

    Each stratum (domain x label by default) receives a quota proportional to
    its size; whole leakage groups are drawn deterministically within a stratum
    until the quota is met. Returns selected row indices and an audit dict.
    """
    n = len(dataset)
    if max_rows is None or n <= max_rows:
        return np.arange(n), {"max_rows": max_rows, "n_rows_selected": n, "stratified": False}

    labels = np.asarray(dataset.labels)
    domains = np.array([str(d) for d in (dataset.domains or dataset.kinds)])
    groups = np.array([str(g) for g in dataset.groups])

    def stratum_key(i: int) -> str:
        parts = []
        if "domain" in strata:
            parts.append(domains[i])
        if "label" in strata:
            parts.append(str(int(labels[i])))
        return "|".join(parts)

    stratum_of = np.array([stratum_key(i) for i in range(n)])
    strata_keys = sorted(set(stratum_of.tolist()))
    selected: list[int] = []
    for sk in strata_keys:
        idx = np.where(stratum_of == sk)[0]
        quota = max(1, int(round(max_rows * len(idx) / n)))
        # Deterministic group ordering within the stratum.
        stratum_groups = groups[idx]
        unique = sorted(set(stratum_groups.tolist()))
        keyed = sorted(
            unique, key=lambda g: hashlib.sha256(f"{seed}:{sk}:{g}".encode()).hexdigest()
        )
        counts = {g: int((stratum_groups == g).sum()) for g in unique}
        chosen: set[str] = set()
        total = 0
        for g in keyed:
            if chosen and total + counts[g] > quota:
                continue
            chosen.add(g)
            total += counts[g]
            if total >= quota:
                break
        keep = idx[np.isin(stratum_groups, list(chosen))]
        selected.extend(keep.tolist())
    selected = np.array(sorted(selected), dtype=int)
    info = {
        "max_rows": max_rows,
        "seed": seed,
        "stratified": True,
        "strata": list(strata),
        "n_strata": len(strata_keys),
        "n_rows_selected": int(len(selected)),
        "n_groups_selected": int(len(set(groups[selected].tolist()))),
    }
    return selected, info


@dataclass
class WindowedCorpus:
    """Per-document windows plus a flat window-level training view."""

    windows: list[list[Window]]  # per document
    labels: np.ndarray  # per document, 0=human 1=AI
    groups: list[str]  # leakage-control group per document
    domains: list[str]
    families: list[str]
    group_keys: list[str]  # GroupDRO/balanced audit key per document
    n_tokens: list[int]
    max_length: int
    overlap: int
    # flat window-level view (built lazily)
    _flat: dict | None = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def n_windows(self) -> int:
        return sum(len(w) for w in self.windows)

    def flat(self) -> dict:
        """Flatten to window-level arrays for training.

        Each window inherits its document's label and audit group key. Returns
        dict of numpy arrays: input_ids, attention_mask, label, doc_idx, and a
        list group_key per window.
        """
        if self._flat is not None:
            return self._flat
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        label: list[int] = []
        doc_idx: list[int] = []
        gk: list[str] = []
        for d, wins in enumerate(self.windows):
            for w in wins:
                input_ids.append(w.input_ids)
                attention_mask.append(w.attention_mask)
                label.append(int(self.labels[d]))
                doc_idx.append(d)
                gk.append(self.group_keys[d])
        self._flat = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_mask, dtype=np.int64),
            "label": np.array(label, dtype=np.int64),
            "doc_idx": np.array(doc_idx, dtype=np.int64),
            "group_key": gk,
        }
        return self._flat


def build_windowed_corpus(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    overlap: int,
    indices: np.ndarray | None = None,
    max_windows: int = 32,
    show_progress: bool = False,
) -> WindowedCorpus:
    """Window every selected document of ``dataset`` (labels stay attached)."""
    if indices is None:
        indices = np.arange(len(dataset))
    indices = np.asarray(indices)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    windows: list[list[Window]] = []
    labels: list[int] = []
    groups: list[str] = []
    domains: list[str] = []
    families: list[str] = []
    group_keys: list[str] = []
    n_tokens: list[int] = []

    dom = dataset.domains if dataset.domains is not None else dataset.kinds
    ds_labels = dataset.datasets
    for pos, i in enumerate(indices):
        i = int(i)
        text = dataset.texts[i]
        label = int(dataset.labels[i])
        raw = document_windows(
            text, tokenizer, max_length=max_length, overlap=overlap, max_windows=max_windows
        )
        wins = pad_windows(raw, pad_id=pad_id, max_length=max_length)
        windows.append(wins)
        labels.append(label)
        groups.append(str(dataset.groups[i]))
        domains.append(str(dom[i]))
        families.append(str(dataset.families[i]))
        group_keys.append(
            group_key(
                dom[i], dataset.families[i], label, dataset=ds_labels[i] if ds_labels else None
            )
        )
        n_tokens.append(max((w.token_end for w in raw), default=0))
        if show_progress and (pos + 1) % 2000 == 0:
            print(f"    windowed {pos + 1}/{len(indices)} docs", flush=True)

    return WindowedCorpus(
        windows=windows,
        labels=np.array(labels, dtype=np.int64),
        groups=groups,
        domains=domains,
        families=families,
        group_keys=group_keys,
        n_tokens=n_tokens,
        max_length=max_length,
        overlap=overlap,
    )
