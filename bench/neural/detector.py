"""Trainable neural detector as a bench ``Detector`` for LOCO transport runs.

Leave-one-cohort-out representation transport retrains the encoder on each
source split (protocol v2.1 Phase 6: "leave-one-cohort-out neural runs for
genuinely unseen populations"). This adapter wraps the frozen pilot-winner
configuration in the bench ``fit(dataset, idx)`` / ``predict_proba(dataset,
idx)`` protocol so the existing ``bench.transport.leave_one_cohort_out`` runs
the neural tier unchanged.

The data firewall is preserved: ``fit`` sees only the supplied training indices
(splitting them group-disjoint into inner-train/inner-dev for early stopping),
and ``predict_proba`` only ever scores. The winner's architecture, objective,
windowing, and aggregation come from the signed pilot selection card — nothing
is retuned here.
"""

from __future__ import annotations

import numpy as np

from bench.datasets import Dataset

# Heavy deps (torch, transformers, bench.neural.training internals) are imported
# lazily inside methods so importing this module stays cheap and CPU-only
# environments can still use the non-neural tiers.


def _import_stack():
    import torch  # noqa: F401
    from transformers import AutoTokenizer  # noqa: F401

    from bench.neural import objectives as obj_mod  # noqa: F401
    from bench.neural.data import build_windowed_corpus  # noqa: F401
    from bench.neural.model import HierarchicalSummaryHead, WindowEncoder  # noqa: F401
    from bench.neural.train import (  # noqa: F401
        PilotConfig,
        encode_corpus,
        summary_head_probabilities,
        train_summary_head,
        train_window_encoder,
    )

    return {
        "torch": torch,
        "AutoTokenizer": AutoTokenizer,
        "obj_mod": obj_mod,
        "build_windowed_corpus": build_windowed_corpus,
        "WindowEncoder": WindowEncoder,
        "HierarchicalSummaryHead": HierarchicalSummaryHead,
        "PilotConfig": PilotConfig,
        "encode_corpus": encode_corpus,
        "summary_head_probabilities": summary_head_probabilities,
        "train_summary_head": train_summary_head,
        "train_window_encoder": train_window_encoder,
    }


def _inner_group_split(
    dataset: Dataset, train_idx: np.ndarray, dev_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Group-disjoint inner-train/inner-dev split of ``train_idx`` for early stopping."""
    groups = np.array([str(dataset.groups[int(i)]) for i in train_idx])
    unique = np.unique(groups)
    if len(unique) < 2:
        # Degenerate: fall back to a tail split (still index-disjoint).
        n_dev = max(1, int(round(len(train_idx) * dev_fraction)))
        return np.asarray(train_idx[:-n_dev], dtype=int), np.asarray(train_idx[-n_dev:], dtype=int)
    rng = np.random.default_rng(seed)
    shuffled = unique[rng.permutation(len(unique))]
    n_dev = max(1, int(round(len(unique) * dev_fraction)))
    dev_groups = set(shuffled[:n_dev].tolist())
    inner_dev = train_idx[np.isin(groups, list(dev_groups))]
    inner_train = train_idx[~np.isin(groups, list(dev_groups))]
    if len(inner_train) == 0 or len(inner_dev) == 0:
        n_dev = max(1, int(round(len(train_idx) * dev_fraction)))
        return np.asarray(train_idx[:-n_dev], dtype=int), np.asarray(train_idx[-n_dev:], dtype=int)
    return np.asarray(inner_train, dtype=int), np.asarray(inner_dev, dtype=int)


class NeuralTrainableDetector:
    """Bench ``Detector`` that trains the frozen-winner neural encoder on ``fit``."""

    def __init__(
        self,
        winner: dict,
        *,
        device: str | None = None,
        max_windows: int = 16,
        inner_dev_fraction: float = 0.15,
        seed: int = 13,
        config_overrides: dict | None = None,
    ):
        self.winner = winner
        self.device = device
        self.max_windows = int(max_windows)
        self.inner_dev_fraction = float(inner_dev_fraction)
        self.seed = int(seed)
        self.config_overrides = dict(config_overrides or {})
        self.name = f"neural-{winner.get('encoder', 'encoder')}-{winner.get('objective', 'erm')}"
        self._model = None
        self._summary_head = None
        self._tokenizer = None
        self._stack = None

    def fit(self, dataset: Dataset, train_idx: np.ndarray) -> NeuralTrainableDetector:
        stack = _import_stack()
        self._stack = stack
        torch = stack["torch"]
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        winner = self.winner
        tokenizer = stack["AutoTokenizer"].from_pretrained(winner["hf"])
        self._tokenizer = tokenizer

        train_idx = np.asarray(train_idx, dtype=int)
        inner_train, inner_dev = _inner_group_split(
            dataset, train_idx, self.inner_dev_fraction, self.seed
        )
        train_corpus = stack["build_windowed_corpus"](
            dataset,
            tokenizer,
            winner["max_length"],
            winner["overlap"],
            indices=inner_train,
            max_windows=self.max_windows,
        )
        dev_corpus = stack["build_windowed_corpus"](
            dataset,
            tokenizer,
            winner["max_length"],
            winner["overlap"],
            indices=inner_dev,
            max_windows=self.max_windows,
        )
        cfg = stack["PilotConfig"](
            seed=self.seed, max_windows=self.max_windows, **self.config_overrides
        )
        model = stack["WindowEncoder"](winner["hf"])
        obj_name, payload = stack["obj_mod"].make_objective(
            winner["objective"], train_corpus.flat()["group_key"], dro_step_size=cfg.dro_step_size
        )
        # OOM resilience: halve the batch (doubling accumulation) until it fits.
        while True:
            try:
                model, history = stack["train_window_encoder"](
                    model,
                    train_corpus,
                    dev_corpus,
                    obj_name,
                    payload,
                    cfg,
                    device,
                    log_prefix=f"  [{self.name}] ",
                )
                break
            except torch.cuda.OutOfMemoryError:
                if cfg.batch_size <= 2:
                    raise
                cfg.batch_size //= 2
                cfg.grad_accum *= 2
                torch.cuda.empty_cache()
        self._model = model
        self._history = history

        # Hierarchical summary head trains on frozen window embeddings.
        if winner.get("aggregation") == "hierarchical_summary_head":
            train_out = stack["encode_corpus"](model, train_corpus, device, cfg.eval_batch_size)
            dev_out = stack["encode_corpus"](model, dev_corpus, device, cfg.eval_batch_size)
            head = stack["HierarchicalSummaryHead"](hidden=model.hidden_size)
            head, _ = stack["train_summary_head"](head, train_out, dev_out, cfg, device)
            self._summary_head = head
        if device == "cuda":
            torch.cuda.empty_cache()
        return self

    def predict_proba(self, dataset: Dataset, idx: np.ndarray) -> np.ndarray:
        if self._model is None or self._stack is None:
            raise RuntimeError("NeuralTrainableDetector must be fit before predict_proba")
        stack = self._stack
        winner = self.winner
        corpus = stack["build_windowed_corpus"](
            dataset,
            self._tokenizer,
            winner["max_length"],
            winner["overlap"],
            indices=np.asarray(idx, dtype=int),
            max_windows=self.max_windows,
        )
        out = stack["encode_corpus"](self._model, corpus, self._device, batch_size=32)
        if self._summary_head is not None:
            return np.asarray(
                stack["summary_head_probabilities"](
                    self._summary_head, out, self._device, self.max_windows
                ),
                dtype=float,
            )
        return np.asarray(out.doc_probabilities_logit_mean(), dtype=float)
