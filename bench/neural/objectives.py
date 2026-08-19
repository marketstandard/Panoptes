"""Training objectives for the neural pilot.

Three preregistered objectives over the audit group (``domain x generator x
label``; see :func:`bench.neural.data.group_key`). The group key is sampling /
reweighting metadata only and is never a model input.

  - ``erm``:            ordinary empirical-risk minimization (mean CE).
  - ``group_balanced``: inverse-frequency per-window loss weights, so each audit
                        group contributes equally to the gradient.
  - ``group_dro``:      Group Distributionally Robust Optimization (Sagawa et
                        al. 2020): an adversarially reweighted worst-group loss
                        via exponentiated-gradient updates on a distribution q
                        over groups.
"""

from __future__ import annotations

import numpy as np
import torch

OBJECTIVES = ("erm", "group_balanced", "group_dro")


def group_counts(group_keys: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for g in group_keys:
        counts[g] = counts.get(g, 0) + 1
    return counts


def group_balanced_weights(group_keys: list[str]) -> np.ndarray:
    """Inverse-frequency per-window weights, normalized to mean 1."""
    counts = group_counts(group_keys)
    w = np.array([1.0 / counts[g] for g in group_keys], dtype=np.float64)
    w = w / w.mean()
    return w.astype(np.float32)


class GroupDRO:
    """Online GroupDRO weight distribution over audit groups.

    ``q`` starts uniform; each ``loss`` call computes the q-weighted batch loss
    and then updates q by exponentiated-gradient ascent on the observed per-group
    mean losses, so the objective tracks the worst group over training.
    """

    def __init__(self, group_keys: list[str], step_size: float = 0.05):
        self.groups = sorted(set(group_keys))
        self.g2i = {g: i for i, g in enumerate(self.groups)}
        self.step_size = float(step_size)
        self.q = np.full(len(self.groups), 1.0 / len(self.groups), dtype=np.float64)

    def loss(self, per_window_loss: torch.Tensor, batch_group_keys: list[str]) -> torch.Tensor:
        device = per_window_loss.device
        present = sorted(set(batch_group_keys))
        keys = np.array(batch_group_keys)
        group_mean: dict[str, torch.Tensor] = {}
        for g in present:
            mask = torch.from_numpy(keys == g).to(device)
            group_mean[g] = per_window_loss[mask].mean()
        q = torch.from_numpy(self.q).to(device=device, dtype=per_window_loss.dtype)
        total = torch.stack([q[self.g2i[g]] * group_mean[g] for g in present]).sum()
        with torch.no_grad():
            for g in present:
                self.q[self.g2i[g]] *= float(
                    torch.exp(torch.tensor(self.step_size) * group_mean[g].detach().cpu())
                )
            self.q /= self.q.sum()
        return total


def make_objective(name: str, train_group_keys: list[str], dro_step_size: float = 0.05):
    """Return ``(kind, payload)`` for the named objective.

    ``kind`` is one of OBJECTIVES; ``payload`` is ``None`` for ERM, a per-window
    weight array for ``group_balanced``, or a :class:`GroupDRO` for
    ``group_dro``.
    """
    if name not in OBJECTIVES:
        raise ValueError(f"unknown objective {name!r}; choose from {OBJECTIVES}")
    if name == "erm":
        return name, None
    if name == "group_balanced":
        return name, group_balanced_weights(train_group_keys)
    return name, GroupDRO(train_group_keys, step_size=dro_step_size)
