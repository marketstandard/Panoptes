"""Tests for bench/panoptes_v0.py (skipped when torch is absent).

Run from the repository root: python -m pytest bench/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from bench import evaluate  # noqa: E402
from bench.panoptes_v0 import PanoptesV0, PanoptesV0Net, evidential_loss  # noqa: E402


def clustered_data(n: int = 60, d: int = 17, seed: int = 5):
    rng = np.random.default_rng(seed)
    y = np.array([i % 2 for i in range(n)])
    X = rng.normal(0, 0.4, (n, d))
    X[y == 1, 0] += 2.0
    X[y == 0, 0] -= 2.0
    return X, y


def test_forward_pass_shapes():
    net = PanoptesV0Net(n_features=17)
    x = torch.randn(4, 17)
    out = net(x)
    assert out["alpha"].shape == (4, 2)
    assert out["p"].shape == (4,)
    assert out["vacuity"].shape == (4,)
    assert out["dissonance"].shape == (4,)
    assert torch.all(out["alpha"] >= 1.0)  # softplus evidence + 1
    assert torch.all((out["vacuity"] > 0) & (out["vacuity"] <= 1.0))


def test_evidential_loss_decreases():
    torch.manual_seed(0)
    net = PanoptesV0Net(n_features=17)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    X, y = clustered_data()
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    first = None
    last = None
    for epoch in range(30):
        opt.zero_grad()
        loss = evidential_loss(net(xt)["alpha"], yt, epoch)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
        last = float(loss.detach())
    assert last < first


def test_vacuity_higher_on_ood_noise_than_in_distribution():
    X, y = clustered_data()
    model = PanoptesV0(epochs=120, seed=13).fit(X, y)
    in_dist = model.uncertainty(X)["vacuity"]
    rng = np.random.default_rng(99)
    noise = rng.uniform(-8, 8, (20, X.shape[1]))  # far outside the training clusters
    ood = model.uncertainty(noise)["vacuity"]
    assert float(ood.mean()) > float(in_dist.mean())


def test_conformal_empirical_coverage_within_tolerance():
    X, y = clustered_data(n=60)
    # manual 2-fold out-of-fold probabilities
    oof = np.zeros(60)
    for train, test in [(np.arange(30), np.arange(30, 60)), (np.arange(30, 60), np.arange(30))]:
        model = PanoptesV0(epochs=80, seed=13).fit(X[train], y[train])
        oof[test] = model.predict_proba(X[test])
    result = evaluate.conformal_sets(y, oof, alpha=0.1)
    assert result["empirical_coverage"] >= 0.75  # marginal guarantee with small-n tolerance
    assert 0.0 <= result["abstention_rate"] <= 1.0
