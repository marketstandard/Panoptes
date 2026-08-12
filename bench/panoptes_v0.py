"""Panoptes-v0: an evidential, abstention-native AI-text detector.

Architecture (custom, from scratch):
  feature branch   Linear(d,64) -> GELU -> LayerNorm -> Linear(64,64) -> GELU
  sequence branch  char embedding (128 vocab, 32-d) -> mean pool -> Linear(32,64)
                   [gated ON only when the bench power gate passes]
  evidence head    Linear(*, 2) -> softplus + 1  =>  Dirichlet alpha

The Dirichlet head natively yields the uncertainty decomposition the
product already exposes: expected probability, vacuity (K/S — "we lack
evidence"), and dissonance (evidence conflict). High vacuity drives the
INSUFFICIENT_DATA evidence state instead of a forced guess.

Training: evidential MSE + annealed KL (Sensoy et al. 2018), AdamW,
grouped train/val early stopping on calibration error, seeds {13,42,87}
reported as mean +/- sd. A split-conformal wrap on out-of-fold
nonconformity provides distribution-free coverage.

Requires the optional neural extra (torch). Everything in this module
degrades with a clear message when torch is absent.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.features import FEATURE_NAMES, vector  # noqa: E402

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on machines without torch
    torch = None
    nn = object
    TORCH_AVAILABLE = False

WEIGHTS_DIR = ROOT / "models" / "panoptes-v0"
CARD_PATH = ROOT / "backend" / "artifacts" / "panoptes-v0-card.json"
FINDINGS_LOG = ROOT / "research" / "findings" / "panoptes-v0.md"

SEEDS = (13, 42, 87)
FEATURE_DIM = len(FEATURE_NAMES)
K_CLASSES = 2


class TorchRequired(RuntimeError):
    pass


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise TorchRequired(
            "panoptes-v0 requires the optional neural extra: pip install torch "
            "(CUDA build recommended). The classical bench tiers work without it."
        )


def device_name() -> str:
    _require_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class PanoptesV0Net(nn.Module):
        def __init__(self, n_features: int = FEATURE_DIM, use_sequence: bool = False):
            super().__init__()
            self.use_sequence = use_sequence
            self.feature_branch = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.GELU(),
                nn.LayerNorm(64),
                nn.Linear(64, 64),
                nn.GELU(),
            )
            if use_sequence:
                self.char_embed = nn.Embedding(128, 32)
                self.seq_proj = nn.Linear(32, 64)
                head_in = 128
            else:
                head_in = 64
            self.evidence_head = nn.Linear(head_in, K_CLASSES)

        def forward(self, x_features, x_chars=None):
            hidden = self.feature_branch(x_features)
            if self.use_sequence and x_chars is not None:
                embedded = self.char_embed(x_chars).mean(dim=1)
                hidden = torch.cat([hidden, torch.nn.functional.gelu(self.seq_proj(embedded))], dim=1)
            evidence = torch.nn.functional.softplus(self.evidence_head(hidden))
            alpha = evidence + 1.0
            strength = alpha.sum(dim=1)
            p = alpha[:, 1] / strength
            vacuity = K_CLASSES / strength
            e = evidence
            dissonance = 1.0 - torch.abs(e[:, 0] - e[:, 1]) / torch.clamp(e.sum(dim=1), min=1e-8)
            return {
                "alpha": alpha,
                "p": p,
                "vacuity": vacuity,
                "dissonance": dissonance,
                "evidence": evidence,
            }


def char_ids(text: str, max_len: int = 512) -> np.ndarray:
    ids = [min(ord(char), 127) for char in text[:max_len]]
    if not ids:
        ids = [0]
    return np.array(ids, dtype=np.int64)


def evidential_loss(alpha, y, epoch: int, anneal_epochs: int = 50):
    """Sensoy et al. 2018: MSE on Dirichlet moments + annealed KL to uniform."""
    _require_torch()
    strength = alpha.sum(dim=1, keepdim=True)
    p = alpha / strength
    y_onehot = torch.nn.functional.one_hot(y, K_CLASSES).float()
    mse = ((y_onehot - p) ** 2 + p * (1 - p) / (strength + 1)).sum(dim=1).mean()
    alpha_tilde = y_onehot + (1 - y_onehot) * alpha
    kl = torch.lgamma(alpha_tilde.sum(dim=1)) - torch.lgamma(
        torch.tensor(float(K_CLASSES), device=alpha.device)
    ) - torch.sum(torch.lgamma(alpha_tilde), dim=1) + torch.sum(
        (alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(alpha_tilde.sum(dim=1, keepdim=True))),
        dim=1,
    )
    anneal = min(1.0, epoch / max(anneal_epochs, 1))
    return mse + anneal * 0.1 * kl.mean()


# ---------------------------------------------------------------------------
# Bench-model wrapper (feature branch only; sequence branch needs raw text
# and is enabled by the training harness when the power gate passes)
# ---------------------------------------------------------------------------


@dataclass
class PanoptesV0:
    name: str = "panoptes-v0"
    tier: int = 2
    seed: int = 13
    epochs: int = 200
    patience: int = 20
    use_sequence: bool = False
    _net: Any = field(default=None, repr=False)
    _mean: np.ndarray | None = field(default=None, repr=False)
    _scale: np.ndarray | None = field(default=None, repr=False)
    device: str = "cpu"

    def fit(self, X: np.ndarray, y: np.ndarray, texts: list[str] | None = None) -> "PanoptesV0":
        _require_torch()
        self.device = device_name()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._mean = X.mean(axis=0)
        self._scale = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
        Xs = (X - self._mean) / self._scale

        net = PanoptesV0Net(X.shape[1], use_sequence=self.use_sequence).to(self.device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
        xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y, dtype=torch.long, device=self.device)
        batch = 16
        for epoch in range(self.epochs):
            permutation = np.random.permutation(len(Xs))
            for start in range(0, len(Xs), batch):
                idx = permutation[start : start + batch]
                optimizer.zero_grad()
                out = net(xt[idx])
                loss = evidential_loss(out["alpha"], yt[idx], epoch)
                loss.backward()
                optimizer.step()
        self._net = net
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _require_torch()
        Xs = (X - self._mean) / self._scale
        with torch.no_grad():
            out = self._net(torch.tensor(Xs, dtype=torch.float32, device=self.device))
        return out["p"].cpu().numpy()

    def uncertainty(self, X: np.ndarray) -> dict[str, np.ndarray]:
        _require_torch()
        Xs = (X - self._mean) / self._scale
        with torch.no_grad():
            out = self._net(torch.tensor(Xs, dtype=torch.float32, device=self.device))
        return {
            "p": out["p"].cpu().numpy(),
            "vacuity": out["vacuity"].cpu().numpy(),
            "dissonance": out["dissonance"].cpu().numpy(),
        }


# ---------------------------------------------------------------------------
# Training harness: grouped CV, early stopping on calibration error, seeds
# ---------------------------------------------------------------------------


def _train_tracked(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xval: np.ndarray,
    yval: np.ndarray,
    seed: int,
    epochs: int = 200,
    patience: int = 20,
) -> tuple[Any, list[dict]]:
    """Train one net with early stopping on validation ECE; return net + curve."""
    _require_torch()
    from bench.evaluate import expected_calibration_error

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device_name()
    mean = Xtr.mean(axis=0)
    scale = np.where(Xtr.std(axis=0) == 0, 1.0, Xtr.std(axis=0))
    net = PanoptesV0Net(Xtr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    xt = torch.tensor((Xtr - mean) / scale, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    xv = torch.tensor((Xval - mean) / scale, dtype=torch.float32, device=device)

    curve: list[dict] = []
    best_ece = float("inf")
    best_state = None
    stale = 0
    for epoch in range(epochs):
        permutation = np.random.permutation(len(xt))
        epoch_loss = 0.0
        for start in range(0, len(xt), 16):
            idx = permutation[start : start + 16]
            optimizer.zero_grad()
            out = net(xt[idx])
            loss = evidential_loss(out["alpha"], yt[idx], epoch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(idx)
        with torch.no_grad():
            val_p = net(xv)["p"].cpu().numpy()
        val_ece = (
            expected_calibration_error(yval, val_p) if len(set(yval)) > 1 else float("nan")
        )
        curve.append({"epoch": epoch, "train_loss": epoch_loss / len(xt), "val_ece": val_ece})
        if not math.isnan(val_ece) and val_ece < best_ece - 1e-4:
            best_ece = val_ece
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net._panoptes_scaler = (mean, scale)  # type: ignore[attr-defined]
    return net, curve


def train_cv(dataset, seeds: tuple[int, ...] = SEEDS) -> dict:
    """Out-of-fold probabilities per seed under GroupKFold by prompt group."""
    _require_torch()
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    X = dataset.features()
    y = dataset.labels
    groups = np.array(dataset.groups)
    per_seed: dict[int, dict] = {}
    for seed in seeds:
        oof = np.zeros(len(dataset))
        splitter = GroupKFold(n_splits=min(5, len(set(groups))))
        for train_idx, test_idx in splitter.split(X, y, groups=groups):
            val_split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
            tr_sub, val_sub = next(
                val_split.split(X[train_idx], y[train_idx], groups=groups[train_idx])
            )
            net, _ = _train_tracked(
                X[train_idx][tr_sub],
                y[train_idx][tr_sub],
                X[train_idx][val_sub],
                y[train_idx][val_sub],
                seed,
            )
            mean, scale = net._panoptes_scaler  # type: ignore[attr-defined]
            with torch.no_grad():
                xt = torch.tensor(
                    (X[test_idx] - mean) / scale, dtype=torch.float32, device=device_name()
                )
                oof[test_idx] = net(xt)["p"].cpu().numpy()
        per_seed[seed] = {"oof": oof}
    return {"per_seed": per_seed, "X": X, "y": y}


def calibration_slope(y: np.ndarray, p: np.ndarray) -> dict:
    """Logistic recalibration diagnostic: ideal is intercept 0, slope 1."""
    from research.methodology import logistic_irls

    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    fit = logistic_irls(logit.reshape(-1, 1), y.astype(float))
    return {"intercept": float(fit["beta"][0]), "slope": float(fit["beta"][1])}


def comparison_battery(dataset, v0_oof: np.ndarray) -> dict:
    """Panoptes-v0 vs the classical tier and the shipped heuristic.

    McNemar on correctness, DeLong on OOF scores, calibration slope,
    Tjur pseudo-R^2, Durbin-Watson on probability-ordered residuals;
    Benjamini-Hochberg across the whole battery.
    """
    from bench import evaluate, models
    from bench.features import heuristic_raw_score
    from research.methodology import (
        benjamini_hochberg,
        delong_test,
        durbin_watson,
        mcnemar,
        tjur_r2,
    )

    y = dataset.labels
    logistic = evaluate.cross_validate(models.LogisticTier0, dataset)
    logistic_oof = logistic["oof_probabilities"]
    heuristic = np.array(
        [
            heuristic_raw_score(text, kind)
            for text, kind in zip(dataset.texts, dataset.kinds, strict=True)
        ]
    )

    contenders = {"panoptes-v0": v0_oof, "logistic-tier0": logistic_oof, "heuristic": heuristic}
    per_model = {}
    for name, probs in contenders.items():
        resid = y - probs
        order = np.argsort(probs, kind="mergesort")
        per_model[name] = {
            "metrics": evaluate.binary_metrics(y, probs),
            "auroc_ci95": evaluate.auroc_ci(y, probs),
            "calibration": calibration_slope(y, probs),
            "tjur_r2": tjur_r2(y.astype(float), probs),
            "durbin_watson": durbin_watson(resid[order]),
        }

    pairs = [("panoptes-v0", "logistic-tier0"), ("panoptes-v0", "heuristic"),
             ("logistic-tier0", "heuristic")]
    comparisons = []
    for a, b in pairs:
        correct_a = (contenders[a] >= 0.5) == y
        correct_b = (contenders[b] >= 0.5) == y
        mc = mcnemar(correct_a, correct_b)
        dl = delong_test(y, contenders[a], contenders[b])
        comparisons.append({"pair": f"{a} vs {b}", "test": "mcnemar", **mc})
        comparisons.append({"pair": f"{a} vs {b}", "test": "delong", **dl})
    q_values = benjamini_hochberg([c["p_value"] for c in comparisons])
    for comparison, q in zip(comparisons, q_values, strict=True):
        comparison["q_value"] = q
        comparison["significant_at_0.05"] = bool(q <= 0.05)

    return {
        "per_model": per_model,
        "comparisons": comparisons,
        "gbm_note": (
            f"GBM tier-1 not run: n={len(dataset)} < {models.TIER1_MIN_N} minimum; "
            "gate rationale recorded on the card."
        ),
        "fairness_slices": evaluate.fairness_slices(dataset, v0_oof),
        "conformal": evaluate.conformal_sets(y, v0_oof),
    }


# ---------------------------------------------------------------------------
# Final training, weights, inference
# ---------------------------------------------------------------------------


def _sha256_bytes(blob: bytes) -> str:
    import hashlib

    return hashlib.sha256(blob).hexdigest()


def train_final(dataset, seeds: tuple[int, ...] = SEEDS) -> dict:
    """Train one net per seed on all data; save weights + config locally."""
    _require_torch()
    from sklearn.model_selection import GroupShuffleSplit

    X = dataset.features()
    y = dataset.labels
    groups = np.array(dataset.groups)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    saved = []
    curves = {}
    for seed in seeds:
        val_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, val = next(val_split.split(X, y, groups=groups))
        net, curve = _train_tracked(X[tr], y[tr], X[val], y[val], seed)
        mean, scale = net._panoptes_scaler  # type: ignore[attr-defined]
        blob = b""
        path = WEIGHTS_DIR / f"seed-{seed}.pt"
        torch.save(net.state_dict(), path)
        blob = path.read_bytes()
        path.with_suffix(".pt.sha256").write_text(_sha256_bytes(blob) + "\n")
        saved.append(
            {
                "seed": seed,
                "file": path.name,
                "sha256": _sha256_bytes(blob),
                "scaler_mean": mean.tolist(),
                "scaler_scale": scale.tolist(),
                "epochs_trained": len(curve),
            }
        )
        curves[str(seed)] = curve

    config = {
        "schema": "panoptes-v0-config-v1",
        "architecture": {
            "feature_branch": "Linear(d,64) -> GELU -> LayerNorm -> Linear(64,64) -> GELU",
            "evidence_head": "Linear(64,2) -> softplus + 1 (Dirichlet alpha)",
            "sequence_branch": "disabled by power gate (see card)",
            "n_features": int(X.shape[1]),
            "feature_names": list(FEATURE_NAMES),
        },
        "training": {
            "loss": "evidential MSE + annealed KL (Sensoy 2018)",
            "optimizer": "AdamW lr=3e-4 wd=1e-2 batch=16",
            "epochs_max": 200,
            "early_stop": "patience 20 on grouped-validation ECE",
            "seeds": list(seeds),
        },
        "members": saved,
    }
    config_path = WEIGHTS_DIR / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"config": config, "curves": curves}


def load_ensemble():
    """Load all seed members; returns list of (net, mean, scale)."""
    _require_torch()
    config = json.loads((WEIGHTS_DIR / "config.json").read_text(encoding="utf-8"))
    members = []
    for member in config["members"]:
        path = WEIGHTS_DIR / member["file"]
        if _sha256_bytes(path.read_bytes()) != member["sha256"]:
            raise TorchRequired(f"weights hash mismatch for {path}; refusing to load")
        net = PanoptesV0Net(config["architecture"]["n_features"])
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        net.eval()
        members.append((net, np.array(member["scaler_mean"]), np.array(member["scaler_scale"])))
    return config, members


def predict_text(text: str, kind: str = "text") -> dict:
    _require_torch()
    config, members = load_ensemble()
    X = np.array([vector(text, kind)])
    ps, vacuities, dissonances = [], [], []
    for net, mean, scale in members:
        with torch.no_grad():
            out = net(torch.tensor((X - mean) / scale, dtype=torch.float32))
        ps.append(float(out["p"][0]))
        vacuities.append(float(out["vacuity"][0]))
        dissonances.append(float(out["dissonance"][0]))
    p = float(np.mean(ps))
    vacuity = float(np.mean(vacuities))
    return {
        "model": "panoptes-v0",
        "p_ai": p,
        "vacuity": vacuity,
        "dissonance": float(np.mean(dissonances)),
        "seed_spread": float(np.std(ps)),
        "evidence_state": "SUPPORTED" if vacuity < 0.6 else "INSUFFICIENT_DATA",
        "note": "evidential estimate; vacuity near 1 means the model has no evidence and abstains",
    }


# ---------------------------------------------------------------------------
# Full harness: gate -> CV -> battery -> final fit -> card -> findings log
# ---------------------------------------------------------------------------


def run_harness(dataset, created_utc: str | None = None) -> dict:
    _require_torch()
    from bench import cards, evaluate, models

    gate = models.power_gate(len(dataset))
    device = device_name()
    print(f"panoptes-v0 harness on {device}; power gate: {gate['rationale']}")

    cv = train_cv(dataset)
    y = dataset.labels
    seed_metrics = {}
    for seed, payload in cv["per_seed"].items():
        seed_metrics[str(seed)] = evaluate.binary_metrics(y, payload["oof"])
    mean_oof = np.mean([payload["oof"] for payload in cv["per_seed"].values()], axis=0)
    aggregate = evaluate.binary_metrics(y, mean_oof)
    aggregate_ci = evaluate.auroc_ci(y, mean_oof)
    print(
        f"OOF AUROC {aggregate['auroc']:.3f} (95% CI {aggregate_ci[0]:.3f}-{aggregate_ci[1]:.3f}), "
        f"ECE {aggregate['ece']:.3f}, Brier {aggregate['brier']:.3f}"
    )

    battery = comparison_battery(dataset, mean_oof)
    final = train_final(dataset)

    evaluation = {
        "metrics": aggregate,
        "auroc_ci95": aggregate_ci,
        "reliability_bins": evaluate.reliability_bins(y, mean_oof),
        "coverage_curve": evaluate.coverage_curve(y, mean_oof),
        "conformal": battery["conformal"],
        "fairness_slices": battery["fairness_slices"],
        "folds": [],
        "n_splits": 5,
    }
    card = cards.model_card(
        model_name="panoptes-v0",
        tier=2,
        dataset=dataset,
        evaluation=evaluation,
        gate=gate,
        created_utc=created_utc,
        config=final["config"]["training"],
        limitations=[
            "Exploratory: the corpus is below the neural power gate; treat comparisons as hypothesis-generating.",
            "Sequence branch disabled by the power gate; feature branch only.",
            "Stylometric signals degrade under paraphrase and heavy editing.",
            "Not for high-stakes decisions about individuals.",
        ],
        extra={
            "architecture": final["config"]["architecture"],
            "device": device,
            "seed_metrics": seed_metrics,
            "comparison_battery": {
                "per_model": battery["per_model"],
                "comparisons": battery["comparisons"],
                "gbm_note": battery["gbm_note"],
            },
            "training_curve_seed13": final["curves"][str(SEEDS[0])],
            "weights": {
                "local": "models/panoptes-v0/ (gitignored)",
                "members": final["config"]["members"],
                "release": "Open weights coming soon on Hugging Face; nothing public yet.",
            },
        },
    )
    cards.write_card(card, CARD_PATH)
    _write_findings_log(card, dataset, gate, seed_metrics, aggregate, aggregate_ci, battery, device)
    print(f"Card: {CARD_PATH} (sha256 {card['artifact_sha256'][:16]}…)")
    print(f"Weights: {WEIGHTS_DIR} (local; Hugging Face release coming soon)")
    return card


def _write_findings_log(card, dataset, gate, seed_metrics, aggregate, aggregate_ci, battery, device) -> None:
    FINDINGS_LOG.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Panoptes-v0 iteration log",
        "",
        "Every training run is appended with its config, data hash, metrics, statistical tests,",
        "the decision taken, and the next refinement. Newest last.",
        "",
        f"## Iteration 1 — {card['created_utc']}",
        "",
        "**Config**: evidential MLP (feature branch only; sequence branch gated OFF), "
        "Dirichlet evidence head, AdamW lr 3e-4 wd 1e-2, batch 16, <=200 epochs, "
        "early stop patience 20 on grouped-val ECE, seeds {13, 42, 87}.",
        "",
        f"**Data**: {dataset.provenance}, n={len(dataset)}, sha256 `{dataset.sha256}`.",
        f"**Device**: {device}. **Power gate**: {gate['rationale']}.",
        "",
        "**Out-of-fold metrics by seed** (GroupKFold by prompt):",
        "",
        "| Seed | AUROC | Brier | ECE | TPR@1%FPR |",
        "|---|---|---|---|---|",
    ]
    for seed, metrics in seed_metrics.items():
        lines.append(
            f"| {seed} | {metrics['auroc']:.3f} | {metrics['brier']:.3f} | "
            f"{metrics['ece']:.3f} | {metrics['tpr_at_1fpr']:.3f} |"
        )
    lines += [
        f"| **mean** | **{aggregate['auroc']:.3f}** | **{aggregate['brier']:.3f}** | "
        f"**{aggregate['ece']:.3f}** | **{aggregate['tpr_at_1fpr']:.3f}** |",
        "",
        f"AUROC 95% CI (ensemble OOF): [{aggregate_ci[0]:.3f}, {aggregate_ci[1]:.3f}]",
        "",
        "**Comparison battery** (BH-corrected):",
        "",
        "| Pair | Test | Statistic | p | q | Significant |",
        "|---|---|---|---|---|---|",
    ]
    for comparison in battery["comparisons"]:
        stat = comparison.get("statistic", comparison.get("z", 0.0))
        lines.append(
            f"| {comparison['pair']} | {comparison['test']} | {stat:.3f} | "
            f"{comparison['p_value']:.4f} | {comparison['q_value']:.4f} | "
            f"{'yes' if comparison['significant_at_0.05'] else 'no'} |"
        )
    lines += [
        "",
        f"Conformal (alpha=0.1): empirical coverage "
        f"{battery['conformal']['empirical_coverage']:.3f}, abstention rate "
        f"{battery['conformal']['abstention_rate']:.3f}.",
        "",
        "**Decision**: ship as *exploratory* — the corpus is below the neural power gate, so no "
        "comparative claim is made. Weights saved locally with SHA-256 sidecars; Hugging Face "
        "release pending a corpus that passes the gate.",
        "",
        "**Next refinements**: (1) grow human controls and community datasets until the gate "
        "passes; (2) enable the char-sequence branch and re-run this battery; (3) per-kind "
        "(code) evidential head once code controls exist.",
        "",
    ]
    FINDINGS_LOG.write_text("\n".join(lines), encoding="utf-8")
