"""Exploratory 7-class source attribution on Defactify.

Beyond binary detection: given a text, predict WHICH source produced it —
human (NYT) or one of six LLM families (Gemma-2-9B, GPT-4o, Llama-8B,
Mistral-7B, Qwen-2-72B, Yi-Large). Roy et al. 2026 report 5-9% attribution
accuracy for their baselines on this dataset; it is deliberately hard.

Two contenders, both evaluated out-of-fold under story-grouped GroupKFold:
  - multinomial logistic regression on the 17-feature stylometric vector
  - a K=7 Dirichlet variant of Panoptes-v0 (feature + char-sequence branches)

The signed card also derives binary detection metrics (P(AI) = 1 - P(human))
so the attribution model is comparable with the detection zoo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import cards, evaluate, models  # noqa: E402
from bench.datasets import Dataset, grouped_splits  # noqa: E402

ATTRIBUTION_CLASSES = [
    "human",
    "gemma-2-9b",
    "gpt-4o",
    "llama-8b",
    "mistral-7b",
    "qwen-2-72b",
    "yi-large",
]
ROY_ET_AL_ATTRIBUTION = {
    "low": 0.05,
    "high": 0.09,
    "note": "Roy et al. 2026 attribution baselines (paper Table 4)",
}


class AttributionError(RuntimeError):
    pass


def _class_index(dataset: Dataset) -> np.ndarray:
    unknown = sorted(set(dataset.families) - set(ATTRIBUTION_CLASSES))
    if unknown:
        raise AttributionError(f"dataset families not in the 7-class attribution set: {unknown}")
    lookup = {name: i for i, name in enumerate(ATTRIBUTION_CLASSES)}
    return np.array([lookup[f] for f in dataset.families])


def _f1_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score

    per_family = f1_score(
        y_true, y_pred, average=None, labels=list(range(len(ATTRIBUTION_CLASSES)))
    )
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(np.mean(y_true == y_pred)),
        "per_family_f1": {name: float(per_family[i]) for i, name in enumerate(ATTRIBUTION_CLASSES)},
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(ATTRIBUTION_CLASSES)))
        ).tolist(),
        "classes": ATTRIBUTION_CLASSES,
    }


def logistic_attribution_cv(dataset: Dataset) -> dict:
    """Out-of-fold 7-class probabilities from multinomial logistic regression."""
    from sklearn.linear_model import LogisticRegression

    X = dataset.features()
    y7 = _class_index(dataset)
    oof = np.zeros((len(dataset), len(ATTRIBUTION_CLASSES)))
    for train, test in grouped_splits(dataset, 5):
        mean = X[train].mean(axis=0)
        scale = np.where(X[train].std(axis=0) == 0, 1.0, X[train].std(axis=0))
        clf = LogisticRegression(C=1.0, max_iter=5000, random_state=13)
        clf.fit((X[train] - mean) / scale, y7[train])
        oof[test] = clf.predict_proba((X[test] - mean) / scale)
    report = _f1_report(y7, oof.argmax(axis=1))
    report["oof_probabilities"] = oof
    return report


def dirichlet_attribution_cv(dataset: Dataset, seeds: tuple[int, ...] = (13, 42, 87)) -> dict:
    """Out-of-fold 7-class probabilities from the K=7 Dirichlet Panoptes-v0 variant."""
    from bench.panoptes_v0 import (
        TORCH_AVAILABLE,
        PanoptesV0Net,
        _schedule,
        char_matrix,
        device_name,
        evidential_loss,
    )

    if not TORCH_AVAILABLE:
        raise AttributionError("the Dirichlet attribution variant requires torch")
    import torch
    from sklearn.model_selection import GroupShuffleSplit

    X = dataset.features()
    y7 = _class_index(dataset)
    groups = np.array(dataset.groups)
    schedule = _schedule(len(dataset))
    chars = char_matrix(dataset.texts)
    device = device_name()
    k = len(ATTRIBUTION_CLASSES)

    oof_seeds = []
    for seed in seeds:
        oof = np.zeros((len(dataset), k))
        for train, test in grouped_splits(dataset, 5):
            val_split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
            tr_sub, val_sub = next(val_split.split(X[train], y7[train], groups=groups[train]))
            tr, val = train[tr_sub], train[val_sub]

            torch.manual_seed(seed)
            np.random.seed(seed)
            mean = X[tr].mean(axis=0)
            scale = np.where(X[tr].std(axis=0) == 0, 1.0, X[tr].std(axis=0))
            net = PanoptesV0Net(X.shape[1], use_sequence=True, n_classes=k).to(device)
            optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
            xt = torch.tensor((X[tr] - mean) / scale, dtype=torch.float32, device=device)
            yt = torch.tensor(y7[tr], dtype=torch.long, device=device)
            ct = torch.tensor(chars[tr], dtype=torch.long)
            xv = torch.tensor((X[val] - mean) / scale, dtype=torch.float32, device=device)
            cv_chars = torch.tensor(chars[val], dtype=torch.long, device=device)

            best_loss = float("inf")
            best_state = None
            stale = 0
            for epoch in range(schedule["epochs"]):
                permutation = np.random.permutation(len(xt))
                for start in range(0, len(xt), schedule["batch"]):
                    idx = permutation[start : start + schedule["batch"]]
                    optimizer.zero_grad()
                    out = net(xt[idx], ct[idx].to(device))
                    loss = evidential_loss(out["alpha"], yt[idx], epoch, n_classes=k)
                    loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    val_probs = net(xv, cv_chars)["probs"]
                    val_loss = float(
                        torch.nn.functional.cross_entropy(
                            torch.log(torch.clamp(val_probs, min=1e-8)),
                            torch.tensor(y7[val], dtype=torch.long, device=device),
                        )
                    )
                if val_loss < best_loss - 1e-4:
                    best_loss = val_loss
                    best_state = {k_: v.detach().clone() for k_, v in net.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= schedule["patience"]:
                        break
            if best_state is not None:
                net.load_state_dict(best_state)
            with torch.no_grad():
                for start in range(0, len(test), 4096):
                    sl = slice(start, min(start + 4096, len(test)))
                    xt_ = torch.tensor(
                        (X[test][sl] - mean) / scale, dtype=torch.float32, device=device
                    )
                    ct_ = torch.tensor(chars[test][sl], dtype=torch.long, device=device)
                    oof[test][sl] = net(xt_, ct_)["probs"].cpu().numpy()
        oof_seeds.append(oof)
    mean_oof = np.mean(oof_seeds, axis=0)
    report = _f1_report(y7, mean_oof.argmax(axis=1))
    report["oof_probabilities"] = mean_oof
    report["seeds"] = list(seeds)
    return report


def run_attribution(
    dataset: Dataset, created_utc: str | None = None, skip_dirichlet: bool = False
) -> dict:
    """Full attribution experiment: both contenders, signed card."""
    gate = models.power_gate(len(dataset))
    print(f"attribution experiment: {len(ATTRIBUTION_CLASSES)} classes, n={len(dataset)}")

    logistic = logistic_attribution_cv(dataset)
    print(
        f"multinomial logistic: macro-F1 {logistic['macro_f1']:.3f}, "
        f"accuracy {logistic['accuracy']:.3f}"
    )

    contenders = {
        "multinomial-logistic": {k: v for k, v in logistic.items() if k != "oof_probabilities"}
    }
    best_oof = logistic["oof_probabilities"]
    best_name = "multinomial-logistic"

    if not skip_dirichlet:
        dirichlet = dirichlet_attribution_cv(dataset)
        print(
            f"K=7 Dirichlet net: macro-F1 {dirichlet['macro_f1']:.3f}, "
            f"accuracy {dirichlet['accuracy']:.3f}"
        )
        contenders["panoptes-v0-dirichlet-k7"] = {
            k: v for k, v in dirichlet.items() if k != "oof_probabilities"
        }
        if dirichlet["macro_f1"] >= logistic["macro_f1"]:
            best_oof = dirichlet["oof_probabilities"]
            best_name = "panoptes-v0-dirichlet-k7"

    # Derived binary detection from the best 7-class model: P(AI) = 1 - P(human).
    p_ai = 1.0 - best_oof[:, 0]
    binary = evaluate.binary_metrics(dataset.labels, p_ai)
    binary_ci = evaluate.auroc_ci(dataset.labels, p_ai)

    evaluation = {
        "protocol": (
            "GroupKFold(5) by reconstructed story group; 7-class out-of-fold; "
            "binary metrics derived as P(AI) = 1 - P(human)."
        ),
        "metrics": binary,
        "auroc_ci95": binary_ci,
        "reliability_bins": evaluate.reliability_bins(dataset.labels, p_ai),
        "coverage_curve": evaluate.coverage_curve(dataset.labels, p_ai),
        "conformal": evaluate.conformal_sets(dataset.labels, p_ai),
        "fairness_slices": evaluate.fairness_slices(dataset, p_ai),
        "folds": [],
        "n_splits": 5,
    }
    card = cards.model_card(
        model_name="panoptes-v0-attribution-k7",
        tier=2,
        dataset=dataset,
        evaluation=evaluation,
        gate=gate,
        created_utc=created_utc,
        config={
            "task": "7-class source attribution (human + 6 LLM families)",
            "classes": ATTRIBUTION_CLASSES,
            "contenders": ["multinomial-logistic", "panoptes-v0-dirichlet-k7"],
            "best_by_macro_f1": best_name,
        },
        limitations=[
            "Exploratory: attribution is far harder than detection; Roy et al. baselines "
            "are 5-9% accuracy.",
            "AI texts are single-prompt rewrites of NYT stories; family cues may be "
            "prompt-specific.",
            "Family labels are the generating model names, not verified provenance.",
            "Not for accusing individuals of using a specific model.",
        ],
        extra={
            "attribution": {
                "contenders": contenders,
                "best_by_macro_f1": best_name,
                "external_reference": ROY_ET_AL_ATTRIBUTION,
            },
            **(
                {"story_groups": dataset.meta["group_reconstruction"]}
                if dataset.meta.get("group_reconstruction")
                else {}
            ),
            **(
                {"leakage_audit": dataset.meta["leakage_audit"]}
                if dataset.meta.get("leakage_audit")
                else {}
            ),
        },
    )
    out = ROOT / "backend" / "artifacts" / "cards" / "attribution-defactify.json"
    cards.write_card(card, out)
    print(f"Card: {out} (sha256 {card['artifact_sha256'][:16]}…)")
    return card
