"""Panoptes bench CLI.

  python -m bench train --model logistic|gbm|panoptes-v0 --data corpus
  python -m bench evaluate --model models/logistic-tier0/model.pkl
  python -m bench validate --dataset your.csv
  python -m bench contribute --dataset your.csv --name my-dataset
  python -m bench predict --model models/logistic-tier0/model.pkl --text "..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import cards, datasets, evaluate, models  # noqa: E402
from bench.features import FEATURE_NAMES, heuristic_raw_score, vector  # noqa: E402

MODELS_DIR = ROOT / "models"
CARDS_DIR = ROOT / "backend" / "artifacts" / "cards"
DATASET_MANIFESTS = ROOT / "datasets" / "manifests"
CALIBRATION_ARTIFACT = ROOT / "backend" / "artifacts" / "baseline-calibration.json"


class CliError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset(spec: str) -> datasets.Dataset:
    if spec == "corpus":
        return datasets.load_verified_corpus()
    if spec == "defactify":
        return datasets.load_defactify()
    return datasets.load_user_dataset(Path(spec))


def _make_model(name: str):
    if name == "logistic":
        return models.LogisticTier0()
    if name == "gbm":
        return models.GbmTier1()
    if name == "panoptes-v0":
        try:
            from bench.panoptes_v0 import PanoptesV0
        except ImportError as exc:
            raise CliError(
                "panoptes-v0 requires the optional neural extra: pip install torch "
                "(see bench/README.md). The classical tiers work without it."
            ) from exc
        return PanoptesV0()
    raise CliError(f"unknown model {name!r}: choose logistic, gbm, or panoptes-v0")


def _save_model(model, name: str, dataset: datasets.Dataset, out_dir: Path | None) -> Path:
    out_dir = out_dir or MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.pkl"
    payload = {
        "schema": "panoptes-bench-model-v1",
        "name": model.name,
        "tier": model.tier,
        "feature_names": list(FEATURE_NAMES),
        "trained_on": {"provenance": dataset.provenance, "sha256": dataset.sha256, "n": len(dataset)},
        "created_utc": _utc_now(),
        "estimator": model,
    }
    blob = pickle.dumps(payload)
    path.write_bytes(blob)
    path.with_suffix(".pkl.sha256").write_text(hashlib.sha256(blob).hexdigest() + "\n")
    return path


def _load_model(path: Path):
    blob = Path(path).read_bytes()
    sidecar = Path(path).with_suffix(".pkl.sha256")
    if sidecar.exists():
        expected = sidecar.read_text().strip()
        actual = hashlib.sha256(blob).hexdigest()
        if actual != expected:
            raise CliError(f"model hash mismatch: {path} fails its .sha256 sidecar check")
    payload = pickle.loads(blob)  # noqa: S301 - local artifacts with hash sidecars
    return payload["estimator"], payload


def _corpus_created_utc(data_arg: str) -> str | None:
    """Deterministic timestamp for pinned datasets: the newest run manifest's
    declared time for the corpus, the fetch manifest's time for Defactify.
    User datasets keep the wall clock."""
    if data_arg == "corpus":
        from research.baseline_corpus import run_manifests

        return max(manifest["created_utc"] for manifest in run_manifests())
    if data_arg == "defactify":
        return datasets.defactify_created_utc()
    return None


def cmd_train(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.data)
    gate = models.power_gate(len(dataset))
    created_utc = _corpus_created_utc(args.data)
    print(f"Dataset: {dataset.provenance} (n={len(dataset)}, sha256={dataset.sha256[:16]}…)")
    if dataset.meta.get("group_reconstruction"):
        stats = dataset.meta["group_reconstruction"]
        print(
            f"Story groups: {stats['n_groups']} (mean size {stats['group_size_mean']:.1f}, "
            f"singletons {stats['singletons']}, threshold {stats['threshold']})"
        )
    if dataset.meta.get("leakage_audit"):
        audit = dataset.meta["leakage_audit"]
        print(
            f"Official-split leakage audit: {audit['official_test_rows_with_train_near_duplicate']}"
            f"/{audit['official_test_rows']} test rows share a story with train "
            f"({audit['official_split_story_leakage_rate']:.1%})"
        )
    print(f"Power gate: {gate['rationale']}")

    if args.model == "panoptes-v0":
        try:
            from bench.panoptes_v0 import run_harness
        except ImportError as exc:
            raise CliError(
                "panoptes-v0 requires the optional neural extra: pip install torch "
                "(see bench/README.md). The classical tiers work without it."
            ) from exc
        run_harness(dataset, created_utc=created_utc)
        return 0

    model = _make_model(args.model)
    if model.tier == 1 and len(dataset) < models.TIER1_MIN_N:
        print(f"WARNING: tier-1 model with n={len(dataset)} < {models.TIER1_MIN_N}; results are exploratory.")
    if model.tier == 2 and not gate["passes"]:
        print("WARNING: neural tier below the power gate; results are exploratory, not comparative.")

    suffix = "-defactify" if args.data == "defactify" else ""
    evaluation = evaluate.cross_validate(lambda: _make_model(args.model), dataset)
    final = _make_model(args.model).fit(dataset.features(), dataset.labels)
    out_dir = Path(args.out) if args.out else MODELS_DIR / f"{model.name}{suffix}"
    model_path = _save_model(final, model.name, dataset, out_dir)

    card = cards.model_card(
        model_name=model.name,
        tier=model.tier,
        dataset=dataset,
        evaluation=evaluation,
        gate=gate,
        config={"model": args.model, "data": args.data},
        created_utc=created_utc,
        extra={"story_groups": dataset.meta["group_reconstruction"], "leakage_audit": dataset.meta["leakage_audit"]}
        if dataset.meta.get("group_reconstruction")
        else None,
    )
    card_name = args.card or f"{model.name}{suffix}.json"
    card_path = cards.write_card(card, CARDS_DIR / card_name)

    metrics = evaluation["metrics"]
    print(f"Out-of-fold: AUROC {metrics['auroc']:.3f} "
          f"(95% CI {evaluation['auroc_ci95'][0]:.3f}–{evaluation['auroc_ci95'][1]:.3f}), "
          f"Brier {metrics['brier']:.3f}, ECE {metrics['ece']:.3f}, "
          f"TPR@1%FPR {metrics['tpr_at_1fpr']:.3f}")
    print(f"Model: {model_path}")
    print(f"Card:  {card_path} (sha256 {card['artifact_sha256'][:16]}…)")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    estimator, payload = _load_model(Path(args.model))
    dataset = _load_dataset(args.data)
    evaluation = evaluate.cross_validate(lambda: _fresh_like(estimator), dataset)
    metrics = evaluation["metrics"]
    print(f"Evaluating {payload['name']} on {dataset.provenance} (n={len(dataset)})")
    print(json.dumps({**metrics, "auroc_ci95": evaluation["auroc_ci95"]}, indent=2, sort_keys=True))
    return 0


def _fresh_like(estimator):
    if isinstance(estimator, models.LogisticTier0):
        return models.LogisticTier0(C=estimator.C, seed=estimator.seed)
    if isinstance(estimator, models.GbmTier1):
        return models.GbmTier1(seed=estimator.seed)
    raise CliError("evaluate supports logistic and gbm artifacts; panoptes-v0 has its own bench harness")


def _shipped_probabilities(dataset: datasets.Dataset) -> np.ndarray:
    raw = np.array(
        [heuristic_raw_score(text, kind) for text, kind in zip(dataset.texts, dataset.kinds, strict=True)]
    )
    if CALIBRATION_ARTIFACT.exists():
        artifact = json.loads(CALIBRATION_ARTIFACT.read_text(encoding="utf-8"))
        calibrator = artifact["binary_calibrator"]
        return np.interp(raw, calibrator["x_thresholds"], calibrator["y_thresholds"])
    return raw


def cmd_validate(args: argparse.Namespace) -> int:
    dataset = datasets.load_user_dataset(Path(args.dataset))
    print(f"Dataset valid: n={len(dataset)}, sha256={dataset.sha256[:16]}…")
    if args.model:
        estimator, payload = _load_model(Path(args.model))
        X = dataset.features()
        probabilities = estimator.predict_proba(X)
        label = payload["name"]
    else:
        probabilities = _shipped_probabilities(dataset)
        label = "shipped heuristic + corpus calibration"
    metrics = evaluate.binary_metrics(dataset.labels, probabilities)
    slices = evaluate.fairness_slices(dataset, probabilities)
    print(f"Model: {label}")
    print(json.dumps({"metrics": metrics, "fairness_slices": slices}, indent=2, sort_keys=True))
    return 0


def cmd_contribute(args: argparse.Namespace) -> int:
    path = Path(args.dataset)
    dataset = datasets.load_user_dataset(path)  # validates against the schema
    name = args.name.strip().lower()
    if not name or not all(c.isalnum() or c in "._-" for c in name):
        raise CliError("--name must be a slug: lowercase letters, digits, dots, dashes")
    manifest = {
        "schema": "panoptes-dataset-pointer-v1",
        "name": name,
        "created_utc": _utc_now(),
        "contributor": args.contributor or "anonymous",
        "dataset": {
            "file_name": path.name,
            "sha256": _file_sha256(path),
            "content_sha256": dataset.sha256,
            "n_rows": len(dataset),
            "label_counts": {
                "human": int((dataset.labels == 0).sum()),
                "ai": int((dataset.labels == 1).sum()),
            },
            "kinds": sorted(set(dataset.kinds)),
        },
        "raw_data": "stays with the contributor; this manifest is a hash pointer only",
        "license_note": args.license or "contributor asserts the right to share derived statistics",
    }
    cards.sign(manifest)
    DATASET_MANIFESTS.mkdir(parents=True, exist_ok=True)
    out = DATASET_MANIFESTS / f"{name}.json"
    if out.exists():
        raise CliError(f"{out} already exists; dataset names are first-come")
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(f"pointer sha256: {manifest['artifact_sha256']}")
    print("Open a PR adding this file to datasets/manifests/ to contribute.")
    return 0


def cmd_external_validate(args: argparse.Namespace) -> int:
    """Score an external dataset with the *shipped* runtime (heuristic +
    corpus-fitted calibration) and write a signed benchmark card. The
    heuristic is training-free, so the full external set is a valid holdout."""
    dataset = _load_dataset(args.data)
    probabilities = _shipped_probabilities(dataset)
    metrics = evaluate.binary_metrics(dataset.labels, probabilities)
    ci = evaluate.auroc_ci(dataset.labels, probabilities)
    card = {
        "schema": "panoptes-benchmark-card-v1",
        "dataset": dataset.provenance,
        "metrics": {
            "auroc": metrics["auroc"],
            "auroc_ci95_lo": ci[0],
            "auroc_ci95_hi": ci[1],
            "brier": metrics["brier"],
            "ece": metrics["ece"],
            "accuracy": metrics["accuracy"],
            "tpr_at_1fpr": metrics["tpr_at_1fpr"],
            "tpr_at_5fpr": metrics["tpr_at_5fpr"],
            "n": float(len(dataset)),
        },
        "limitations": [
            "Shipped runtime = heuristic raw score + corpus-fitted isotonic calibration; it never saw this dataset.",
            "Domain shift: calibration was fitted on the 104-record project corpus, not on NYT prose.",
            "External validation measures transportability, not the bench-trained tiers (see their cards).",
        ],
    }
    cards.sign(card)
    out = CARDS_DIR / args.card
    cards.write_card(card, out)
    print(f"External validation on {dataset.provenance} (n={len(dataset)})")
    print(json.dumps(card["metrics"], indent=2, sort_keys=True))
    print(f"Card: {out} (sha256 {card['artifact_sha256'][:16]}…)")
    return 0


def cmd_attribute(args: argparse.Namespace) -> int:
    from bench import attribution

    dataset = _load_dataset(args.data)
    created_utc = _corpus_created_utc(args.data)
    attribution.run_attribution(dataset, created_utc=created_utc, skip_dirichlet=args.skip_dirichlet)
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    from bench.measure import run_measurement

    dataset = _load_dataset(args.data)
    card = run_measurement(dataset)
    card["created_utc"] = _corpus_created_utc(args.data) or _utc_now()
    cards.sign(card)
    out = Path(args.out) if args.out else CARDS_DIR / "measurement-protocol.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Measurement protocol on {dataset.provenance} (n={len(dataset)})")
    for name, block in card["metrics"].items():
        metrics = block["metrics"]
        print(
            f"  {name}: AUROC {metrics['auroc']:.3f}  Brier {metrics['brier']:.3f}  "
            f"ECE {metrics['ece']:.3f}  slope {metrics['calibration_slope']}"
        )
    print(f"Card: {out} (sha256 {card['artifact_sha256'][:16]}…)")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    if args.model == "panoptes-v0":
        from bench.panoptes_v0 import predict_text

        result = predict_text(args.text, kind=args.kind)
    else:
        estimator, payload = _load_model(Path(args.model))
        X = np.array([vector(args.text, args.kind)])
        result = {"p_ai": float(estimator.predict_proba(X)[0]), "model": payload["name"]}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="cross-validate, fit, and card a model")
    train.add_argument("--model", required=True, choices=["logistic", "gbm", "panoptes-v0"])
    train.add_argument("--data", default="corpus", help="'corpus', 'defactify', or a CSV/JSONL path")
    train.add_argument("--out", default=None, help="model output directory")
    train.add_argument("--card", default=None, help="card filename override (default: <model>[-defactify].json)")
    train.set_defaults(func=cmd_train)

    evaluate_cmd = sub.add_parser("evaluate", help="re-evaluate a saved model on the corpus")
    evaluate_cmd.add_argument("--model", required=True)
    evaluate_cmd.add_argument("--data", default="corpus")
    evaluate_cmd.set_defaults(func=cmd_evaluate)

    validate = sub.add_parser("validate", help="score your dataset with the shipped model")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--model", default=None)
    validate.set_defaults(func=cmd_validate)

    contribute = sub.add_parser("contribute", help="write a signed hash-pointer manifest")
    contribute.add_argument("--dataset", required=True)
    contribute.add_argument("--name", required=True)
    contribute.add_argument("--contributor", default=None)
    contribute.add_argument("--license", dest="license", default=None)
    contribute.set_defaults(func=cmd_contribute)

    external = sub.add_parser("external-validate", help="score an external dataset with the shipped runtime")
    external.add_argument("--data", default="defactify", help="'defactify', 'corpus', or a CSV/JSONL path")
    external.add_argument("--card", default="defactify-external-validation.json", help="card filename under backend/artifacts/cards/")
    external.set_defaults(func=cmd_external_validate)

    attribute = sub.add_parser("attribute", help="exploratory 7-class source attribution experiment")
    attribute.add_argument("--data", default="defactify", help="'defactify' (the only 7-family dataset) or a CSV/JSONL path")
    attribute.add_argument("--skip-dirichlet", action="store_true", help="run only the multinomial logistic contender")
    attribute.set_defaults(func=cmd_attribute)

    measure = sub.add_parser("measure", help="run the frozen measurement protocol (train/cal/test)")
    measure.add_argument("--data", default="corpus", help="'corpus', 'defactify', or a CSV/JSONL path")
    measure.add_argument("--out", default=None, help="output card path")
    measure.set_defaults(func=cmd_measure)

    predict = sub.add_parser("predict", help="score one text with a saved model")
    predict.add_argument("--model", required=True)
    predict.add_argument("--text", required=True)
    predict.add_argument("--kind", default="text", choices=["text", "code"])
    predict.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (CliError, datasets.DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
