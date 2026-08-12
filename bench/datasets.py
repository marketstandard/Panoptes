"""Dataset loading for the bench.

Two sources:
  - the hash-verified baseline corpus (via research/baseline_corpus)
  - user datasets (CSV or JSONL) validated against schemas/bench-dataset.schema.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.features import FEATURE_NAMES, length_bucket, vector, word_tokens  # noqa: E402

SCHEMA_PATH = ROOT / "schemas" / "bench-dataset.schema.json"

_LABELS = {0: 0, 1: 1, "0": 0, "1": 1, "human": 0, "ai": 1}


class DatasetError(ValueError):
    pass


@dataclass
class Dataset:
    texts: list[str]
    labels: np.ndarray
    families: list[str]
    kinds: list[str]
    groups: list[str]
    buckets: list[str]
    provenance: str
    sha256: str

    def __len__(self) -> int:
        return len(self.texts)

    def features(self) -> np.ndarray:
        return np.array(
            [vector(text, kind) for text, kind in zip(self.texts, self.kinds, strict=True)]
        )

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)


def _dataset_hash(texts: list[str], labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for text, label in zip(texts, labels, strict=True):
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(int(label)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_verified_corpus() -> Dataset:
    """The project corpus; every document re-hashed against its run manifest."""
    from research.baseline_corpus import load_corpus

    records = load_corpus()
    texts = [record.text for record in records]
    labels = np.array([record.label for record in records])
    return Dataset(
        texts=texts,
        labels=labels,
        families=[record.family for record in records],
        kinds=[record.kind for record in records],
        # Group by prompt: the same prompt across models shares a topic, so
        # prompt groups keep topic leakage out of the cross-validation folds.
        groups=[record.prompt_id for record in records],
        buckets=[record.length_bucket for record in records],
        provenance="panoptes-verified-corpus",
        sha256=_dataset_hash(texts, labels),
    )


def _validate_row(row: dict, schema: dict, where: str) -> dict:
    import jsonschema

    try:
        jsonschema.validate(row, schema)
    except jsonschema.ValidationError as exc:
        raise DatasetError(f"{where}: {exc.message}") from exc
    return row


def load_user_dataset(path: Path) -> Dataset:
    """Load and schema-validate a community CSV or JSONL dataset."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"{path} does not exist")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    rows: list[dict] = []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "text" not in reader.fieldnames:
                raise DatasetError("CSV must have a 'text' column")
            for index, raw in enumerate(reader):
                row = {key: value for key, value in raw.items() if value not in (None, "")}
                rows.append(_validate_row(row, schema, f"row {index + 2}"))
    elif path.suffix.lower() in {".jsonl", ".json"}:
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"line {index + 1}: invalid JSON: {exc}") from exc
            rows.append(_validate_row(row, schema, f"line {index + 1}"))
    else:
        raise DatasetError("dataset must be .csv or .jsonl")

    if len(rows) < 8:
        raise DatasetError(f"dataset has {len(rows)} rows; at least 8 are required")

    texts = [row["text"] for row in rows]
    labels = np.array([_LABELS[row["label"]] for row in rows])
    families = [row.get("family", "unknown") for row in rows]
    kinds = [row.get("kind", "text") for row in rows]
    groups = [row.get("group", f"row-{index}") for index, row in enumerate(rows)]
    return Dataset(
        texts=texts,
        labels=labels,
        families=families,
        kinds=kinds,
        groups=groups,
        buckets=[length_bucket(len(word_tokens(text))) for text in texts],
        provenance=str(path),
        sha256=_dataset_hash(texts, labels),
    )


def grouped_splits(dataset: Dataset, n_splits: int = 5, seed: int = 13):
    """GroupKFold splits; falls back to fewer splits when groups are scarce."""
    from sklearn.model_selection import GroupKFold

    groups = np.array(dataset.groups)
    n_groups = len(set(dataset.groups))
    splits = max(2, min(n_splits, n_groups))
    splitter = GroupKFold(n_splits=splits)
    X = np.zeros((len(dataset), 1))  # placeholder; split only needs groups
    yield from splitter.split(X, dataset.labels, groups=groups)
