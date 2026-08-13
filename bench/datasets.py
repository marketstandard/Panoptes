"""Dataset loading for the bench.

Three sources:
  - the hash-verified baseline corpus (via research/baseline_corpus)
  - user datasets (CSV or JSONL) validated against schemas/bench-dataset.schema.json
  - the Defactify_Text_Dataset (Roy et al. 2026, arXiv:2510.22874), fetched and
    hygiene-filtered locally by research/fetch_defactify.py (raw text gitignored)
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.features import FEATURE_NAMES, length_bucket, vector, word_tokens  # noqa: E402

SCHEMA_PATH = ROOT / "schemas" / "bench-dataset.schema.json"
DEFACTIFY_DIR = ROOT / "datasets" / "local" / "defactify"

_LABELS = {0: 0, 1: 1, "0": 0, "1": 1, "human": 0, "ai": 1}

DEFACTIFY_FAMILIES = {
    "Human_Story": "human",
    "Gemma-2-9B": "gemma-2-9b",
    "Mistral-7B": "mistral-7b",
    "Qwen-2-72B": "qwen-2-72b",
    "Llama-8B": "llama-8b",
    "Yi-Large": "yi-large",
    "GPT-4o": "gpt-4o",
}


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
    meta: dict = field(default_factory=dict)
    authors: list[str] | None = None
    domains: list[str] | None = None
    enforce_author_disjoint: bool = False

    def __post_init__(self) -> None:
        n = len(self.texts)
        if self.authors is None:
            self.authors = [
                "human" if int(label) == 0 else f"generator:{family}"
                for label, family in zip(self.labels, self.families, strict=True)
            ]
            self.enforce_author_disjoint = False
        else:
            self.enforce_author_disjoint = any(str(a).strip() for a in self.authors)
        if not self.domains:
            self.domains = list(self.kinds)
        if self.authors is None or self.domains is None:
            raise DatasetError("authors and domains must be populated")
        if len(self.authors) != n or len(self.domains) != n:
            raise DatasetError("authors and domains must align with texts")

    def __len__(self) -> int:
        return len(self.texts)

    def features(self) -> np.ndarray:
        return np.array(
            [vector(text, kind) for text, kind in zip(self.texts, self.kinds, strict=True)]
        )

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)

    def subset(self, indices: np.ndarray | list[int]) -> "Dataset":
        idx = [int(i) for i in indices]
        return Dataset(
            texts=[self.texts[i] for i in idx],
            labels=np.asarray(self.labels)[idx],
            families=[self.families[i] for i in idx],
            kinds=[self.kinds[i] for i in idx],
            groups=[self.groups[i] for i in idx],
            buckets=[self.buckets[i] for i in idx],
            provenance=self.provenance,
            sha256=self.sha256,
            meta=dict(self.meta),
            authors=(
                [self.authors[i] for i in idx]
                if self.enforce_author_disjoint and self.authors is not None
                else None
            ),
            domains=[self.domains[i] for i in idx] if self.domains is not None else None,
        )


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
    groups = [row.get("group") or row.get("source") or f"row-{index}" for index, row in enumerate(rows)]
    authors = [row.get("author", "") for row in rows]
    domains = [row.get("domain", "") for row in rows]
    return Dataset(
        texts=texts,
        labels=labels,
        families=families,
        kinds=kinds,
        groups=groups,
        buckets=[length_bucket(len(word_tokens(text))) for text in texts],
        provenance=str(path),
        sha256=_dataset_hash(texts, labels),
        authors=authors if any(authors) else None,
        domains=domains if any(domains) else None,
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


def reconstruct_story_groups(
    texts: list[str], threshold: float = 0.45, max_features: int = 60000, chunk: int = 4000
) -> tuple[list[str], dict]:
    """Rebuild Defactify's story groups via TF-IDF near-duplicate clustering.

    Each human story in the dataset has up to six LLM rewrites, and the
    upstream documentation does not state whether the official splits are
    story-disjoint. Rows whose TF-IDF cosine similarity meets `threshold`
    are unioned into one group; every group is one underlying story. This
    keeps near-duplicate rewrites out of different cross-validation folds.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True
    )
    tfidf = vectorizer.fit_transform(texts)
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    neighbors.fit(tfidf)

    parent = list(range(len(texts)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    radius = 1.0 - threshold
    edges = 0
    for start in range(0, len(texts), chunk):
        distances, indices = neighbors.radius_neighbors(tfidf[start : start + chunk], radius=radius)
        for offset, (dists, nbrs) in enumerate(zip(distances, indices, strict=True)):
            i = start + offset
            for j in nbrs:
                j = int(j)
                if j == i:
                    continue
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
                    edges += 1

    roots: dict[int, str] = {}
    groups: list[str] = []
    for i in range(len(texts)):
        root = find(i)
        if root not in roots:
            roots[root] = f"story-{len(roots)}"
        groups.append(roots[root])

    _, sizes = np.unique(np.array(groups), return_counts=True)
    stats = {
        "threshold": threshold,
        "n_groups": len(roots),
        "edges": edges,
        "group_size_mean": float(sizes.mean()),
        "group_size_median": float(np.median(sizes)),
        "group_size_max": int(sizes.max()),
        "singletons": int((sizes == 1).sum()),
    }
    return groups, stats


def defactify_created_utc() -> str | None:
    """Deterministic timestamp for Defactify-derived artifacts."""
    manifest = DEFACTIFY_DIR / "fetch-manifest.json"
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text(encoding="utf-8")).get("created_utc")


def load_defactify(
    splits: tuple[str, ...] = ("train", "validation", "test"), threshold: float = 0.45
) -> Dataset:
    """Load the hygiene-filtered Defactify splits with reconstructed story groups.

    Groups are reconstructed jointly across splits so the leakage audit can
    detect stories that the official split places in more than one partition.
    """
    import pandas as pd

    frames: list = []
    for split in splits:
        path = DEFACTIFY_DIR / f"{split}-clean.parquet"
        if not path.exists():
            raise DatasetError(f"{path} missing; run python research/fetch_defactify.py first")
        frame = pd.read_parquet(path)
        frame["official_split"] = split
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    texts = data["text"].tolist()
    labels = np.array(data["label"].tolist(), dtype=int)
    unknown = sorted(set(data["family"]) - set(DEFACTIFY_FAMILIES))
    if unknown:
        raise DatasetError(f"unmapped Defactify families: {unknown}")
    families = [DEFACTIFY_FAMILIES[f] for f in data["family"]]

    dataset_hash = _dataset_hash(texts, labels)
    cache_path = DEFACTIFY_DIR / f"story-groups-t{threshold}.json"
    cached = None
    if cache_path.exists():
        candidate = json.loads(cache_path.read_text(encoding="utf-8"))
        if candidate.get("dataset_sha256") == dataset_hash and candidate.get("splits") == list(splits):
            cached = candidate
    if cached is not None:
        groups = cached["groups"]
        group_stats = cached["group_reconstruction"]
        audit = cached["leakage_audit"]
    else:
        groups, group_stats = reconstruct_story_groups(texts, threshold=threshold)
        split_arr = data["official_split"].to_numpy()
        group_arr = np.array(groups)
        audit = {}
        if "train" in splits and "test" in splits:
            train_groups = set(group_arr[split_arr == "train"])
            test_mask = split_arr == "test"
            leaked = np.isin(group_arr[test_mask], list(train_groups))
            audit = {
                "official_test_rows": int(test_mask.sum()),
                "official_test_rows_with_train_near_duplicate": int(leaked.sum()),
                "official_split_story_leakage_rate": float(leaked.mean()) if test_mask.sum() else 0.0,
            }
        cache_path.write_text(
            json.dumps(
                {
                    "dataset_sha256": dataset_hash,
                    "splits": list(splits),
                    "groups": groups,
                    "group_reconstruction": group_stats,
                    "leakage_audit": audit,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    split_arr = data["official_split"].to_numpy()

    fetch_manifest = DEFACTIFY_DIR / "fetch-manifest.json"
    meta = {
        "group_reconstruction": group_stats,
        "leakage_audit": audit,
        "fetch_manifest_sha256": (
            json.loads(fetch_manifest.read_text(encoding="utf-8"))["artifact_sha256"]
            if fetch_manifest.exists()
            else None
        ),
        "official_split_counts": {split: int((split_arr == split).sum()) for split in splits},
    }
    return Dataset(
        texts=texts,
        labels=labels,
        families=families,
        kinds=["text"] * len(texts),
        groups=groups,
        buckets=[length_bucket(len(word_tokens(text))) for text in texts],
        provenance="defactify-text (Roy et al. 2026, arXiv:2510.22874; hygiene-filtered, hash-pinned)",
        sha256=dataset_hash,
        meta=meta,
        domains=["journalism-nyt"] * len(texts),
    )
