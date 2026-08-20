"""Dataset loading for the bench.

Sources:
  - the hash-verified baseline corpus (via bench.baseline_corpus)
  - user datasets (CSV or JSONL) validated against schemas/bench-dataset.schema.json
  - the Defactify_Text_Dataset (Roy et al. 2026, arXiv:2510.22874), fetched and
    hygiene-filtered locally by python -m bench.fetch_defactify (raw text gitignored)
  - RAID (Dugan et al. 2024), M4GT-Bench (Wang et al. 2024), EvoBench
    (ACL 2025 Findings), CoAuthor, and MAGE, fetched and hygiene-filtered locally by
    python -m bench.fetch_raid / fetch_m4gt / fetch_evobench / fetch_coauthor / fetch_mage
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
RAID_DIR = ROOT / "datasets" / "local" / "raid"
M4GT_DIR = ROOT / "datasets" / "local" / "m4gt"
EVOBENCH_DIR = ROOT / "datasets" / "local" / "evobench"
MAGE_DIR = ROOT / "datasets" / "local" / "mage"
COAUTHOR_DIR = ROOT / "datasets" / "local" / "coauthor"

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
    datasets: list[str] | None = None
    enforce_author_disjoint: bool = False
    _feature_cache: np.ndarray | None = field(default=None, repr=False, compare=False)

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
        if self.datasets is not None and len(self.datasets) != n:
            raise DatasetError("datasets must align with texts")

    def __len__(self) -> int:
        return len(self.texts)

    def features(self) -> np.ndarray:
        if self._feature_cache is None:
            self._feature_cache = np.array(
                [vector(text, kind) for text, kind in zip(self.texts, self.kinds, strict=True)]
            )
        return self._feature_cache

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)

    def subset(self, indices: np.ndarray | list[int]) -> Dataset:
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
            datasets=[self.datasets[i] for i in idx] if self.datasets is not None else None,
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
    from bench.baseline_corpus import load_corpus

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
    groups = [
        row.get("group") or row.get("source") or f"row-{index}" for index, row in enumerate(rows)
    ]
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
        for offset, (_dists, nbrs) in enumerate(zip(distances, indices, strict=True)):
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
    return _fetch_created_utc(DEFACTIFY_DIR)


def _fetch_created_utc(directory: Path) -> str | None:
    """Deterministic timestamp from a fetch manifest, when present."""
    manifest = directory / "fetch-manifest.json"
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text(encoding="utf-8")).get("created_utc")


def _fetch_manifest_sha256(directory: Path) -> str | None:
    manifest = directory / "fetch-manifest.json"
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text(encoding="utf-8")).get("artifact_sha256")


def _group_subsample_mask(groups: list[str], max_rows: int, seed: int) -> tuple[np.ndarray, dict]:
    """Deterministically select whole groups until `max_rows` is reached.

    Subsampling at the group level keeps every row that shares a leakage-
    control group in the evaluation cohort together.
    """
    unique = sorted(set(groups))
    keyed = sorted(unique, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    counts = {g: 0 for g in unique}
    for g in groups:
        counts[g] += 1
    chosen: set[str] = set()
    total = 0
    for g in keyed:
        if total + counts[g] > max_rows and chosen:
            break
        chosen.add(g)
        total += counts[g]
        if total >= max_rows:
            break
    mask = np.array([g in chosen for g in groups])
    info = {
        "max_rows": max_rows,
        "seed": seed,
        "n_groups_available": len(unique),
        "n_groups_selected": len(chosen),
        "n_rows_selected": int(total),
    }
    return mask, info


_RAID_FRAME_CACHE = None
_RAID_FRAME_CACHE_DIR = None


def _raid_frame():
    """Process-local cache of the full RAID clean parquet.

    The 4.3 GB parquet holds every attack arm; Phase 6 evaluates many attack
    cells, so reading it once and filtering per attack avoids repeated 12 GB
    transient reads. The cached frame is never mutated (``.loc`` copies). The
    cache is keyed on ``RAID_DIR`` so tests that monkeypatch the directory to a
    missing path still exercise the not-found branch.
    """
    global _RAID_FRAME_CACHE, _RAID_FRAME_CACHE_DIR
    if _RAID_FRAME_CACHE is None or _RAID_FRAME_CACHE_DIR != RAID_DIR:
        import pandas as pd

        path = RAID_DIR / "train-clean.parquet"
        if not path.exists():
            raise DatasetError(f"{path} missing; run python -m bench.fetch_raid first")
        _RAID_FRAME_CACHE = pd.read_parquet(path)
        _RAID_FRAME_CACHE_DIR = RAID_DIR
    return _RAID_FRAME_CACHE


def load_raid(attack: str = "none", max_rows: int = 150_000, seed: int = 13) -> Dataset:
    """Load the hygiene-filtered RAID train split, subsampled at the source_id level.

    `attack` selects the AI rows: "none" is the clean evaluation cohort;
    any other value yields human rows plus AI rows attacked with that method
    (the train-clean/test-attacked robustness cell). Human rows are always
    attack-free and always included.
    """
    frame = _raid_frame()
    is_human = frame["family"] == "human"
    frame = frame.loc[is_human | (frame["attack"] == attack)].reset_index(drop=True)
    rows_before = len(frame)

    groups = frame["group"].astype(str).tolist()
    subsample = {"max_rows": None, "n_rows_selected": rows_before}
    if rows_before > max_rows:
        mask, subsample = _group_subsample_mask(groups, max_rows, seed)
        frame = frame.loc[mask].reset_index(drop=True)
        groups = frame["group"].astype(str).tolist()

    texts = frame["text"].tolist()
    labels = np.array(frame["label"].tolist(), dtype=int)
    families = frame["family"].astype(str).tolist()
    domains = frame["domain"].astype(str).tolist()
    meta = {
        "attack": attack,
        "rows_before_subsample": int(rows_before),
        "subsample": subsample,
        "fetch_manifest_sha256": _fetch_manifest_sha256(RAID_DIR),
    }
    return Dataset(
        texts=texts,
        labels=labels,
        families=families,
        kinds=["text"] * len(texts),
        groups=groups,
        buckets=[length_bucket(len(word_tokens(text))) for text in texts],
        provenance=(
            f"raid (Dugan et al. 2024, arXiv:2405.07940; attack={attack}, "
            "hygiene-filtered, hash-pinned)"
        ),
        sha256=_dataset_hash(texts, labels),
        meta=meta,
        domains=domains,
    )


def raid_attacks() -> list[str]:
    """Attack values present in the clean RAID parquet (excluding 'none')."""
    import pandas as pd

    path = RAID_DIR / "train-clean.parquet"
    if not path.exists():
        raise DatasetError(f"{path} missing; run python -m bench.fetch_raid first")
    values = pd.read_parquet(path, columns=["attack"])["attack"].unique().tolist()
    return sorted(v for v in (str(v) for v in values) if v and v != "none")


def _load_m4gt_file(
    file_stem: str, provenance: str, max_rows: int | None = None, seed: int = 13
) -> Dataset:
    import pandas as pd

    path = M4GT_DIR / f"{file_stem}-clean.parquet"
    if not path.exists():
        raise DatasetError(f"{path} missing; run python -m bench.fetch_m4gt first")
    frame = pd.read_parquet(path)
    rows_before = len(frame)
    subsample = {"max_rows": None, "n_rows_selected": rows_before}
    if max_rows is not None and rows_before > max_rows:
        groups = frame["group"].astype(str).tolist()
        mask, subsample = _group_subsample_mask(groups, max_rows, seed)
        frame = frame.loc[mask].reset_index(drop=True)
    texts = frame["text"].tolist()
    labels = np.array(frame["label"].tolist(), dtype=int)
    return Dataset(
        texts=texts,
        labels=labels,
        families=frame["family"].astype(str).tolist(),
        kinds=["text"] * len(texts),
        groups=frame["group"].astype(str).tolist(),
        buckets=[length_bucket(len(word_tokens(text))) for text in texts],
        provenance=provenance,
        sha256=_dataset_hash(texts, labels),
        meta={
            "fetch_manifest_sha256": _fetch_manifest_sha256(M4GT_DIR),
            "rows_before_subsample": int(rows_before),
            "subsample": subsample,
        },
        domains=frame["domain"].astype(str).tolist(),
    )


def load_m4gt(max_rows: int | None = None, seed: int = 13) -> Dataset:
    """M4GT-Bench Subtask A (English, 5 domains, 6 generators)."""
    return _load_m4gt_file(
        "subtask_a",
        "m4gt (Wang et al. 2024, arXiv:2403.14822; English, hygiene-filtered, hash-pinned)",
        max_rows=max_rows,
        seed=seed,
    )


def load_m4gtml(max_rows: int | None = None, seed: int = 13) -> Dataset:
    """M4GT-Bench Subtask A multilingual (16 sources, 8 generators)."""
    return _load_m4gt_file(
        "subtask_a_multilingual",
        "m4gt-multilingual (Wang et al. 2024, arXiv:2403.14822; hygiene-filtered, hash-pinned)",
        max_rows=max_rows,
        seed=seed,
    )


def load_evobench(max_rows: int | None = None, seed: int = 13) -> Dataset:
    """EvoBench: families are LLM versions, so leave-one-family-out is the
    generator-generation shift experiment."""
    import pandas as pd

    path = EVOBENCH_DIR / "clean.parquet"
    if not path.exists():
        raise DatasetError(f"{path} missing; run python -m bench.fetch_evobench first")
    frame = pd.read_parquet(path)
    rows_before = len(frame)
    groups = frame["group"].astype(str).tolist()
    subsample = {"max_rows": None, "n_rows_selected": rows_before}
    if max_rows is not None and rows_before > max_rows:
        mask, subsample = _group_subsample_mask(groups, max_rows, seed)
        frame = frame.loc[mask].reset_index(drop=True)
    texts = frame["text"].tolist()
    labels = np.array(frame["label"].tolist(), dtype=int)
    return Dataset(
        texts=texts,
        labels=labels,
        families=frame["family"].astype(str).tolist(),
        kinds=["text"] * len(texts),
        groups=frame["group"].astype(str).tolist(),
        buckets=[length_bucket(len(word_tokens(text))) for text in texts],
        provenance="evobench (ACL 2025 Findings; hygiene-filtered, pinned-commit)",
        sha256=_dataset_hash(texts, labels),
        meta={
            "fetch_manifest_sha256": _fetch_manifest_sha256(EVOBENCH_DIR),
            "family_groups": sorted(frame["family_group"].astype(str).unique().tolist()),
            "rows_before_subsample": int(rows_before),
            "subsample": subsample,
        },
        domains=frame["domain"].astype(str).tolist(),
    )


COAUTHOR_SPLITS = ("train", "development", "calibration", "test")


def load_coauthor(split: str = "test", max_rows: int | None = None, seed: int = 13) -> Dataset:
    """Load an author-disjoint CoAuthor split with contribution-fraction ground truth.

    CoAuthor (Lee et al. 2022) is a ``mixed_task`` cohort: every session is
    human+GPT-3 collaborative, so ``label`` is 1 for every row and the dataset
    is NEVER a binary fully-AI positive set. The real ground truth is
    ``ai_contribution_fraction`` (ai_chars / (ai_chars + human_chars), prompt
    excluded), stored per-row in ``meta`` and aligned with ``texts``. Groups and
    authors are the Mechanical Turk ``worker_id``, so the four-way split is
    author-disjoint by construction (a writer never crosses the firewall).
    """
    import pandas as pd

    if split not in COAUTHOR_SPLITS:
        raise DatasetError(f"unknown CoAuthor split {split!r}; expected one of {COAUTHOR_SPLITS}")
    path = COAUTHOR_DIR / "coauthor.parquet"
    if not path.exists():
        raise DatasetError(f"{path} missing; run python -m bench.fetch_coauthor first")
    frame = pd.read_parquet(path)
    frame = frame.loc[frame["split"] == split].reset_index(drop=True)

    rows_before = len(frame)
    groups = frame["worker_id"].astype(str).tolist()
    subsample = {"max_rows": None, "n_rows_selected": rows_before}
    if max_rows is not None and rows_before > max_rows:
        mask, subsample = _group_subsample_mask(groups, max_rows, seed)
        frame = frame.loc[mask].reset_index(drop=True)
        groups = frame["worker_id"].astype(str).tolist()

    texts = frame["text"].tolist()
    labels = np.array(frame["label"].tolist(), dtype=int)
    prompts = frame["prompt_code"].astype(str).tolist()
    meta = {
        "split": split,
        "cohort_role": "mixed_task",
        "ai_contribution_fraction": [float(v) for v in frame["ai_contribution_fraction"].tolist()],
        "human_chars": [int(v) for v in frame["human_chars"].tolist()],
        "ai_chars": [int(v) for v in frame["ai_chars"].tolist()],
        "prompt_chars": [int(v) for v in frame["prompt_chars"].tolist()],
        "written_by_human_official": [
            float(v) for v in frame["written_by_human_official"].tolist()
        ],
        "rows_before_subsample": int(rows_before),
        "subsample": subsample,
        "fetch_manifest_sha256": _fetch_manifest_sha256(COAUTHOR_DIR),
    }
    return Dataset(
        texts=texts,
        labels=labels,
        families=["gpt-3-assisted"] * len(texts),
        kinds=["mixed_task"] * len(texts),
        groups=groups,
        buckets=[length_bucket(len(word_tokens(text))) for text in texts],
        provenance=(
            f"coauthor (Lee et al. 2022, arXiv:2201.06796; split={split}, "
            "author-disjoint, hash-pinned)"
        ),
        sha256=_dataset_hash(texts, labels),
        meta=meta,
        authors=groups,
        domains=prompts,
    )


MAGE_SPLITS = ("train", "valid", "test", "ood", "ood_para")


def load_mage(
    split: str = "test",
    domains: tuple[str, ...] | None = None,
    prompt_modes: tuple[str, ...] | None = None,
    max_rows: int | None = None,
    seed: int = 13,
) -> Dataset:
    """Load a hygiene-filtered MAGE split with near-duplicate leakage groups.

    `split` is one of train/valid/test (in-domain, 10 domains, 27 generators),
    `ood` (4 held-out domains, GPT-4), or `ood_para` (held-out domains plus a
    paraphrase attack). Groups are MinHash-LSH near-duplicate clusters built
    across all splits jointly, so a human source and its machine continuation
    never cross partitions. `max_rows` subsamples whole groups deterministically.
    """
    import pandas as pd

    if split not in MAGE_SPLITS:
        raise DatasetError(f"unknown MAGE split {split!r}; expected one of {MAGE_SPLITS}")
    path = MAGE_DIR / f"{split}-clean.parquet"
    if not path.exists():
        raise DatasetError(f"{path} missing; run python -m bench.fetch_mage first")
    frame = pd.read_parquet(path)

    if domains is not None:
        frame = frame.loc[frame["domain"].isin(list(domains))].reset_index(drop=True)
    if prompt_modes is not None:
        frame = frame.loc[frame["prompt_mode"].isin(list(prompt_modes))].reset_index(drop=True)

    rows_before = len(frame)
    groups = frame["group"].astype(str).tolist()
    subsample = {"max_rows": None, "n_rows_selected": rows_before}
    if max_rows is not None and rows_before > max_rows:
        mask, subsample = _group_subsample_mask(groups, max_rows, seed)
        frame = frame.loc[mask].reset_index(drop=True)
        groups = frame["group"].astype(str).tolist()

    texts = frame["text"].tolist()
    labels = np.array(frame["label"].tolist(), dtype=int)
    families = frame["family"].astype(str).tolist()
    domains_list = frame["domain"].astype(str).tolist()
    meta = {
        "split": split,
        "prompt_modes": sorted(frame["prompt_mode"].astype(str).unique().tolist()),
        "paraphrased_rows": int(frame["paraphrased"].sum()),
        "rows_before_subsample": int(rows_before),
        "subsample": subsample,
        "fetch_manifest_sha256": _fetch_manifest_sha256(MAGE_DIR),
    }
    return Dataset(
        texts=texts,
        labels=labels,
        families=families,
        kinds=["text"] * len(texts),
        groups=groups,
        buckets=[length_bucket(len(word_tokens(text))) for text in texts],
        provenance=(
            f"mage (Li et al. 2023, arXiv:2305.13242; split={split}, hygiene-filtered, hash-pinned)"
        ),
        sha256=_dataset_hash(texts, labels),
        meta=meta,
        domains=domains_list,
    )


def load_defactify(
    splits: tuple[str, ...] = ("train", "validation", "test"),
    threshold: float = 0.45,
    split: str | None = None,
    max_rows: int | None = None,
    seed: int = 13,
) -> Dataset:
    """Load the hygiene-filtered Defactify splits with reconstructed story groups.

    Groups are reconstructed jointly across splits so the leakage audit can
    detect stories that the official split places in more than one partition.
    ``split`` (one of train/validation/test) subsets to a single official
    partition after joint group reconstruction; ``max_rows`` then applies a
    deterministic group-stratified subsample.
    """
    import pandas as pd

    frames: list = []
    for split in splits:
        path = DEFACTIFY_DIR / f"{split}-clean.parquet"
        if not path.exists():
            raise DatasetError(f"{path} missing; run python -m bench.fetch_defactify first")
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
        if candidate.get("dataset_sha256") == dataset_hash and candidate.get("splits") == list(
            splits
        ):
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
                "official_split_story_leakage_rate": float(leaked.mean())
                if test_mask.sum()
                else 0.0,
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

    if split is not None:
        if split not in splits:
            raise DatasetError(f"split {split!r} not among loaded splits {splits}")
        keep = np.where(split_arr == split)[0]
        texts = [texts[int(i)] for i in keep]
        labels = labels[keep]
        families = [families[int(i)] for i in keep]
        groups = [groups[int(i)] for i in keep]

    rows_before = len(texts)
    subsample = {"max_rows": None, "n_rows_selected": rows_before}
    if max_rows is not None and rows_before > max_rows:
        mask, subsample = _group_subsample_mask(groups, max_rows, seed)
        keep_idx = np.where(mask)[0]
        texts = [texts[int(i)] for i in keep_idx]
        labels = labels[keep_idx]
        families = [families[int(i)] for i in keep_idx]
        groups = [groups[int(i)] for i in keep_idx]

    fetch_manifest = DEFACTIFY_DIR / "fetch-manifest.json"
    meta = {
        "group_reconstruction": group_stats,
        "leakage_audit": audit,
        "split": split,
        "rows_before_subsample": int(rows_before),
        "subsample": subsample,
        "fetch_manifest_sha256": (
            json.loads(fetch_manifest.read_text(encoding="utf-8"))["artifact_sha256"]
            if fetch_manifest.exists()
            else None
        ),
        "official_split_counts": {s: int((split_arr == s).sum()) for s in splits},
    }
    return Dataset(
        texts=texts,
        labels=labels,
        families=families,
        kinds=["text"] * len(texts),
        groups=groups,
        buckets=[length_bucket(len(word_tokens(text))) for text in texts],
        provenance=(
            "defactify-text (Roy et al. 2026, arXiv:2510.22874; hygiene-filtered, hash-pinned)"
        ),
        sha256=dataset_hash if split is None and max_rows is None else _dataset_hash(texts, labels),
        meta=meta,
        domains=["journalism-nyt"] * len(texts),
    )


# --- v2.1 pooled public-weight training pool -----------------------------------

#: Cohorts whose licenses permit model training and redistribution of derived
#: weights (protocol v2.1 "Data roles"). EvoBench (no license) and M4GT
#: (NOASSERTION, weak pairing groups) are evaluation-only and never enter the pool.
POOL_TRAIN_COHORTS = ("mage", "raid", "defactify")


def combine_datasets(named: list[tuple[str, Dataset]], provenance: str) -> Dataset:
    """Combine named cohorts into one pooled Dataset.

    Each cohort's leakage groups are prefixed with its dataset id so a group
    never spans cohorts (cross-cohort group disjointness), and a per-row
    ``datasets`` label records cohort identity for the ``dataset`` cohort axis
    (leave-one-cohort-out over datasets) and the pooled balanced/GroupDRO
    objective. Dataset IDs remain sampling/audit metadata, never model inputs.
    """
    texts: list[str] = []
    families: list[str] = []
    kinds: list[str] = []
    groups: list[str] = []
    buckets: list[str] = []
    domains: list[str] = []
    datasets: list[str] = []
    label_parts: list[np.ndarray] = []
    for name, ds in named:
        n = len(ds)
        texts.extend(ds.texts)
        label_parts.append(np.asarray(ds.labels, dtype=int))
        families.extend(ds.families)
        kinds.extend(ds.kinds)
        groups.extend(f"{name}:{g}" for g in ds.groups)
        buckets.extend(ds.buckets)
        domains.extend(list(ds.domains) if ds.domains is not None else list(ds.kinds))
        datasets.extend([name] * n)
    labels = np.concatenate(label_parts) if label_parts else np.array([], dtype=int)
    return Dataset(
        texts=texts,
        labels=labels,
        families=families,
        kinds=kinds,
        groups=groups,
        buckets=buckets,
        provenance=provenance,
        sha256=_dataset_hash(texts, labels),
        meta={"cohorts": {name: int(len(ds)) for name, ds in named}},
        domains=domains,
        datasets=datasets,
    )


def concat_partitions(parts: list[Dataset], provenance: str) -> Dataset:
    """Concatenate already-group-prefixed pooled partitions without re-prefixing.

    Unlike :func:`combine_datasets`, the inputs are partitions of the *same*
    pooled pool (e.g. pooled train + pooled calibration) whose leakage groups
    are already prefixed and mutually group-disjoint, so groups are concatenated
    verbatim. The per-row ``datasets`` label is preserved so the combined pool
    still supports the ``dataset`` cohort axis (leave-one-dataset-out). The
    sealed pooled test partition is never passed here.
    """
    texts: list[str] = []
    families: list[str] = []
    kinds: list[str] = []
    groups: list[str] = []
    buckets: list[str] = []
    domains: list[str] = []
    datasets: list[str] = []
    label_parts: list[np.ndarray] = []
    cohorts: dict[str, int] = {}
    for ds in parts:
        n = len(ds)
        texts.extend(ds.texts)
        label_parts.append(np.asarray(ds.labels, dtype=int))
        families.extend(ds.families)
        kinds.extend(ds.kinds)
        groups.extend(ds.groups)
        buckets.extend(ds.buckets)
        domains.extend(list(ds.domains) if ds.domains is not None else list(ds.kinds))
        row_ds = list(ds.datasets) if ds.datasets is not None else [ds.provenance] * n
        datasets.extend(row_ds)
        for d in row_ds:
            cohorts[d] = cohorts.get(d, 0) + 1
    labels = np.concatenate(label_parts) if label_parts else np.array([], dtype=int)
    return Dataset(
        texts=texts,
        labels=labels,
        families=families,
        kinds=kinds,
        groups=groups,
        buckets=buckets,
        provenance=provenance,
        sha256=_dataset_hash(texts, labels),
        meta={"cohorts": cohorts, "n_partitions": len(parts)},
        domains=domains,
        datasets=datasets,
    )


def _group_partition_indices(
    groups: list[str], fracs: tuple[float, ...], seed: int
) -> list[np.ndarray]:
    """Deterministic group-disjoint partition into ``len(fracs)`` index arrays.

    Whole leakage groups are assigned to partitions in a seeded hash order so a
    group never crosses partitions. ``fracs`` need not sum to 1 (normalized).
    """
    uniq = sorted(set(str(g) for g in groups))
    keyed = sorted(uniq, key=lambda g: hashlib.sha256(f"{seed}:partition:{g}".encode()).hexdigest())
    n = len(keyed)
    total = float(sum(fracs))
    cuts = [0]
    acc = 0.0
    for f in fracs[:-1]:
        acc += f
        cuts.append(int(round(n * acc / total)))
    cuts.append(n)
    part_of: dict[str, int] = {}
    for pi in range(len(fracs)):
        for g in keyed[cuts[pi] : cuts[pi + 1]]:
            part_of[g] = pi
    part_arr = np.array([part_of[str(g)] for g in groups], dtype=int)
    return [np.where(part_arr == pi)[0] for pi in range(len(fracs))]


def _exclude_groups(ds: Dataset, exclude: set[str]) -> Dataset:
    """Subset ``ds`` to rows whose (unprefixed) leakage group is not in ``exclude``."""
    if not exclude:
        return ds
    keep = np.array([str(g) not in exclude for g in ds.groups])
    return ds.subset(np.where(keep)[0])


def _cap_rows(ds: Dataset, max_rows: int | None, seed: int) -> Dataset:
    """Deterministic group-stratified cap on rows (no-op when small enough)."""
    if max_rows is None or len(ds) <= max_rows:
        return ds
    mask, _ = _group_subsample_mask([str(g) for g in ds.groups], max_rows, seed)
    return ds.subset(np.where(mask)[0])


def load_pooled_partitions(
    train_rows: int,
    cal_rows: int,
    test_rows: int | None = None,
    seed: int = 13,
    cohorts: tuple[str, ...] = POOL_TRAIN_COHORTS,
    mage_cal_exclude: set[str] | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Pooled training pool, held-out calibration, and held-out test (protocol v2.1).

    Training cohorts (license permits derived weights): MAGE, RAID clean,
    DeFactify. Partition strategy per cohort, always group-disjoint:

      - MAGE uses its official train/valid/test splits; because those splits
        share a small number of near-duplicate clusters, calibration and test
        rows whose leakage group appears in an earlier partition are dropped.
      - DeFactify's official splits are story-leaky by construction (AI rewrites
        of the same NYT stories span partitions), so all rows are re-split by
        reconstructed story group.
      - RAID clean has no official evaluation split, so it is re-split by
        source_id group; RAID attack cells remain a separate robustness eval.

    ``mage_cal_exclude`` drops additional MAGE-valid groups (the pilot's
    development subsample) so calibration stays disjoint from development.
    Returns ``(pooled_train, pooled_calibration, pooled_test)``.
    """
    test_rows = test_rows if test_rows is not None else cal_rows
    train_parts: list[tuple[str, Dataset]] = []
    cal_parts: list[tuple[str, Dataset]] = []
    test_parts: list[tuple[str, Dataset]] = []

    if "mage" in cohorts:
        tr = load_mage(split="train", max_rows=train_rows, seed=seed)
        tr_groups = set(tr.groups)
        cal = load_mage(split="valid", max_rows=cal_rows * 2, seed=seed)
        cal = _exclude_groups(cal, tr_groups | (mage_cal_exclude or set()))
        cal = _cap_rows(cal, cal_rows, seed)
        te = load_mage(split="test", max_rows=test_rows * 2, seed=seed)
        te = _exclude_groups(te, tr_groups | set(cal.groups))
        te = _cap_rows(te, test_rows, seed)
        train_parts.append(("mage", tr))
        cal_parts.append(("mage", cal))
        test_parts.append(("mage", te))

    if "defactify" in cohorts:
        pool = load_defactify(max_rows=train_rows + cal_rows + test_rows, seed=seed)
        tr_i, cal_i, te_i = _group_partition_indices(
            pool.groups, (train_rows, cal_rows, test_rows), seed
        )
        train_parts.append(("defactify", pool.subset(tr_i)))
        cal_parts.append(("defactify", pool.subset(cal_i)))
        test_parts.append(("defactify", pool.subset(te_i)))

    if "raid" in cohorts:
        pool = load_raid(attack="none", max_rows=train_rows + cal_rows + test_rows, seed=seed)
        tr_i, cal_i, te_i = _group_partition_indices(
            pool.groups, (train_rows, cal_rows, test_rows), seed
        )
        train_parts.append(("raid", pool.subset(tr_i)))
        cal_parts.append(("raid", pool.subset(cal_i)))
        test_parts.append(("raid", pool.subset(te_i)))

    pooled_train = combine_datasets(
        train_parts, "pooled-train-v2.1 (mage+raid-clean+defactify; derived-weight-licensed)"
    )
    pooled_cal = combine_datasets(
        cal_parts, "pooled-calibration-v2.1 (group-disjoint from pooled train)"
    )
    pooled_test = combine_datasets(
        test_parts, "pooled-test-v2.1 (seen-cohort holdout, group-disjoint from train/cal)"
    )
    return pooled_train, pooled_cal, pooled_test
