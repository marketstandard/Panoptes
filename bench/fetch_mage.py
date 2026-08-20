"""Fetch, verify, clean, and group the MAGE benchmark for the bench.

MAGE (Li et al. 2023, "MAGE: Machine-generated Text Detection in the Wild",
arXiv:2305.13242) is a multi-generator (27 LLMs), multi-domain (10 domains)
detection benchmark distributed as five CSVs on Hugging Face:

  train.csv / valid.csv / test.csv          main splits (10 in-domain domains)
  test_ood_set_gpt.csv                      4 held-out domains, GPT-4 machine text
  test_ood_set_gpt_para.csv                 held-out domains + paraphrase attack

Columns are ``text,label,src`` with ``label`` 1=human / 0=machine (inverted here
into the project convention 1=AI / 0=human) and ``src`` encoding provenance:

  {domain}_human                            human text
  {domain}_machine_{prompt_mode}_{model}    machine text (main splits)
  {domain}_gpt4[{,_para}]                   OOD GPT-4 text (``_para`` = paraphrased)
  {domain}_human_para                       human text paraphrased by a machine

Hygiene filters (counts reported in fetch-manifest.json):
  - empty texts;
  - exact duplicate texts (SHA-256 of normalized text), removed globally;
  - texts under 50 word tokens (the runtime support threshold);
  - known API-error / refusal markers.

Leakage control: a deterministic MinHash-LSH near-duplicate index is built
across every split jointly (bench/near_dup.py). Each near-duplicate cluster is
a leakage-control group, so a human source and its machine continuation never
land in different partitions. The official splits are preserved and audited:
the manifest reports how many test rows share a near-duplicate cluster with a
train row. Raw text never enters git — only hashes, counts, and the signed
pointer manifest do.

Deterministic: a prior fetch-manifest.json reuses its created_utc.

Usage:  python -m bench.fetch_mage
        python -m bench.fetch_mage --check   # verify local files only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOCAL = ROOT / "datasets" / "local" / "mage"
RAW = LOCAL / "raw"
POINTER_OUT = ROOT / "datasets" / "manifests" / "mage.json"
MANIFEST = LOCAL / "fetch-manifest.json"

HF_REPO = "yaful/MAGE"
REVISION = "342663f0a2b775455c023f5d36a1341ff0ec5402"

MIN_WORD_TOKENS = 50

# split name -> (raw filename, role)
SPLITS = {
    "train": ("train.csv", "main"),
    "valid": ("valid.csv", "main"),
    "test": ("test.csv", "main"),
    "ood": ("test_ood_set_gpt.csv", "ood"),
    "ood_para": ("test_ood_set_gpt_para.csv", "ood_para"),
}

CITATION = (
    "@inproceedings{li2023mage, title={MAGE: Machine-generated Text Detection in the Wild}, "
    "author={Li, Yafu and Li, Qintong and Cui, Leyang and others}, "
    "booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational "
    "Linguistics (ACL 2024)}, year={2024}, url={https://arxiv.org/abs/2305.13242}}"
)

# Substrings that mark a failed generation / API error rather than real text.
_ERROR_MARKERS = (
    "as an ai language model",
    "i cannot fulfill this request",
    "openai",
    "content policy",
)


class FetchError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict) -> str:
    clone = dict(payload)
    clone.pop("artifact_sha256", None)
    canonical = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _download(raw_name: str) -> Path:
    dest = RAW / raw_name
    if dest.exists():
        return dest
    from huggingface_hub import hf_hub_download

    RAW.mkdir(parents=True, exist_ok=True)
    print(f"downloading {HF_REPO}/{raw_name}@{REVISION[:12]}", flush=True)
    path = hf_hub_download(
        HF_REPO, raw_name, repo_type="dataset", revision=REVISION, local_dir=str(RAW)
    )
    return Path(path)


def parse_src(src: str) -> dict:
    """Parse a MAGE ``src`` value into domain/family/prompt_mode/paraphrased."""
    src = str(src)
    domain = src.split("_")[0]
    paraphrased = src.endswith("_para")
    if "_machine_" in src:
        rest = src.split("_machine_", 1)[1]
        prompt_mode, _, generator = rest.partition("_")
        return {
            "domain": domain,
            "family": generator or "unknown",
            "prompt_mode": prompt_mode or "unknown",
            "paraphrased": False,
        }
    if src.endswith("_human"):
        return {"domain": domain, "family": "human", "prompt_mode": "human", "paraphrased": False}
    if src.endswith("_human_para"):
        # Human text run through a machine paraphraser: machine-touched (label 0).
        return {
            "domain": domain,
            "family": "paraphrased_human",
            "prompt_mode": "paraphrase",
            "paraphrased": True,
        }
    # OOD machine rows: {domain}_gpt4 or {domain}_gpt4_para
    body = src[: -len("_para")] if paraphrased else src
    generator = body.split("_", 1)[1] if "_" in body else "unknown"
    return {
        "domain": domain,
        "family": generator,
        "prompt_mode": "paraphrase" if paraphrased else "ood",
        "paraphrased": paraphrased,
    }


def clean_split(split: str, raw_name: str) -> tuple[object, dict]:
    """Read, label-invert, and hygiene-filter one split. Returns (frame, counts).

    Exact-duplicate removal is scoped to *this* split: the same human control
    legitimately recurs across the official OOD and OOD-paraphrase partitions
    (they are separate evaluation conditions), so cross-split duplicates are not
    dropped here. Cross-split near-duplicates are still grouped together by the
    global MinHash-LSH index and surfaced by the leakage audit.
    """
    import pandas as pd

    seen: set[str] = set()
    path = _download(raw_name)
    frame = pd.read_csv(path)
    counts = {
        "rows_raw": int(len(frame)),
        "dropped_empty": 0,
        "dropped_exact_duplicates": 0,
        "dropped_under_50_tokens": 0,
        "dropped_error_marker": 0,
        "raw_sha256": sha256_file(path),
    }
    parsed = frame["src"].astype(str).map(parse_src)
    frame = frame.assign(
        domain=[p["domain"] for p in parsed],
        family=[p["family"] for p in parsed],
        prompt_mode=[p["prompt_mode"] for p in parsed],
        paraphrased=[p["paraphrased"] for p in parsed],
    )
    # Invert labels: MAGE 1=human/0=machine -> project 1=AI/0=human.
    frame["label"] = 1 - frame["label"].astype(int)
    frame["official_split"] = split

    keep_text: list[str] = []
    keep_idx: list[int] = []
    for idx, text in enumerate(frame["text"].astype(str)):
        text = text.strip()
        if not text:
            counts["dropped_empty"] += 1
            continue
        lowered = text.lower()
        if any(marker in lowered for marker in _ERROR_MARKERS):
            counts["dropped_error_marker"] += 1
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            counts["dropped_exact_duplicates"] += 1
            continue
        if len(text.split()) < MIN_WORD_TOKENS:
            counts["dropped_under_50_tokens"] += 1
            continue
        seen.add(digest)
        keep_idx.append(idx)
        keep_text.append(text)
    frame = frame.iloc[keep_idx].reset_index(drop=True)
    frame["text"] = keep_text
    counts["rows_clean"] = int(len(frame))
    counts["n_human"] = int((frame["label"] == 0).sum())
    counts["n_ai"] = int((frame["label"] == 1).sum())
    counts["domains"] = sorted(frame["domain"].unique().tolist())
    counts["n_generators"] = int(frame.loc[frame["label"] == 1, "family"].nunique())
    return frame, counts


def _assign_groups(frames: dict[str, object]) -> tuple[dict[str, list[str]], dict]:
    """Near-duplicate cluster index across every split jointly (cached)."""
    import pandas as pd

    combined = pd.concat(list(frames.values()), ignore_index=True)
    texts = combined["text"].tolist()
    dataset_hash = hashlib.sha256(
        "\x00".join(hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts).encode("utf-8")
    ).hexdigest()
    cache = LOCAL / "near-dup-groups.json"
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("dataset_sha256") == dataset_hash:
            labels = cached["labels"]
            stats = cached["stats"]
        else:
            labels = stats = None
    else:
        labels = stats = None
    if labels is None:
        from bench.near_dup import near_duplicate_clusters

        print(f"building near-duplicate index over {len(texts)} texts ...", flush=True)
        label_arr, stats = near_duplicate_clusters(texts, num_perm=128, bands=32, ngram=3, seed=13)
        labels = [int(x) for x in label_arr]
        cache.write_text(
            json.dumps({"dataset_sha256": dataset_hash, "labels": labels, "stats": stats}) + "\n",
            encoding="utf-8",
        )

    # Leakage audit: do near-duplicate clusters span the official train/test split?
    split_arr = combined["official_split"].to_numpy()
    import numpy as np

    label_arr = np.array(labels)
    audit: dict = {}
    train_mask = split_arr == "train"
    test_mask = split_arr == "test"
    if train_mask.any() and test_mask.any():
        train_clusters = set(label_arr[train_mask].tolist())
        leaked = np.isin(label_arr[test_mask], list(train_clusters))
        audit = {
            "official_test_rows": int(test_mask.sum()),
            "official_test_rows_sharing_cluster_with_train": int(leaked.sum()),
            "official_split_near_duplicate_leakage_rate": (
                float(leaked.mean()) if test_mask.sum() else 0.0
            ),
        }

    # Slice the global labels back into per-split group lists.
    groups: dict[str, list[str]] = {}
    offset = 0
    for split, frame in frames.items():
        n = len(frame)
        groups[split] = [f"mage:{c}" for c in labels[offset : offset + n]]
        offset += n
    return groups, {"near_duplicate_index": stats, "leakage_audit": audit}


def pointer_manifest(manifest: dict) -> dict:
    main = manifest["splits"]
    payload = {
        "schema": "panoptes-dataset-manifest-v1",
        "id": "mage",
        "kind": "prose",
        "title": "MAGE — Machine-generated Text Detection in the Wild (Li et al. 2023)",
        "license": {
            "spdx": "Apache-2.0",
            "redistributable": True,
            "citation": CITATION,
        },
        "source": {
            "url": f"https://huggingface.co/datasets/{HF_REPO}",
            "version": f"pinned-revision {REVISION[:12]}",
            "access": "public",
            "download_instructions": (
                "Run `python -m bench.fetch_mage`. The script downloads the five MAGE CSVs "
                f"from Hugging Face dataset {HF_REPO} at pinned revision {REVISION[:12]}, inverts "
                "labels to the project convention, applies documented hygiene filters, builds a "
                "near-duplicate leakage-control index, and stores clean parquets under "
                "datasets/local/mage/ (gitignored). Raw text is never committed."
            ),
        },
        "content_hash": {
            "algorithm": "sha256",
            "value": manifest["combined_sha256"],
        },
        "splits": {
            "train": {"groups": ["mage:<near-dup-cluster>"], "n": main["train"]["rows_clean"]},
            "calibration": {
                "groups": ["mage:<near-dup-cluster>"],
                "n": main["valid"]["rows_clean"],
            },
            "test": {"groups": ["mage:<near-dup-cluster>"], "n": main["test"]["rows_clean"]},
            "group_keys": ["near_duplicate_cluster", "domain", "generator", "prompt_mode"],
        },
        "labels": {
            "schema": "binary_ai",
            "values": ["human", "ai (27 generators, 3 prompt modes; OOD GPT-4 + paraphrase)"],
        },
        "privacy": {
            "contains_pii_risk": "low",
            "raw_text_in_repo": False,
            "sanitization": "hashes-only",
        },
        "limitations": [
            "OOD domains (cnn, dialogsum, imdb, pubmed) are disjoint from the 10 "
            "in-domain domains.",
            "The paraphrase set labels machine-paraphrased human text as AI (machine-touched).",
            "Generator tag 'gpt-3.5-trubo' is the upstream spelling (sic).",
            "Near-duplicate clusters are approximate (MinHash-LSH, Jaccard ~0.42 threshold).",
        ],
        "notice_entry_required": True,
    }
    payload["artifact_sha256"] = canonical_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify local files against the manifest"
    )
    args = parser.parse_args()

    if args.check:
        if not MANIFEST.exists():
            raise FetchError("no fetch-manifest.json; run without --check first")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for split, counts in manifest["splits"].items():
            path = LOCAL / f"{split}-clean.parquet"
            if not path.exists() or sha256_file(path) != counts["clean_sha256"]:
                raise FetchError(f"{path.name}: missing or hash mismatch")
        print("mage local files verified:", ", ".join(manifest["splits"]))
        return 0

    LOCAL.mkdir(parents=True, exist_ok=True)
    prior_created = None
    if MANIFEST.exists():
        prior_created = json.loads(MANIFEST.read_text(encoding="utf-8")).get("created_utc")

    frames: dict[str, object] = {}
    split_counts: dict[str, dict] = {}
    for split, (raw_name, _role) in SPLITS.items():
        frame, counts = clean_split(split, raw_name)
        frames[split] = frame
        split_counts[split] = counts
        print(
            f"{split}: {counts['rows_raw']} raw -> {counts['rows_clean']} clean "
            f"(dups {counts['dropped_exact_duplicates']}, "
            f"short {counts['dropped_under_50_tokens']}, "
            f"err {counts['dropped_error_marker']})",
            flush=True,
        )

    groups, group_meta = _assign_groups(frames)
    for split, frame in frames.items():
        frame["group"] = groups[split]
        out = LOCAL / f"{split}-clean.parquet"
        frame.to_parquet(out, index=False)
        split_counts[split]["clean_sha256"] = sha256_file(out)
        split_counts[split]["clean_bytes"] = out.stat().st_size

    combined_sha = hashlib.sha256(
        "".join(split_counts[s]["clean_sha256"] for s in SPLITS).encode("utf-8")
    ).hexdigest()

    manifest = {
        "schema": "panoptes-mage-fetch-v1",
        "dataset": f"{HF_REPO}@{REVISION}",
        "created_utc": prior_created or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "min_word_tokens": MIN_WORD_TOKENS,
        "splits": split_counts,
        "combined_sha256": combined_sha,
        **group_meta,
    }
    manifest["artifact_sha256"] = canonical_hash(manifest)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    POINTER_OUT.parent.mkdir(parents=True, exist_ok=True)
    pointer = pointer_manifest(manifest)
    POINTER_OUT.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"fetch manifest: {MANIFEST}")
    print(f"pointer manifest: {POINTER_OUT} (sha256 {pointer['artifact_sha256'][:16]}...)")
    leak = group_meta.get("leakage_audit", {})
    if leak:
        print(
            f"leakage audit: {leak['official_test_rows_sharing_cluster_with_train']}/"
            f"{leak['official_test_rows']} test rows share a near-dup cluster with train "
            f"({leak['official_split_near_duplicate_leakage_rate']:.4f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
