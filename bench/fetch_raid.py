"""Fetch, verify, and clean the RAID benchmark train split for the bench.

Downloads `train.csv` (11.8 GB, labeled, includes the adversarial-attack
rows) from Hugging Face into the gitignored local directory
`datasets/local/raid/`, verifying it against the pinned SHA-256 of the
upstream LFS object. Raw text never enters git — only hashes, row counts,
and the signed pointer manifest do.

The unlabeled leaderboard `test.csv` is not fetched (no labels to evaluate
against); `extra.csv` (code, Czech, German) is out of scope for the English
prose comparison here.

Hygiene filters (counts reported in fetch-manifest.json):
  - rows with empty generations;
  - exact duplicate generations (SHA-256 of normalized text);
  - generations under 50 word tokens (the runtime support threshold).

Cleaning is streamed in chunks and written incrementally with pyarrow so
the 11.8 GB CSV never has to fit in memory. Deterministic: if a previous
fetch-manifest.json exists, its created_utc is reused so downstream
artifacts stay byte-identical on regeneration from the same upstream file.

Usage:  python -m bench.fetch_raid
        python -m bench.fetch_raid --check   # verify local files only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOCAL = ROOT / "datasets" / "local" / "raid"
POINTER_OUT = ROOT / "datasets" / "manifests" / "raid.json"
MANIFEST = LOCAL / "fetch-manifest.json"

HF_BASE = "https://huggingface.co/datasets/liamdugan/raid/resolve/main"
SPLITS: dict[str, dict] = {
    "train": {
        "file": "train.csv",
        "sha256": "52f04ceebc126064e68fbd22d8b736964065745464f4bfd52e488150b49f84e4",
        "bytes": 11779491051,
    },
}

MIN_WORD_TOKENS = 50
CHUNK_ROWS = 200_000

CITATION = (
    "@inproceedings{dugan-etal-2024-raid, title={RAID: A Shared Benchmark for Robust "
    "Evaluation of Machine-Generated Text Detectors}, author={Dugan, Liam and Hwang, "
    "Alyssa and Trhlik, Filip and Zhu, Andrew and Ludan, Josh Magnus and Xu, Hainiu and "
    "Ippolito, Daphne and Callison-Burch, Chris}, booktitle={Proceedings of the 62nd "
    "Annual Meeting of the Association for Computational Linguistics (Volume 1: Long "
    "Papers)}, pages={12463--12492}, year={2024}, address={Bangkok, Thailand}, "
    "publisher={Association for Computational Linguistics}, "
    "url={https://aclanthology.org/2024.acl-long.674}, doi={10.18653/v1/2024.acl-long.674}}"
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


def _download(url: str, dest: Path, expected_sha256: str, expected_bytes: int) -> None:
    if dest.exists():
        if sha256_file(dest) == expected_sha256:
            return
        dest.unlink()
    print(f"downloading {url} -> {dest.name} ({expected_bytes / 1e9:.2f} GB)", flush=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        while True:
            chunk = response.read(1 << 22)
            if not chunk:
                break
            out.write(chunk)
    actual = sha256_file(dest)
    if actual != expected_sha256:
        dest.unlink(missing_ok=True)
        raise FetchError(
            f"{dest.name}: sha256 {actual} != pinned {expected_sha256}; "
            "refusing to use unverified data"
        )


def clean_split(raw_path: Path, clean_path: Path) -> dict:
    """Stream the CSV in chunks, filter, and write clean parquet incrementally."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = ["id", "source_id", "model", "decoding", "attack", "domain", "generation"]
    counts: dict[str, int] = {
        "rows_raw": 0,
        "dropped_empty": 0,
        "dropped_exact_duplicates": 0,
        "dropped_under_50_tokens": 0,
    }
    seen: set[str] = set()
    schema = pa.schema(
        [
            ("text", pa.string()),
            ("label", pa.int64()),
            ("family", pa.string()),
            ("domain", pa.string()),
            ("attack", pa.string()),
            ("decoding", pa.string()),
            ("group", pa.string()),
        ]
    )
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(clean_path, schema) as writer:
        for frame in pd.read_csv(
            raw_path,
            usecols=columns,
            chunksize=CHUNK_ROWS,
            dtype={"attack": "string", "decoding": "string"},
            keep_default_na=False,
        ):
            counts["rows_raw"] += len(frame)
            frame["generation"] = frame["generation"].astype(str).str.strip()
            empty_mask = frame["generation"].str.len() == 0
            counts["dropped_empty"] += int(empty_mask.sum())
            frame = frame.loc[~empty_mask]
            hashes = frame["generation"].map(
                lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()
            )
            dup_mask = hashes.isin(seen) | hashes.duplicated()
            seen.update(hashes[~dup_mask].tolist())
            counts["dropped_exact_duplicates"] += int(dup_mask.sum())
            frame = frame.loc[~dup_mask]

            token_counts = frame["generation"].str.split().str.len()
            short_mask = token_counts < MIN_WORD_TOKENS
            counts["dropped_under_50_tokens"] += int(short_mask.sum())
            frame = frame.loc[~short_mask]
            if frame.empty:
                continue

            out = pd.DataFrame(
                {
                    "text": frame["generation"],
                    "label": (frame["model"] != "human").astype("int64"),
                    "family": frame["model"].astype(str),
                    "domain": frame["domain"].astype(str),
                    "attack": frame["attack"].astype(str),
                    "decoding": frame["decoding"].astype(str),
                    # source_id links an AI generation to the human text it was
                    # prompted from; it is the leakage-control group key.
                    "group": frame["source_id"].astype(str),
                }
            )
            writer.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False))

    counts["rows_clean"] = int(pq.read_metadata(clean_path).num_rows)
    counts["clean_sha256"] = sha256_file(clean_path)
    counts["clean_bytes"] = clean_path.stat().st_size
    return counts


def pointer_manifest(manifest: dict) -> dict:
    splits = {
        name: {
            "groups": ["source_id (AI generation shares the human source prompt id)"],
            "n": entry["rows_clean"],
        }
        for name, entry in manifest["splits"].items()
    }
    payload = {
        "schema": "panoptes-dataset-manifest-v1",
        "id": "raid",
        "kind": "prose",
        "title": (
            "RAID — shared benchmark for robust evaluation of MGT detectors (Dugan et al. 2024)"
        ),
        "license": {
            "spdx": "MIT",
            "redistributable": False,
            "citation": CITATION,
        },
        "source": {
            "url": "https://huggingface.co/datasets/liamdugan/raid",
            "version": f"pinned-sha256 train:{SPLITS['train']['sha256'][:12]}",
            "access": "public",
            "download_instructions": (
                "Run `python -m bench.fetch_raid`. The script downloads the labeled train.csv "
                "from Hugging Face, verifies it against the pinned SHA-256 hash, applies "
                "documented hygiene filters, and stores a clean parquet under "
                "datasets/local/raid/ (gitignored). Raw text is never committed to this "
                "repository."
            ),
        },
        "content_hash": {
            "algorithm": "sha256",
            "value": manifest["combined_sha256"],
        },
        "splits": {
            "train": splits["train"],
            "calibration": splits["train"],
            "test": splits["train"],
            "group_keys": ["source_id"],
        },
        "labels": {
            "schema": "binary_ai",
            "values": ["human (model=human)", "ai (model in 11 generator names)"],
        },
        "privacy": {
            "contains_pii_risk": "low",
            "raw_text_in_repo": False,
            "sanitization": "hashes-only",
        },
        "limitations": [
            "Only the labeled train split is fetched; the leaderboard test split ships "
            "without labels.",
            "Adversarial-attack rows are retained and flagged via the attack column; "
            "clean evaluation uses attack=none unless stated.",
            "RAID covers English prose domains plus code/Czech/German in extra.csv, "
            "which is not fetched here.",
        ],
        "notice_entry_required": True,
    }
    payload["artifact_sha256"] = canonical_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify local files against the manifest without downloading",
    )
    args = parser.parse_args()

    if args.check:
        if not MANIFEST.exists():
            raise FetchError("no fetch-manifest.json; run without --check first")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for name, entry in manifest["splits"].items():
            clean = LOCAL / f"{name}-clean.parquet"
            if not clean.exists() or sha256_file(clean) != entry["clean_sha256"]:
                raise FetchError(f"{clean.name}: missing or hash mismatch")
        print("raid local files verified:", ", ".join(manifest["splits"]))
        return 0

    LOCAL.mkdir(parents=True, exist_ok=True)
    prior_created = None
    if MANIFEST.exists():
        prior_created = json.loads(MANIFEST.read_text(encoding="utf-8")).get("created_utc")

    splits: dict[str, dict] = {}
    for name, spec in SPLITS.items():
        raw = LOCAL / spec["file"]
        _download(f"{HF_BASE}/{spec['file']}", raw, spec["sha256"], spec["bytes"])
        counts = clean_split(raw, LOCAL / f"{name}-clean.parquet")
        counts["source_file"] = spec["file"]
        counts["source_sha256"] = spec["sha256"]
        counts["source_bytes"] = spec["bytes"]
        splits[name] = counts
        print(
            f"{name}: {counts['rows_raw']} raw -> {counts['rows_clean']} clean "
            f"(dups {counts['dropped_exact_duplicates']}, "
            f"short {counts['dropped_under_50_tokens']})",
            flush=True,
        )

    combined = hashlib.sha256(
        "".join(splits[name]["clean_sha256"] for name in sorted(splits)).encode("ascii")
    ).hexdigest()
    manifest = {
        "schema": "panoptes-raid-fetch-v1",
        "dataset": "liamdugan/raid",
        "created_utc": prior_created or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "min_word_tokens": MIN_WORD_TOKENS,
        "splits": splits,
        "combined_sha256": combined,
    }
    manifest["artifact_sha256"] = canonical_hash(manifest)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    POINTER_OUT.parent.mkdir(parents=True, exist_ok=True)
    pointer = pointer_manifest(manifest)
    POINTER_OUT.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"fetch manifest: {MANIFEST}")
    print(f"pointer manifest: {POINTER_OUT} (sha256 {pointer['artifact_sha256'][:16]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
