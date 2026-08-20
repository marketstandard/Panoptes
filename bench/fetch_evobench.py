"""Fetch, verify, and clean the EvoBench benchmark for the bench.

EvoBench (ACL 2025 Findings) tracks detector generalization across evolving
LLM versions: 7 families, ~30 versions, 5 domains, distributed as a GitHub
repository of per-(version, domain) JSON files of the form
``{"original": [150 human texts], "sampled": [150 machine texts]}``.

The repository is downloaded as a tarball at a pinned commit (git content
addressing guarantees the file contents), ``*.raw_data.json`` files are
extracted (generation-config ``args/`` and ``ablation/`` duplicates are
skipped), and each file's SHA-256 is recorded in the fetch manifest. Raw
text never enters git — only hashes, row counts, and the signed pointer
manifest do.

Hygiene filters (counts reported in fetch-manifest.json):
  - empty texts;
  - exact duplicate texts (SHA-256 of normalized text);
  - texts under 50 word tokens (the runtime support threshold).

Deterministic: if a previous fetch-manifest.json exists, its created_utc is
reused so downstream artifacts stay byte-identical on regeneration.

Usage:  python -m bench.fetch_evobench
        python -m bench.fetch_evobench --check   # verify local files only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOCAL = ROOT / "datasets" / "local" / "evobench"
POINTER_OUT = ROOT / "datasets" / "manifests" / "evobench.json"
MANIFEST = LOCAL / "fetch-manifest.json"

REPO = "happy-Moer/EvoBench"
COMMIT = "4c866cd211499744d21d34cbf2f32594064047c0"
TARBALL_URL = f"https://codeload.github.com/{REPO}/tar.gz/{COMMIT}"

MIN_WORD_TOKENS = 50

CITATION = (
    "@inproceedings{moer2025evobench, title={EvoBench: Towards Real-world LLM-Generated "
    "Text Detection Benchmarking for Evolving Large Language Models}, author={Mo, Yuan and "
    "Liu, Ziyi and Zhang, Yichi and others}, booktitle={Findings of the Association for "
    "Computational Linguistics: ACL 2025}, year={2025}, "
    "publisher={Association for Computational Linguistics}, "
    "url={https://aclanthology.org/2025.findings-acl.754}, doi={10.18653/v1/2025.findings-acl.754}}"
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


def _download_tarball(dest: Path) -> None:
    if dest.exists():
        return
    print(f"downloading {TARBALL_URL} -> {dest.name}", flush=True)
    with urllib.request.urlopen(TARBALL_URL) as response, dest.open("wb") as out:
        while True:
            chunk = response.read(1 << 22)
            if not chunk:
                break
            out.write(chunk)


def _iter_raw_data(tarball: Path):
    """Yield (family_group, domain, version, payload) for each raw_data.json."""
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".raw_data.json"):
                continue
            parts = member.name.split("/")[1:]  # strip the repo-prefix directory
            rel = "/".join(parts)
            if "/args/" in f"/{rel}" or "/ablation/" in f"/{rel}" or "/agrs/" in f"/{rel}":
                continue
            stem = Path(parts[-1]).name[: -len(".raw_data.json")]
            if "_" not in stem:
                continue
            domain, version = stem.split("_", 1)
            family_group = parts[0]
            payload = json.loads(tar.extractfile(member).read().decode("utf-8"))
            yield family_group, domain, version, payload, rel


def clean(tarball: Path, clean_path: Path) -> dict:
    import pandas as pd

    rows: list[dict] = []
    counts: dict[str, int] = {
        "rows_raw": 0,
        "dropped_empty": 0,
        "dropped_exact_duplicates": 0,
        "dropped_under_50_tokens": 0,
    }
    seen: set[str] = set()
    n_files = 0
    for family_group, domain, version, payload, _rel in _iter_raw_data(tarball):
        n_files += 1
        originals = payload.get("original") or []
        sampled = payload.get("sampled") or []
        for _index, (human_text, ai_text) in enumerate(zip(originals, sampled, strict=False)):
            # Group by the human original's content: originals repeat across
            # version files, and every AI sample must share a fold with the
            # human text it was generated from, whichever file supplied it.
            group_key = hashlib.sha256(str(human_text).strip().encode("utf-8")).hexdigest()[:16]
            for text, label, family in (
                (human_text, 0, "human"),
                (ai_text, 1, version),
            ):
                counts["rows_raw"] += 1
                text = str(text).strip()
                if not text:
                    counts["dropped_empty"] += 1
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in seen:
                    counts["dropped_exact_duplicates"] += 1
                    continue
                seen.add(digest)
                if len(text.split()) < MIN_WORD_TOKENS:
                    counts["dropped_under_50_tokens"] += 1
                    continue
                rows.append(
                    {
                        "text": text,
                        "label": label,
                        "family": family,
                        "family_group": family_group,
                        "domain": domain,
                        "group": f"{domain}:{group_key}",
                    }
                )
    frame = pd.DataFrame(rows)
    counts["rows_clean"] = int(len(frame))
    counts["n_human"] = int((frame["label"] == 0).sum())
    counts["n_ai"] = int((frame["label"] == 1).sum())
    counts["n_source_files"] = n_files
    counts["family_groups"] = sorted(frame["family_group"].unique().tolist())
    counts["versions"] = sorted(v for v in frame["family"].unique().tolist() if v != "human")
    counts["domains"] = sorted(frame["domain"].unique().tolist())

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(clean_path, index=False)
    counts["clean_sha256"] = sha256_file(clean_path)
    counts["clean_bytes"] = clean_path.stat().st_size
    return counts


def pointer_manifest(manifest: dict) -> dict:
    payload = {
        "schema": "panoptes-dataset-manifest-v1",
        "id": "evobench",
        "kind": "prose",
        "title": (
            "EvoBench — detector generalization across evolving LLM versions (ACL 2025 Findings)"
        ),
        "license": {
            "spdx": "NOASSERTION",
            "redistributable": False,
            "citation": CITATION,
        },
        "source": {
            "url": f"https://github.com/{REPO}",
            "version": f"pinned-commit {COMMIT[:12]}",
            "access": "public",
            "download_instructions": (
                "Run `python -m bench.fetch_evobench`. The script downloads the repository "
                f"tarball at pinned commit {COMMIT[:12]} from GitHub, extracts the raw_data.json "
                "files, applies documented hygiene filters, and stores a clean parquet under "
                "datasets/local/evobench/ (gitignored). Raw text is never committed."
            ),
        },
        "content_hash": {
            "algorithm": "sha256",
            "value": manifest["combined_sha256"],
        },
        "splits": {
            "train": {
                "groups": ["domain:sha256(human original)[:16]"],
                "n": manifest["splits"]["clean"]["rows_clean"],
            },
            "calibration": {
                "groups": ["domain:sha256(human original)[:16]"],
                "n": manifest["splits"]["clean"]["rows_clean"],
            },
            "test": {
                "groups": ["domain:sha256(human original)[:16]"],
                "n": manifest["splits"]["clean"]["rows_clean"],
            },
            "group_keys": ["pair"],
        },
        "labels": {
            "schema": "binary_ai",
            "values": ["human (original)", "ai (sampled from an evolving LLM version)"],
        },
        "privacy": {
            "contains_pii_risk": "low",
            "raw_text_in_repo": False,
            "sanitization": "hashes-only",
        },
        "limitations": [
            "The 'harmful' domain is paraphrased social-media content; label semantics "
            "differ from the continuation domains.",
            "Human originals are shared across LLM versions within a domain; pair "
            "groups keep them out of different folds.",
            "Repository declares no license; we redistribute hashes and fitted parameters only.",
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
        clean_path = LOCAL / "clean.parquet"
        if (
            not clean_path.exists()
            or sha256_file(clean_path) != manifest["splits"]["clean"]["clean_sha256"]
        ):
            raise FetchError(f"{clean_path.name}: missing or hash mismatch")
        print("evobench local files verified: clean.parquet")
        return 0

    LOCAL.mkdir(parents=True, exist_ok=True)
    prior_created = None
    if MANIFEST.exists():
        prior_created = json.loads(MANIFEST.read_text(encoding="utf-8")).get("created_utc")

    tarball = LOCAL / f"evobench-{COMMIT[:12]}.tar.gz"
    _download_tarball(tarball)
    counts = clean(tarball, LOCAL / "clean.parquet")
    print(
        f"clean: {counts['rows_raw']} raw -> {counts['rows_clean']} clean "
        f"({counts['n_source_files']} files, {len(counts['versions'])} versions, "
        f"dups {counts['dropped_exact_duplicates']}, short {counts['dropped_under_50_tokens']})",
        flush=True,
    )

    manifest = {
        "schema": "panoptes-evobench-fetch-v1",
        "dataset": f"{REPO}@{COMMIT}",
        "created_utc": prior_created or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "min_word_tokens": MIN_WORD_TOKENS,
        "splits": {"clean": counts},
        "combined_sha256": counts["clean_sha256"],
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
