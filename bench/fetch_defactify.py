"""Fetch, verify, and clean the Defactify_Text_Dataset for the bench.

Downloads the three parquet splits from Hugging Face into the gitignored
local directory `datasets/local/defactify/`, verifying each file against the
pinned SHA-256 of the upstream LFS object. Raw text never enters git — only
hashes, row counts, and the signed pointer manifest do.

Hygiene filters (counts reported per split in fetch-manifest.json):
  - error artifacts: rows whose text is an API/client error string
    (the preview shows "Error communicating with OpenAI..." rows labeled
    as GPT-4o output);
  - exact duplicate texts (SHA-256 of normalized text);
  - texts under 50 word tokens (the runtime support threshold).

Deterministic: if a previous fetch-manifest.json exists, its created_utc is
reused so downstream artifacts (cards, calibration) stay byte-identical on
regeneration from the same upstream files.

Usage:  python -m bench.fetch_defactify
        python -m bench.fetch_defactify --check   # verify local files only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOCAL = ROOT / "datasets" / "local" / "defactify"
POINTER_OUT = ROOT / "datasets" / "manifests" / "defactify-text.json"
MANIFEST = LOCAL / "fetch-manifest.json"

HF_BASE = (
    "https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Text_Dataset/resolve/main/data"
)
SPLITS: dict[str, dict] = {
    "train": {
        "file": "train-00000-of-00001.parquet",
        "sha256": "9820a98d995f98a9bccd88a3ac73ad140b53755f23d7ffd531e9568a0638298a",
        "bytes": 86244339,
    },
    "validation": {
        "file": "validation-00000-of-00001.parquet",
        "sha256": "9dc5a986d8eff2f32cda6d438f95973576f248aea7d1381d55238e9f1e6dcecc",
        "bytes": 19634102,
    },
    "test": {
        "file": "test-00000-of-00001.parquet",
        "sha256": "253f8fb9e049d8084954bf0cb8a9e2d8ea0b59f4774b65a4802e90ee84421ec8",
        "bytes": 22487902,
    },
}

ERROR_PATTERNS = re.compile(
    r"Error communicating with|api\.openai\.com|NameResolutionError|Max retries exceeded|"
    r"HTTPSConnectionPool|OpenAI API error|RateLimitError|APIConnectionError",
    re.IGNORECASE,
)
MIN_WORD_TOKENS = 50

CITATION = (
    "@misc{roy2026comprehensivedatasethumanvs, "
    "title={A Comprehensive Dataset for Human vs. AI Generated Text Detection}, "
    "author={Rajarshi Roy and Gurpreet Singh and Ashhar Aziz and Shashwat Bajpai and "
    "Nasrin Imanpour and Shwetangshu Biswas and Kapil Wanaskar and Parth Patwa and "
    "Subhankar Ghosh and Shreyas Dixit and Nilesh Ranjan Pal and Vipula Rawte and "
    "Ritvik Garimella and Gaytri Jena and Amitava Das and Amit Sheth and Vasu Sharma and "
    "Aishwarya Naresh Reganti and Vinija Jain and Aman Chadha}, year={2026}, "
    "eprint={2510.22874}, archivePrefix={arXiv}, primaryClass={cs.CL}}"
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
    print(f"downloading {url} -> {dest.name} ({expected_bytes / 1e6:.1f} MB)")
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        while True:
            chunk = response.read(1 << 20)
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
    import pandas as pd

    frame = pd.read_parquet(raw_path, columns=["Text", "Label_A", "Label_B"])
    frame = frame.rename(columns={"Text": "text", "Label_A": "label", "Label_B": "family"})
    counts: dict[str, int] = {"rows_raw": int(len(frame))}

    error_mask = frame["text"].str.contains(ERROR_PATTERNS, na=False)
    counts["dropped_error_artifacts"] = int(error_mask.sum())
    frame = frame.loc[~error_mask]

    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame.loc[frame["text"].str.len() > 0]

    dup_mask = (
        frame["text"].map(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()).duplicated()
    )
    counts["dropped_exact_duplicates"] = int(dup_mask.sum())
    frame = frame.loc[~dup_mask]

    token_counts = frame["text"].str.split().str.len()
    short_mask = token_counts < MIN_WORD_TOKENS
    counts["dropped_under_50_tokens"] = int(short_mask.sum())
    frame = frame.loc[~short_mask]

    frame["label"] = frame["label"].astype(int)
    counts["rows_clean"] = int(len(frame))
    counts["n_human"] = int((frame["label"] == 0).sum())
    counts["n_ai"] = int((frame["label"] == 1).sum())
    counts["families"] = sorted(frame["family"].unique().tolist())

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(clean_path, index=False)
    counts["clean_sha256"] = sha256_file(clean_path)
    counts["clean_bytes"] = clean_path.stat().st_size
    return counts


def pointer_manifest(manifest: dict) -> dict:
    splits = {}
    for name, entry in manifest["splits"].items():
        splits[name] = {
            "groups": ["story (reconstructed at load time via TF-IDF clustering)"],
            "n": entry["rows_clean"],
        }
    payload = {
        "schema": "panoptes-dataset-manifest-v1",
        "id": "defactify-text",
        "kind": "prose",
        "title": (
            "Defactify_Text_Dataset — NYT human articles vs six LLM families (Roy et al. 2026)"
        ),
        "license": {
            "spdx": "CC-BY-4.0",
            "redistributable": False,
            "citation": CITATION,
        },
        "source": {
            "url": "https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Text_Dataset",
            "version": (
                f"pinned-sha256 train:{SPLITS['train']['sha256'][:12]} "
                f"validation:{SPLITS['validation']['sha256'][:12]} "
                f"test:{SPLITS['test']['sha256'][:12]}"
            ),
            "access": "public",
            "download_instructions": (
                "Run `python -m bench.fetch_defactify`. The script downloads the three "
                "parquet splits from Hugging Face, verifies them against pinned SHA-256 "
                "hashes, applies documented hygiene filters, and stores clean parquet "
                "files under datasets/local/defactify/ (gitignored). Raw text is never "
                "committed to this repository."
            ),
        },
        "content_hash": {
            "algorithm": "sha256",
            "value": manifest["combined_sha256"],
        },
        "splits": {
            "train": splits["train"],
            "calibration": splits["validation"],
            "test": splits["test"],
            "group_keys": ["story"],
        },
        "labels": {
            "schema": "binary_ai",
            "values": [
                "human (Label_A=0, Label_B=Human_Story)",
                "ai (Label_A=1, Label_B in {Gemma-2-9B, Mistral-7B, Qwen-2-72B, Llama-8B, "
                "Yi-Large, GPT-4o})",
            ],
        },
        "privacy": {
            "contains_pii_risk": "low",
            "raw_text_in_repo": False,
            "sanitization": "hashes-only",
        },
        "limitations": [
            "Domain is New York Times news prose; findings do not transfer automatically "
            "to other registers.",
            "Upstream dataset contained API-error artifacts labeled as GPT-4o output; "
            "filtered at fetch time (counts in fetch-manifest.json).",
            "AI texts are single-prompt rewrites of the human stories; story groups are "
            "reconstructed at load time via TF-IDF near-duplicate clustering.",
            "Dataset repository declares no separate license; the associated paper is "
            "CC BY 4.0. We redistribute hashes and fitted parameters only.",
        ],
        "notice_entry_required": True,
    }
    payload["artifact_sha256"] = canonical_hash(payload)
    return payload


def write_summary() -> Path:
    """Signed aggregate summary for the UI (counts and statistics only)."""
    from bench.datasets import load_defactify

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    dataset = load_defactify()
    family_counts: dict[str, int] = {}
    for family in dataset.families:
        family_counts[family] = family_counts.get(family, 0) + 1
    hygiene = {
        "rows_raw": 0,
        "dropped_error_artifacts": 0,
        "dropped_exact_duplicates": 0,
        "dropped_under_50_tokens": 0,
    }
    for entry in manifest["splits"].values():
        for key in hygiene:
            hygiene[key] += int(entry[key])
    group_stats = dict(dataset.meta["group_reconstruction"])
    group_stats.pop("edges", None)
    group_stats.pop("group_size_median", None)
    payload = {
        "schema": "panoptes-defactify-summary-v1",
        "dataset": "Rajarshi-Roy-research/Defactify_Text_Dataset",
        "created_utc": manifest["created_utc"],
        "n_records": len(dataset),
        "n_human": int((dataset.labels == 0).sum()),
        "n_ai": int((dataset.labels == 1).sum()),
        "families": dict(sorted(family_counts.items())),
        "splits": dataset.meta["official_split_counts"],
        "hygiene": hygiene,
        "group_reconstruction": group_stats,
        "leakage_audit": dataset.meta["leakage_audit"],
        "citation": CITATION,
    }
    payload["artifact_sha256"] = canonical_hash(payload)
    out = ROOT / "backend" / "artifacts" / "defactify-summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify local files against the manifest without downloading",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="write the signed defactify-summary.json artifact for the UI",
    )
    args = parser.parse_args()

    if args.summary:
        out = write_summary()
        print(f"summary artifact: {out}")
        return 0

    if args.check:
        if not MANIFEST.exists():
            raise FetchError("no fetch-manifest.json; run without --check first")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for name, entry in manifest["splits"].items():
            clean = LOCAL / f"{name}-clean.parquet"
            if not clean.exists() or sha256_file(clean) != entry["clean_sha256"]:
                raise FetchError(f"{clean.name}: missing or hash mismatch")
        print("defactify local files verified:", ", ".join(manifest["splits"]))
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
            f"(errors {counts['dropped_error_artifacts']}, "
            f"dups {counts['dropped_exact_duplicates']}, "
            f"short {counts['dropped_under_50_tokens']})"
        )

    combined = hashlib.sha256(
        "".join(splits[name]["clean_sha256"] for name in sorted(splits)).encode("ascii")
    ).hexdigest()
    manifest = {
        "schema": "panoptes-defactify-fetch-v1",
        "dataset": "Rajarshi-Roy-research/Defactify_Text_Dataset",
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
