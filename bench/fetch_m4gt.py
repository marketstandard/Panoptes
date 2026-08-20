"""Fetch, verify, and clean the M4GT-Bench Subtask A files for the bench.

Downloads `SubtaskA.jsonl` (English, 5 domains, 6 generators) and
`SubtaskA_multilingual.jsonl` (16 sources, 8 generators) from the authors'
public Google Drive folder into the gitignored local directory
`datasets/local/m4gt/`, verifying each against its pinned SHA-256. Raw text
never enters git — only hashes, row counts, and the signed pointer manifest
do.

Google Drive serves large files behind an interstitial confirmation form;
the downloader parses the form's hidden inputs and follows the real
download URL.

Hygiene filters (counts reported in fetch-manifest.json):
  - rows with empty texts;
  - exact duplicate texts (SHA-256 of normalized text);
  - texts under 50 word tokens (the runtime support threshold).

Group keys: each row's group is the SHA-256 of its normalized text, so
exact duplicates share a fold. The upstream files carry no identifier
linking a parallel human/AI pair, so paired rows cannot be group-linked;
this is recorded in the pointer manifest's limitations.

Deterministic: if a previous fetch-manifest.json exists, its created_utc is
reused so downstream artifacts stay byte-identical on regeneration.

Usage:  python -m bench.fetch_m4gt
        python -m bench.fetch_m4gt --check   # verify local files only
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
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOCAL = ROOT / "datasets" / "local" / "m4gt"
POINTER_OUT = ROOT / "datasets" / "manifests" / "m4gt.json"
MANIFEST = LOCAL / "fetch-manifest.json"

DRIVE_FILES: dict[str, dict] = {
    "subtask_a": {
        "file": "SubtaskA.jsonl",
        "drive_id": "1WS2tNQ50sjXucWSSLmrah87be39huRKw",
        "sha256": "fbc0dbbb4045d8cde332bc4d9bed183f36704bca859d40bf750c5e3ea8d4b7f9",
        "bytes": 503815389,
    },
    "subtask_a_multilingual": {
        "file": "SubtaskA_multilingual.jsonl",
        "drive_id": "1P0nSHpkB1MFzvglqm7W_hIshE2Qpoiuo",
        "sha256": "bbace883094cd8cb5c5e44a954d066d0f77acbfffe9faa58fcf8d1e16aa50a58",
        "bytes": 727037804,
    },
}

MIN_WORD_TOKENS = 50

CITATION = (
    "@inproceedings{wang-etal-2024-m4gt, title={M4GT-Bench: Evaluation Benchmark for "
    "Black-Box Machine-Generated Text Detection}, author={Wang, Yuxia and Mansurov, "
    "Jonibek and Ivanov, Petar and Su, Jinyan and Shelmanov, Artem and Tsvigun, Akim and "
    "Mohammed, Osama and Elmadany, Abdelrahman and Whitehouse, Chenxi and Aji, Alham Fikri "
    "and Gurevych, Iryna and Nakov, Preslav}, booktitle={Proceedings of the 62nd Annual "
    "Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)}, "
    "pages={3964--3992}, year={2024}, address={Bangkok, Thailand}, publisher={Association "
    "for Computational Linguistics}, url={https://aclanthology.org/2024.acl-long.218}, "
    "doi={10.18653/v1/2024.acl-long.218}}"
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


def _drive_download(file_id: str, dest: Path, expected_sha256: str, expected_bytes: int) -> None:
    if dest.exists():
        if sha256_file(dest) == expected_sha256:
            return
        dest.unlink()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"downloading drive:{file_id} -> {dest.name} ({expected_bytes / 1e6:.0f} MB)", flush=True)
    with opener.open(request) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            with dest.open("wb") as out:
                while True:
                    chunk = response.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)
        else:
            page = response.read()
            action = re.search(rb'<form[^>]+action="([^"]+)"', page)
            if not action:
                dest.unlink(missing_ok=True)
                raise FetchError(f"{dest.name}: could not parse Drive confirmation page")
            base = action.group(1).decode().replace("&amp;", "&")
            params: dict[str, str] = {}
            for field_name, value in re.findall(
                rb'<input[^>]+type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', page
            ):
                params[field_name.decode()] = value.decode()
            for field_name, value in re.findall(
                rb'<input[^>]+name="([^"]+)"[^>]*type="hidden"[^>]*value="([^"]*)"', page
            ):
                params[field_name.decode()] = value.decode()
            real_url = base + ("&" if "?" in base else "?") + urlencode(params)
            request2 = urllib.request.Request(real_url, headers={"User-Agent": "Mozilla/5.0"})
            with opener.open(request2) as response2, dest.open("wb") as out:
                while True:
                    chunk = response2.read(1 << 22)
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
    """Stream the JSONL, filter, and write a clean parquet."""
    import pandas as pd

    counts: dict[str, int] = {
        "rows_raw": 0,
        "dropped_empty": 0,
        "dropped_exact_duplicates": 0,
        "dropped_under_50_tokens": 0,
    }
    seen: set[str] = set()
    rows: list[dict] = []
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            counts["rows_raw"] += 1
            text = str(record.get("text") or "").strip()
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
            label = int(record["label"])
            model = str(record.get("model") or "unknown")
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "family": "human" if label == 0 else model,
                    "domain": str(record.get("source") or "unknown"),
                    "group": digest[:16],
                }
            )
    frame = pd.DataFrame(rows)
    counts["rows_clean"] = int(len(frame))
    counts["n_human"] = int((frame["label"] == 0).sum())
    counts["n_ai"] = int((frame["label"] == 1).sum())
    counts["families"] = sorted(frame["family"].unique().tolist())
    counts["domains"] = sorted(frame["domain"].unique().tolist())

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(clean_path, index=False)
    counts["clean_sha256"] = sha256_file(clean_path)
    counts["clean_bytes"] = clean_path.stat().st_size
    return counts


def pointer_manifest(manifest: dict) -> dict:
    splits = {
        name: {
            "groups": ["sha256(normalized text)[:16] — exact-duplicate-level grouping"],
            "n": entry["rows_clean"],
        }
        for name, entry in manifest["splits"].items()
    }
    payload = {
        "schema": "panoptes-dataset-manifest-v1",
        "id": "m4gt",
        "kind": "prose",
        "title": (
            "M4GT-Bench Subtask A — multi-domain, multilingual, multi-generator MGT "
            "detection (Wang et al. 2024)"
        ),
        "license": {
            "spdx": "NOASSERTION",
            "redistributable": False,
            "citation": CITATION,
        },
        "source": {
            "url": "https://github.com/mbzuai-nlp/M4GT-Bench",
            "version": (
                f"pinned-sha256 subtask_a:{DRIVE_FILES['subtask_a']['sha256'][:12]} "
                f"multilingual:{DRIVE_FILES['subtask_a_multilingual']['sha256'][:12]}"
            ),
            "access": "public",
            "download_instructions": (
                "Run `python -m bench.fetch_m4gt`. The script downloads SubtaskA.jsonl and "
                "SubtaskA_multilingual.jsonl from the authors' Google Drive folder (confirm-form "
                "handled), verifies them against pinned SHA-256 hashes, applies documented hygiene "
                "filters, and stores clean parquet files under datasets/local/m4gt/ (gitignored). "
                "Raw text is never committed to this repository."
            ),
        },
        "content_hash": {
            "algorithm": "sha256",
            "value": manifest["combined_sha256"],
        },
        "splits": {
            "train": splits["subtask_a"],
            "calibration": splits["subtask_a"],
            "test": splits["subtask_a"],
            "group_keys": ["text_hash"],
        },
        "labels": {
            "schema": "binary_ai",
            "values": [
                "human (label=0)",
                "ai (label=1; model in {bloomz, chatGPT, cohere, davinci, dolly, gpt4} "
                "plus llama2-fine-tuned and jais-30b in the multilingual file)",
            ],
        },
        "privacy": {
            "contains_pii_risk": "low",
            "raw_text_in_repo": False,
            "sanitization": "hashes-only",
        },
        "limitations": [
            "The upstream files carry no identifier linking a parallel human/AI pair, "
            "so paired rows cannot be group-linked; groups are exact-duplicate-level only.",
            "The multilingual file overlaps the English file's domains; the two are evaluated as "
            "separate datasets (m4gt, m4gtml) and never mixed in one split.",
            "Repository declares no license for the data files; we redistribute hashes and fitted "
            "parameters only.",
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
        print("m4gt local files verified:", ", ".join(manifest["splits"]))
        return 0

    LOCAL.mkdir(parents=True, exist_ok=True)
    prior_created = None
    if MANIFEST.exists():
        prior_created = json.loads(MANIFEST.read_text(encoding="utf-8")).get("created_utc")

    splits: dict[str, dict] = {}
    for name, spec in DRIVE_FILES.items():
        raw = LOCAL / spec["file"]
        _drive_download(spec["drive_id"], raw, spec["sha256"], spec["bytes"])
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
        "schema": "panoptes-m4gt-fetch-v1",
        "dataset": "mbzuai-nlp/M4GT-Bench Subtask A",
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
