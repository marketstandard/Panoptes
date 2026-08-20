"""Fetch, reconstruct, and split-lock the CoAuthor dataset for Panoptes v2.1.

CoAuthor (Lee et al. 2022, arXiv:2201.06796) is a keystroke-level dataset of
1,447 human+GPT-3 collaborative writing sessions. Each session is a JSONL event
log of Quill deltas with an ``eventSource`` field (``user`` or ``api``) that gives
*character-level* provenance: which characters of the final document were written
by the human and which by GPT-3.

This is Panoptes's ground truth for the ``ai_contribution_fraction`` and
``ai_participation`` labels — the "AI-assisted middle" that binary detectors
cannot express. Per the v2.1 protocol, CoAuthor is a ``mixed_task`` cohort and is
NEVER treated as binary fully-AI positives.

What this script does:
  1. Downloads the session zip (Google Drive, via gdown) and the public metadata
     sheet (worker_id, prompt_code, official written_by_human).
  2. Reconstructs every session's final text + per-character authorship mask by
     replaying the Quill deltas. Prompt characters are tracked separately ('P').
  3. VALIDATES the reconstruction against the official ``written_by_human``
     (expects Pearson r > 0.95); refuses to proceed if reconstruction is wrong.
  4. Filters degenerate sessions (too little authored content, reconstruction
     failures, missing metadata).
  5. Assigns AUTHOR-DISJOINT (worker_id) train/development/calibration/test
     groups via a frozen hash — the same writer never crosses the firewall.
  6. Writes clean parquet + a fetch manifest + a pointer manifest.

Raw text is never committed to git; only hashes, counts, and fitted parameters.

Usage:
    python -m research.fetch_coauthor
"""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COAUTHOR_DIR = ROOT / "datasets" / "local" / "coauthor"
ZIP_PATH = COAUTHOR_DIR / "coauthor-dataset.zip"
META_CSV = COAUTHOR_DIR / "coauthor-metadata.csv"
FETCH_MANIFEST = COAUTHOR_DIR / "fetch-manifest.json"
POINTER_MANIFEST = ROOT / "datasets" / "manifests" / "coauthor.json"

# Google Drive file id for the CoAuthor session zip (from coauthor.stanford.edu).
GDRIVE_FILE_ID = "16czJfjqHfcsJkG9cRl1hJ4e3L0OlvkXr"
METADATA_SHEET_ID = "1O3EXJm52TQHfFSbzVGZmNIzzdu5ow6IjnOBrGTUY02o"

# Frozen split proportions for the four-way firewall (train/dev/calibration/test).
# CoAuthor is primarily an evaluation cohort, but we still lock author-disjoint
# partitions so any calibration is never scored on the final test writers.
SPLIT_PROPORTIONS = {"train": 0.40, "development": 0.15, "calibration": 0.15, "test": 0.30}
SPLIT_SEED = 20260819  # frozen; changing it re-partitions writers and voids the lock

MIN_AUTHORED_CHARS = 200  # drop sessions with almost no human+AI authored content
VALIDATION_MIN_PEARSON = 0.95


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(payload: dict) -> str:
    clone = dict(payload)
    clone.pop("artifact_sha256", None)
    canonical = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _download_zip() -> None:
    if ZIP_PATH.exists():
        print(f"[fetch_coauthor] zip present: {ZIP_PATH.name}")
        return
    COAUTHOR_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError as e:  # pragma: no cover
        raise SystemExit("gdown is required: pip install gdown") from e
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"[fetch_coauthor] downloading {url}")
    gdown.download(url, str(ZIP_PATH), quiet=False, fuzzy=True)
    if not ZIP_PATH.exists():
        raise SystemExit("CoAuthor zip download failed")


def _download_metadata() -> None:
    if META_CSV.exists():
        print(f"[fetch_coauthor] metadata present: {META_CSV.name}")
        return
    COAUTHOR_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://docs.google.com/spreadsheets/d/{METADATA_SHEET_ID}/export?format=csv"
    print(f"[fetch_coauthor] downloading metadata sheet")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=60).read()
    META_CSV.write_bytes(data)


def _load_metadata() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    with open(META_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = (row.get("session_id") or "").strip()
            if sid:
                meta[sid] = row
    return meta


def reconstruct_session(lines: list[str]) -> tuple[str, str] | None:
    """Replay Quill deltas -> (final_text, per-char mask of 'P'/'U'/'A').

    'P' = prompt (initial document, human-provided task), 'U' = user-authored,
    'A' = api-authored (GPT-3). Returns None on structural failure.
    """
    chars: list[list] = []
    try:
        for line in lines:
            obj = json.loads(line)
            if obj.get("eventName") == "system-initialize":
                chars = [[c, "P"] for c in obj.get("currentDoc", "")]
                continue
            delta = obj.get("textDelta")
            if not delta or "ops" not in delta:
                continue
            source = "A" if obj.get("eventSource") == "api" else "U"
            pos = 0
            for op in delta["ops"]:
                if "retain" in op:
                    pos += int(op["retain"])
                elif "insert" in op:
                    for c in str(op["insert"]):
                        chars.insert(pos, [c, source])
                        pos += 1
                elif "delete" in op:
                    del chars[pos : pos + int(op["delete"])]
    except Exception:
        return None
    return "".join(c for c, _ in chars), "".join(s for _, s in chars)


def _assign_split(worker_id: str) -> str:
    """Deterministic author-disjoint split from a frozen hash of the worker id."""
    h = int(hashlib.sha256(f"{SPLIT_SEED}:{worker_id}".encode()).hexdigest(), 16) % 10_000 / 10_000.0
    acc = 0.0
    for split, prop in SPLIT_PROPORTIONS.items():
        acc += prop
        if h < acc:
            return split
    return "test"


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / max(sx * sy, 1e-9)


def main() -> None:
    _download_zip()
    _download_metadata()
    meta = _load_metadata()
    print(f"[fetch_coauthor] metadata rows: {len(meta)}")

    z = zipfile.ZipFile(ZIP_PATH)
    names = sorted(n for n in z.namelist() if n.endswith(".jsonl"))
    print(f"[fetch_coauthor] session files: {len(names)}")

    records: list[dict] = []
    val_my: list[float] = []
    val_official: list[float] = []
    n_fail = n_nometa = n_short = 0
    for i, fname in enumerate(names):
        sid = fname.split("/")[-1][: -len(".jsonl")]
        m = meta.get(sid)
        if m is None:
            n_nometa += 1
            continue
        with z.open(fname) as f:
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        rec = reconstruct_session(lines)
        if rec is None:
            n_fail += 1
            continue
        text, mask = rec
        u = mask.count("U")
        a = mask.count("A")
        p = mask.count("P")
        authored = u + a
        if authored < MIN_AUTHORED_CHARS:
            n_short += 1
            continue
        ai_frac = a / authored
        human_frac = u / authored
        try:
            official = float(m.get("written_by_human", "") or "nan") / 100.0
        except ValueError:
            official = float("nan")
        if official == official:  # not NaN
            val_my.append(human_frac)
            val_official.append(official)
        records.append(
            {
                "id": sid,
                "text": text,
                "label": 1,  # every CoAuthor session is AI-assisted (mixed_task; never binary)
                "group": m.get("worker_id", "") or sid,
                "split": _assign_split(m.get("worker_id", "") or sid),
                "worker_id": m.get("worker_id", ""),
                "prompt_code": m.get("prompt_code", ""),
                "ai_contribution_fraction": round(ai_frac, 6),
                "human_chars": u,
                "ai_chars": a,
                "prompt_chars": p,
                "num_query": m.get("num_query", ""),
                "num_selected": m.get("num_selected", ""),
                "written_by_human_official": official,
                "mask": mask,
            }
        )
        if (i + 1) % 300 == 0:
            print(f"  ... {i + 1}/{len(names)} sessions")

    pearson = _pearson(val_my, val_official) if len(val_my) > 2 else 0.0
    mae = sum(abs(x - y) for x, y in zip(val_my, val_official)) / max(len(val_my), 1)
    print(f"[fetch_coauthor] reconstruction validation vs official written_by_human:")
    print(f"    n={len(val_my)}  pearson_r={pearson:.4f}  MAE={mae:.4f}")
    if pearson < VALIDATION_MIN_PEARSON:
        raise SystemExit(
            f"Reconstruction validation FAILED (pearson {pearson:.3f} < "
            f"{VALIDATION_MIN_PEARSON}). Refusing to lock untrustworthy provenance."
        )

    df = pd.DataFrame(records)
    COAUTHOR_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = COAUTHOR_DIR / "coauthor.parquet"
    df.drop(columns=["mask"]).to_parquet(parquet_path, index=False)
    # The mask is large; store it in a separate compressed file for span-level work.
    mask_path = COAUTHOR_DIR / "coauthor-masks.jsonl.gz"
    import gzip

    with gzip.open(mask_path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"id": r["id"], "mask": r["mask"]}) + "\n")

    split_counts = df["split"].value_counts().to_dict()
    n_workers = df["worker_id"].nunique()
    # Leakage audit: confirm no worker crosses splits.
    worker_splits = df.groupby("worker_id")["split"].nunique()
    cross_split_workers = int((worker_splits > 1).sum())

    print(f"[fetch_coauthor] kept {len(df)} sessions | workers={n_workers}")
    print(f"    splits: {split_counts}")
    print(f"    dropped: no_meta={n_nometa} recon_fail={n_fail} too_short={n_short}")
    print(f"    cross-split workers (must be 0): {cross_split_workers}")
    print(f"    ai_fraction mean={df['ai_contribution_fraction'].mean():.3f}")

    POINTER_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    split_n = {k: int(v) for k, v in split_counts.items()}
    pointer = {
        "schema": "panoptes-dataset-manifest-v1",
        "id": "coauthor",
        "kind": "prose",
        "title": "CoAuthor — Human+GPT-3 collaborative writing with keystroke provenance (Lee et al. 2022)",
        "license": {
            "spdx": "NOASSERTION",
            "redistributable": False,
            "citation": "@inproceedings{lee2022coauthor, title={CoAuthor: Designing a Human-AI Collaborative Writing Dataset for Exploring Language Model Capabilities}, author={Lee, Mina and Liang, Percy and Yang, Qian}, booktitle={CHI Conference on Human Factors in Computing Systems}, year={2022}, url={https://arxiv.org/abs/2201.06796}}",
        },
        "source": {
            "url": "https://coauthor.stanford.edu",
            "version": "v1.0 (2021 collection; metadata sheet pinned)",
            "access": "public",
            "download_instructions": "Run `python -m bench.fetch_coauthor`. The script downloads the session zip from Google Drive and the public metadata sheet, replays each session's Quill deltas to reconstruct the final text plus a per-character authorship mask (prompt/user/api), validates the reconstruction against the official written_by_human (Pearson r >= 0.95), filters degenerate sessions, and locks author-disjoint (worker_id) train/development/calibration/test splits. Clean parquet + masks live under datasets/local/coauthor/ (gitignored). Raw text is never committed.",
        },
        "content_hash": {"algorithm": "sha256", "value": _sha256(parquet_path)},
        "splits": {
            "train": {"groups": ["coauthor:<worker_id>"], "n": split_n.get("train", 0)},
            "calibration": {"groups": ["coauthor:<worker_id>"], "n": split_n.get("calibration", 0)},
            "test": {"groups": ["coauthor:<worker_id>"], "n": split_n.get("test", 0)},
            "group_keys": ["worker_id", "prompt_code"],
        },
        "labels": {
            "schema": "binary_ai",
            "values": [
                "ai-assisted (every session is human+GPT-3; label is always 1)",
                "ground truth is ai_contribution_fraction in [0,1], prompt excluded",
            ],
        },
        "privacy": {
            "contains_pii_risk": "low",
            "raw_text_in_repo": False,
            "sanitization": "hashes-only",
        },
        "limitations": [
            "mixed_task cohort: every session used GPT-3, so CoAuthor is NEVER a binary fully-AI positive set.",
            "Only the 830 sessions with released author metadata (worker_id, written_by_human) are used; 617 sessions without worker ids are excluded to keep splits author-disjoint.",
            f"development split ({split_n.get('development', 0)} sessions) is held between train and calibration; see the v2.1 split manifest for the four-way firewall.",
            "Reconstruction is validated against the official written_by_human (see fetch-manifest reconstruction_validation).",
            "GPT-3 suggestions are text-davinci-002 era (2021); contribution fractions reflect that generator.",
        ],
        "notice_entry_required": True,
    }
    pointer["artifact_sha256"] = canonical_hash(pointer)
    POINTER_MANIFEST.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")
    print(f"[fetch_coauthor] wrote pointer manifest {POINTER_MANIFEST.name}")

    # Reuse the prior created_utc so the manifest hash is reproducible across re-runs.
    prior_created = None
    if FETCH_MANIFEST.exists():
        try:
            prior_created = json.loads(FETCH_MANIFEST.read_text(encoding="utf-8")).get("created_utc")
        except Exception:
            prior_created = None
    fetch_manifest = {
        "dataset": "coauthor",
        "version": "v1.0",
        "created_utc": prior_created or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "https://coauthor.stanford.edu (Lee et al. 2022, arXiv:2201.06796)",
        "license": "NOASSERTION (publicly available for research; no explicit data license; see NOTICE)",
        "gdrive_file_id": GDRIVE_FILE_ID,
        "metadata_sheet_id": METADATA_SHEET_ID,
        "zip_sha256": _sha256(ZIP_PATH),
        "metadata_csv_sha256": _sha256(META_CSV),
        "n_session_files": len(names),
        "n_kept": int(len(df)),
        "n_workers": int(n_workers),
        "split_proportions": SPLIT_PROPORTIONS,
        "split_seed": SPLIT_SEED,
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "cross_split_workers": cross_split_workers,
        "min_authored_chars": MIN_AUTHORED_CHARS,
        "reconstruction_validation": {
            "n": len(val_my),
            "pearson_r": round(pearson, 4),
            "mae": round(mae, 4),
            "min_pearson_required": VALIDATION_MIN_PEARSON,
        },
        "ai_fraction_mean": round(float(df["ai_contribution_fraction"].mean()), 4),
    }
    fetch_manifest["artifact_sha256"] = canonical_hash(fetch_manifest)
    FETCH_MANIFEST.write_text(json.dumps(fetch_manifest, indent=2))
    print(f"[fetch_coauthor] wrote {parquet_path.name}, fetch-manifest.json, coauthor.json")


if __name__ == "__main__":
    main()
