"""Build the verified baseline corpus from reference runs and controls.

Every document admitted to the corpus is re-hashed against its run
manifest (SHA-256 per file, Merkle root over the output set, and the
manifest's own canonical hash). A run that fails verification is
rejected as a whole — the corpus never silently ingests tampered or
corrupted data. This is the only ingestion path used by calibration,
the methodology layer, and the bench.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.features import FEATURE_NAMES, extract, length_bucket, word_tokens  # noqa: E402

DEFAULT_ROOTS = (ROOT / "baselines" / "reference", ROOT / "baselines" / "controls")
REGISTRY = ROOT / "baselines" / "catalog" / "registry.jsonl"
SUMMARY_OUT = ROOT / "backend" / "artifacts" / "corpus-summary.json"
MANIFEST_NAME = "run.manifest.json"


class CorpusError(RuntimeError):
    """Raised when a run directory fails hash verification."""


@dataclass(frozen=True)
class CorpusRecord:
    text: str
    label: int  # 0 = human, 1 = ai
    family: str  # model slug, or "human" for controls
    kind: str  # "text" | "code"
    length_bucket: str
    prompt_id: str
    run_id: str
    sha256: str
    watermark_status: str = "unknown"  # declared-none | declared-active | suspected | unknown
    watermark_scheme: str | None = None
    watermark_notes: str | None = None


def canonical_hash(payload: dict) -> str:
    """Canonical SHA-256, identical to baselines/baseline.py."""
    clone = dict(payload)
    clone.pop("artifact_sha256", None)
    canonical = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merkle_root(hashes: list[str]) -> str:
    """Pairwise SHA-256 Merkle root, identical to baselines/baseline.py."""
    if not hashes:
        return hashlib.sha256(b"").hexdigest()
    level = sorted(hashes)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256((left + right).encode("ascii")).hexdigest()
            for left, right in zip(level[::2], level[1::2], strict=True)
        ]
    return level[0]


def verify_run(run_dir: Path) -> list[CorpusRecord]:
    """Re-hash every output against the manifest and return corpus records.

    The whole run is rejected (CorpusError) on any mismatch: manifest
    self-hash, per-file SHA-256, per-file byte count, or Merkle root.
    """
    manifest_path = run_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise CorpusError(f"{run_dir}: missing {MANIFEST_NAME}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    recorded_hash = manifest.get("artifact_sha256")
    if recorded_hash != canonical_hash(manifest):
        raise CorpusError(f"{run_dir}: manifest self-hash mismatch (tampered manifest)")

    interface = manifest["model"]["interface"]
    slug = manifest["model"]["slug"]
    label = 0 if interface == "human" else 1
    family = "human" if interface == "human" else slug
    watermark = manifest.get("watermark") or {}
    watermark_status = watermark.get("status", "unknown")
    watermark_scheme = watermark.get("scheme")
    watermark_notes = watermark.get("notes")

    records: list[CorpusRecord] = []
    for output in manifest["outputs"]:
        path = run_dir / output["file"]
        if not path.exists():
            raise CorpusError(f"{run_dir}: missing output {output['file']}")
        actual = sha256_file(path)
        if actual != output["sha256"]:
            raise CorpusError(
                f"{run_dir}: SHA-256 mismatch for {output['file']} "
                f"(manifest {output['sha256'][:12]}…, disk {actual[:12]}…)"
            )
        if path.stat().st_size != output["bytes"]:
            raise CorpusError(f"{run_dir}: byte-count mismatch for {output['file']}")
        text = path.read_text(encoding="utf-8")
        prompt_id = output["prompt_id"]
        kind = "code" if prompt_id.startswith("code-") else "text"
        records.append(
            CorpusRecord(
                text=text,
                label=label,
                family=family,
                kind=kind,
                length_bucket=length_bucket(len(word_tokens(text))),
                prompt_id=prompt_id,
                run_id=manifest["run_id"],
                sha256=actual,
                watermark_status=watermark_status,
                watermark_scheme=watermark_scheme,
                watermark_notes=watermark_notes,
            )
        )

    if merkle_root([record.sha256 for record in records]) != manifest["merkle_root"]:
        raise CorpusError(f"{run_dir}: Merkle root mismatch over output set")
    return records


def load_corpus(roots: tuple[Path, ...] | list[Path] = DEFAULT_ROOTS) -> list[CorpusRecord]:
    """Verify and load every run found under the given roots."""
    records: list[CorpusRecord] = []
    for root in roots:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / MANIFEST_NAME).exists():
                records.extend(verify_run(child))
    if not records:
        raise CorpusError(f"no verified runs found under {[str(root) for root in roots]}")
    return records


def run_manifests(roots: tuple[Path, ...] | list[Path] = DEFAULT_ROOTS) -> list[dict]:
    """Return the run manifests themselves (already hash-checked by load_corpus)."""
    manifests: list[dict] = []
    for root in roots:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            manifest_path = child / MANIFEST_NAME
            if child.is_dir() and manifest_path.exists():
                manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    return manifests


def catalog_entry_count(registry: Path = REGISTRY) -> int:
    if not registry.exists():
        return 0
    return sum(1 for line in registry.read_text(encoding="utf-8").splitlines() if line.strip())


def summarize(
    records: list[CorpusRecord],
    catalog_entries: int,
    *,
    screening: dict | None = None,
) -> dict:
    groups: dict[tuple[str, str], list[CorpusRecord]] = {}
    for record in records:
        groups.setdefault((record.family, record.kind), []).append(record)

    cohorts = []
    contaminated_cohorts: list[dict] = []
    for (family, kind), group in sorted(groups.items()):
        features = [extract(record.text, record.kind) for record in group]
        stats = {}
        for name in FEATURE_NAMES:
            values = [row[name] for row in features]
            stats[name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
        # Prefer the strongest contamination signal in the cohort.
        statuses = {record.watermark_status for record in group}
        if "declared-active" in statuses:
            watermark_status = "declared-active"
        elif "suspected" in statuses:
            watermark_status = "suspected"
        elif "declared-none" in statuses and statuses <= {"declared-none"}:
            watermark_status = "declared-none"
        else:
            watermark_status = (
                "unknown" if "unknown" in statuses or not statuses else next(iter(statuses))
            )
        cohorts.append(
            {
                "family": family,
                "kind": kind,
                "label": group[0].label,
                "n": len(group),
                "runs": sorted({record.run_id for record in group}),
                "length_buckets": {
                    bucket: sum(1 for record in group if record.length_bucket == bucket)
                    for bucket in ("lt50", "50-149", "150-499", "500plus")
                },
                "watermark_status": watermark_status,
                "features": stats,
            }
        )
        if watermark_status in {"declared-active", "suspected"}:
            notes = next(
                (record.watermark_notes for record in group if record.watermark_notes),
                None,
            )
            entry = {
                "family": family,
                "kind": kind,
                "watermark_status": watermark_status,
            }
            if notes:
                entry["notes"] = notes
            contaminated_cohorts.append(entry)

    payload = {
        "schema": "panoptes-corpus-summary-v1",
        "n_records": len(records),
        "n_runs": len({record.run_id for record in records}),
        "n_human": sum(1 for record in records if record.label == 0),
        "n_ai": sum(1 for record in records if record.label == 1),
        "families": sorted({record.family for record in records}),
        "kinds": sorted({record.kind for record in records}),
        "catalog_entries": catalog_entries,
        "cohorts": cohorts,
        "contaminated_cohorts": contaminated_cohorts,
    }
    if screening is not None:
        payload["watermark_screening"] = screening
    return payload


def save_signed_summary(payload: dict, output: Path) -> None:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    from bench.watermark_screening import screen_corpus

    records = load_corpus()
    screening = screen_corpus(records)
    summary = summarize(records, catalog_entry_count(), screening=screening)
    save_signed_summary(summary, SUMMARY_OUT)
    print(
        json.dumps(
            {
                "n_records": summary["n_records"],
                "n_runs": summary["n_runs"],
                "n_human": summary["n_human"],
                "n_ai": summary["n_ai"],
                "families": summary["families"],
                "catalog_entries": summary["catalog_entries"],
                "contaminated_cohorts": summary.get("contaminated_cohorts", []),
                "artifact_sha256": summary["artifact_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
