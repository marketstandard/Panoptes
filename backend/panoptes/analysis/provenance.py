from __future__ import annotations

import base64
import json
from dataclasses import dataclass

from panoptes.schemas import ProvenanceResult


@dataclass(frozen=True)
class UploadedFile:
    filename: str
    content: bytes


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg"}


def decode_upload(file_base64: str | None, filename: str | None) -> UploadedFile | None:
    if not file_base64:
        return None
    content = base64.b64decode(file_base64, validate=True)
    return UploadedFile(filename=filename or "upload.bin", content=content)


def verify_provenance(upload: UploadedFile | None) -> ProvenanceResult:
    if upload is None:
        return ProvenanceResult(
            status="not_applicable",
            summary="No file was supplied for signed provenance verification.",
        )

    suffix = "." + upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ""
    if suffix not in SUPPORTED_EXTENSIONS:
        return ProvenanceResult(
            status="unsupported_file",
            summary="This file type is not supported by the provenance verifier.",
        )

    try:
        from c2pa import Reader  # type: ignore[import-not-found]
    except Exception:
        return _heuristic_manifest_detection(upload)

    try:
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(upload.content)
            temp_path = Path(handle.name)
        try:
            with Reader(str(temp_path)) as reader:
                manifest = json.loads(reader.json())
        finally:
            temp_path.unlink(missing_ok=True)
    except Exception:
        return ProvenanceResult(
            status="not_present",
            summary="No readable signed provenance manifest was found.",
        )

    active = manifest.get("active_manifest")
    manifests = manifest.get("manifests", {})
    active_manifest = manifests.get(active, {}) if active else {}
    validation = manifest.get("validation_status") or manifest.get("validation", [])
    tampered = bool(validation)
    claim_generator = active_manifest.get("claim_generator") or active_manifest.get("issuer")
    actions = [
        str(action.get("action"))
        for action in active_manifest.get("actions", [])
        if isinstance(action, dict) and action.get("action")
    ]
    return ProvenanceResult(
        status="tampered" if tampered else "verified",
        summary=(
            "A signed provenance manifest was found, with validation warnings."
            if tampered
            else "A signed provenance manifest was found and validated."
        ),
        issuer=str(claim_generator) if claim_generator else None,
        timestamp=active_manifest.get("timestamp"),
        actions=actions,
    )


def _heuristic_manifest_detection(upload: UploadedFile) -> ProvenanceResult:
    content = upload.content[:2_000_000]
    if b"c2pa" in content.lower() or b"contentcredentials" in content.lower():
        return ProvenanceResult(
            status="verified",
            summary=(
                "C2PA provenance markers were found, but the optional c2pa-python package is not "
                "installed, so cryptographic validation was not completed."
            ),
            issuer=None,
            timestamp=None,
            actions=[],
        )
    return ProvenanceResult(
        status="not_present",
        summary="No C2PA provenance markers were found in the supported file bytes.",
    )
