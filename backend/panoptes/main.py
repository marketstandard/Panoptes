from __future__ import annotations

import json
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from panoptes.analysis.pipeline import analyze
from panoptes.registry import enabled_registrations
from panoptes.schemas import AnalysisRequest, AnalysisResponse, RuntimeInfo
from panoptes.settings import Settings, get_settings

app = FastAPI(
    title="Panoptes",
    version="0.1.0",
    description="Local-first evidence workbench for AI text, watermark, and provenance analysis.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
def analyze_endpoint(
    request: AnalysisRequest,
    settings: Settings = Depends(get_settings),
) -> AnalysisResponse:
    if request.text and len(request.text) > settings.max_characters:
        raise HTTPException(status_code=413, detail="Input exceeds configured character limit.")
    return analyze(request, settings)


@app.get("/api/v1/capabilities")
def capabilities(settings: Settings = Depends(get_settings)) -> dict:
    registrations = enabled_registrations(settings.profile.value)
    return {
        "profile": settings.profile.value,
        "detectors": [registration.__dict__ for registration in registrations],
        "max_characters": settings.max_characters,
        "max_file_bytes": settings.max_file_bytes,
    }


@app.get("/api/v1/runtime", response_model=RuntimeInfo)
def runtime(settings: Settings = Depends(get_settings)) -> RuntimeInfo:
    return RuntimeInfo(
        profile=settings.profile,
        device="cpu" if "gpu" not in settings.profile.value else "gpu",
        models_loaded=[],
        calibration_bundles=[],
    )


# Signed research artifacts the UI may render. Allowlisted names only; the
# files contain aggregate statistics and signatures, never raw corpus text.
_ARTIFACT_ALLOWLIST = {
    "baseline-calibration": "baseline-calibration.json",
    "corpus-summary": "corpus-summary.json",
    "methodology-report": "methodology-report.json",
    "panoptes-v0-card": "panoptes-v0-card.json",
    "logistic-tier0-card": "cards/logistic-tier0.json",
    "gbm-tier1-card": "cards/gbm-tier1.json",
}


@app.get("/api/v1/artifacts/{name}")
def artifact(name: str, settings: Settings = Depends(get_settings)) -> dict:
    if name not in _ARTIFACT_ALLOWLIST:
        raise HTTPException(status_code=404, detail="Unknown artifact.")
    relative = _ARTIFACT_ALLOWLIST[name]
    package_root = Path(__file__).resolve().parents[1]  # backend/
    candidates = [
        Path(settings.artifact_dir) / relative,
        package_root / "artifacts" / relative,
        package_root.parent / "artifacts" / relative,
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Artifact not generated yet.")


@app.get("/metrics")
def metrics(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics are disabled.")
    expected = f"Bearer {settings.operator_token}" if settings.operator_token else None
    if expected and authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid operator token.")
    return {"status": "enabled"}


def mount_frontend() -> None:
    dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if not dist.exists():
        return
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        candidate = dist / path
        if path and candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")


mount_frontend()
