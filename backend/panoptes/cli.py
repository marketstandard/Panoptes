from __future__ import annotations

import json
import platform
import shutil
import sys
import webbrowser
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from panoptes.analysis.pipeline import analyze
from panoptes.schemas import AnalysisRequest, RuntimeProfile
from panoptes.settings import Settings

app = typer.Typer(help="Panoptes local evidence workbench")
console = Console()


@app.command()
def up(
    profile: RuntimeProfile = typer.Option(RuntimeProfile.FIXTURE, "--profile"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
) -> None:
    settings = Settings(profile=profile, host=host, port=port)
    console.print(f"Starting Panoptes at http://{host}:{port} with profile {profile.value}")
    if open_browser:
        webbrowser.open(f"http://{host}:{port}")
    uvicorn.run(
        "panoptes.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


@app.command()
def doctor() -> None:
    table = Table(title="Panoptes environment")
    table.add_column("Check")
    table.add_column("Value")
    table.add_row("OS", platform.platform())
    table.add_row("Architecture", platform.machine())
    table.add_row("Python", sys.version.split()[0])
    table.add_row("Node", shutil.which("node") or "not found")
    table.add_row("pnpm", shutil.which("pnpm") or "not found")
    table.add_row("Docker", shutil.which("docker") or "not found")
    table.add_row("CUDA visible", _cuda_visible())
    table.add_row("Default profile", Settings().profile.value)
    console.print(table)


@app.command()
def analyze_command(
    path: Path | None = typer.Argument(None),
    profile: RuntimeProfile = typer.Option(RuntimeProfile.FIXTURE, "--profile"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    text = path.read_text(encoding="utf-8") if path else sys.stdin.read()
    response = analyze(AnalysisRequest(text=text), Settings(profile=profile))
    if json_output:
        console.print_json(json.dumps(response.public_dict()))
        return
    console.print(response.summary.plain_language)
    console.print_json(json.dumps(response.summary.overall.model_dump()))


@app.command(name="fixtures")
def fixtures_command() -> None:
    for name in ("human-prose", "ai-prose", "code"):
        response = analyze(AnalysisRequest(fixture=name), Settings(profile=RuntimeProfile.FIXTURE))
        console.print(f"{name}: {response.summary.plain_language}")


models_app = typer.Typer(help="Inspect and verify pinned model artifacts")
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list() -> None:
    from panoptes.registry import DETECTOR_REGISTRY

    table = Table(title="Registered detectors")
    table.add_column("ID")
    table.add_column("Kind")
    table.add_column("Status")
    table.add_column("License")
    for item in DETECTOR_REGISTRY:
        table.add_row(item.id, item.kind, item.status, item.license or "")
    console.print(table)


@models_app.command("verify")
def models_verify() -> None:
    console.print("Artifact verification is configured for pinned bundles; no external artifacts are required in fixture mode.")


plugins_app = typer.Typer(help="Inspect plugin loading")
app.add_typer(plugins_app, name="plugins")


@plugins_app.command("list")
def plugins_list() -> None:
    settings = Settings()
    if not settings.plugin_paths:
        console.print("No plugin paths configured.")
        return
    for path in settings.plugin_paths:
        console.print(path)


def _cuda_visible() -> str:
    try:
        import torch

        return "yes" if torch.cuda.is_available() else "no"
    except Exception:
        return "unknown (torch not installed)"


if __name__ == "__main__":
    app()
