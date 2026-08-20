"""Git-repo evaluation harness: evaluate any external system straight from a repo.

Point Panoptes at a git URL and get a signed evaluation card. The external
system is described by an **adapter contract** and executed in a subprocess so
cloned code never runs in the Panoptes process.

Adapter contract (a ``panoptes.adapter.json`` at the repo root)::

    {
      "kind": "watermark-remover" | "watermark-scheme" | "detector",
      "name": "...", "version": "...",
      "entry": {"type": "python-function", "module": "panoptes_adapter",
                "callable": "transform"},
      "requires_network": false
    }

If no manifest is present, a ``panoptes_adapter.py`` at the root is auto-detected
and must expose the conventional callable for the kind. Kind contracts:

* ``watermark-remover``: ``transform(text) -> str``
* ``watermark-scheme``:  ``detect(text) -> {"score": float, "p_value": float}``
* ``detector``:          ``score(text) -> float``

SECURITY: this clones and executes arbitrary code. Only run repositories you
trust. Isolation is a subprocess with a wall-clock limit, a scrubbed environment
(common secret env vars are dropped), and no network *enforcement* unless you
pass ``--docker`` (which requires an image containing the repo's dependencies).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOS_DIR = ROOT / ".panoptes" / "repos"
RUNNER = Path(__file__).resolve().parent / "_repo_adapter_runner.py"

KINDS = ("watermark-remover", "watermark-scheme", "detector")
DEFAULT_CALLABLE = {
    "watermark-remover": "transform",
    "watermark-scheme": "detect",
    "detector": "score",
}
TRUSTED_WARNING = (
    "evaluate-repo clones and executes code from the given repository in a "
    "subprocess. Only run repositories you trust. Subprocess isolation drops "
    "common secret environment variables and enforces a wall-clock limit, but "
    "does NOT block network access unless you pass --docker."
)

# Env vars that commonly hold credentials; dropped from the subprocess env so a
# malicious repo cannot read them from its process environment.
_SECRET_ENV_PREFIXES = (
    "AWS_",
    "AZURE_",
    "GCP_",
    "GOOGLE_",
    "OPENAI",
    "ANTHROPIC",
    "HF_",
    "HUGGING",
    "STRIPE",
    "GITHUB_",
    "GH_",
    "SLACK_",
    "SUPABASE",
    "VERCEL",
    "SENTRY",
    "DATABASE_URL",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "API_KEY",
)


@dataclass
class AdapterSpec:
    kind: str
    module: str
    callable: str
    name: str = "external"
    version: str = ""
    requires_network: bool = False
    source: str = "auto"  # manifest | auto | injected


def _repo_sha(url: str, ref: str | None) -> str:
    return hashlib.sha256(f"{url}@{ref or 'HEAD'}".encode()).hexdigest()[:16]


def clone_repo(url: str, ref: str | None = None, dest_root: Path = REPOS_DIR) -> Path:
    """Clone ``url`` (optionally at ``ref``) into an isolated dir; reuse if present."""
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / _repo_sha(url, ref)
    if dest.exists():
        return dest
    cmd = ["git", "clone", "--depth", "1", url, str(dest)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    if ref:
        subprocess.run(
            ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", ref],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(dest), "checkout", ref], check=True, capture_output=True, text=True
        )
    return dest


def find_adapter(repo_dir: Path) -> AdapterSpec | None:
    """Resolve the adapter contract: manifest first, then auto-detect."""
    manifest = repo_dir / "panoptes.adapter.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        entry = data.get("entry", {})
        return AdapterSpec(
            kind=data["kind"],
            module=entry.get("module", "panoptes_adapter"),
            callable=entry.get("callable", DEFAULT_CALLABLE.get(data["kind"], "transform")),
            name=data.get("name", "external"),
            version=data.get("version", ""),
            requires_network=bool(data.get("requires_network", False)),
            source="manifest",
        )
    auto = repo_dir / "panoptes_adapter.py"
    if auto.exists():
        # Kind unknown until the caller supplies it; module/callable conventional.
        return AdapterSpec(kind="", module="panoptes_adapter", callable="", source="auto")
    return None


def inject_adapter(repo_dir: Path, adapter_path: Path, kind: str) -> AdapterSpec:
    """Copy a user-supplied adapter into the cloned repo and build its spec.

    This is how you evaluate a repo that does not ship its own Panoptes adapter:
    supply a thin ``panoptes_adapter.py`` that wires the repo's code to the
    conventional callable for the kind.
    """
    shutil.copyfile(adapter_path, repo_dir / "panoptes_adapter.py")
    return AdapterSpec(
        kind=kind,
        module="panoptes_adapter",
        callable=DEFAULT_CALLABLE[kind],
        name=adapter_path.parent.name or "external",
        source="injected",
    )


def _sandbox_env() -> dict:
    env = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if any(upper.startswith(p) or p in upper for p in _SECRET_ENV_PREFIXES):
            continue
        env[key] = value
    env["PANOPTES_REPO_SANDBOX"] = "1"
    return env


def run_adapter(
    repo_dir: Path,
    spec: AdapterSpec,
    texts: list[str],
    *,
    timeout: int = 180,
    docker: bool = False,
    docker_image: str = "python:3.12-slim",
) -> list:
    """Run the adapter over ``texts`` in a subprocess; return one result per text."""
    with tempfile.TemporaryDirectory(prefix="panoptes-adapter-") as td:
        inp = Path(td) / "input.json"
        outp = Path(td) / "output.json"
        inp.write_text(json.dumps(texts), encoding="utf-8")
        cmd = [
            sys.executable,
            str(RUNNER),
            "--repo",
            str(repo_dir),
            "--module",
            spec.module,
            "--callable",
            spec.callable,
            "--kind",
            spec.kind,
            "--input",
            str(inp),
            "--output",
            str(outp),
        ]
        if docker:
            cmd = _docker_wrap(cmd, repo_dir, Path(td), docker_image)
        proc = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            env=None if docker else _sandbox_env(),
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"adapter subprocess exited {proc.returncode}: {proc.stderr[-2000:]}"
            )
        if not outp.exists():
            raise RuntimeError(f"adapter produced no output file: {proc.stdout[-1000:]}")
        return json.loads(outp.read_text(encoding="utf-8"))


def _docker_wrap(cmd: list[str], repo_dir: Path, td: Path, image: str) -> list[str]:
    """Wrap the runner in a network-disabled container. The image must contain
    the repo's dependencies; the repo and I/O dir are mounted read-only/read-write."""
    inner = cmd[1:]  # drop sys.executable; use the image's python
    return [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "-v",
        f"{repo_dir}:/repo:ro",
        "-v",
        f"{td}:/io",
        image,
        "python",
        *inner,
    ]


# --- kind routing ------------------------------------------------------------


def evaluate_remover(repo_dir: Path, spec: AdapterSpec, generations_card: dict, **run_kw) -> dict:
    from panoptes.analysis.watermarks import KGWReferenceAdapter
    from panoptes.schemas import ContentType

    from bench.watermark_unicode import detect_unicode_watermark, embed_unicode_watermark

    samples = generations_card["samples"]
    wm = [s["text"] for s in samples if s["kind"] == "watermarked"]
    ctrl = [s["text"] for s in samples if s["kind"] == "control"]
    uni_in = [embed_unicode_watermark(t) for t in ctrl]

    outputs = run_adapter(repo_dir, spec, wm + uni_in, **run_kw)
    wm_out, uni_out = outputs[: len(wm)], outputs[len(wm) :]

    det = KGWReferenceAdapter()

    def kgw_rate(texts: list[str]) -> tuple[float, float]:
        rows = [det.detect(t, ContentType.PROSE)[0] for t in texts]
        tested = [r for r in rows if r.status == "tested" and r.p_value is not None]
        if not tested:
            return (0.0, 0.0)
        return (
            sum(1 for r in tested if r.p_value < 0.05) / len(tested),
            sum(r.z or 0.0 for r in tested) / len(tested),
        )

    def uni_rate(texts: list[str]) -> float:
        rows = [detect_unicode_watermark(t) for t in texts]
        return sum(1 for r in rows if r["present"]) / len(rows) if rows else 0.0

    kgw_before, z_before = kgw_rate(wm)
    kgw_after, z_after = kgw_rate(wm_out)
    return {
        "kgw": {
            "n": len(wm),
            "detection_rate_before": kgw_before,
            "detection_rate_after": kgw_after,
            "mean_z_before": z_before,
            "mean_z_after": z_after,
        },
        "unicode": {
            "n": len(uni_in),
            "present_rate_before": uni_rate(uni_in),
            "present_rate_after": uni_rate(uni_out),
        },
    }


def evaluate_scheme(repo_dir: Path, spec: AdapterSpec, generations_card: dict, **run_kw) -> dict:
    samples = generations_card["samples"]
    wm = [s["text"] for s in samples if s["kind"] == "watermarked"]
    ctrl = [s["text"] for s in samples if s["kind"] == "control"]
    outputs = run_adapter(repo_dir, spec, wm + ctrl, **run_kw)
    wm_out, ctrl_out = outputs[: len(wm)], outputs[len(wm) :]

    def detected(res) -> bool:
        try:
            return float(res.get("p_value", 1.0)) < 0.05
        except (AttributeError, TypeError, ValueError):
            return False

    return {
        "n_watermarked": len(wm),
        "n_control": len(ctrl),
        "tpr": sum(1 for r in wm_out if detected(r)) / len(wm) if wm else None,
        "fpr": sum(1 for r in ctrl_out if detected(r)) / len(ctrl) if ctrl else None,
    }


def evaluate_detector(repo_dir: Path, spec: AdapterSpec, corpus, **run_kw) -> dict:

    from bench.evaluate import evaluate_protocol
    from bench.external_baselines import PrecomputedScoreDetector

    outputs = run_adapter(repo_dir, spec, list(corpus.texts), **run_kw)
    scores = {i: float(s) for i, s in enumerate(outputs)}
    result = evaluate_protocol(lambda: PrecomputedScoreDetector(scores), corpus)
    return {"n": len(corpus), "protocol": result.get("metrics", result)}


def evaluate_repo(
    url: str,
    kind: str,
    *,
    ref: str | None = None,
    adapter_path: Path | None = None,
    generations_card: dict | None = None,
    corpus=None,
    docker: bool = False,
    timeout: int = 180,
    dest_root: Path = REPOS_DIR,
) -> dict:
    """Clone ``url``, resolve its adapter, run it, and route to the eval for ``kind``."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")
    repo_dir = clone_repo(url, ref, dest_root)
    if adapter_path is not None:
        spec = inject_adapter(repo_dir, Path(adapter_path), kind)
    else:
        spec = find_adapter(repo_dir)
        if spec is None:
            raise RuntimeError(
                f"no panoptes.adapter.json or panoptes_adapter.py in {url}; "
                "supply one with --adapter-path"
            )
        if not spec.kind:
            spec.kind = kind
        if not spec.callable:
            spec.callable = DEFAULT_CALLABLE[kind]

    run_kw = {"timeout": timeout, "docker": docker}
    if kind == "watermark-remover":
        result = evaluate_remover(repo_dir, spec, generations_card, **run_kw)
    elif kind == "watermark-scheme":
        result = evaluate_scheme(repo_dir, spec, generations_card, **run_kw)
    else:
        result = evaluate_detector(repo_dir, spec, corpus, **run_kw)

    return {
        "schema": "panoptes-external-repo-eval-v1",
        "repo": {"url": url, "ref": ref or "HEAD", "dir_sha": _repo_sha(url, ref)},
        "adapter": {
            "kind": spec.kind,
            "name": spec.name,
            "version": spec.version,
            "module": spec.module,
            "callable": spec.callable,
            "source": spec.source,
        },
        "kind": kind,
        "result": result,
        "trusted_warning": TRUSTED_WARNING,
        "docker": docker,
    }
