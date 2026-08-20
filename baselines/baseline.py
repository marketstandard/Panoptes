"""Panoptes baseline run tooling.

Standard-library-only CLI for producing, hashing, cataloging, and verifying
model baseline runs. See baselines/README.md for the full workflow.

Commands: scaffold, run, finalize, anchor, promote, submit, verify-catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_MANIFEST = ROOT / "baselines" / "prompts" / "prompts.manifest.json"
VALIDATOR = ROOT / "bench" / "validate_submission.py"
RUNS = ROOT / "baselines" / "runs"
REFERENCE = ROOT / "baselines" / "reference"
CATALOG = ROOT / "baselines" / "catalog"
REGISTRY = CATALOG / "registry.jsonl"
MANIFESTS = CATALOG / "manifests"

TOOL = "baseline.py/1.0.0"
SCHEMA_ID = "panoptes-baseline-run-v1"
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
MANIFEST_NAME = "run.manifest.json"
SCAFFOLD_STATE = "_run.json"
CHECKLIST_NAME = "_CHECKLIST.md"
KINDS = ("text", "code")
INTERFACES = ("chat-ui", "api", "agent-chat", "human")


class BaselineError(Exception):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_hash(payload: dict) -> str:
    """Canonical SHA-256, identical to bench/validate_submission.py."""
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
    """Pairwise SHA-256 Merkle root over sorted hex digest strings.

    An odd leaf duplicates itself at each level. The empty set hashes the
    empty byte string so the result is still a well-defined digest.
    """
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


def load_prompt_manifest() -> tuple[dict, str]:
    payload = json.loads(PROMPTS_MANIFEST.read_text(encoding="utf-8"))
    return payload, canonical_hash(payload)


def prompts_for_kind(manifest: dict, kind: str) -> list[dict]:
    prompts = [p for p in manifest["prompts"] if p["kind"] == kind]
    if not prompts:
        raise BaselineError(f"no prompts of kind {kind!r} in {PROMPTS_MANIFEST}")
    return prompts


def check_slug(slug: str) -> str:
    slug = slug.strip().lower()
    if not SLUG_RE.match(slug):
        raise BaselineError(
            f"invalid model slug {slug!r}: use lowercase letters, digits, dots, dashes "
            "(e.g. gpt-5.6-sol, claude-opus-5)"
        )
    return slug


def run_dir_for(slug: str, kind: str) -> Path:
    return RUNS / f"{slug}_{kind}"


def scaffold_state_path(run_dir: Path) -> Path:
    return run_dir / SCAFFOLD_STATE


def write_scaffold_state(run_dir: Path, state: dict) -> None:
    scaffold_state_path(run_dir).write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_scaffold_state(run_dir: Path) -> dict:
    path = scaffold_state_path(run_dir)
    if not path.exists():
        raise BaselineError(
            f"{run_dir} is not a scaffolded run folder (missing {SCAFFOLD_STATE}); "
            "run `scaffold` first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def checklist_text(kind: str, prompts: list[dict]) -> str:
    lines = [
        f"# Baseline run checklist — {kind}",
        "",
        "Protocol (full version in baselines/prompts/):",
        "",
        "1. Fresh session per prompt; no history, memory, system prompt, or custom instructions.",
        "2. Provider default sampling settings; single turn; no browsing or tools.",
        "3. Paste the prompt verbatim from baselines/prompts/" + kind + ".md.",
        "4. Save the raw, unedited reply into the file named below.",
        "5. Record the exact model version string shown by the product in `_run.json`.",
        "",
        "Files:",
        "",
    ]
    lines += [f"- [ ] `{p['id']}.md` — {p['title']}" for p in prompts]
    lines += [
        "",
        "When every box is checked:",
        "",
        "```bash",
        f"python baselines/baseline.py finalize --run <this folder>",
        "```",
        "",
    ]
    return "\n".join(lines)


def cmd_scaffold(args: argparse.Namespace) -> int:
    kind = args.kind
    manifest, prompts_hash = load_prompt_manifest()
    prompts = prompts_for_kind(manifest, kind)
    slug = check_slug(args.model) if args.model else None
    run_dir = run_dir_for(slug, kind) if slug else RUNS / f"_pending_{kind}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise BaselineError(f"{run_dir} already exists and is not empty; refusing to overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    for prompt in prompts:
        (run_dir / f"{prompt['id']}.md").touch()
    state = {
        "kind": kind,
        "model": slug,
        "provider": args.provider,
        "reported_version": None,
        "interface": None,
        "prompts_version": manifest["version"],
        "prompts_sha256": prompts_hash,
        "scaffolded_utc": utc_now(),
    }
    write_scaffold_state(run_dir, state)
    (run_dir / CHECKLIST_NAME).write_text(checklist_text(kind, prompts), encoding="utf-8")
    print(f"Scaffolded {kind} run at {run_dir}")
    print(f"Fill {len(prompts)} files, then: python baselines/baseline.py finalize --run {run_dir}")
    return 0


def _post_json(url: str, headers: dict, body: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise BaselineError(f"API request failed ({error.code}): {detail}") from None


def _call_provider(provider: str, model_id: str, prompt: str, timeout: int) -> tuple[str, str]:
    """Return (output_text, reported_version). Keys come from the environment only."""
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise BaselineError("OPENAI_API_KEY is not set")
        payload = _post_json(
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {key}"},
            {"model": model_id, "messages": [{"role": "user", "content": prompt}]},
            timeout,
        )
        return payload["choices"][0]["message"]["content"], payload.get("model", model_id)
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise BaselineError("ANTHROPIC_API_KEY is not set")
        payload = _post_json(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
            {
                "model": model_id,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout,
        )
        text = "".join(block.get("text", "") for block in payload.get("content", []))
        return text, payload.get("model", model_id)
    raise BaselineError(f"unsupported provider {provider!r} (supported: openai, anthropic)")


def cmd_run(args: argparse.Namespace) -> int:
    kind = args.kind
    slug = check_slug(args.model)
    model_id = args.model_id or slug
    manifest, prompts_hash = load_prompt_manifest()
    prompts = prompts_for_kind(manifest, kind)
    run_dir = run_dir_for(slug, kind)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        raise BaselineError(f"{run_dir} already exists; pass --force to rerun and overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    reported_version = model_id
    for prompt in prompts:
        output_path = run_dir / f"{prompt['id']}.md"
        print(f"Running {prompt['id']} via {args.provider} ({model_id}) ...")
        text, reported_version = _call_provider(args.provider, model_id, prompt["prompt"], args.timeout)
        if not text.strip():
            raise BaselineError(f"empty response for {prompt['id']}; aborting before finalize")
        output_path.write_text(text, encoding="utf-8")
    write_scaffold_state(
        run_dir,
        {
            "kind": kind,
            "model": slug,
            "provider": args.provider,
            "reported_version": reported_version,
            "interface": "api",
            "prompts_version": manifest["version"],
            "prompts_sha256": prompts_hash,
            "scaffolded_utc": utc_now(),
        },
    )
    print(f"Wrote {len(prompts)} outputs to {run_dir}")
    print(f"Next: python baselines/baseline.py finalize --run {run_dir}")
    return 0


def build_manifest(run_dir: Path, state: dict) -> dict:
    kind = state["kind"]
    manifest, prompts_hash = load_prompt_manifest()
    if state.get("prompts_sha256") != prompts_hash:
        raise BaselineError(
            "the prompt manifest has changed since this run was scaffolded; "
            "re-scaffold so runs stay pinned to one prompts version"
        )
    prompts = prompts_for_kind(manifest, kind)
    outputs = []
    missing = []
    for prompt in prompts:
        path = run_dir / f"{prompt['id']}.md"
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            missing.append(prompt["id"])
            continue
        outputs.append(
            {
                "prompt_id": prompt["id"],
                "file": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    if missing:
        raise BaselineError(f"run is incomplete; missing or empty outputs: {', '.join(missing)}")
    stamp = utc_now()
    run_stamp = stamp.replace(":", "").replace("-", "").lower()
    return {
        "schema": SCHEMA_ID,
        "run_id": f"{state['model']}_{kind}-{run_stamp}",
        "model": {
            "slug": state["model"],
            "provider": state.get("provider") or "unspecified",
            "reported_version": state.get("reported_version") or state["model"],
            "interface": state["interface"],
        },
        "prompts": {"version": manifest["version"], "sha256": prompts_hash},
        "created_utc": stamp,
        "environment": {
            "os": platform.system().lower(),
            "python": platform.python_version(),
            "tool": TOOL,
        },
        "outputs": outputs,
        "merkle_root": merkle_root([o["sha256"] for o in outputs]),
    }


def write_manifest(run_dir: Path, manifest: dict) -> Path:
    manifest["artifact_sha256"] = canonical_hash(manifest)
    path = run_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def cmd_finalize(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    state = read_scaffold_state(run_dir)
    if args.model:
        state["model"] = check_slug(args.model)
    if args.interface:
        if args.interface not in INTERFACES:
            raise BaselineError(f"interface must be one of {', '.join(INTERFACES)}")
        state["interface"] = args.interface
    if args.provider:
        state["provider"] = args.provider
    if args.reported_version:
        state["reported_version"] = args.reported_version
    if not state.get("model"):
        raise BaselineError(
            "no model slug recorded; pass --model <slug> (the person running the test "
            "declares the model; the tool never guesses)"
        )
    if not state.get("interface"):
        raise BaselineError("no interface recorded; pass --interface chat-ui|api|agent-chat|human")
    manifest = build_manifest(run_dir, state)
    manifest_path = write_manifest(run_dir, manifest)
    write_scaffold_state(run_dir, state)

    target = run_dir_for(state["model"], state["kind"])
    if run_dir.name.startswith("_pending_") and run_dir != target:
        if target.exists():
            raise BaselineError(f"{target} already exists; move or remove it first")
        run_dir.rename(target)
        manifest_path = target / MANIFEST_NAME
        print(f"Renamed run folder to {target}")
    print(f"Wrote {manifest_path}")
    print(f"manifest sha256: {manifest['artifact_sha256']}")
    print(f"merkle root:     {manifest['merkle_root']}")
    return 0


def cmd_anchor(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    manifest_path = run_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise BaselineError(f"{manifest_path} not found; run finalize first")
    ots = shutil.which("ots")
    if not ots:
        print("OpenTimestamps client not found. Install it and rerun:")
        print("  pip install opentimestamps-client")
        print("  ots stamp <manifest>   # creates a Bitcoin-anchored timestamp proof")
        return 1
    subprocess.run([ots, "stamp", str(manifest_path)], check=True)
    proof = manifest_path.with_suffix(manifest_path.suffix + ".ots")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["anchor"] = {"type": "opentimestamps", "proof_file": proof.name}
    write_manifest(run_dir, {k: v for k, v in manifest.items() if k != "artifact_sha256"})
    print(f"Stamped {manifest_path}; proof at {proof}")
    print("Note: the manifest hash changed because the anchor field was added; resubmit if needed.")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    manifest_path = run_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise BaselineError(f"{manifest_path} not found; run finalize first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dest = REFERENCE / run_dir.name
    if dest.exists():
        raise BaselineError(f"{dest} already exists; refusing to overwrite a reference run")
    dest.mkdir(parents=True)
    for output in manifest["outputs"]:
        shutil.copy2(run_dir / output["file"], dest / output["file"])
    shutil.copy2(manifest_path, dest / MANIFEST_NAME)
    proof = manifest_path.with_suffix(manifest_path.suffix + ".ots")
    if proof.exists():
        shutil.copy2(proof, dest / proof.name)
    print(f"Promoted {run_dir.name} to {dest}")
    return 0


def _validate_artifacts(paths: list[Path]) -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), *[str(p) for p in paths]],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise BaselineError(f"artifact validation failed:\n{result.stdout}{result.stderr}")


def cmd_submit(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    manifest_path = run_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise BaselineError(f"{manifest_path} not found; run finalize first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.contributor:
        manifest["contributor"] = args.contributor
        manifest_path = write_manifest(run_dir, manifest)
    expected = manifest.get("artifact_sha256")
    if expected != canonical_hash(manifest):
        raise BaselineError("manifest artifact_sha256 does not match its contents; refinalize")
    _validate_artifacts([manifest_path])

    kind = manifest["outputs"][0]["prompt_id"].rsplit("-", 1)[0]
    line = {
        "run_id": manifest["run_id"],
        "model_slug": manifest["model"]["slug"],
        "kind": kind,
        "prompts_sha256": manifest["prompts"]["sha256"],
        "manifest_sha256": expected,
        "merkle_root": manifest["merkle_root"],
        "submitted_utc": utc_now(),
        "contributor": manifest.get("contributor", "anonymous"),
    }
    proof = manifest_path.with_suffix(manifest_path.suffix + ".ots")
    if proof.exists():
        line["ots_proof"] = f"{expected}.ots"

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    existing = [
        json.loads(entry)
        for entry in REGISTRY.read_text(encoding="utf-8").splitlines()
        if entry.strip()
    ] if REGISTRY.exists() else []
    if any(entry["run_id"] == line["run_id"] for entry in existing):
        raise BaselineError(f"run_id {line['run_id']!r} is already in the registry")
    stored_manifest = MANIFESTS / f"{expected}.json"
    shutil.copy2(manifest_path, stored_manifest)
    if proof.exists():
        shutil.copy2(proof, MANIFESTS / f"{expected}.ots")
    with REGISTRY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")
    print(f"Registered run {line['run_id']}")
    print(f"  manifest: {stored_manifest}")
    print(f"  registry line appended to {REGISTRY}")
    print("Raw outputs were NOT copied; only hashes entered the catalog.")
    return 0


def cmd_verify_catalog(args: argparse.Namespace) -> int:
    errors: list[str] = []
    lines = []
    if REGISTRY.exists():
        for number, raw in enumerate(REGISTRY.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                lines.append(json.loads(raw))
            except json.JSONDecodeError as error:
                errors.append(f"registry.jsonl line {number}: invalid JSON ({error})")
    seen_run_ids: set[str] = set()
    manifest_paths: list[Path] = []
    for entry in lines:
        run_id = entry.get("run_id", "<missing>")
        if run_id in seen_run_ids:
            errors.append(f"duplicate run_id {run_id!r} in registry")
        seen_run_ids.add(run_id)
        manifest_hash = entry.get("manifest_sha256", "")
        manifest_path = MANIFESTS / f"{manifest_hash}.json"
        if not manifest_path.exists():
            errors.append(f"{run_id}: manifest {manifest_hash}.json missing from catalog")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_paths.append(manifest_path)
        if manifest.get("artifact_sha256") != manifest_hash:
            errors.append(f"{run_id}: manifest filename does not match its artifact_sha256")
        if canonical_hash(manifest) != manifest_hash:
            errors.append(f"{run_id}: manifest contents do not hash to {manifest_hash}")
        if manifest.get("merkle_root") != entry.get("merkle_root"):
            errors.append(f"{run_id}: merkle_root mismatch between registry and manifest")
        if manifest.get("run_id") != run_id:
            errors.append(f"{run_id}: run_id mismatch between registry and manifest")
        if "ots_proof" in entry and not (MANIFESTS / entry["ots_proof"]).exists():
            errors.append(f"{run_id}: missing OTS proof {entry['ots_proof']}")
    stray = {p.name for p in MANIFESTS.glob("*.json")} - {f"{e['manifest_sha256']}.json" for e in lines}
    for name in sorted(stray):
        errors.append(f"manifest {name} is not referenced by any registry line")
    if manifest_paths:
        _validate_artifacts(manifest_paths)
    if errors:
        for error in errors:
            print(f"error: {error}")
        return 1
    print(f"Catalog OK: {len(lines)} registered run(s), {len(manifest_paths)} manifest(s).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Panoptes baseline run tooling (see baselines/README.md)"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    scaffold = commands.add_parser("scaffold", help="create an empty run folder to fill manually")
    scaffold.add_argument("--model", help="model slug, e.g. gpt-5.6-sol (omit for agent-assisted runs)")
    scaffold.add_argument("--kind", choices=KINDS, required=True)
    scaffold.add_argument("--provider", default=None, help="provider label recorded in the manifest")
    scaffold.set_defaults(func=cmd_scaffold)

    run = commands.add_parser("run", help="execute the prompt set against a provider API")
    run.add_argument("--model", required=True, help="model slug used for the folder and manifest")
    run.add_argument("--provider", choices=("openai", "anthropic"), required=True)
    run.add_argument("--kind", choices=KINDS, required=True)
    run.add_argument("--model-id", help="provider API model id if it differs from the slug")
    run.add_argument("--timeout", type=int, default=120)
    run.add_argument("--force", action="store_true", help="overwrite an existing run folder")
    run.set_defaults(func=cmd_run)

    finalize = commands.add_parser("finalize", help="hash outputs and write run.manifest.json")
    finalize.add_argument("--run", required=True, help="run folder path")
    finalize.add_argument("--model", help="model slug (required if not set at scaffold time)")
    finalize.add_argument("--interface", choices=INTERFACES)
    finalize.add_argument("--provider")
    finalize.add_argument("--reported-version", help="exact model version string shown by the product")
    finalize.set_defaults(func=cmd_finalize)

    anchor = commands.add_parser("anchor", help="OpenTimestamps-stamp the manifest (optional)")
    anchor.add_argument("--run", required=True)
    anchor.set_defaults(func=cmd_anchor)

    promote = commands.add_parser("promote", help="copy a finalized run into baselines/reference/")
    promote.add_argument("--run", required=True)
    promote.set_defaults(func=cmd_promote)

    submit = commands.add_parser("submit", help="append the run to the community catalog (hashes only)")
    submit.add_argument("--run", required=True)
    submit.add_argument("--contributor", help="handle recorded in the registry")
    submit.set_defaults(func=cmd_submit)

    verify = commands.add_parser("verify-catalog", help="validate registry and manifests")
    verify.set_defaults(func=cmd_verify_catalog)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BaselineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
