"""Runtime plugin loader for Panoptes.

Plugins extend Panoptes without changing the core evidence contract. They are
loaded only from explicit local paths in ``Settings.plugin_paths`` — never from
remote URLs. Failures become abstentions; a plugin must not crash ``analyze()``.

Supported surfaces: WatermarkAdapter, DetectorAdapter, FeatureExtractor,
Calibrator, Exporter. See plugins/README.md.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from panoptes.analysis.detectors import DetectorAdapter, DetectorScore
from panoptes.analysis.watermarks import WatermarkAdapter, WatermarkToken, _empty_result
from panoptes.registry import DetectorRegistration
from panoptes.schemas import ContentType, WatermarkResult
from panoptes.settings import Settings

PluginKind = Literal[
    "watermark",
    "detector",
    "feature_extractor",
    "calibrator",
    "exporter",
]

REQUIRED_METADATA = ("id", "version", "license", "content_types", "limitations")


@dataclass
class LoadedPlugin:
    kind: PluginKind
    path: str
    module_name: str
    status: Literal["ok", "error"]
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    instance: Any = None
    registration: DetectorRegistration | None = None


@dataclass
class PluginRegistry:
    watermarks: list[WatermarkAdapter] = field(default_factory=list)
    detectors: list[DetectorAdapter] = field(default_factory=list)
    loaded: list[LoadedPlugin] = field(default_factory=list)

    def watermark_registrations(self) -> list[DetectorRegistration]:
        return [p.registration for p in self.loaded if p.registration and p.kind == "watermark"]

    def detector_registrations(self) -> list[DetectorRegistration]:
        return [p.registration for p in self.loaded if p.registration and p.kind == "detector"]


class SafeWatermarkAdapter(WatermarkAdapter):
    """Wrap a plugin watermark adapter so exceptions become abstentions."""

    def __init__(self, inner: Any, plugin_id: str):
        self._inner = inner
        self.id = f"plugin:{plugin_id}"
        self.origin = "plugin"

    def detect(
        self,
        text: str,
        content_type: ContentType,
        include_tokens: bool = False,
    ) -> tuple[WatermarkResult, list[WatermarkToken]]:
        try:
            result, tokens = self._inner.detect(text, content_type, include_tokens=include_tokens)
        except Exception:
            empty = _empty_result(self.id, "adapter_unavailable", 0)
            return empty.model_copy(update={"scheme": self.id, "origin": "plugin"}), []
        if not isinstance(result, WatermarkResult):
            return _empty_result(self.id, "adapter_unavailable", 0).model_copy(
                update={"scheme": self.id, "origin": "plugin"}
            ), []
        # Namespace scheme and enforce token span shape.
        safe_tokens: list[WatermarkToken] = []
        for token in tokens or []:
            if not all(hasattr(token, attr) for attr in ("start", "end", "green")):
                continue
            safe_tokens.append(
                WatermarkToken(
                    token=getattr(token, "token", ""),
                    start=int(token.start),
                    end=int(token.end),
                    green=bool(token.green),
                )
            )
        updates: dict[str, Any] = {"scheme": self.id, "origin": "plugin"}
        if result.tokens is not None and include_tokens:
            from panoptes.schemas import WatermarkTokenSpan

            updates["tokens"] = [
                WatermarkTokenSpan(start=t.start, end=t.end, green=t.green) for t in result.tokens
            ]
        elif not include_tokens:
            updates["tokens"] = None
        return result.model_copy(update=updates), safe_tokens


class SafeDetectorAdapter(DetectorAdapter):
    """Wrap a plugin detector so exceptions become abstentions."""

    def __init__(self, inner: Any, plugin_id: str):
        self._inner = inner
        self.id = f"plugin:{plugin_id}"
        self.min_tokens = int(getattr(inner, "min_tokens", 1) or 1)
        raw_types = getattr(inner, "content_types", ()) or ()
        self.content_types = tuple(
            ContentType(t) if not isinstance(t, ContentType) else t for t in raw_types
        )
        self.languages = tuple(getattr(inner, "languages", ()) or ())

    def score(self, text: str, content_type: ContentType, language: str) -> DetectorScore:
        try:
            return self._inner.score(text, content_type, language)
        except Exception as exc:  # noqa: BLE001
            from panoptes.schemas import OutcomeDistribution

            return DetectorScore(
                distribution=OutcomeDistribution(
                    human=1 / 3, ai_generated=1 / 3, ai_refined_or_mixed=1 / 3
                ),
                raw_score=0.5,
                detector_id=self.id,
                abstain_reason=f"Plugin error: {exc}",
            )


def _read_metadata(obj: Any) -> dict[str, Any] | str:
    meta: dict[str, Any] = {}
    for key in REQUIRED_METADATA:
        if not hasattr(obj, key):
            return f"missing required metadata field {key!r}"
        meta[key] = getattr(obj, key)
    if not isinstance(meta["id"], str) or not meta["id"]:
        return "id must be a non-empty string"
    if not isinstance(meta["version"], str):
        return "version must be a string"
    if not isinstance(meta["license"], str):
        return "license must be a string"
    if not isinstance(meta["content_types"], (list, tuple)):
        return "content_types must be a list or tuple"
    if not isinstance(meta["limitations"], (list, tuple)):
        return "limitations must be a list or tuple"
    return meta


def _import_path(path: Path) -> tuple[Any | None, str | None]:
    if not path.exists():
        return None, f"path does not exist: {path}"
    if path.is_dir():
        candidate = path / "__init__.py"
        if not candidate.exists():
            # Also accept a single plugin.py inside the directory.
            candidate = path / "plugin.py"
            if not candidate.exists():
                return None, f"directory has no importable plugin modules (*.py): {path}"
        path = candidate
    if path.suffix != ".py":
        return None, f"plugin path must be a .py file or package directory: {path}"
    module_name = f"panoptes_plugin_{path.stem}_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None, f"could not create import spec for {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        return None, f"import failed: {exc}"
    return module, None


def _discover_exports(module: Any) -> list[tuple[PluginKind, Any]]:
    found: list[tuple[PluginKind, Any]] = []
    for attr_name in ("PLUGIN", "WATERMARK_ADAPTER", "DETECTOR_ADAPTER", "ADAPTER"):
        obj = getattr(module, attr_name, None)
        if obj is None:
            continue
        kind = _infer_kind(obj)
        if kind:
            found.append((kind, obj))
    # Also accept a factory: register() -> list of (kind, instance) or instances
    register = getattr(module, "register", None)
    if callable(register):
        try:
            items = register()
        except Exception:
            items = []
        for item in items or []:
            if isinstance(item, tuple) and len(item) == 2:
                kind, obj = item
                if kind in {"watermark", "detector", "feature_extractor", "calibrator", "exporter"}:
                    found.append((kind, obj))  # type: ignore[arg-type]
            else:
                kind = _infer_kind(item)
                if kind:
                    found.append((kind, item))
    # Module-level adapter instances (must look like intentional exports)
    for value in vars(module).values():
        if isinstance(value, type):
            continue
        kind = _infer_kind(value)
        if not kind or (kind, value) in found:
            continue
        method = "detect" if kind == "watermark" else "score"
        if hasattr(value, "id") and callable(getattr(value, method, None)):
            found.append((kind, value))
    return found


def _infer_kind(obj: Any) -> PluginKind | None:
    if obj is None:
        return None
    instance = obj
    if isinstance(obj, type):
        try:
            instance = obj()
        except Exception:
            return None
    if callable(getattr(instance, "detect", None)) and hasattr(instance, "id"):
        return "watermark"
    if callable(getattr(instance, "score", None)) and hasattr(instance, "id"):
        return "detector"
    if callable(getattr(instance, "extract", None)):
        return "feature_extractor"
    if callable(getattr(instance, "calibrate", None)):
        return "calibrator"
    if callable(getattr(instance, "export", None)):
        return "exporter"
    return None


def _instantiate(obj: Any) -> Any:
    if isinstance(obj, type):
        return obj()
    return obj


def _expand_paths(path_list: Iterable[str]) -> list[Path]:
    """Resolve configured paths to plugin files.

    A directory without ``__init__.py``/``plugin.py`` is scanned for top-level
    ``*.py`` modules (skipping ``_``-prefixed files); each module is one plugin.
    """
    expanded: list[Path] = []
    for raw in path_list:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            # Resolve relative to CWD (operator-declared).
            path = path.resolve()
        if (
            path.is_dir()
            and not (path / "__init__.py").exists()
            and not (path / "plugin.py").exists()
        ):
            modules = sorted(p for p in path.glob("*.py") if not p.name.startswith(("_", ".")))
            if modules:
                expanded.extend(modules)
                continue
        expanded.append(path)
    return expanded


def load_plugins(
    settings: Settings | None = None, paths: list[str] | None = None
) -> PluginRegistry:
    """Load and validate plugins from explicit local paths."""
    settings = settings or Settings()
    path_list = paths if paths is not None else list(settings.plugin_paths or [])
    registry = PluginRegistry()
    for path in _expand_paths(path_list):
        module, error = _import_path(path)
        if error or module is None:
            registry.loaded.append(
                LoadedPlugin(
                    kind="watermark",
                    path=str(path),
                    module_name="",
                    status="error",
                    error=error or "unknown import error",
                )
            )
            continue
        exports = _discover_exports(module)
        if not exports:
            registry.loaded.append(
                LoadedPlugin(
                    kind="watermark",
                    path=str(path),
                    module_name=module.__name__,
                    status="error",
                    error="no plugin adapter export found (expected PLUGIN / register())",
                )
            )
            continue
        for kind, raw_obj in exports:
            try:
                instance = _instantiate(raw_obj)
            except Exception as exc:  # noqa: BLE001
                registry.loaded.append(
                    LoadedPlugin(
                        kind=kind,
                        path=str(path),
                        module_name=module.__name__,
                        status="error",
                        error=f"instantiation failed: {exc}",
                    )
                )
                continue
            meta = _read_metadata(instance)
            if isinstance(meta, str):
                registry.loaded.append(
                    LoadedPlugin(
                        kind=kind,
                        path=str(path),
                        module_name=module.__name__,
                        status="error",
                        error=meta,
                    )
                )
                continue
            plugin_id = str(meta["id"])
            registration: DetectorRegistration | None = None
            if kind in {"watermark", "detector"}:
                registration = DetectorRegistration(
                    id=f"plugin:{plugin_id}",
                    version=str(meta["version"]),
                    kind="watermark" if kind == "watermark" else "prose",
                    status="enabled",
                    license=str(meta["license"]),
                    content_types=tuple(str(t) for t in meta["content_types"]),
                    known_limitations=tuple(str(x) for x in meta["limitations"]),
                )
            if kind == "watermark":
                if not callable(getattr(instance, "detect", None)):
                    registry.loaded.append(
                        LoadedPlugin(
                            kind=kind,
                            path=str(path),
                            module_name=module.__name__,
                            status="error",
                            error="watermark plugin missing detect()",
                            metadata=meta,
                        )
                    )
                    continue
                safe = SafeWatermarkAdapter(instance, plugin_id)
                registry.watermarks.append(safe)
            elif kind == "detector":
                if not callable(getattr(instance, "score", None)):
                    registry.loaded.append(
                        LoadedPlugin(
                            kind=kind,
                            path=str(path),
                            module_name=module.__name__,
                            status="error",
                            error="detector plugin missing score()",
                            metadata=meta,
                        )
                    )
                    continue
                safe_det = SafeDetectorAdapter(instance, plugin_id)
                registry.detectors.append(safe_det)
            registry.loaded.append(
                LoadedPlugin(
                    kind=kind,
                    path=str(path),
                    module_name=module.__name__,
                    status="ok",
                    metadata=meta,
                    instance=instance,
                    registration=registration,
                )
            )
    return registry


# Process-level cache keyed by frozenset of paths.
_CACHE: dict[frozenset[str], PluginRegistry] = {}


def get_plugin_registry(settings: Settings) -> PluginRegistry:
    key = frozenset(settings.plugin_paths or [])
    if key not in _CACHE:
        _CACHE[key] = load_plugins(settings)
    return _CACHE[key]


def clear_plugin_cache() -> None:
    _CACHE.clear()
