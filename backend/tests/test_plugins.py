"""Tests for the runtime plugin loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from panoptes.analysis.pipeline import analyze
from panoptes.analysis.watermarks import watermark_adapters
from panoptes.plugins import clear_plugin_cache, load_plugins
from panoptes.schemas import AnalysisRequest, ContentType, RuntimeProfile
from panoptes.settings import Settings

FIXTURES = Path(__file__).parent / "plugin_fixtures"


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_plugin_cache()
    yield
    clear_plugin_cache()


def test_loads_valid_watermark_plugin(tmp_path: Path):
    plugin = tmp_path / "valid_wm.py"
    plugin.write_text(
        """
from panoptes.analysis.watermarks import WatermarkAdapter, _empty_result
from panoptes.schemas import ContentType

class DemoWatermark(WatermarkAdapter):
    id = "demo-wm"
    version = "1.0.0"
    license = "MIT"
    content_types = ("prose",)
    limitations = ("fixture plugin",)

    def detect(self, text, content_type, include_tokens=False):
        return _empty_result("demo-wm", "insufficient_data", 0), []

PLUGIN = DemoWatermark()
""",
        encoding="utf-8",
    )
    registry = load_plugins(paths=[str(plugin)])
    assert len(registry.watermarks) == 1
    assert registry.watermarks[0].id == "plugin:demo-wm"
    assert registry.loaded[0].status == "ok"


def test_rejects_missing_metadata(tmp_path: Path):
    plugin = tmp_path / "bad_meta.py"
    plugin.write_text(
        """
class Broken:
    id = "broken"
    def detect(self, text, content_type, include_tokens=False):
        return None, []

PLUGIN = Broken()
""",
        encoding="utf-8",
    )
    registry = load_plugins(paths=[str(plugin)])
    assert registry.watermarks == []
    assert registry.loaded[0].status == "error"
    assert "missing required metadata" in (registry.loaded[0].error or "")


def test_raising_plugin_becomes_abstention(tmp_path: Path):
    plugin = tmp_path / "raising.py"
    plugin.write_text(
        """
class Boom:
    id = "boom"
    version = "0.0.1"
    license = "MIT"
    content_types = ("prose",)
    limitations = ("raises",)

    def detect(self, text, content_type, include_tokens=False):
        raise RuntimeError("explode")

PLUGIN = Boom()
""",
        encoding="utf-8",
    )
    registry = load_plugins(paths=[str(plugin)])
    assert len(registry.watermarks) == 1
    result, _ = registry.watermarks[0].detect("hello world", ContentType.PROSE)
    assert result.status == "adapter_unavailable"
    assert result.origin == "plugin"
    assert result.scheme == "plugin:boom"


def test_pipeline_merges_plugin_watermark(tmp_path: Path):
    plugin = tmp_path / "pipe_wm.py"
    plugin.write_text(
        """
from panoptes.analysis.watermarks import _empty_result

class Demo:
    id = "pipe-demo"
    version = "1.0.0"
    license = "Apache-2.0"
    content_types = ("prose",)
    limitations = ("test",)

    def detect(self, text, content_type, include_tokens=False):
        return _empty_result("pipe-demo", "insufficient_data", 3), []

PLUGIN = Demo()
""",
        encoding="utf-8",
    )
    settings = Settings(profile=RuntimeProfile.FIXTURE, plugin_paths=[str(plugin)])
    response = analyze(
        AnalysisRequest(
            text="human-written " + ("word " * 60),
            include_text=False,
        ),
        settings,
    )
    schemes = [w.scheme for w in response.watermarks]
    assert "plugin:pipe-demo" in schemes
    plugin_result = next(w for w in response.watermarks if w.scheme.startswith("plugin:"))
    assert plugin_result.origin == "plugin"


def test_watermark_adapters_without_settings_stays_builtin():
    adapters = watermark_adapters()
    assert [a.id for a in adapters] == ["kgw-v1", "claude-text-watermark"]


def test_missing_path_recorded_as_error():
    registry = load_plugins(paths=["/nonexistent/plugin/path.py"])
    assert registry.loaded[0].status == "error"


def test_plugin_paths_env_accepts_plain_path(monkeypatch, tmp_path: Path):
    plugin = tmp_path / "p.py"
    plugin.write_text("# empty\n", encoding="utf-8")
    monkeypatch.setenv("PANOPTES_PLUGIN_PATHS", str(plugin))
    assert Settings().plugin_paths == [str(plugin)]


def test_plugin_paths_env_accepts_json_and_pathsep(monkeypatch):
    monkeypatch.setenv("PANOPTES_PLUGIN_PATHS", '["/a/b.py", "/c"]')
    assert Settings().plugin_paths == ["/a/b.py", "/c"]
    monkeypatch.setenv("PANOPTES_PLUGIN_PATHS", f"/a/b.py{__import__('os').pathsep}/c")
    assert Settings().plugin_paths == ["/a/b.py", "/c"]
    monkeypatch.setenv("PANOPTES_PLUGIN_PATHS", "")
    assert Settings().plugin_paths == []


def test_directory_path_is_scanned_for_modules(tmp_path: Path):
    (tmp_path / "one.py").write_text(
        """
from panoptes.analysis.watermarks import _empty_result

class One:
    id = "one"
    version = "1.0.0"
    license = "MIT"
    content_types = ("prose",)
    limitations = ("test",)

    def detect(self, text, content_type, include_tokens=False):
        return _empty_result("one", "insufficient_data", 0), []

PLUGIN = One()
""",
        encoding="utf-8",
    )
    (tmp_path / "_skip.py").write_text("raise RuntimeError('must not import')\n", encoding="utf-8")
    registry = load_plugins(paths=[str(tmp_path)])
    ok = [p for p in registry.loaded if p.status == "ok"]
    assert [p.metadata["id"] for p in ok] == ["one"]
    assert all("_skip" not in p.path for p in registry.loaded)
