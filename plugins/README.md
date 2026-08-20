# Panoptes plugins

Plugins extend Panoptes without changing the core evidence contract.

Supported plugin surfaces:

1. `WatermarkAdapter` — returns per-scheme statistics and optional per-token evidence.
2. `DetectorAdapter` — returns calibrated class distributions for a supported content type.
3. `FeatureExtractor` — returns numeric segment features for calibration and source-family analysis.
4. `Calibrator` — maps raw scores to probabilities with a versioned calibration bundle.
5. `Exporter` — converts an `AnalysisResponse` into an external report format.

## Loader semantics

Configure local paths via `PANOPTES_PLUGIN_PATHS` or `Settings.plugin_paths`. The
env var accepts a single path, an `os.pathsep`-separated list (`;` on Windows,
`:` on POSIX), or a JSON list:

```bash
export PANOPTES_PLUGIN_PATHS=./plugins/my_watermark.py
# or several at once: export PANOPTES_PLUGIN_PATHS='./plugins/a.py:./plugins/b.py'
# or JSON:              export PANOPTES_PLUGIN_PATHS='["./plugins/my_watermark.py"]'
panoptes plugins list
panoptes plugins doctor
```

Each path may be a single `.py` file, a package directory (with `__init__.py` or
`plugin.py`), or a plain directory — plain directories are scanned for top-level
`*.py` modules (skipping `_`-prefixed files), and each module is loaded as one
plugin.

Rules:

- Plugins are loaded only from explicit local paths (never remote URLs).
- Plugins must declare `id`, `version`, `license`, `content_types`, and `limitations`.
- Plugins must not fetch remote *code* at runtime.
- Plugins must not store submitted text unless the user explicitly enables report retention.
- Probability fields must use the shared response schema.
- Watermark plugins should export token spans only as `{start, end, green}` offsets, never raw secret keys.
- Loaded watermark schemes are namespaced as `plugin:<id>` and marked `origin: "plugin"` in the analysis response.
- Exceptions inside a plugin become abstention / `adapter_unavailable` results — they must not crash `analyze()`.
- New marker schemes still require a watermark evaluation card and a dataset pointer manifest before registry enablement of a *built-in* adapter; plugins are opt-in operator local extensions.

### External oracle plugins (future)

When a provider ships a public detection API (e.g. Anthropic SynthID-Text detector),
an opt-in plugin may call that API for *data*. The "no remote fetch" rule covers
**code**, not authenticated oracle queries the operator explicitly enables. Such a
plugin must declare network use in `limitations` and remain disabled by default.

## Minimal watermark adapter

```python
from panoptes.analysis.watermarks import WatermarkAdapter, _empty_result

class MyWatermark:
    id = "my-watermark"
    version = "0.1.0"
    license = "MIT"
    content_types = ("prose",)
    limitations = ("example plugin",)

    def detect(self, text, content_type, include_tokens=False):
        # Return WatermarkResult + optional WatermarkToken list
        return _empty_result("my-watermark", "insufficient_data", 0), []

PLUGIN = MyWatermark()
```

A package directory with `__init__.py` or `plugin.py` also works, as does a plain
directory of `.py` files. Alternatively export `register()` returning
`[(kind, instance), ...]`.

## Minimal detector adapter

```python
class MyDetector:
    id = "my-detector"
    version = "0.1.0"
    license = "MIT"
    min_tokens = 50
    content_types = ("prose",)
    languages = ("en",)
    limitations = ("example",)

    def score(self, text, content_type, language):
        ...
```
