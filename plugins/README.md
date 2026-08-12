# Panoptes plugins

Plugins extend Panoptes without changing the core evidence contract.

Supported plugin surfaces in the initial architecture:

1. `WatermarkAdapter` — returns per-scheme statistics and optional per-token evidence.
2. `DetectorAdapter` — returns calibrated class distributions for a supported content type.
3. `FeatureExtractor` — returns numeric segment features for calibration and source-family analysis.
4. `Calibrator` — maps raw scores to probabilities with a versioned calibration bundle.
5. `Exporter` — converts an `AnalysisResponse` into an external report format.

Plugin rules:

- Plugins are loaded only from explicit local paths.
- Plugins must declare ID, version, license, supported content types, and limitations.
- Plugins must not fetch remote code at runtime.
- Plugins must not store submitted text unless the user explicitly enables report retention.
- Probability fields must use the shared response schema.
- Watermark plugins should export token spans only as `{start, end, green}` offsets, never raw secret keys.
- New marker schemes require a watermark evaluation card and a dataset pointer manifest before registry enablement.

A minimal detector adapter:

```python
class MyDetector:
    id = "my-detector"
    min_tokens = 50
    content_types = ("prose",)
    languages = ("en",)

    def score(self, text, content_type, language):
        ...
```
