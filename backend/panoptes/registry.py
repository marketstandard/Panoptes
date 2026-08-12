from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class DetectorRegistration:
    id: str
    version: str
    kind: Literal["prose", "code", "watermark", "provenance"]
    status: Literal["enabled", "fixture", "disabled"]
    model_revision: str | None = None
    license: str | None = None
    content_types: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    min_tokens: int = 0
    requirements: tuple[str, ...] = ()
    known_limitations: tuple[str, ...] = field(default_factory=tuple)


DETECTOR_REGISTRY: tuple[DetectorRegistration, ...] = (
    DetectorRegistration(
        id="fixture-detector",
        version="1.0.0",
        kind="prose",
        status="fixture",
        content_types=("prose", "code", "mixed"),
        languages=("en",),
        min_tokens=1,
        known_limitations=("Deterministic fixture scores are not calibrated evidence.",),
    ),
    DetectorRegistration(
        id="desklib-ai-text-detector-v1.01",
        version="hf-pinned",
        kind="prose",
        status="enabled",
        model_revision="pinned-at-build",
        license="MIT",
        content_types=("prose",),
        languages=("en",),
        min_tokens=80,
        requirements=("torch", "transformers"),
        known_limitations=(
            "English prose calibration only.",
            "Short and heavily rewritten text can be unreliable.",
        ),
    ),
    DetectorRegistration(
        id="droiddetect-base-ternary",
        version="hf-pinned",
        kind="code",
        status="enabled",
        model_revision="pinned-at-build",
        license="Apache-2.0",
        content_types=("code",),
        languages=("c", "cpp", "csharp", "go", "java", "javascript", "python"),
        min_tokens=40,
        requirements=("torch", "transformers"),
        known_limitations=(
            "Formatting and semantic-preserving edits can change evidence.",
            "Very small snippets receive low power.",
        ),
    ),
    DetectorRegistration(
        id="kgw-v1",
        version="reference",
        kind="watermark",
        status="enabled",
        content_types=("prose", "code"),
        min_tokens=50,
        known_limitations=(
            "Requires the same tokenizer and configuration used during generation.",
            "Cannot detect private vendor watermarks without their published detector.",
        ),
    ),
    DetectorRegistration(
        id="claude-text-watermark",
        version="pending",
        kind="watermark",
        status="disabled",
        content_types=("prose", "code"),
        known_limitations=("Awaiting public Anthropic detector specification and API.",),
    ),
)


def enabled_registrations(profile: str) -> list[DetectorRegistration]:
    if profile == "fixture":
        return [item for item in DETECTOR_REGISTRY if item.status == "fixture"]
    return [item for item in DETECTOR_REGISTRY if item.status != "fixture"]
