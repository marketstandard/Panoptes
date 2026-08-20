import json
import os
from typing import Annotated, Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from panoptes.schemas import RuntimeProfile


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PANOPTES_", extra="ignore")

    profile: RuntimeProfile = RuntimeProfile.FIXTURE
    host: str = "127.0.0.1"
    port: int = 8000
    allow_origin: str = "http://127.0.0.1:5173"
    max_characters: int = 120_000
    max_file_bytes: int = 10_000_000
    enable_metrics: bool = False
    operator_token: str | None = None
    artifact_dir: str = "artifacts"
    calibration_bundle: str = "baseline-calibration.json"
    neural_enabled: bool = True
    neural_artifact_dir: str = "models/neural"
    # NoDecode: accept a plain path string from PANOPTES_PLUGIN_PATHS, not just JSON.
    plugin_paths: Annotated[list[str], NoDecode] = []

    @field_validator("plugin_paths", mode="before")
    @classmethod
    def _parse_plugin_paths(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                parsed = json.loads(text)
                return [str(item) for item in parsed]
            return [part for part in text.split(os.pathsep) if part]
        return [str(item) for item in value]

    @property
    def is_cloud(self) -> bool:
        return self.profile in {RuntimeProfile.CLOUD_CPU, RuntimeProfile.CLOUD_GPU}


def get_settings() -> Settings:
    return Settings()
