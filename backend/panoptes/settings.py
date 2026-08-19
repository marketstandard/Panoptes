from pydantic_settings import BaseSettings, SettingsConfigDict

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
    plugin_paths: list[str] = []

    @property
    def is_cloud(self) -> bool:
        return self.profile in {RuntimeProfile.CLOUD_CPU, RuntimeProfile.CLOUD_GPU}


def get_settings() -> Settings:
    return Settings()
