"""Configuration loader for FTAAS services."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "settings.yaml"


class ServiceEndpoint(BaseModel):
    host: str = "0.0.0.0"
    port: int
    db_url: str | None = None
    storage_root: str | None = None


class Integrations(BaseModel):
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment: str = "ftaas-finetune"
    ray_address: str = "auto"
    ray_mock: bool = True
    airflow_enabled: bool = False
    gcs_mock: bool = True
    gcs_local_mirror: str = "./data/gcs_mirror"


class Settings(BaseSettings):
    env: str = "local"
    config_path: str = str(DEFAULT_CONFIG)
    # Unified gateway (single process). Override only if splitting services.
    jobs_url: str = "http://127.0.0.1:8080"
    datasets_url: str = "http://127.0.0.1:8080"
    pipelines_url: str = "http://127.0.0.1:8080"
    serving_url: str = "http://127.0.0.1:8080"
    data_dir: str = str(ROOT / "data")

    class Config:
        env_prefix = "FTAAS_"
        extra = "ignore"


class PlatformConfig(BaseModel):
    raw: dict[str, Any] = Field(default_factory=dict)
    services: dict[str, ServiceEndpoint] = Field(default_factory=dict)
    integrations: Integrations = Field(default_factory=Integrations)
    frameworks: list[str] = Field(default_factory=list)
    techniques: dict[str, list[str]] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_platform_config() -> PlatformConfig:
    settings = get_settings()
    path = Path(os.environ.get("FTAAS_CONFIG", settings.config_path))
    raw = _load_yaml(path)

    services: dict[str, ServiceEndpoint] = {}
    for name, cfg in (raw.get("services") or {}).items():
        services[name] = ServiceEndpoint(**cfg)

    integ_raw = raw.get("integrations") or {}
    integrations = Integrations(
        mlflow_tracking_uri=(integ_raw.get("mlflow") or {}).get("tracking_uri", "http://127.0.0.1:5000"),
        mlflow_experiment=(integ_raw.get("mlflow") or {}).get("experiment", "ftaas-finetune"),
        ray_address=(integ_raw.get("ray") or {}).get("address", "auto"),
        ray_mock=(integ_raw.get("ray") or {}).get("mock", True),
        airflow_enabled=(integ_raw.get("airflow") or {}).get("enabled", False),
        gcs_mock=(integ_raw.get("gcs") or {}).get("mock", True),
        gcs_local_mirror=(integ_raw.get("gcs") or {}).get("local_mirror", "./data/gcs_mirror"),
    )

    return PlatformConfig(
        raw=raw,
        services=services,
        integrations=integrations,
        frameworks=list(raw.get("frameworks") or []),
        techniques=dict(raw.get("techniques") or {}),
        defaults=dict(raw.get("defaults") or {}),
    )


def ensure_data_dirs() -> Path:
    settings = get_settings()
    root = Path(settings.data_dir)
    for sub in (
        "",
        "datasets",
        "outputs",
        "gcs_mirror",
        "models",
        "mlruns",
        "endpoints",
    ):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root
