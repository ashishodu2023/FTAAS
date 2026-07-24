"""Remote GPU trainer client — offload train from control/UI plane."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ftaas.config import get_settings
from ftaas.models import Framework, HyperParameters, Technique
from training.frameworks.registry import TrainResult

logger = logging.getLogger("ftaas.remote_trainer")


def remote_trainer_enabled() -> bool:
    s = get_settings()
    return (s.train_mode or "").lower() == "remote" and bool((s.trainer_url or "").strip())


def submit_remote_training(
    *,
    job_id: str,
    model_name: str,
    dataset_path: str,
    framework: Framework | str,
    technique: Technique | str,
    params: HyperParameters,
    dataset_download_url: str | None = None,
) -> TrainResult:
    """
    POST train request to GPU worker. Worker reports progress to control and
    uploads adapter artifacts back; returns a TrainResult with model_uri on control.
    """
    settings = get_settings()
    trainer = settings.trainer_url.rstrip("/")
    control = (settings.public_url or settings.control_url).rstrip("/")
    payload: dict[str, Any] = {
        "job_id": job_id,
        "model_name": model_name,
        "framework": str(framework.value if hasattr(framework, "value") else framework),
        "technique": str(technique.value if hasattr(technique, "value") else technique),
        "parameters": params.model_dump(mode="json"),
        "control_url": control,
        "dataset_path": dataset_path,
        "dataset_download_url": dataset_download_url,
    }
    timeout = httpx.Timeout(connect=30.0, read=None, write=120.0, pool=30.0)
    logger.info("Offloading job %s to remote trainer %s", job_id, trainer)
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{trainer}/v1/train", json=payload)
        resp.raise_for_status()
        body = resp.json()
    return TrainResult(
        output_dir=body.get("output_dir") or body.get("model_uri") or "",
        metrics={k: float(v) for k, v in (body.get("metrics") or {}).items() if isinstance(v, (int, float))},
        parameters=dict(body.get("parameters") or {}),
        model_uri=body.get("model_uri") or body.get("output_dir"),
        statistics=dict(body.get("statistics") or {}),
    )
