"""FTAAS GPU trainer worker — runs training off the control/UI plane."""

from __future__ import annotations

import io
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ftaas.config import ensure_data_dirs, get_settings
from ftaas.models import Framework, HyperParameters, Technique
from training.device import device_report
from training.frameworks.registry import get_trainer

logger = logging.getLogger("ftaas.trainer_worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(
    title="FTAAS Trainer Worker",
    version="0.1.0",
    description="GPU fine-tune worker: QLoRA / LoRA → upload adapters to control",
)


class TrainRequest(BaseModel):
    job_id: str
    model_name: str
    framework: str = "transformers"
    technique: str = "lora"
    parameters: dict[str, Any] = Field(default_factory=dict)
    control_url: str
    dataset_path: Optional[str] = None
    dataset_download_url: Optional[str] = None


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "trainer_worker", "device": device_report()}


def _to_file_url(download_url: str) -> str:
    """Map .../datasets/{id}/download → .../datasets/{id}/file (keep query)."""
    parsed = urlparse(download_url)
    path = parsed.path.replace("/download", "/file")
    return urlunparse(parsed._replace(path=path))


def _materialize_dataset(req: TrainRequest, work: Path) -> Path:
    if req.dataset_path and Path(req.dataset_path).exists():
        return Path(req.dataset_path)

    if not req.dataset_download_url:
        raise RuntimeError("No dataset_path or dataset_download_url available")

    # Prefer raw /file endpoint for remote workers (no shared disk).
    file_url = _to_file_url(req.dataset_download_url)
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        if file_url != req.dataset_download_url:
            fr = client.get(file_url)
            if fr.status_code == 200 and fr.content and "application/json" not in fr.headers.get(
                "content-type", ""
            ):
                out = work / "train.jsonl"
                out.write_bytes(fr.content)
                return out

        r = client.get(req.dataset_download_url)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "application/json" in ctype:
            info = r.json()
            local = info.get("local_path")
            if local and Path(local).exists():
                return Path(local)
            ds_id = info.get("dataset_id")
            ver = info.get("version") or "1"
            if ds_id:
                fr = client.get(
                    f"{req.control_url.rstrip('/')}/v1/datasets/{ds_id}/file",
                    params={"version": ver},
                )
                if fr.status_code == 200:
                    out = work / f"{ds_id}.jsonl"
                    out.write_bytes(fr.content)
                    return out
            raise RuntimeError(
                "Remote trainer could not fetch dataset bytes. "
                "Expose GET /v1/datasets/{id}/file on the control plane."
            )
        out = work / "dataset.bin"
        out.write_bytes(r.content)
        return out


def _zip_dir(src: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in src.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(src)))
    return buf.getvalue()


def _upload_artifacts(control_url: str, job_id: str, model_dir: Path) -> str:
    data = _zip_dir(model_dir)
    with httpx.Client(timeout=300.0) as client:
        r = client.post(
            f"{control_url.rstrip('/')}/v1/jobs/{job_id}/artifacts",
            files={"file": ("adapter.zip", data, "application/zip")},
        )
        r.raise_for_status()
        return r.json()["model_uri"]


@app.post("/v1/train")
def train(req: TrainRequest) -> dict[str, Any]:
    """Synchronous train on this GPU box; upload adapters to control."""
    os.environ["FTAAS_CONTROL_URL"] = req.control_url.rstrip("/")
    os.environ["FTAAS_JOB_ID"] = req.job_id
    # Worker always trains locally (never recurse to another remote).
    os.environ["FTAAS_TRAIN_MODE"] = "local"
    get_settings.cache_clear()
    settings = get_settings()
    ensure_data_dirs()

    work = Path(tempfile.mkdtemp(prefix=f"ftaas_train_{req.job_id}_"))
    try:
        ds_path = _materialize_dataset(req, work)
        params = HyperParameters(**(req.parameters or {}))
        params.output_dir = str(ensure_data_dirs() / "outputs")
        Path(params.output_dir).mkdir(parents=True, exist_ok=True)

        if not os.environ.get("FTAAS_TRAIN_DEVICE"):
            os.environ["FTAAS_TRAIN_DEVICE"] = "auto"
            get_settings.cache_clear()

        trainer = get_trainer(Framework(req.framework))
        result = trainer.train(
            req.model_name,
            str(ds_path),
            Technique(req.technique),
            params,
            req.job_id,
        )
        local_uri = Path(result.model_uri or result.output_dir)
        if not local_uri.exists():
            raise RuntimeError(f"Trainer produced no artifacts at {local_uri}")

        model_uri = _upload_artifacts(req.control_url, req.job_id, local_uri)
        return {
            "job_id": req.job_id,
            "model_uri": model_uri,
            "output_dir": model_uri,
            "metrics": result.metrics,
            "parameters": result.parameters,
            "statistics": result.statistics,
            "device": device_report(),
            "trainer": settings.public_url or "trainer_worker",
        }
    except Exception as exc:
        logger.exception("Remote train failed for %s", req.job_id)
        raise HTTPException(500, str(exc)) from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    import uvicorn

    port = int(os.environ.get("FTAAS_TRAINER_PORT") or os.environ.get("PORT") or "8090")
    os.environ["FTAAS_TRAIN_MODE"] = "local"
    get_settings.cache_clear()
    uvicorn.run("trainer_worker.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
