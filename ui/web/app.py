"""FTAAS web UI — job setup, config, tracking, dataset management, prompting."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ftaas.config import get_platform_config, get_settings

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="FTAAS UI — Fine Tuning as a Service")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
static_dir = ROOT / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def _urls():
    s = get_settings()
    return s.jobs_url, s.datasets_url, s.serving_url


@app.get("/health")
async def health():
    return {"status": "ok", "service": "web"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    jobs_url, datasets_url, serving_url = _urls()
    catalog = jobs = datasets = models = endpoints = {}
    err = None
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            catalog = (await c.get(f"{jobs_url}/v1/catalog")).json()
            jobs = (await c.get(f"{jobs_url}/v1/jobs")).json()
            models = (await c.get(f"{jobs_url}/v1/models")).json()
            datasets = (await c.get(f"{datasets_url}/v1/datasets")).json()
            endpoints = (await c.get(f"{serving_url}/v1/endpoints")).json()
    except Exception as e:
        err = str(e)
        catalog = get_platform_config().model_dump()
        jobs, datasets, models, endpoints = [], [], [], []

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "catalog": catalog,
            "jobs": jobs,
            "datasets": datasets,
            "models": models,
            "endpoints": endpoints,
            "error": err,
            "defaults": (catalog.get("defaults") if isinstance(catalog, dict) else {}) or {},
        },
    )


@app.post("/register-dataset")
async def register_dataset(
    gcs_path: str = Form(...),
    name: Optional[str] = Form(None),
):
    _, datasets_url, _ = _urls()
    async with httpx.AsyncClient(timeout=30.0) as c:
        await c.post(
            f"{datasets_url}/v1/datasets/register",
            json={"gcs_path": gcs_path, "name": name or None, "format": "jsonl"},
        )
    return RedirectResponse("/", status_code=303)


@app.post("/create-job")
async def create_job(
    model_name: str = Form(...),
    framework: str = Form("transformers"),
    technique: str = Form("lora"),
    dataset_id: str = Form(...),
    dataset_version: str = Form("1"),
    max_steps: int = Form(10),
    learning_rate: float = Form(2e-4),
    lora_r: int = Form(8),
):
    jobs_url, _, _ = _urls()
    payload = {
        "model_name": model_name,
        "framework": framework,
        "technique": technique,
        "dataset": {"dataset_id": dataset_id, "version": dataset_version},
        "parameters": {
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "lora_r": lora_r,
            "lora_alpha": lora_r * 2,
        },
    }
    async with httpx.AsyncClient(timeout=60.0) as c:
        await c.post(f"{jobs_url}/v1/jobs/finetune", json=payload)
    return RedirectResponse("/", status_code=303)


@app.post("/deploy")
async def deploy(
    model_name: str = Form(...),
    inference_framework: str = Form("vllm"),
):
    _, _, serving_url = _urls()
    async with httpx.AsyncClient(timeout=30.0) as c:
        await c.post(
            f"{serving_url}/v1/endpoints",
            json={
                "model_name": model_name,
                "inference_framework": inference_framework,
                "use_adapters": True,
            },
        )
    return RedirectResponse("/", status_code=303)


@app.post("/prompt")
async def prompt(endpoint_id: str = Form(...), prompt: str = Form(...)):
    _, _, serving_url = _urls()
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(
            f"{serving_url}/v1/endpoints/{endpoint_id}/prompt",
            json={"prompt": prompt},
        )
        r.raise_for_status()
        completion = r.json().get("completion", "")
    return HTMLResponse(
        f"<html><body style='font-family:system-ui;padding:2rem'>"
        f"<h2>Prompt result</h2><p><b>In:</b> {prompt}</p>"
        f"<p><b>Out:</b> {completion}</p>"
        f"<p><a href='/'>← Back</a></p></body></html>"
    )


def main() -> None:
    import uvicorn

    cfg = get_platform_config()
    port = cfg.services.get("web").port if cfg.services.get("web") else 8080
    uvicorn.run("ui.web.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
