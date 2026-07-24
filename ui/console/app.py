"""FTAAS Console — serves UI; browser talks to /v1/* APIs on the same gateway."""

from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ftaas.config import get_platform_config, get_settings

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Console — Fine Tuning as a Service")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
static_dir = ROOT / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def _urls():
    s = get_settings()
    return s.control_url, s.registry_url, s.deploy_url


@app.get("/health")
async def health():
    return {"status": "ok", "service": "console"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """SSR bootstrap; console.js refreshes live from /v1 APIs."""
    control_url, registry_url, deploy_url = _urls()
    catalog: dict = {}
    jobs: list = []
    datasets: list = []
    models: list = []
    endpoints: list = []
    err = None
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            catalog = (await c.get(f"{control_url}/v1/catalog")).json()
            jobs = (await c.get(f"{control_url}/v1/jobs")).json()
            models = (await c.get(f"{control_url}/v1/models")).json()
            datasets = (await c.get(f"{registry_url}/v1/datasets")).json()
            endpoints = (await c.get(f"{deploy_url}/v1/endpoints")).json()
    except Exception as e:
        err = str(e)
        raw = get_platform_config()
        catalog = {
            "frameworks": raw.frameworks,
            "techniques": raw.techniques,
            "defaults": raw.defaults,
            "phases": [
                {"phase": 0, "name": "Fine-Tuning & RL Templates", "status": "available"},
                {"phase": 1, "name": "Fine-Tuning UI", "status": "available"},
                {"phase": 2, "name": "Fine-Tune & Evaluate", "status": "available"},
                {"phase": 3, "name": "Resource Optimization", "status": "planned"},
                {"phase": 4, "name": "Sweeps & Optimization", "status": "planned"},
            ],
        }

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "catalog": catalog,
            "jobs": jobs,
            "datasets": datasets,
            "models": models,
            "endpoints": endpoints,
            "error": err,
            "defaults": (catalog.get("defaults") if isinstance(catalog, dict) else {}) or {},
            "flash": request.query_params.get("flash"),
            "flash_type": request.query_params.get("type", "ok"),
        },
    )


def main() -> None:
    import uvicorn

    cfg = get_platform_config()
    port = cfg.services.get("console").port if cfg.services.get("console") else 8080
    uvicorn.run("ui.console.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
