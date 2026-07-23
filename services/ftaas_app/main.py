"""Unified FTAAS gateway — one process for datasets, jobs, pipelines, serving, and UI."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.routing import APIRoute, Mount

from ftaas.config import ensure_data_dirs, get_platform_config


async def _run_startup(sub: FastAPI) -> None:
    for handler in sub.router.on_startup:
        result = handler()
        if hasattr(result, "__await__"):
            await result


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dirs()
    for key in ("datasets_app", "jobs_app", "pipelines_app", "serving_app", "web_app"):
        await _run_startup(app.state[key])
    yield


def create_app() -> FastAPI:
    from datasets.main import app as datasets_app
    from jobs.main import app as jobs_app
    from pipelines.main import app as pipelines_app
    from serving.main import app as serving_app
    from ui.web.app import app as web_app

    app = FastAPI(
        title="Fine Tuning as a Service",
        version="0.1.0",
        description="Unified gateway: dataset → orchestrate → train → track → deploy → UI/API",
        lifespan=lifespan,
    )
    app.state.datasets_app = datasets_app
    app.state.jobs_app = jobs_app
    app.state.pipelines_app = pipelines_app
    app.state.serving_app = serving_app
    app.state.web_app = web_app

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "ftaas",
            "components": ["datasets", "jobs", "pipelines", "serving", "web"],
        }

    skip = {"/docs", "/redoc", "/openapi.json", "/health"}

    for sub in (datasets_app, jobs_app, pipelines_app, serving_app, web_app):
        for route in sub.routes:
            path = getattr(route, "path", None)
            if path in skip:
                continue
            if isinstance(route, (APIRoute, Mount)):
                app.routes.append(route)

    return app


app = create_app()


def main() -> None:
    import os

    import uvicorn

    cfg = get_platform_config()
    port = int(
        os.environ.get(
            "FTAAS_PORT",
            cfg.services.get("ftaas").port if cfg.services.get("ftaas") else 8080,
        )
    )
    uvicorn.run("ftaas_app.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
