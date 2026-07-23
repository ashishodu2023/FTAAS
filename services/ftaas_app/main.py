"""Unified FTAAS gateway — one process for MDS + MDLC + Pipelineserv + Aimlopsserv + Cosmos UI."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.routing import APIRoute, Mount

from mdlc.config import ensure_data_dirs, get_platform_config


async def _run_startup(sub: FastAPI) -> None:
    for handler in sub.router.on_startup:
        result = handler()
        if hasattr(result, "__await__"):
            await result


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dirs()
    for key in ("mds_app", "mdlc_app", "pipe_app", "aiml_app", "ui_app"):
        await _run_startup(app.state[key])
    yield


def create_app() -> FastAPI:
    from aimlopsserv.main import app as aiml_app
    from mdlc_server.main import app as mdlc_app
    from mds.main import app as mds_app
    from pipelineserv.main import app as pipe_app
    from ui.cosmos_ui.app import app as ui_app

    app = FastAPI(
        title="Fine Tuning as a Service",
        version="0.1.0",
        description="Unified gateway: dataset → orchestrate → train → track → deploy → UI/API",
        lifespan=lifespan,
    )
    app.state.mds_app = mds_app
    app.state.mdlc_app = mdlc_app
    app.state.pipe_app = pipe_app
    app.state.aiml_app = aiml_app
    app.state.ui_app = ui_app

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "ftaas",
            "components": ["mds", "mdlc_server", "pipelineserv", "aimlopsserv", "cosmos_ui"],
        }

    skip = {"/docs", "/redoc", "/openapi.json", "/health"}

    # API modules first, then UI (so / is the Cosmos UI)
    for sub in (mds_app, mdlc_app, pipe_app, aiml_app, ui_app):
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
    port = int(os.environ.get("FTAAS_PORT", cfg.services.get("ftaas").port if cfg.services.get("ftaas") else 8080))
    uvicorn.run("ftaas_app.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
