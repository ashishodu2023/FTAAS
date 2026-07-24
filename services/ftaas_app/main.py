"""Unified FTAAS gateway — one process for registry, control, workflow, deploy, and console."""

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
    for key in ("registry_app", "control_app", "workflow_app", "deploy_app", "console_app"):
        await _run_startup(app.state[key])
    yield


def create_app() -> FastAPI:
    from registry.main import app as registry_app
    from control.main import app as control_app
    from workflow.main import app as workflow_app
    from deploy.main import app as deploy_app
    from ui.console.app import app as console_app

    app = FastAPI(
        title="Fine Tuning as a Service",
        version="0.1.0",
        description="Unified gateway: dataset → orchestrate → train → track → deploy → UI/API",
        lifespan=lifespan,
    )
    app.state.registry_app = registry_app
    app.state.control_app = control_app
    app.state.workflow_app = workflow_app
    app.state.deploy_app = deploy_app
    app.state.console_app = console_app

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "ftaas",
            "components": ["registry", "control", "workflow", "deploy", "console"],
        }

    skip = {"/docs", "/redoc", "/openapi.json", "/health"}

    for sub in (registry_app, control_app, workflow_app, deploy_app, console_app):
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
    # Railway/Render/Fly set PORT; FTAAS_PORT overrides for local.
    port = int(
        os.environ.get("PORT")
        or os.environ.get("FTAAS_PORT")
        or (cfg.services.get("ftaas").port if cfg.services.get("ftaas") else 8080)
    )
    # Same-process API calls (runner ↔ control/registry) on the container loopback.
    base = f"http://127.0.0.1:{port}"
    os.environ.setdefault("FTAAS_CONTROL_URL", base)
    os.environ.setdefault("FTAAS_REGISTRY_URL", base)
    os.environ.setdefault("FTAAS_WORKFLOW_URL", base)
    os.environ.setdefault("FTAAS_DEPLOY_URL", os.environ.get("FTAAS_PUBLIC_URL", base))
    get_settings.cache_clear()
    get_platform_config.cache_clear()
    uvicorn.run("ftaas_app.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
