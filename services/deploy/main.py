"""Deploy — endpoints, evaluation path (vLLM / adapters / Ray Serve)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from sqlalchemy import DateTime, Integer, String, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ftaas.config import ensure_data_dirs, get_platform_config, get_settings, sqlite_url
from ftaas.models import (
    CreateEndpointRequest,
    EndpointInfo,
    PromptRequest,
    PromptResponse,
    new_id,
    utcnow,
)

app = FastAPI(
    title="FTAAS Deploy",
    version="0.1.0",
    description="Create endpoint → inference framework → model deploy → UI/API prompt",
)


class Base(DeclarativeBase):
    pass


class EndpointRow(Base):
    __tablename__ = "endpoints"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    endpoint_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    model_name: Mapped[str] = mapped_column(String(256))
    model_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    inference_framework: Mapped[str] = mapped_column(String(64))
    url: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), default="ready")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


engine = None
SessionLocal: async_sessionmaker[AsyncSession] | None = None


@app.on_event("startup")
async def startup() -> None:
    global engine, SessionLocal
    ensure_data_dirs()
    engine = create_async_engine(sqlite_url("deploy"), echo=False)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "deploy"}


@app.post("/v1/endpoints", response_model=EndpointInfo)
async def create_endpoint(req: CreateEndpointRequest) -> EndpointInfo:
    """deployment path: Create Endpoint → Select Inference framework → Model Deploy."""
    assert SessionLocal is not None
    settings = get_settings()

    # Resolve model from Jobs registry
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {"version": req.model_version or "latest"}
            r = await client.get(
                f"{settings.control_url}/v1/models/{req.model_name}",
                params=params,
            )
            if r.status_code == 404:
                raise HTTPException(404, f"Model {req.model_name} not registered")
            r.raise_for_status()
            model = r.json()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Control lookup failed: {exc}") from exc

    endpoint_id = new_id("ep_")
    # evaluation path uses vllm → adapters → Ray Serve; deployment uses deploy API
    fw = req.inference_framework
    url = f"{settings.deploy_url.rstrip('/')}/v1/endpoints/{endpoint_id}/prompt"
    info = EndpointInfo(
        endpoint_id=endpoint_id,
        model_name=model["model_name"],
        model_version=model["version"],
        inference_framework=fw,
        url=url,
        status="ready",
        created_at=utcnow(),
    )
    async with SessionLocal() as session:
        session.add(
            EndpointRow(
                endpoint_id=info.endpoint_id,
                model_name=info.model_name,
                model_version=info.model_version,
                inference_framework=info.inference_framework,
                url=info.url,
                status=info.status,
                created_at=info.created_at,
            )
        )
        await session.commit()
    return info


@app.get("/v1/endpoints", response_model=list[EndpointInfo])
async def list_endpoints() -> list[EndpointInfo]:
    assert SessionLocal is not None
    async with SessionLocal() as session:
        rows = (
            await session.execute(select(EndpointRow).order_by(EndpointRow.id.desc()))
        ).scalars().all()
    return [
        EndpointInfo(
            endpoint_id=r.endpoint_id,
            model_name=r.model_name,
            model_version=r.model_version,
            inference_framework=r.inference_framework,
            url=r.url,
            status=r.status,
            created_at=r.created_at,
        )
        for r in rows
    ]


@app.post("/v1/endpoints/{endpoint_id}/prompt", response_model=PromptResponse)
async def prompt_endpoint(endpoint_id: str, req: PromptRequest) -> PromptResponse:
    """End-user access: prompt on UI / API."""
    assert SessionLocal is not None
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(EndpointRow).where(EndpointRow.endpoint_id == endpoint_id)
            )
        ).scalars().first()
    if not row:
        raise HTTPException(404, "Endpoint not found")

    # Demo completion — swap for vLLM / Ray Serve when wired to real infra
    completion = (
        f"[{row.inference_framework}:{row.model_name}@v{row.model_version}] "
        f"Echo: {req.prompt[:200]}"
    )
    return PromptResponse(
        endpoint_id=endpoint_id,
        prompt=req.prompt,
        completion=completion,
        model_name=row.model_name,
    )


@app.get("/v1/eval/stack")
async def eval_stack() -> dict:
    """Documents the evaluation serving path from the flow diagram."""
    return {
        "path": ["vllm", "adapters", "ray_serve"],
        "supported_by": {"mlflow": "adapters", "ray": "ray_serve"},
        "access": ["prompt_on_ui", "api"],
    }


def main() -> None:
    import uvicorn

    cfg = get_platform_config()
    port = cfg.services.get("deploy").port if cfg.services.get("deploy") else 8003
    uvicorn.run("deploy.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
