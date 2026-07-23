"""Pipelineserv — creates & tracks fine-tune pipelines (Airflow DAG bindings)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from sqlalchemy import DateTime, Integer, String, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from mdlc.config import ensure_data_dirs, get_platform_config, get_settings
from mdlc.models import (
    CreatePipelineRequest,
    JobStatus,
    PipelineInfo,
    new_id,
    utcnow,
)

app = FastAPI(title="FTAAS Pipelineserv", version="0.1.0")


class Base(DeclarativeBase):
    pass


class PipelineRow(Base):
    __tablename__ = "pipelines"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pipeline_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    job_id: Mapped[str] = mapped_column(String(64), index=True)
    dag_id: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default=JobStatus.PENDING.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


engine = None
SessionLocal: async_sessionmaker[AsyncSession] | None = None


@app.on_event("startup")
async def startup() -> None:
    global engine, SessionLocal
    root = ensure_data_dirs()
    cfg = get_platform_config()
    svc = cfg.services.get("pipelineserv")
    db_url = svc.db_url if svc else f"sqlite+aiosqlite:///{root}/pipelines.db"
    engine = create_async_engine(db_url, echo=False)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "pipelineserv"}


@app.post("/v1/pipelines", response_model=PipelineInfo)
async def create_pipeline(req: CreatePipelineRequest) -> PipelineInfo:
    assert SessionLocal is not None
    info = PipelineInfo(
        pipeline_id=new_id("pl_"),
        job_id=req.job_id,
        dag_id=req.dag_id,
        status=JobStatus.QUEUED,
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    async with SessionLocal() as session:
        session.add(
            PipelineRow(
                pipeline_id=info.pipeline_id,
                job_id=info.job_id,
                dag_id=info.dag_id,
                status=info.status.value,
                created_at=info.created_at,
                updated_at=info.updated_at,
            )
        )
        await session.commit()

    # MDLC schedules the Airflow/local runner after persisting pipeline_id.
    return info


@app.get("/v1/pipelines/{pipeline_id}", response_model=PipelineInfo)
async def get_pipeline(pipeline_id: str) -> PipelineInfo:
    assert SessionLocal is not None
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(PipelineRow).where(PipelineRow.pipeline_id == pipeline_id)
            )
        ).scalars().first()
    if not row:
        raise HTTPException(404, "Pipeline not found")
    return PipelineInfo(
        pipeline_id=row.pipeline_id,
        job_id=row.job_id,
        dag_id=row.dag_id,
        status=JobStatus(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@app.post("/v1/pipelines/{pipeline_id}/complete", response_model=PipelineInfo)
async def complete_pipeline(pipeline_id: str, status: str = "succeeded") -> PipelineInfo:
    assert SessionLocal is not None
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(PipelineRow).where(PipelineRow.pipeline_id == pipeline_id)
            )
        ).scalars().first()
        if not row:
            raise HTTPException(404, "Pipeline not found")
        row.status = status
        row.updated_at = utcnow()
        await session.commit()
        info = PipelineInfo(
            pipeline_id=row.pipeline_id,
            job_id=row.job_id,
            dag_id=row.dag_id,
            status=JobStatus(row.status),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    # Propagate completion to mdlc server
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                f"{settings.mdlc_url}/v1/internal/job_complete",
                json={"job_id": info.job_id, "status": status},
            )
    except Exception:
        pass
    return info


def main() -> None:
    import uvicorn

    cfg = get_platform_config()
    port = cfg.services.get("pipelineserv").port if cfg.services.get("pipelineserv") else 8002
    uvicorn.run("pipelineserv.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
