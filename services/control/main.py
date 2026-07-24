"""Control — fine-tune job lifecycle and model registry."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import JSON, DateTime, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ftaas.config import ensure_data_dirs, get_platform_config, get_settings, sqlite_url
from ftaas.models import (
    CreateFinetuneJobRequest,
    CreatePipelineRequest,
    FinetuneJob,
    JobStatus,
    ModelInfo,
    RegisterModelRequest,
    new_id,
    utcnow,
)

app = FastAPI(
    title="FTAAS Control",
    version="0.1.0",
    description="Fine Tuning as a Service — job lifecycle, model registry, catalog",
)


class Base(DeclarativeBase):
    pass


class JobRow(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default=JobStatus.PENDING.value)
    pipeline_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    ray_cluster: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    mlflow_experiment_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    registered_model_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    registered_model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ModelRow(Base):
    __tablename__ = "models"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    model_name: Mapped[str] = mapped_column(String(256), index=True)
    version: Mapped[str] = mapped_column(String(32))
    job_id: Mapped[str] = mapped_column(String(64))
    experiment_id: Mapped[str] = mapped_column(String(128))
    run_id: Mapped[str] = mapped_column(String(128))
    model_uri: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


engine = None
SessionLocal: async_sessionmaker[AsyncSession] | None = None


def _row_to_job(row: JobRow) -> FinetuneJob:
    payload = dict(row.payload)
    return FinetuneJob(
        job_id=row.job_id,
        model_name=payload["model_name"],
        framework=payload["framework"],
        technique=payload["technique"],
        dataset=payload["dataset"],
        parameters=payload["parameters"],
        status=JobStatus(row.status),
        pipeline_id=row.pipeline_id,
        ray_cluster=row.ray_cluster,
        mlflow_run_id=row.mlflow_run_id,
        mlflow_experiment_id=row.mlflow_experiment_id,
        registered_model_name=row.registered_model_name,
        registered_model_version=row.registered_model_version,
        error=row.error,
        metrics=row.metrics or {},
        created_at=row.created_at,
        updated_at=row.updated_at,
        tags=payload.get("tags") or {},
    )


@app.on_event("startup")
async def startup() -> None:
    global engine, SessionLocal
    ensure_data_dirs()
    engine = create_async_engine(sqlite_url("control"), echo=False)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "control"}


@app.get("/v1/catalog")
async def catalog() -> dict[str, Any]:
    cfg = get_platform_config()
    return {
        "frameworks": cfg.frameworks,
        "techniques": cfg.techniques,
        "defaults": cfg.defaults,
        "phases": [
            {"phase": 0, "name": "Fine-Tuning & RL Templates", "status": "available"},
            {"phase": 1, "name": "Fine-Tuning UI", "status": "available"},
            {"phase": 2, "name": "Fine-Tune & Evaluate", "status": "available"},
            {"phase": 3, "name": "Resource Optimization", "status": "planned"},
            {"phase": 4, "name": "Sweeps & Optimization", "status": "planned"},
        ],
    }


@app.post("/v1/jobs/finetune", response_model=FinetuneJob)
async def create_finetune_job(
    req: CreateFinetuneJobRequest,
    background: BackgroundTasks,
) -> FinetuneJob:
    """UI/SDK → create_finetune_job(model, framework, dataset, parameters)."""
    assert SessionLocal is not None
    settings = get_settings()
    job_id = new_id("job_")
    now = utcnow()
    payload = req.model_dump(mode="json")

    async with SessionLocal() as session:
        session.add(
            JobRow(
                job_id=job_id,
                payload=payload,
                status=JobStatus.PENDING.value,
                metrics={},
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()

    # create_pipeline via workflow
    pipeline_id = None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            pr = CreatePipelineRequest(
                job_id=job_id,
                framework=req.framework,
                dag_id="ftaas_finetune",
            )
            resp = await client.post(
                f"{settings.workflow_url}/v1/pipelines",
                json=pr.model_dump(mode="json"),
            )
            resp.raise_for_status()
            pipeline_id = resp.json()["pipeline_id"]
    except Exception as exc:
        async with SessionLocal() as session:
            row = (
                await session.execute(select(JobRow).where(JobRow.job_id == job_id))
            ).scalars().first()
            if row:
                row.status = JobStatus.FAILED.value
                row.error = f"pipeline create failed: {exc}"
                row.updated_at = utcnow()
                await session.commit()
        raise HTTPException(502, f"Failed to create pipeline: {exc}") from exc

    async with SessionLocal() as session:
        row = (
            await session.execute(select(JobRow).where(JobRow.job_id == job_id))
        ).scalars().first()
        assert row
        row.pipeline_id = pipeline_id
        row.status = JobStatus.QUEUED.value
        row.updated_at = utcnow()
        await session.commit()
        job = _row_to_job(row)

    # Schedule according to framework (Airflow or local runner)
    background.add_task(_schedule_job, job_id, pipeline_id)
    return job


async def _schedule_job(job_id: str, pipeline_id: str) -> None:
    """schedule job according to framework → airflow / local runner."""
    from runner.local.runner import run_finetune_pipeline

    await asyncio.to_thread(run_finetune_pipeline, job_id, pipeline_id)


@app.get("/v1/jobs", response_model=list[FinetuneJob])
async def list_jobs() -> list[FinetuneJob]:
    assert SessionLocal is not None
    async with SessionLocal() as session:
        rows = (await session.execute(select(JobRow).order_by(JobRow.id.desc()))).scalars().all()
    return [_row_to_job(r) for r in rows]


@app.get("/v1/jobs/{job_id}", response_model=FinetuneJob)
async def get_job_status(job_id: str) -> FinetuneJob:
    assert SessionLocal is not None
    async with SessionLocal() as session:
        row = (
            await session.execute(select(JobRow).where(JobRow.job_id == job_id))
        ).scalars().first()
    if not row:
        raise HTTPException(404, "Job not found")
    return _row_to_job(row)


class StatusUpdate(BaseModel):
    status: JobStatus
    ray_cluster: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    mlflow_experiment_id: Optional[str] = None
    metrics: Optional[dict[str, float]] = None
    error: Optional[str] = None


@app.patch("/v1/jobs/{job_id}/status", response_model=FinetuneJob)
async def update_job_status(job_id: str, upd: StatusUpdate) -> FinetuneJob:
    assert SessionLocal is not None
    async with SessionLocal() as session:
        row = (
            await session.execute(select(JobRow).where(JobRow.job_id == job_id))
        ).scalars().first()
        if not row:
            raise HTTPException(404, "Job not found")
        row.status = upd.status.value
        if upd.ray_cluster is not None:
            row.ray_cluster = upd.ray_cluster
        if upd.mlflow_run_id is not None:
            row.mlflow_run_id = upd.mlflow_run_id
        if upd.mlflow_experiment_id is not None:
            row.mlflow_experiment_id = upd.mlflow_experiment_id
        if upd.metrics is not None:
            row.metrics = {**(row.metrics or {}), **upd.metrics}
        if upd.error is not None:
            row.error = upd.error
        row.updated_at = utcnow()
        await session.commit()
        return _row_to_job(row)


@app.post("/v1/models/register", response_model=ModelInfo)
async def register_model(req: RegisterModelRequest) -> ModelInfo:
    """airflow → register_model(exp_id, run_id, model_name, ...)."""
    assert SessionLocal is not None
    # bump version
    async with SessionLocal() as session:
        existing = (
            await session.execute(
                select(ModelRow)
                .where(ModelRow.model_name == req.model_name)
                .order_by(ModelRow.id.desc())
            )
        ).scalars().first()
        next_ver = str(int(existing.version) + 1) if existing else "1"
        info = ModelInfo(
            model_id=new_id("mo_"),
            model_name=req.model_name,
            version=next_ver,
            job_id=req.job_id,
            experiment_id=req.experiment_id,
            run_id=req.run_id,
            model_uri=req.model_uri,
            metrics=req.metrics,
            created_at=utcnow(),
        )
        session.add(
            ModelRow(
                model_id=info.model_id,
                model_name=info.model_name,
                version=info.version,
                job_id=info.job_id,
                experiment_id=info.experiment_id,
                run_id=info.run_id,
                model_uri=info.model_uri,
                metrics=info.metrics,
                created_at=info.created_at,
            )
        )
        job = (
            await session.execute(select(JobRow).where(JobRow.job_id == req.job_id))
        ).scalars().first()
        if job:
            job.registered_model_name = info.model_name
            job.registered_model_version = info.version
            job.mlflow_run_id = req.run_id
            job.mlflow_experiment_id = req.experiment_id
            job.metrics = {**(job.metrics or {}), **req.metrics}
            job.status = JobStatus.REGISTERING.value
            job.updated_at = utcnow()
        await session.commit()
    return info


@app.get("/v1/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    assert SessionLocal is not None
    async with SessionLocal() as session:
        rows = (await session.execute(select(ModelRow).order_by(ModelRow.id.desc()))).scalars().all()
    return [
        ModelInfo(
            model_id=r.model_id,
            model_name=r.model_name,
            version=r.version,
            job_id=r.job_id,
            experiment_id=r.experiment_id,
            run_id=r.run_id,
            model_uri=r.model_uri,
            metrics=r.metrics or {},
            created_at=r.created_at,
        )
        for r in rows
    ]


@app.get("/v1/models/{model_name}", response_model=ModelInfo)
async def get_model(model_name: str, version: str = Query("latest")) -> ModelInfo:
    assert SessionLocal is not None
    async with SessionLocal() as session:
        q = select(ModelRow).where(ModelRow.model_name == model_name)
        if version != "latest":
            q = q.where(ModelRow.version == version)
        q = q.order_by(ModelRow.id.desc())
        row = (await session.execute(q)).scalars().first()
    if not row:
        raise HTTPException(404, f"Model {model_name} not found")
    return ModelInfo(
        model_id=row.model_id,
        model_name=row.model_name,
        version=row.version,
        job_id=row.job_id,
        experiment_id=row.experiment_id,
        run_id=row.run_id,
        model_uri=row.model_uri,
        metrics=row.metrics or {},
        created_at=row.created_at,
    )


class ScheduleRequest(BaseModel):
    job_id: str
    pipeline_id: str


@app.post("/v1/internal/schedule")
async def internal_schedule(req: ScheduleRequest, background: BackgroundTasks) -> dict:
    background.add_task(_schedule_job, req.job_id, req.pipeline_id)
    return {"scheduled": True, "job_id": req.job_id}


class JobCompleteRequest(BaseModel):
    job_id: str
    status: str = "succeeded"


@app.post("/v1/internal/job_complete")
async def internal_job_complete(req: JobCompleteRequest) -> dict:
    assert SessionLocal is not None
    async with SessionLocal() as session:
        row = (
            await session.execute(select(JobRow).where(JobRow.job_id == req.job_id))
        ).scalars().first()
        if not row:
            raise HTTPException(404, "Job not found")
        row.status = req.status
        row.updated_at = utcnow()
        await session.commit()
    return {"ok": True, "job_id": req.job_id, "status": req.status}


def main() -> None:
    import uvicorn

    cfg = get_platform_config()
    port = cfg.services.get("control").port if cfg.services.get("control") else 8000
    uvicorn.run("control.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
