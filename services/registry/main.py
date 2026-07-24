"""Registry — dataset registration & versioning (path → id:version)."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import DateTime, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ftaas.config import ensure_data_dirs, get_platform_config, get_settings, resolve_storage_root, sqlite_url
from ftaas.models import DatasetInfo, RegisterDatasetRequest, new_id, utcnow

app = FastAPI(title="FTAAS Registry", version="0.1.0", description="Dataset registry (GCS path → id:version)")


class Base(DeclarativeBase):
    pass


class DatasetRow(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(32))
    gcs_path: Mapped[str] = mapped_column(String(1024))
    local_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    format: Mapped[str] = mapped_column(String(32), default="jsonl")
    num_rows: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


engine = None
SessionLocal: async_sessionmaker[AsyncSession] | None = None


def _resolve_local_path(gcs_path: str, storage_root: Path) -> Path:
    """Map gs:// or file:// / local paths into the registry storage mirror."""
    if gcs_path.startswith("gs://"):
        parsed = urlparse(gcs_path)
        rel = Path(parsed.netloc) / parsed.path.lstrip("/")
        return storage_root / "gcs_mirror" / rel
    if gcs_path.startswith("file://"):
        return Path(urlparse(gcs_path).path)
    p = Path(gcs_path)
    if p.exists():
        return p.resolve()
    # treat as relative under gcs_mirror
    return storage_root / "gcs_mirror" / gcs_path


def _count_rows(path: Path, fmt: str) -> Optional[int]:
    if not path.exists() or not path.is_file():
        return None
    try:
        if fmt == "jsonl":
            return sum(1 for line in path.open() if line.strip())
        if fmt == "csv":
            with path.open() as f:
                return max(0, sum(1 for _ in f) - 1)
        if fmt == "json":
            data = json.loads(path.read_text())
            return len(data) if isinstance(data, list) else 1
    except Exception:
        return None
    return None


def _materialize(src: Path, dest_dir: Path, dataset_id: str, version: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if src.exists() and src.is_file():
        dest = dest_dir / f"{dataset_id}_v{version}{src.suffix or '.jsonl'}"
        shutil.copy2(src, dest)
        return dest
    # create a tiny placeholder dataset for demo when GCS mock & missing
    dest = dest_dir / f"{dataset_id}_v{version}.jsonl"
    sample = [
        {"instruction": "What is FTAAS?", "input": "", "output": "Fine Tuning as a Service."},
        {"instruction": "Explain LoRA", "input": "", "output": "Low-Rank Adaptation for PEFT."},
        {"instruction": "Name a framework", "input": "", "output": "Hugging Face Transformers."},
    ]
    with dest.open("w") as f:
        for row in sample:
            f.write(json.dumps(row) + "\n")
    return dest


@app.on_event("startup")
async def startup() -> None:
    global engine, SessionLocal
    ensure_data_dirs()
    engine = create_async_engine(sqlite_url("registry"), echo=False)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "registry"}


@app.post("/v1/datasets/register", response_model=DatasetInfo)
async def register_dataset(req: RegisterDatasetRequest) -> DatasetInfo:
    assert SessionLocal is not None
    root = ensure_data_dirs()
    storage = resolve_storage_root("datasets")

    dataset_id = new_id("ds_")
    version = "1"
    src = _resolve_local_path(req.gcs_path, root)
    local = _materialize(src, Path(storage), dataset_id, version)
    num_rows = _count_rows(local, req.format)

    info = DatasetInfo(
        dataset_id=dataset_id,
        version=version,
        gcs_path=req.gcs_path,
        local_path=str(local),
        name=req.name or local.stem,
        description=req.description,
        format=req.format,
        num_rows=num_rows,
        created_at=utcnow(),
    )

    async with SessionLocal() as session:
        row = DatasetRow(
            dataset_id=info.dataset_id,
            version=info.version,
            gcs_path=info.gcs_path,
            local_path=info.local_path,
            name=info.name,
            description=info.description,
            format=info.format,
            num_rows=info.num_rows,
            created_at=info.created_at,
        )
        session.add(row)
        await session.commit()
    return info


@app.get("/v1/datasets", response_model=list[DatasetInfo])
async def list_datasets() -> list[DatasetInfo]:
    assert SessionLocal is not None
    async with SessionLocal() as session:
        rows = (await session.execute(select(DatasetRow).order_by(DatasetRow.id.desc()))).scalars().all()
    return [
        DatasetInfo(
            dataset_id=r.dataset_id,
            version=r.version,
            gcs_path=r.gcs_path,
            local_path=r.local_path,
            name=r.name,
            description=r.description,
            format=r.format,
            num_rows=r.num_rows,
            created_at=r.created_at,
        )
        for r in rows
    ]


@app.get("/v1/datasets/{dataset_id}", response_model=DatasetInfo)
async def get_dataset(dataset_id: str, version: str = Query("latest")) -> DatasetInfo:
    assert SessionLocal is not None
    async with SessionLocal() as session:
        q = select(DatasetRow).where(DatasetRow.dataset_id == dataset_id)
        if version != "latest":
            q = q.where(DatasetRow.version == version)
        q = q.order_by(DatasetRow.id.desc())
        row = (await session.execute(q)).scalars().first()
    if not row:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    return DatasetInfo(
        dataset_id=row.dataset_id,
        version=row.version,
        gcs_path=row.gcs_path,
        local_path=row.local_path,
        name=row.name,
        description=row.description,
        format=row.format,
        num_rows=row.num_rows,
        created_at=row.created_at,
    )


class DownloadResponse(BaseModel):
    dataset_id: str
    version: str
    local_path: str
    num_rows: Optional[int] = None


@app.get("/v1/datasets/{dataset_id}/download", response_model=DownloadResponse)
async def download_dataset(dataset_id: str, version: str = Query("1")) -> DownloadResponse:
    """Called by Airflow / local runner: download_dataset(id, version)."""
    info = await get_dataset(dataset_id, version=version)
    if not info.local_path or not Path(info.local_path).exists():
        raise HTTPException(404, "Dataset file missing on disk")
    return DownloadResponse(
        dataset_id=info.dataset_id,
        version=info.version,
        local_path=info.local_path,
        num_rows=info.num_rows,
    )


def main() -> None:
    import uvicorn

    cfg = get_platform_config()
    port = cfg.services.get("registry").port if cfg.services.get("registry") else 8001
    uvicorn.run("registry.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
