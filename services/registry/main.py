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


def _locate_source(gcs_path: str) -> Path:
    """Resolve a register/preview path to an existing local file."""
    root = ensure_data_dirs()
    src = _resolve_local_path(gcs_path, root)
    candidates = [src]
    if not src.is_absolute():
        from ftaas.config import ROOT

        candidates.append(ROOT / src)
        candidates.append(Path.cwd() / src)
        candidates.append(ROOT / gcs_path)
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"Dataset path not found: {gcs_path}. Provide an existing local file or sync gs:// to the mirror."
    )


def _materialize(src: Path, dest_dir: Path, dataset_id: str, version: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    located = src if src.exists() and src.is_file() else _locate_source(str(src))
    dest = dest_dir / f"{dataset_id}_v{version}{located.suffix or '.jsonl'}"
    shutil.copy2(located, dest)
    return dest


def _preview_file(path: Path, fmt: str, limit: int = 5) -> dict:
    """Load a few sample rows + schema hints without registering."""
    samples: list[dict] = []
    columns: list[str] = []
    warnings: list[str] = []
    fmt = (fmt or "jsonl").lower()

    if fmt == "jsonl":
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    warnings.append(f"Invalid JSON on a sample line: {exc}")
                    break
                if isinstance(row, dict):
                    samples.append(row)
                    for k in row:
                        if k not in columns:
                            columns.append(k)
                else:
                    samples.append({"value": row})
                    if "value" not in columns:
                        columns.append("value")
                if len(samples) >= limit:
                    break
    elif fmt == "json":
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else [data]
        for row in rows[:limit]:
            if isinstance(row, dict):
                samples.append(row)
                for k in row:
                    if k not in columns:
                        columns.append(k)
            else:
                samples.append({"value": row})
                if "value" not in columns:
                    columns.append("value")
    elif fmt == "csv":
        import csv

        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                samples.append(dict(row))
    else:
        warnings.append(f"Preview for format '{fmt}' shows raw text only.")
        text = path.read_text(errors="replace")[:2000]
        samples = [{"preview": text}]
        columns = ["preview"]

    num_rows = _count_rows(path, fmt if fmt in {"jsonl", "json", "csv"} else "jsonl")
    preferred = {"instruction", "input", "output", "text", "prompt", "response", "messages"}
    if columns and not (preferred & set(columns)):
        warnings.append(
            "Columns look unusual for SFT — expected keys like instruction/input/output "
            f"(or text/prompt). Found: {', '.join(columns)}"
        )
    if num_rows == 0:
        warnings.append("File appears empty (0 rows).")

    return {
        "resolved_path": str(path),
        "format": fmt,
        "num_rows": num_rows,
        "columns": columns,
        "samples": samples,
        "warnings": warnings,
        "preview_limit": limit,
    }


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


class PreviewDatasetRequest(BaseModel):
    gcs_path: str
    format: str = "jsonl"
    limit: int = 5


@app.post("/v1/datasets/preview")
async def preview_dataset(req: PreviewDatasetRequest) -> dict:
    """Inspect a dataset path before register_dataset — samples + row count, no side effects."""
    limit = max(1, min(int(req.limit or 5), 20))
    try:
        path = _locate_source(req.gcs_path)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        preview = _preview_file(path, req.format, limit=limit)
    except Exception as exc:
        raise HTTPException(400, f"Failed to preview dataset: {exc}") from exc
    preview["gcs_path"] = req.gcs_path
    preview["ok"] = True
    return preview


@app.post("/v1/datasets/register", response_model=DatasetInfo)
async def register_dataset(req: RegisterDatasetRequest) -> DatasetInfo:
    assert SessionLocal is not None
    storage = resolve_storage_root("datasets")

    dataset_id = new_id("ds_")
    version = "1"
    try:
        src = _locate_source(req.gcs_path)
        local = _materialize(src, Path(storage), dataset_id, version)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
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
