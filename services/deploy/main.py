"""Deploy — endpoints, evaluation path (vLLM / adapters / Ray Serve)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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
    # Serving is always local HF/PEFT generate today; keep the requested label for UI/API.
    fw = req.inference_framework or "transformers"
    if fw in {"vllm", "ray_serve"}:
        fw = "transformers"
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
    """End-user access: real generation from the registered fine-tuned adapter."""
    assert SessionLocal is not None
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(EndpointRow).where(EndpointRow.endpoint_id == endpoint_id)
            )
        ).scalars().first()
    if not row:
        raise HTTPException(404, "Endpoint not found")

    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                f"{settings.control_url}/v1/models/{row.model_name}",
                params={"version": row.model_version or "latest"},
            )
            r.raise_for_status()
            model = r.json()
    except Exception as exc:
        raise HTTPException(502, f"Model lookup failed: {exc}") from exc

    model_uri = model.get("model_uri")
    if not model_uri:
        raise HTTPException(500, "Registered model has no model_uri")

    try:
        completion = _generate(
            model_uri,
            req.prompt,
            max_new_tokens=int(req.max_tokens or 64),
            temperature=float(req.temperature if req.temperature is not None else 0.7),
        )
    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {exc}") from exc

    return PromptResponse(
        endpoint_id=endpoint_id,
        prompt=req.prompt,
        completion=completion,
        model_name=row.model_name,
    )


_GEN_CACHE: dict[str, tuple[Any, Any]] = {}


def _generate(
    model_uri: str,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
) -> str:
    """Load PEFT/HF weights from model_uri and generate (CPU/GPU)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_uri not in _GEN_CACHE:
        tok = AutoTokenizer.from_pretrained(model_uri)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Adapter dirs contain adapter_config.json; base model is in config
        adapter_cfg = Path(model_uri) / "adapter_config.json"
        if adapter_cfg.exists():
            import json as _json

            base = _json.loads(adapter_cfg.read_text()).get("base_model_name_or_path")
            if not base:
                raise RuntimeError("adapter_config.json missing base_model_name_or_path")
            base_model = AutoModelForCausalLM.from_pretrained(base)
            model = PeftModel.from_pretrained(base_model, model_uri)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_uri)
        model.eval()
        _GEN_CACHE[model_uri] = (model, tok)
    model, tok = _GEN_CACHE[model_uri]
    inputs = tok(prompt, return_tensors="pt")
    temp = max(float(temperature), 1e-5)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt) :].lstrip()
    return text or tok.decode(out[0], skip_special_tokens=True)


@app.get("/v1/eval/stack")
async def eval_stack() -> dict:
    """Documents serving path: transformers is live; vLLM/Ray Serve planned."""
    return {
        "live": ["transformers"],
        "planned": ["vllm", "adapters", "ray_serve"],
        "access": ["prompt_on_ui", "api"],
        "note": "create_endpoint currently serves via local HF/PEFT generate regardless of inference_framework label",
    }


def main() -> None:
    import uvicorn

    cfg = get_platform_config()
    port = cfg.services.get("deploy").port if cfg.services.get("deploy") else 8003
    uvicorn.run("deploy.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
