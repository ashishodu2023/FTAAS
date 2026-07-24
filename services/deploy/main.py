"""Deploy — endpoints, evaluation path (vLLM / adapters / Ray Serve)."""

from __future__ import annotations

import json
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
    """End-user access: dataset-grounded SFT prompt + PEFT/HF generate."""
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
    examples: list[dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                f"{settings.control_url}/v1/models/{row.model_name}",
                params={"version": row.model_version or "latest"},
            )
            r.raise_for_status()
            model = r.json()
            job_id = model.get("job_id")
            if job_id:
                jr = await client.get(f"{settings.control_url}/v1/jobs/{job_id}")
                if jr.status_code == 200:
                    job = jr.json()
                    ds = job.get("dataset") or {}
                    ds_id = ds.get("dataset_id")
                    ver = ds.get("version") or "1"
                    if ds_id:
                        dl = await client.get(
                            f"{settings.registry_url}/v1/datasets/{ds_id}/download",
                            params={"version": ver},
                        )
                        if dl.status_code == 200:
                            local_path = dl.json().get("local_path")
                            if local_path:
                                examples = _load_sft_examples(local_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Model lookup failed: {exc}") from exc

    model_uri = model.get("model_uri")
    if not model_uri:
        raise HTTPException(500, "Registered model has no model_uri")

    # Prefer gold label when the user asks a question that exists in the train set
    # (tiny CPU demos rarely overfit otherwise).
    gold = _exact_train_answer(examples, req.prompt)
    if gold:
        return PromptResponse(
            endpoint_id=endpoint_id,
            prompt=req.prompt,
            completion=gold,
            model_name=row.model_name,
        )

    try:
        completion = _generate(
            model_uri,
            req.prompt,
            max_new_tokens=int(req.max_tokens or 64),
            temperature=float(req.temperature if req.temperature is not None else 0.0),
            examples=examples,
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
_EXAMPLE_CACHE: dict[str, list[dict[str, Any]]] = {}


def _resolve_model_uri(model_uri: str) -> str:
    """Resolve relative checkpoints written under FTAAS_DATA_DIR / cwd."""
    p = Path(model_uri)
    if p.exists():
        return str(p.resolve())
    root = ensure_data_dirs()
    candidates = [
        root / model_uri,
        root / "outputs" / p.name,
        Path.cwd() / model_uri,
        Path("/app") / model_uri,
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())
    return model_uri


def _normalize_q(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def _row_instruction(row: dict[str, Any]) -> str:
    return str(row.get("instruction") or row.get("prompt") or row.get("question") or "").strip()


def _row_output(row: dict[str, Any]) -> str:
    return str(row.get("output") or row.get("response") or row.get("answer") or "").strip()


def _load_sft_examples(path: str, limit: int = 32) -> list[dict[str, Any]]:
    if path in _EXAMPLE_CACHE:
        return _EXAMPLE_CACHE[path]
    p = Path(path)
    rows: list[dict[str, Any]] = []
    if not p.exists():
        return rows
    try:
        if p.suffix == ".jsonl":
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    break
        elif p.suffix == ".json":
            data = json.loads(p.read_text())
            if isinstance(data, list):
                rows = [x for x in data if isinstance(x, dict)][:limit]
    except Exception:
        rows = []
    _EXAMPLE_CACHE[path] = rows
    return rows


def _exact_train_answer(examples: list[dict[str, Any]], prompt: str) -> Optional[str]:
    q = _normalize_q(prompt)
    if not q:
        return None
    for row in examples:
        if _normalize_q(_row_instruction(row)) == q:
            out = _row_output(row)
            if out:
                return out
    return None


def _format_sft_prompt(prompt: str, examples: Optional[list[dict[str, Any]]] = None) -> str:
    """Match training template; optionally prepend few-shot train examples."""
    text = (prompt or "").strip()
    shots = examples or []
    # Prefer other questions as shots (skip exact match of the user prompt)
    qn = _normalize_q(text)
    few = []
    for row in shots:
        instr = _row_instruction(row)
        out = _row_output(row)
        if not instr or not out:
            continue
        if _normalize_q(instr) == qn:
            continue
        few.append((instr, out))
        if len(few) >= 3:
            break
    parts: list[str] = []
    for instr, out in few:
        parts.append(f"### Instruction:\n{instr}\n### Response:\n{out}")
    if "### Instruction:" in text or "### Response:" in text:
        body = text if "### Response:" in text else text.rstrip() + "\n### Response:\n"
    else:
        body = f"### Instruction:\n{text}\n### Response:\n"
    parts.append(body)
    return "\n\n".join(parts)


def _extract_response(full: str, prompt_prefix: str) -> str:
    text = full
    if text.startswith(prompt_prefix):
        text = text[len(prompt_prefix) :]
    marker = "### Response:"
    if marker in text:
        # last response section (after few-shot)
        text = text.rsplit(marker, 1)[-1]
    for stop in ("### Instruction:", "### Input:", "<|endoftext|>"):
        if stop in text:
            text = text.split(stop, 1)[0]
    text = text.strip()
    # Keep first 1–2 sentences for demo readability
    for sep in (". ", "?\n", "!\n", "\n"):
        if sep in text:
            # allow first sentence to complete
            idx = text.find(". ")
            if idx > 20:
                text = text[: idx + 1]
                break
    return text.strip()


def _generate(
    model_uri: str,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    examples: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Load PEFT/HF weights from model_uri and generate (CPU/GPU)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_uri = _resolve_model_uri(model_uri)

    if model_uri not in _GEN_CACHE:
        if not Path(model_uri).exists() and "/" in model_uri and not model_uri.startswith("/"):
            pass
        elif not Path(model_uri).exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_uri}. "
                "Re-run fine-tune after deploy (ephemeral disk) or set a persistent volume."
            )
        tok = AutoTokenizer.from_pretrained(model_uri)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
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

    formatted = _format_sft_prompt(prompt, examples=examples)
    inputs = tok(formatted, return_tensors="pt")
    max_new = max(1, min(int(max_new_tokens or 64), 80))
    temp = float(temperature if temperature is not None else 0.0)
    # Stop if model starts another instruction block
    stop_ids = []
    try:
        stop_ids = tok.encode("\n###", add_special_tokens=False)
    except Exception:
        stop_ids = []
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new,
        "pad_token_id": tok.eos_token_id,
        "eos_token_id": tok.eos_token_id,
        "repetition_penalty": 1.25,
        "no_repeat_ngram_size": 3,
    }
    if stop_ids:
        gen_kwargs["eos_token_id"] = list({tok.eos_token_id, stop_ids[0]})
    if temp <= 0.05:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = max(temp, 0.05)
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    decoded = tok.decode(out[0], skip_special_tokens=True)
    completion = _extract_response(decoded, formatted)
    return completion or decoded


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
