FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# App + MLflow first
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" pydantic pydantic-settings httpx \
    sqlalchemy aiosqlite greenlet python-multipart jinja2 aiofiles \
    pyyaml rich typer tenacity mlflow

# Real fine-tune stack (CPU wheels — Railway has no GPU by default)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch \
 && pip install --no-cache-dir \
    "transformers>=4.46.0" "datasets>=3.0.0" "peft>=0.13.0" \
    "accelerate>=1.0.0" "trl>=0.12.0" \
    "tokenizers>=0.20.0" "sentencepiece>=0.2.0"

COPY . .

ENV PYTHONPATH=/app:/app/packages/ftaas_sdk:/app/services
ENV FTAAS_DATA_DIR=/app/data
ENV PORT=8080
ENV HF_HOME=/app/data/hf_cache
ENV TRANSFORMERS_CACHE=/app/data/hf_cache

RUN mkdir -p /app/data /app/data/hf_cache \
 && test -f /app/examples/data/alpaca_sample.jsonl

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
  CMD curl -sf "http://127.0.0.1:${PORT:-8080}/health" || exit 1

CMD ["python", "-m", "ftaas_app.main"]
