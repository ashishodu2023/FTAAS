FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" pydantic pydantic-settings httpx \
    sqlalchemy aiosqlite greenlet python-multipart jinja2 aiofiles \
    pyyaml rich typer tenacity mlflow

COPY . .

ENV PYTHONPATH=/app:/app/packages/ftaas_sdk:/app/services
ENV FTAAS_DATA_DIR=/app/data
ENV PORT=8080

RUN mkdir -p /app/data

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -sf "http://127.0.0.1:${PORT:-8080}/health" || exit 1

CMD ["python", "-m", "ftaas_app.main"]
