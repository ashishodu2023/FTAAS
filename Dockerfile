FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] pydantic pydantic-settings httpx sqlalchemy aiosqlite greenlet \
    python-multipart jinja2 aiofiles pyyaml rich typer tenacity mlflow

COPY . .
ENV PYTHONPATH=/app:/app/packages/ftaas_sdk:/app/services
ENV FTAAS_DATA_DIR=/data
