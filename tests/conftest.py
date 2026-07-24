"""Shared fixtures for FTAAS tests."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_JSONL = ROOT / "examples" / "data" / "alpaca_sample.jsonl"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Isolated FTAAS_DATA_DIR + cleared settings caches."""
    d = tmp_path / "ftaas-data"
    d.mkdir()
    monkeypatch.setenv("FTAAS_DATA_DIR", str(d))
    monkeypatch.setenv("FTAAS_CONFIG", str(ROOT / "configs" / "settings.yaml"))
    from ftaas.config import get_platform_config, get_settings

    get_settings.cache_clear()
    get_platform_config.cache_clear()
    yield d
    get_settings.cache_clear()
    get_platform_config.cache_clear()


@pytest.fixture
def sample_dataset(tmp_path) -> Path:
    path = tmp_path / "train.jsonl"
    if SAMPLE_JSONL.exists():
        path.write_text(SAMPLE_JSONL.read_text())
    else:
        rows = [
            {"instruction": "What is LoRA?", "input": "", "output": "Low-Rank Adaptation."},
            {"instruction": "What is FTAAS?", "input": "", "output": "Fine Tuning as a Service."},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


@pytest.fixture
def client_app(data_dir, monkeypatch):
    """FastAPI TestClient against the unified gateway (lifespan on)."""
    from fastapi.testclient import TestClient

    # Import after env is set so settings pick up FTAAS_DATA_DIR
    from ftaas_app.main import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def live_gateway(tmp_path_factory):
    """
    Real uvicorn process for system integration (runner httpx → same port).
    Module-scoped to amortize startup.
    """
    import uvicorn

    data = tmp_path_factory.mktemp("live-ftaas")
    port = _free_port()
    base = f"http://127.0.0.1:{port}"

    env_keys = {
        "FTAAS_DATA_DIR": str(data),
        "FTAAS_CONFIG": str(ROOT / "configs" / "settings.yaml"),
        "PORT": str(port),
        "FTAAS_PORT": str(port),
        "FTAAS_CONTROL_URL": base,
        "FTAAS_REGISTRY_URL": base,
        "FTAAS_WORKFLOW_URL": base,
        "FTAAS_DEPLOY_URL": base,
        "PYTHONPATH": f"{ROOT}:{ROOT / 'packages' / 'ftaas_sdk'}:{ROOT / 'services'}",
    }
    old = {k: os.environ.get(k) for k in env_keys}
    os.environ.update(env_keys)

    from ftaas.config import get_platform_config, get_settings

    get_settings.cache_clear()
    get_platform_config.cache_clear()

    # Re-import app after env
    import importlib
    import ftaas_app.main as main_mod

    importlib.reload(main_mod)
    app = main_mod.create_app()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 30
    import httpx

    while time.time() < deadline:
        try:
            r = httpx.get(f"{base}/health", timeout=1.0)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.2)
    else:
        server.should_exit = True
        raise RuntimeError("live_gateway failed to become healthy")

    yield {"base": base, "data_dir": data, "port": port}

    server.should_exit = True
    thread.join(timeout=5)
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    get_settings.cache_clear()
    get_platform_config.cache_clear()
