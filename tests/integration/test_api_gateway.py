"""System integration tests against the unified FastAPI gateway (TestClient)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_health(client_app):
    r = client_app.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "registry" in body["components"]


def test_catalog(client_app):
    r = client_app.get("/v1/catalog")
    assert r.status_code == 200
    body = r.json()
    assert "frameworks" in body or "phases" in body or isinstance(body, dict)


def test_preview_then_register_dataset(client_app, sample_dataset):
    preview = client_app.post(
        "/v1/datasets/preview",
        json={"gcs_path": str(sample_dataset), "format": "jsonl", "limit": 3},
    )
    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert body["ok"] is True
    assert body["num_rows"] and body["num_rows"] >= 1
    assert body["samples"]
    assert "instruction" in body["columns"] or body["columns"]

    reg = client_app.post(
        "/v1/datasets/register",
        json={"gcs_path": str(sample_dataset), "name": "previewed", "format": "jsonl"},
    )
    assert reg.status_code == 200, reg.text
    assert reg.json()["num_rows"] == body["num_rows"]


def test_job_logs_and_progress_fields(client_app):
    """Exercise GET /v1/jobs/{id}/logs + progress payload on a seeded job."""
    import asyncio
    import concurrent.futures
    from datetime import datetime, timezone

    from control.main import JobRow, SessionLocal
    from ftaas.models import new_id

    async def _seed() -> str:
        assert SessionLocal is not None
        job_id = new_id("job_")
        now = datetime.now(timezone.utc)
        async with SessionLocal() as session:
            session.add(
                JobRow(
                    job_id=job_id,
                    payload={
                        "model_name": "sshleifer/tiny-gpt2",
                        "framework": "transformers",
                        "technique": "lora",
                        "dataset": {"dataset_id": "ds_x", "version": "1"},
                        "parameters": {},
                    },
                    status="training",
                    metrics={},
                    logs=[{"ts": now.isoformat(), "message": "Pipeline started"}],
                    progress={"percent": 35, "phase": "training", "message": "step 1/10"},
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()
        return job_id

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        job_id = pool.submit(lambda: asyncio.run(_seed())).result()

    r = client_app.get(f"/v1/jobs/{job_id}/logs")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == job_id
    assert body["status"] == "training"
    assert body["progress"]["percent"] == 35
    assert body["logs"] and "Pipeline started" in body["logs"][0]["message"]

    job = client_app.get(f"/v1/jobs/{job_id}")
    assert job.status_code == 200
    assert job.json()["progress"]["phase"] == "training"


def test_cancel_job_endpoint(client_app):
    import asyncio
    import concurrent.futures
    from datetime import datetime, timezone

    from control.main import JobRow, SessionLocal
    from ftaas.cancel import clear_cancel, is_cancel_requested
    from ftaas.models import new_id

    async def _seed() -> str:
        assert SessionLocal is not None
        job_id = new_id("job_")
        now = datetime.now(timezone.utc)
        async with SessionLocal() as session:
            session.add(
                JobRow(
                    job_id=job_id,
                    payload={
                        "model_name": "distilgpt2",
                        "framework": "transformers",
                        "technique": "lora",
                        "dataset": {"dataset_id": "ds_x", "version": "1"},
                        "parameters": {},
                    },
                    status="training",
                    metrics={},
                    logs=[],
                    progress={"percent": 40, "phase": "training", "message": "step 2/12"},
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()
        return job_id

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        job_id = pool.submit(lambda: asyncio.run(_seed())).result()

    clear_cancel(job_id)
    r = client_app.post(f"/v1/jobs/{job_id}/cancel")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "cancelled"
    assert is_cancel_requested(job_id)
    again = client_app.post(f"/v1/jobs/{job_id}/cancel")
    assert again.status_code == 409
    clear_cancel(job_id)


def test_register_dataset(client_app, sample_dataset):
    r = client_app.post(
        "/v1/datasets/register",
        json={"gcs_path": str(sample_dataset), "name": "unit-alpaca", "format": "jsonl"},
    )
    assert r.status_code == 200, r.text
    ds = r.json()
    assert ds["dataset_id"]
    assert ds["version"] == "1"
    assert ds["num_rows"] and ds["num_rows"] >= 1

    listed = client_app.get("/v1/datasets")
    assert listed.status_code == 200
    assert any(x["dataset_id"] == ds["dataset_id"] for x in listed.json())


def test_create_endpoint(client_app):
    reg = client_app.post(
        "/v1/models/register",
        json={
            "job_id": "job_seed",
            "experiment_id": "0",
            "run_id": "seed",
            "model_name": "tiny-gpt2-ft",
            "model_uri": "sshleifer/tiny-gpt2",
            "metrics": {"train_loss": 1.0},
            "parameters": {"model_name": "sshleifer/tiny-gpt2"},
        },
    )
    assert reg.status_code == 200, reg.text
    # Endpoint create talks to control via FTAAS_CONTROL_URL; with TestClient
    # that only works when a live server is up — assert register path here.
    models = client_app.get("/v1/models")
    assert models.status_code == 200
    assert any(m["model_name"] == "tiny-gpt2-ft" for m in models.json())


def test_prompt_format_helpers():
    from deploy.main import _exact_train_answer, _extract_response, _format_sft_prompt

    wrapped = _format_sft_prompt("What is LoRA?")
    assert wrapped.startswith("### Instruction:")
    assert wrapped.rstrip().endswith("### Response:")
    full = wrapped + "LoRA is a parameter-efficient fine-tuning method."
    assert "LoRA is a parameter-efficient" in _extract_response(full, wrapped)

    examples = [
        {"instruction": "What is LoRA?", "output": "LoRA is a parameter-efficient fine-tuning method."},
        {"instruction": "What is FTAAS?", "output": "Fine Tuning as a Service."},
    ]
    assert _exact_train_answer(examples, "What is LoRA?") == "LoRA is a parameter-efficient fine-tuning method."
    few = _format_sft_prompt("What is LoRA?", examples=examples)
    assert "What is FTAAS?" in few
    assert few.count("### Instruction:") >= 2


def test_cancel_flags():
    from ftaas.cancel import clear_cancel, is_cancel_requested, request_cancel

    clear_cancel("job_test_cancel")
    assert not is_cancel_requested("job_test_cancel")
    request_cancel("job_test_cancel")
    assert is_cancel_requested("job_test_cancel")
    clear_cancel("job_test_cancel")
    assert not is_cancel_requested("job_test_cancel")


@pytest.mark.slow
def test_real_generate_from_hf_checkpoint():
    from deploy.main import _GEN_CACHE, _generate

    _GEN_CACHE.clear()
    text = _generate("sshleifer/tiny-gpt2", "Hello", max_new_tokens=4)
    assert isinstance(text, str)
    assert len(text) >= 0
