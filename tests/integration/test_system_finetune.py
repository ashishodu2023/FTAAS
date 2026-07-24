"""
System integration: live uvicorn + register → real tiny-gpt2 LoRA → metrics.

Requires torch/transformers/peft (requirements.txt). Marked slow.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "examples" / "data" / "alpaca_sample.jsonl"

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _has_train_deps() -> bool:
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from datasets import Dataset  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_train_deps(), reason="train extras not installed")
def test_live_finetune_tiny_gpt2_with_statistics(live_gateway, monkeypatch):
    monkeypatch.setenv("FTAAS_FORCE_REAL_TRAIN", "1")
    base = live_gateway["base"]
    data_dir = Path(live_gateway["data_dir"])

    with httpx.Client(base_url=base, timeout=600.0) as client:
        assert client.get("/health").json()["status"] == "ok"

        ds = client.post(
            "/v1/datasets/register",
            json={"gcs_path": str(SAMPLE), "name": "alpaca-live", "format": "jsonl"},
        )
        assert ds.status_code == 200, ds.text
        dataset = ds.json()

        job_resp = client.post(
            "/v1/jobs/finetune",
            json={
                "model_name": "sshleifer/tiny-gpt2",
                "framework": "transformers",
                "technique": "lora",
                "dataset": {
                    "dataset_id": dataset["dataset_id"],
                    "version": dataset["version"],
                },
                "parameters": {
                    "max_steps": 3,
                    "per_device_train_batch_size": 1,
                    "logging_steps": 1,
                    "max_seq_length": 64,
                    "lora_r": 4,
                    "lora_alpha": 8,
                    "output_dir": str(data_dir / "outputs"),
                },
                "tags": {"source": "pytest_system"},
            },
        )
        assert job_resp.status_code == 200, job_resp.text
        job = job_resp.json()
        job_id = job["job_id"]

        deadline = time.time() + 600
        final = None
        while time.time() < deadline:
            final = client.get(f"/v1/jobs/{job_id}").json()
            if final["status"] in {"succeeded", "failed"}:
                break
            time.sleep(1.5)

        assert final is not None
        assert final["status"] == "succeeded", final.get("error") or final
        metrics = final.get("metrics") or {}
        assert metrics.get("real", 0.0) == 1.0, f"expected real train, got metrics={metrics}"
        assert "train_loss" in metrics or "final_train_loss" in metrics
        assert metrics.get("steps", 0) >= 1
        assert metrics.get("trainable_params", 0) > 0
        assert metrics.get("total_params", 0) > metrics.get("trainable_params", 0)

        # On-disk statistics.json from trainer
        stats_files = list((data_dir / "outputs").rglob("statistics.json"))
        assert stats_files, "statistics.json missing under outputs"
        stats = json.loads(stats_files[0].read_text())
        assert stats["mode"] == "real"
        assert stats["model_name"] == "sshleifer/tiny-gpt2"
        assert stats["trainable_params"] > 0


@pytest.mark.skipif(not _has_train_deps(), reason="train extras not installed")
def test_trainer_unit_real_tiny_gpt2(data_dir, sample_dataset, tmp_path):
    """Direct trainer call (no HTTP) — fastest proof of real LoRA + stats."""
    from ftaas.models import HyperParameters, Technique
    from training.frameworks.registry import get_trainer

    trainer = get_trainer("transformers")
    params = HyperParameters(
        max_steps=2,
        per_device_train_batch_size=1,
        logging_steps=1,
        max_seq_length=64,
        lora_r=4,
        lora_alpha=8,
        output_dir=str(tmp_path / "out"),
    )
    result = trainer.train(
        "sshleifer/tiny-gpt2",
        str(sample_dataset),
        Technique.LORA,
        params,
        "job_direct_real",
    )
    assert result.statistics["mode"] == "real"
    assert result.metrics.get("train_loss") is not None or result.statistics.get("final_train_loss") is not None
    assert result.statistics["trainable_params"] > 0
    assert (Path(result.output_dir) / "statistics.json").exists()
    assert (Path(result.output_dir) / "adapter_config.json").exists() or any(
        Path(result.output_dir).glob("*.safetensors")
    ) or any(Path(result.output_dir).glob("*.bin"))
