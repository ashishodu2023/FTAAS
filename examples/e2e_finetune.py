#!/usr/bin/env python3
"""End-to-end Fine Tuning as a Service demo via FTAAS SDK."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "packages" / "ftaas_sdk"), str(ROOT / "services")]

from rich.console import Console
from rich.table import Table

from ftaas import FTAASClient, Framework, Technique, HyperParameters

console = Console()
SAMPLE = ROOT / "examples" / "data" / "alpaca_sample.jsonl"


def main() -> int:
    console.rule("[bold cyan]FTAAS E2E")
    with FTAASClient() as client:
        console.print("[bold]1. register_dataset[/]")
        ds = client.register_dataset(
            gcs_path=str(SAMPLE),
            name="alpaca-sample",
            format="jsonl",
        )
        console.print(f"  → {ds.dataset_id}:{ds.version}  rows={ds.num_rows}  path={ds.local_path}")

        console.print("[bold]2. create_finetune_job[/]")
        job = client.create_finetune_job(
            model_name="sshleifer/tiny-gpt2",
            dataset=ds,
            framework=Framework.TRANSFORMERS,
            technique=Technique.LORA,
            parameters=HyperParameters(max_steps=5, per_device_train_batch_size=1),
            tags={"source": "e2e_smoke"},
        )
        console.print(f"  → job_id={job.job_id}  pipeline={job.pipeline_id}  status={job.status}")

        console.print("[bold]3. poll get_job_status[/]")
        job = client.wait_for_job(job.job_id, poll_seconds=1.5, timeout_seconds=300)
        console.print(f"  → status={job.status}  metrics={job.metrics}")
        if job.status.value != "succeeded":
            console.print(f"[red]FAILED[/] {job.error}")
            return 1

        console.print("[bold]4. get_model[/]")
        # registered as <basename>-ft
        model_name = job.registered_model_name or "tiny-gpt2-ft"
        # wait briefly for registry write
        time.sleep(0.5)
        model = client.get_model(model_name)
        console.print(f"  → {model.model_name} v{model.version}  uri={model.model_uri}")

        console.print("[bold]5. create_endpoint (serving)[/]")
        ep = client.create_endpoint(model_name=model.model_name, inference_framework="vllm")
        console.print(f"  → {ep.endpoint_id}  {ep.url}")

        console.print("[bold]6. prompt[/]")
        resp = client.prompt(ep.endpoint_id, "What is LoRA?")
        console.print(f"  → {resp.completion}")

        table = Table(title="E2E OK")
        table.add_column("Step")
        table.add_column("Result")
        table.add_row("dataset", f"{ds.dataset_id}:{ds.version}")
        table.add_row("job", f"{job.job_id} ({job.status})")
        table.add_row("model", f"{model.model_name}:v{model.version}")
        table.add_row("endpoint", ep.endpoint_id)
        console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
