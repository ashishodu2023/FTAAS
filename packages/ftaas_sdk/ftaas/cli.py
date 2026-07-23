"""ftaas CLI — thin Typer wrapper around FTAASClient."""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich import print as rprint

from .client import FTAASClient
from .models import DatasetRef, Framework, HyperParameters, Technique

# re-export client module path for pyproject
from . import client as _client  # noqa: F401

app = typer.Typer(name="ftaas", help="FTAAS Python SDK CLI for Fine Tuning as a Service")


@app.command("register-dataset")
def register_dataset(
    gcs_path: str = typer.Argument(..., help="gs://bucket/path or local file:// /path"),
    name: Optional[str] = typer.Option(None),
    format: str = typer.Option("jsonl"),
) -> None:
    with FTAASClient() as c:
        ds = c.register_dataset(gcs_path, name=name, format=format)
        rprint(ds.model_dump())


@app.command("create-job")
def create_job(
    model_name: str = typer.Option(...),
    dataset_id: str = typer.Option(...),
    version: str = typer.Option("1"),
    framework: str = typer.Option("transformers"),
    technique: str = typer.Option("lora"),
    max_steps: int = typer.Option(10),
) -> None:
    with FTAASClient() as c:
        job = c.create_finetune_job(
            model_name=model_name,
            dataset=DatasetRef(dataset_id=dataset_id, version=version),
            framework=Framework(framework),
            technique=Technique(technique),
            parameters=HyperParameters(max_steps=max_steps),
        )
        rprint(job.model_dump())


@app.command("status")
def status(job_id: str) -> None:
    with FTAASClient() as c:
        job = c.get_job_status(job_id)
        rprint(job.model_dump())


@app.command("wait")
def wait(job_id: str, timeout: float = 600.0) -> None:
    with FTAASClient() as c:
        job = c.wait_for_job(job_id, timeout_seconds=timeout)
        rprint(job.model_dump())


@app.command("catalog")
def catalog() -> None:
    with FTAASClient() as c:
        rprint(json.dumps(c.catalog(), indent=2))


if __name__ == "__main__":
    app()
