"""
Airflow DAG: ftaas_finetune

Mirrors the sequence diagram when airflow.enabled=true.
For local demo, prefer orchestrator.local_runner.runner.
"""

from __future__ import annotations

from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:  # Airflow not installed in local/dev
    DAG = None  # type: ignore
    PythonOperator = None  # type: ignore


def _run(**context):
    conf = context.get("dag_run").conf if context.get("dag_run") else {}
    job_id = conf.get("job_id")
    pipeline_id = conf.get("pipeline_id")
    if not job_id or not pipeline_id:
        raise ValueError("dag_run.conf must include job_id and pipeline_id")
    from orchestrator.local_runner.runner import run_finetune_pipeline

    run_finetune_pipeline(job_id, pipeline_id)


if DAG is not None:
    with DAG(
        dag_id="ftaas_finetune",
        start_date=datetime(2025, 1, 1),
        schedule=None,
        catchup=False,
        tags=["ftaas", "finetune", "ray"],
    ) as dag:
        PythonOperator(
            task_id="run_finetune_pipeline",
            python_callable=_run,
        )
