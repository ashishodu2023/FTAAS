"""
Local orchestrator mirroring the Airflow sequence diagram:

  download_dataset → load_parameters → create_cluster → submit_training_job
  → poll_job_status → log_metrics/params/model (MLflow) → register_model
  → complete pipeline → update job status
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

import httpx

from mdlc.config import ensure_data_dirs, get_platform_config, get_settings
from mdlc.models import Framework, HyperParameters, RegisterModelRequest, Technique
from training.ray_jobs.cluster import (
    create_cluster,
    load_parameters,
    poll_job_status,
    submit_training_job,
)

logger = logging.getLogger("ftaas.local_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _mlflow_log(params: dict[str, Any], metrics: dict[str, float], model_uri: str, experiment: str):
    """log_metrics / log_parameters / log_model — local sqlite store (no MLflow server required)."""
    import json
    import os

    root = ensure_data_dirs()
    artifact_root = root / "mlartifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    # Newer MLflow disables FileStore; use sqlite backend by default.
    local_store = os.environ.get("FTAAS_MLFLOW_TRACKING_URI") or f"sqlite:///{root / 'mlflow.db'}"

    try:
        import mlflow
    except ImportError:
        fb = root / "mlruns_fallback"
        fb.mkdir(parents=True, exist_ok=True)
        run_id = f"local_{Path(model_uri).name}"
        (fb / f"{run_id}.json").write_text(
            json.dumps({"params": params, "metrics": metrics, "model_uri": model_uri})
        )
        return "0", run_id

    if local_store.startswith("http"):
        try:
            import urllib.request

            urllib.request.urlopen(local_store.rstrip("/") + "/health", timeout=1.5)
            tracking = local_store
        except Exception:
            tracking = f"sqlite:///{root / 'mlflow.db'}"
    else:
        tracking = local_store

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=experiment) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                pass
        if model_uri and Path(model_uri).exists():
            try:
                mlflow.log_artifacts(model_uri, artifact_path="model")
            except Exception:
                pass
        return str(run.info.experiment_id), run.info.run_id


def run_finetune_pipeline(job_id: str, pipeline_id: str) -> None:
    settings = get_settings()
    mdlc = settings.mdlc_url.rstrip("/")
    mds = settings.mds_url.rstrip("/")
    pipes = settings.pipelineserv_url.rstrip("/")

    with httpx.Client(timeout=120.0) as client:
        def set_status(status: str, **extra: Any) -> None:
            client.patch(
                f"{mdlc}/v1/jobs/{job_id}/status",
                json={"status": status, **extra},
            ).raise_for_status()

        try:
            job = client.get(f"{mdlc}/v1/jobs/{job_id}").json()
            set_status("running")

            # 11. download_dataset(id, version)
            ds = job["dataset"]
            dl = client.get(
                f"{mds}/v1/datasets/{ds['dataset_id']}/download",
                params={"version": ds.get("version") or "1"},
            )
            dl.raise_for_status()
            dataset_path = dl.json()["local_path"]

            # 12. load_parameters()
            params = HyperParameters(**(job.get("parameters") or {}))
            _ = load_parameters(params)

            # 13-14. create_cluster()
            cluster = create_cluster(job["framework"])
            set_status("training", ray_cluster=cluster.cluster_name)

            # 15-17. submit_training_job + poll until complete
            handle = submit_training_job(
                cluster_name=cluster.cluster_name,
                model_name=job["model_name"],
                dataset_path=dataset_path,
                framework=Framework(job["framework"]),
                technique=Technique(job["technique"]),
                params=params,
                mdlc_job_id=job_id,
            )
            result = poll_job_status(handle.job_id)

            # 18-20. log to mlflow
            set_status("logging")
            exp_name = job.get("tags", {}).get("experiment") or get_platform_config().integrations.mlflow_experiment
            exp_id, run_id = _mlflow_log(result.parameters, result.metrics, result.model_uri or result.output_dir, exp_name)

            # 21-23. register_model
            set_status("registering", mlflow_run_id=run_id, mlflow_experiment_id=str(exp_id), metrics=result.metrics)
            reg = RegisterModelRequest(
                job_id=job_id,
                experiment_id=str(exp_id),
                run_id=run_id,
                model_name=job["model_name"].split("/")[-1] + "-ft",
                model_uri=result.model_uri or result.output_dir,
                metrics=result.metrics,
                parameters=result.parameters,
            )
            client.post(f"{mdlc}/v1/models/register", json=reg.model_dump()).raise_for_status()

            # 24-26. complete pipeline → mdlc updates status
            client.post(f"{pipes}/v1/pipelines/{pipeline_id}/complete", params={"status": "succeeded"})
            set_status("succeeded", metrics=result.metrics)
            logger.info("Job %s succeeded (mock=%s)", job_id, result.mock)

        except Exception as exc:
            logger.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
            try:
                set_status("failed", error=str(exc))
                client.post(f"{pipes}/v1/pipelines/{pipeline_id}/complete", params={"status": "failed"})
            except Exception:
                pass
