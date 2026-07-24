"""
Local orchestrator mirroring the Airflow sequence diagram:

  download_dataset → load_parameters → create_cluster → submit_training_job
  → poll_job_status → log_metrics/params/model (MLflow) → register_model
  → complete pipeline → update job status
"""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from typing import Any

import httpx

from ftaas.config import ensure_data_dirs, get_platform_config, get_settings
from ftaas.models import Framework, HyperParameters, RegisterModelRequest, Technique
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

    root = ensure_data_dirs()
    artifact_root = root / "mlartifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
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
    control_url = settings.control_url.rstrip("/")
    registry_url = settings.registry_url.rstrip("/")
    workflow_url = settings.workflow_url.rstrip("/")

    # Make progress callbacks from the trainer able to reach control.
    os.environ["FTAAS_CONTROL_URL"] = control_url
    os.environ["FTAAS_JOB_ID"] = job_id

    with httpx.Client(timeout=120.0) as client:
        def set_status(status: str | None = None, **extra: Any) -> None:
            body: dict[str, Any] = {**extra}
            if status is not None:
                body["status"] = status
            client.patch(
                f"{control_url}/v1/jobs/{job_id}/status",
                json=body,
            ).raise_for_status()

        def log(message: str, *, percent: float | None = None, phase: str | None = None, **prog: Any) -> None:
            progress = {"message": message}
            if percent is not None:
                progress["percent"] = round(float(percent), 1)
            if phase:
                progress["phase"] = phase
            progress.update(prog)
            set_status(log=message, progress=progress)
            logger.info("[%s] %s", job_id, message)

        try:
            job = client.get(f"{control_url}/v1/jobs/{job_id}").json()
            set_status("running", log="Pipeline started", progress={"percent": 5, "phase": "running", "message": "Pipeline started"})

            ds = job["dataset"]
            log(
                f"Downloading dataset {ds['dataset_id']}:{ds.get('version') or '1'}",
                percent=10,
                phase="download",
            )
            dl = client.get(
                f"{registry_url}/v1/datasets/{ds['dataset_id']}/download",
                params={"version": ds.get("version") or "1"},
            )
            dl.raise_for_status()
            dataset_path = dl.json()["local_path"]
            log(f"Dataset ready at {dataset_path}", percent=20, phase="download")

            params = HyperParameters(**(job.get("parameters") or {}))
            # Always write checkpoints under FTAAS_DATA_DIR so compose/Railway volumes keep models.
            data_root = ensure_data_dirs()
            params.output_dir = str(data_root / "outputs")
            Path(params.output_dir).mkdir(parents=True, exist_ok=True)
            _ = load_parameters(params)
            log(
                f"Loaded hyperparameters max_steps={params.max_steps} lr={params.learning_rate}",
                percent=25,
                phase="prepare",
            )

            cluster = create_cluster(job["framework"])
            set_status(
                "training",
                ray_cluster=cluster.cluster_name,
                log=f"Cluster ready ({cluster.address})",
                progress={
                    "percent": 30,
                    "phase": "training",
                    "message": f"Starting train on {job['model_name']}",
                    "step": 0,
                    "max_steps": params.max_steps,
                },
            )

            handle = submit_training_job(
                cluster_name=cluster.cluster_name,
                model_name=job["model_name"],
                dataset_path=dataset_path,
                framework=Framework(job["framework"]),
                technique=Technique(job["technique"]),
                params=params,
                job_id_ref=job_id,
            )
            log(f"Training job submitted ({handle.job_id})", percent=35, phase="training")
            result = poll_job_status(handle.job_id)
            log(
                f"Training finished backend={(result.statistics or {}).get('backend')}",
                percent=75,
                phase="training",
            )

            stats = result.statistics or {}
            metrics = dict(result.metrics or {})
            for key in (
                "final_train_loss",
                "duration_seconds",
                "trainable_params",
                "total_params",
                "trainable_pct",
                "num_examples",
            ):
                if key in stats and isinstance(stats[key], (int, float)):
                    metrics[key] = float(stats[key])
            if stats.get("loss_curve"):
                metrics["loss_curve_len"] = float(len(stats["loss_curve"]))
                metrics["loss_first"] = float(stats["loss_curve"][0])
                metrics["loss_last"] = float(stats["loss_curve"][-1])
            metrics["real"] = 1.0

            set_status(
                "logging",
                metrics=metrics,
                log="Logging metrics to MLflow",
                progress={"percent": 85, "phase": "logging", "message": "Logging to MLflow"},
            )
            exp_name = job.get("tags", {}).get("experiment") or get_platform_config().integrations.mlflow_experiment
            exp_id, run_id = _mlflow_log(result.parameters, metrics, result.model_uri or result.output_dir, exp_name)

            set_status(
                "registering",
                mlflow_run_id=run_id,
                mlflow_experiment_id=str(exp_id),
                metrics=metrics,
                log=f"Registering model (run={run_id})",
                progress={"percent": 92, "phase": "registering", "message": "Registering model"},
            )
            reg = RegisterModelRequest(
                job_id=job_id,
                experiment_id=str(exp_id),
                run_id=run_id,
                model_name=job["model_name"].split("/")[-1] + "-ft",
                model_uri=result.model_uri or result.output_dir,
                metrics=metrics,
                parameters=result.parameters,
            )
            client.post(f"{control_url}/v1/models/register", json=reg.model_dump()).raise_for_status()

            client.post(f"{workflow_url}/v1/pipelines/{pipeline_id}/complete", params={"status": "succeeded"})
            set_status(
                "succeeded",
                metrics=metrics,
                log="Job succeeded",
                progress={
                    "percent": 100,
                    "phase": "succeeded",
                    "message": "Complete",
                    "step": int(metrics.get("steps") or params.max_steps),
                    "max_steps": params.max_steps,
                },
            )
            logger.info(
                "Job %s succeeded (real train loss=%s steps=%s backend=%s)",
                job_id,
                metrics.get("train_loss") or metrics.get("final_train_loss"),
                metrics.get("steps"),
                (result.statistics or {}).get("backend"),
            )

        except Exception as exc:
            logger.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
            try:
                set_status(
                    "failed",
                    error=str(exc),
                    log=f"FAILED: {exc}",
                    progress={"percent": 100, "phase": "failed", "message": str(exc)[:200]},
                )
                client.post(f"{workflow_url}/v1/pipelines/{pipeline_id}/complete", params={"status": "failed"})
            except Exception:
                pass
