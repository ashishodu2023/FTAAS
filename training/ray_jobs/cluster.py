"""Ray training job helpers — real local or Ray-remote execution (no mock train)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from ftaas.config import get_platform_config
from ftaas.models import Framework, HyperParameters, Technique
from training.frameworks.registry import TrainResult, get_trainer


@dataclass
class RayCluster:
    cluster_name: str
    address: str
    distributed: bool = False


@dataclass
class RayJobHandle:
    job_id: str
    cluster_name: str
    status: str = "PENDING"


_CLUSTERS: dict[str, RayCluster] = {}
_JOBS: dict[str, dict[str, Any]] = {}


def create_cluster(framework: str, num_workers: int = 1) -> RayCluster:
    """Create a real Ray cluster when available; otherwise local in-process execution."""
    cfg = get_platform_config()
    name = f"ray-{framework}-{uuid.uuid4().hex[:8]}"
    prefer_ray = not cfg.integrations.ray_mock
    if prefer_ray:
        try:
            import ray

            if not ray.is_initialized():
                addr = cfg.integrations.ray_address
                if addr and addr != "auto":
                    ray.init(address=addr, ignore_reinit_error=True)
                else:
                    ray.init(ignore_reinit_error=True, include_dashboard=False)
            cluster = RayCluster(
                cluster_name=name,
                address=str(ray.address_info.get("address", "local")),
                distributed=True,
            )
            _CLUSTERS[name] = cluster
            return cluster
        except Exception:
            pass
    cluster = RayCluster(cluster_name=name, address="local://process", distributed=False)
    _CLUSTERS[name] = cluster
    return cluster


def load_parameters(params: HyperParameters | dict) -> dict[str, Any]:
    if isinstance(params, HyperParameters):
        return params.model_dump()
    return dict(params)


def submit_training_job(
    cluster_name: str,
    model_name: str,
    dataset_path: str,
    framework: Framework | str,
    technique: Technique | str,
    params: HyperParameters,
    job_id_ref: str,
) -> RayJobHandle:
    handle = RayJobHandle(job_id=f"rayjob_{uuid.uuid4().hex[:10]}", cluster_name=cluster_name, status="RUNNING")
    _JOBS[handle.job_id] = {
        "handle": handle,
        "model_name": model_name,
        "dataset_path": dataset_path,
        "framework": framework,
        "technique": technique,
        "params": params,
        "job_id_ref": job_id_ref,
        "result": None,
        "error": None,
    }
    cluster = _CLUSTERS.get(cluster_name)
    try:
        if cluster and cluster.distributed:
            import ray

            @ray.remote
            def _remote_train(fw, model, ds, tech, p, jid):
                trainer = get_trainer(fw)
                return trainer.train(model, ds, Technique(tech), HyperParameters(**p), jid)

            ref = _remote_train.remote(
                str(framework),
                model_name,
                dataset_path,
                str(technique),
                params.model_dump(),
                job_id_ref,
            )
            result: TrainResult = ray.get(ref)
        else:
            trainer = get_trainer(framework)
            tech = Technique(technique) if isinstance(technique, str) else technique
            result = trainer.train(model_name, dataset_path, tech, params, job_id_ref)
        _JOBS[handle.job_id]["result"] = result
        handle.status = "SUCCEEDED"
    except Exception as exc:
        _JOBS[handle.job_id]["error"] = str(exc)
        handle.status = "FAILED"
    return handle


def poll_job_status(ray_job_id: str, timeout_s: float = 3600.0, interval_s: float = 1.0) -> TrainResult:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = _JOBS.get(ray_job_id)
        if not job:
            raise KeyError(f"Unknown Ray job {ray_job_id}")
        handle: RayJobHandle = job["handle"]
        if handle.status == "SUCCEEDED":
            return job["result"]
        if handle.status == "FAILED":
            raise RuntimeError(job.get("error") or "Ray training failed")
        time.sleep(interval_s)
    raise TimeoutError(f"Ray job {ray_job_id} timed out")
