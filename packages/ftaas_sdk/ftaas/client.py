"""FTAAS / FTAAS Python SDK — programmatic client for FTAAS workbench & notebooks."""

from __future__ import annotations

from typing import Any, Optional

import httpx

from .config import get_settings
from .models import (
    CreateEndpointRequest,
    CreateFinetuneJobRequest,
    DatasetInfo,
    DatasetRef,
    EndpointInfo,
    FinetuneJob,
    Framework,
    HyperParameters,
    ModelInfo,
    PromptRequest,
    PromptResponse,
    RegisterDatasetRequest,
    Technique,
)


class Client:
    """Client for control + registry + deploy."""

    def __init__(
        self,
        control_url: Optional[str] = None,
        registry_url: Optional[str] = None,
        deploy_url: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        s = get_settings()
        self.control_url = (control_url or s.control_url).rstrip("/")
        self.registry_url = (registry_url or s.registry_url).rstrip("/")
        self.deploy_url = (deploy_url or s.deploy_url).rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ---- Registry ----
    def register_dataset(
        self,
        gcs_path: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        format: str = "jsonl",
    ) -> DatasetInfo:
        payload = RegisterDatasetRequest(
            gcs_path=gcs_path, name=name, description=description, format=format
        )
        r = self._client.post(f"{self.registry_url}/v1/datasets/register", json=payload.model_dump())
        r.raise_for_status()
        return DatasetInfo.model_validate(r.json())

    def get_dataset(self, dataset_id: str, version: str = "latest") -> DatasetInfo:
        r = self._client.get(f"{self.registry_url}/v1/datasets/{dataset_id}", params={"version": version})
        r.raise_for_status()
        return DatasetInfo.model_validate(r.json())

    def list_datasets(self) -> list[DatasetInfo]:
        r = self._client.get(f"{self.registry_url}/v1/datasets")
        r.raise_for_status()
        return [DatasetInfo.model_validate(x) for x in r.json()]

    # ---- Control ----
    def create_finetune_job(
        self,
        model_name: str,
        dataset: DatasetRef | DatasetInfo,
        framework: Framework | str = Framework.TRANSFORMERS,
        technique: Technique | str = Technique.LORA,
        parameters: Optional[HyperParameters | dict[str, Any]] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> FinetuneJob:
        if isinstance(dataset, DatasetInfo):
            ds = DatasetRef(
                dataset_id=dataset.dataset_id,
                version=dataset.version,
                gcs_path=dataset.gcs_path,
                local_path=dataset.local_path,
                name=dataset.name,
                num_rows=dataset.num_rows,
            )
        else:
            ds = dataset

        params = parameters or HyperParameters()
        if isinstance(params, dict):
            params = HyperParameters(**params)

        fw = Framework(framework) if isinstance(framework, str) else framework
        tech = Technique(technique) if isinstance(technique, str) else technique

        payload = CreateFinetuneJobRequest(
            model_name=model_name,
            framework=fw,
            technique=tech,
            dataset=ds,
            parameters=params,
            experiment_name=experiment_name,
            tags=tags or {},
        )
        r = self._client.post(
            f"{self.control_url}/v1/jobs/finetune",
            json=payload.model_dump(mode="json"),
        )
        r.raise_for_status()
        return FinetuneJob.model_validate(r.json())

    def get_job_status(self, job_id: str) -> FinetuneJob:
        r = self._client.get(f"{self.control_url}/v1/jobs/{job_id}")
        r.raise_for_status()
        return FinetuneJob.model_validate(r.json())

    def list_jobs(self) -> list[FinetuneJob]:
        r = self._client.get(f"{self.control_url}/v1/jobs")
        r.raise_for_status()
        return [FinetuneJob.model_validate(x) for x in r.json()]

    def get_model(self, model_name: str, version: str = "latest") -> ModelInfo:
        r = self._client.get(
            f"{self.control_url}/v1/models/{model_name}",
            params={"version": version},
        )
        r.raise_for_status()
        return ModelInfo.model_validate(r.json())

    def list_models(self) -> list[ModelInfo]:
        r = self._client.get(f"{self.control_url}/v1/models")
        r.raise_for_status()
        return [ModelInfo.model_validate(x) for x in r.json()]

    def wait_for_job(
        self,
        job_id: str,
        poll_seconds: float = 2.0,
        timeout_seconds: float = 600.0,
    ) -> FinetuneJob:
        import time

        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            job = self.get_job_status(job_id)
            if job.status.value in {"succeeded", "failed", "cancelled"}:
                return job
            time.sleep(poll_seconds)
        raise TimeoutError(f"Job {job_id} did not finish within {timeout_seconds}s")

    # ---- Deploy ----
    def create_endpoint(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        inference_framework: str = "vllm",
        use_adapters: bool = True,
    ) -> EndpointInfo:
        payload = CreateEndpointRequest(
            model_name=model_name,
            model_version=model_version,
            inference_framework=inference_framework,
            use_adapters=use_adapters,
        )
        r = self._client.post(
            f"{self.deploy_url}/v1/endpoints",
            json=payload.model_dump(),
        )
        r.raise_for_status()
        return EndpointInfo.model_validate(r.json())

    def prompt(
        self,
        endpoint_id: str,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ) -> PromptResponse:
        payload = PromptRequest(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        r = self._client.post(
            f"{self.deploy_url}/v1/endpoints/{endpoint_id}/prompt",
            json=payload.model_dump(),
        )
        r.raise_for_status()
        return PromptResponse.model_validate(r.json())

    def list_endpoints(self) -> list[EndpointInfo]:
        r = self._client.get(f"{self.deploy_url}/v1/endpoints")
        r.raise_for_status()
        return [EndpointInfo.model_validate(x) for x in r.json()]

    def catalog(self) -> dict[str, Any]:
        r = self._client.get(f"{self.control_url}/v1/catalog")
        r.raise_for_status()
        return r.json()


__all__ = ["Client"]
