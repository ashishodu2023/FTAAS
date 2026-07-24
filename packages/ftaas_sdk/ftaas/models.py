"""Shared FTAAS domain models used across services and the SDK."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str = "") -> str:
    uid = uuid4().hex[:12]
    return f"{prefix}{uid}" if prefix else uid


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    TRAINING = "training"
    LOGGING = "logging"
    REGISTERING = "registering"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Framework(str, Enum):
    TRANSFORMERS = "transformers"
    TRL = "trl"
    VERL = "verl"
    LLAMA_FACTORY = "llama-factory"
    UNSLOTH = "unsloth"
    AXOLOTL = "axolotl"


class Technique(str, Enum):
    # Full FT
    FULL_16BIT = "16bit_full"
    FROZEN = "frozen"
    # PEFT
    LORA = "lora"
    QLORA = "qlora"
    DORA = "dora"
    LORA_PLUS = "lora_plus"
    LONG_LORA = "longlora"
    LOFTQ = "loftq"
    PISSA = "pissa"
    PREFIX_TUNING = "prefix_tuning"
    ADAPTER_TUNING = "adapter_tuning"
    BITFIT = "bitfit"
    IA3 = "ia3"
    # Instruction
    SFT = "sft"
    MULTIMODAL_INSTRUCTION = "multimodal_instruction"
    SELF_INSTRUCT = "self_instruct"
    # Alignment
    REWARD_MODELING = "reward_modeling"
    PPO = "ppo"
    DPO = "dpo"
    ORPO = "orpo"


class DatasetRef(BaseModel):
    dataset_id: str
    version: str
    gcs_path: Optional[str] = None
    local_path: Optional[str] = None
    name: Optional[str] = None
    num_rows: Optional[int] = None


class RegisterDatasetRequest(BaseModel):
    gcs_path: str
    name: Optional[str] = None
    description: Optional[str] = None
    format: str = "jsonl"  # jsonl | csv | parquet | hf


class DatasetInfo(BaseModel):
    dataset_id: str
    version: str
    gcs_path: str
    local_path: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    format: str = "jsonl"
    num_rows: Optional[int] = None
    created_at: datetime = Field(default_factory=utcnow)


class HyperParameters(BaseModel):
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    max_steps: int = 10
    warmup_ratio: float = 0.03
    logging_steps: int = 1
    save_steps: int = 50
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_seq_length: int = 512
    output_dir: str = "./data/outputs"
    seed: int = 42
    extra: dict[str, Any] = Field(default_factory=dict)


class CreateFinetuneJobRequest(BaseModel):
    model_name: str
    framework: Framework = Framework.TRANSFORMERS
    technique: Technique = Technique.LORA
    dataset: DatasetRef
    parameters: HyperParameters = Field(default_factory=HyperParameters)
    experiment_name: Optional[str] = None
    tags: dict[str, str] = Field(default_factory=dict)


class FinetuneJob(BaseModel):
    job_id: str
    model_name: str
    framework: Framework
    technique: Technique
    dataset: DatasetRef
    parameters: HyperParameters
    status: JobStatus = JobStatus.PENDING
    pipeline_id: Optional[str] = None
    ray_cluster: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    mlflow_experiment_id: Optional[str] = None
    registered_model_name: Optional[str] = None
    registered_model_version: Optional[str] = None
    error: Optional[str] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    logs: list[dict[str, Any]] = Field(default_factory=list)
    progress: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    tags: dict[str, str] = Field(default_factory=dict)


class CreatePipelineRequest(BaseModel):
    job_id: str
    framework: Framework
    dag_id: str = "ftaas_finetune"


class PipelineInfo(BaseModel):
    pipeline_id: str
    job_id: str
    dag_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class RegisterModelRequest(BaseModel):
    job_id: str
    experiment_id: str
    run_id: str
    model_name: str
    model_uri: Optional[str] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    model_id: str
    model_name: str
    version: str
    job_id: str
    experiment_id: str
    run_id: str
    model_uri: Optional[str] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class CreateEndpointRequest(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    inference_framework: str = "vllm"  # vllm | transformers | ray_serve
    use_adapters: bool = True
    replicas: int = 1


class EndpointInfo(BaseModel):
    endpoint_id: str
    model_name: str
    model_version: Optional[str] = None
    inference_framework: str
    url: str
    status: str = "ready"
    created_at: datetime = Field(default_factory=utcnow)


class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 48
    temperature: float = 0.0


class PromptResponse(BaseModel):
    endpoint_id: str
    prompt: str
    completion: str
    model_name: str
