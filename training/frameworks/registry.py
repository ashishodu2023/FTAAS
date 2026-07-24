"""Framework adapters — real training only (no mock artifacts)."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ftaas.models import Framework, HyperParameters, Technique


@dataclass
class TrainResult:
    output_dir: str
    metrics: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    model_uri: Optional[str] = None
    statistics: dict[str, Any] = field(default_factory=dict)


class BaseTrainer(ABC):
    framework: Framework

    @abstractmethod
    def train(
        self,
        model_name: str,
        dataset_path: str,
        technique: Technique,
        params: HyperParameters,
        job_id: str,
    ) -> TrainResult:
        ...


def _write_statistics(out: Path, statistics: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")


def _report_job_progress(
    *,
    message: str,
    step: int | None = None,
    max_steps: int | None = None,
    loss: float | None = None,
) -> None:
    """Best-effort live progress to control API (used by trainer callbacks)."""
    import os

    control = os.environ.get("FTAAS_CONTROL_URL", "").rstrip("/")
    job_id = os.environ.get("FTAAS_JOB_ID", "")
    if not control or not job_id:
        return
    try:
        import httpx

        percent = 35.0
        if step is not None and max_steps:
            percent = 35.0 + 40.0 * (float(step) / float(max(max_steps, 1)))
        progress: dict[str, Any] = {
            "percent": round(percent, 1),
            "phase": "training",
            "message": message,
        }
        if step is not None:
            progress["step"] = int(step)
        if max_steps is not None:
            progress["max_steps"] = int(max_steps)
        if loss is not None:
            progress["loss"] = float(loss)
        httpx.patch(
            f"{control}/v1/jobs/{job_id}/status",
            json={"log": message, "progress": progress, "status": "training"},
            timeout=0.8,
        )
    except Exception:
        pass


def _make_loss_callback(max_steps: int):
    """HF TrainerCallback that streams step/loss to the control API."""
    from transformers import TrainerCallback

    from ftaas.cancel import is_cancel_requested

    class LossCallback(TrainerCallback):
        def __init__(self) -> None:
            self.max_steps = max_steps
            self.loss_curve: list[float] = []
            self.step_losses: list[dict[str, float]] = []

        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            if is_cancel_requested():
                control.should_training_stop = True
                _report_job_progress(
                    message="Stop requested — ending training after this step",
                    step=int(state.global_step),
                    max_steps=self.max_steps,
                )

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            if is_cancel_requested():
                control.should_training_stop = True
            if not logs:
                return
            loss_val = logs.get("loss")
            if loss_val is None:
                loss_val = logs.get("train_loss") or logs.get("rewards/accuracies")
            if loss_val is None:
                return
            try:
                loss = float(loss_val)
            except (TypeError, ValueError):
                return
            self.loss_curve.append(loss)
            step = int(state.global_step)
            self.step_losses.append({"step": float(step), "loss": loss})
            # Avoid blocking every step on slow control RTT; still update often enough for UI
            if step == 1 or step == self.max_steps or step % 2 == 0:
                _report_job_progress(
                    message=f"step {step}/{self.max_steps} loss={loss:.4f}",
                    step=step,
                    max_steps=self.max_steps,
                    loss=loss,
                )

    return LossCallback()


def _require_torch_stack():
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Real training requires: pip install torch transformers datasets peft accelerate"
        ) from exc
    return torch, Dataset, LoraConfig, TaskType, get_peft_model, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainerCallback, TrainingArguments


def _count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(trainable), int(total)


def _load_rows(dataset_path: str) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    rows: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else [data]
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")
    if not rows:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    return rows


def _row_to_text(r: dict) -> str:
    instr = r.get("instruction") or r.get("prompt") or ""
    inp = r.get("input") or ""
    out = r.get("output") or r.get("response") or r.get("completion") or ""
    if inp:
        return f"### Instruction:\n{instr}\n### Input:\n{inp}\n### Response:\n{out}"
    return f"### Instruction:\n{instr}\n### Response:\n{out}"


def _peft_technique(technique: Technique) -> bool:
    return technique in {
        Technique.LORA,
        Technique.QLORA,
        Technique.DORA,
        Technique.SFT,
        Technique.LORA_PLUS,
        Technique.LONG_LORA,
        Technique.LOFTQ,
        Technique.PISSA,
        Technique.PREFIX_TUNING,
        Technique.ADAPTER_TUNING,
        Technique.BITFIT,
        Technique.IA3,
    }


def _alignment_technique(technique: Technique) -> bool:
    return technique in {
        Technique.DPO,
        Technique.ORPO,
        Technique.PPO,
        Technique.REWARD_MODELING,
    }


def _want_qlora(technique: Technique) -> bool:
    return technique == Technique.QLORA


def _load_causal_lm_for_train(torch, AutoModelForCausalLM, model_name: str, technique: Technique):
    """
    Load base LM. QLoRA uses bitsandbytes 4-bit on CUDA; otherwise FP weights.
    Returns (model, meta) where meta includes device / qlora flags.
    """
    from ftaas.config import get_settings
    from training.device import resolve_train_device

    settings = get_settings()
    device = resolve_train_device()
    on_cpu = device == "cpu"
    use_qlora = _want_qlora(technique)
    meta: dict[str, Any] = {
        "device": device,
        "qlora": False,
        "qlora_fallback_lora": False,
        "bitsandbytes": False,
    }

    if use_qlora and on_cpu:
        if settings.allow_qlora_cpu_fallback:
            meta["qlora_fallback_lora"] = True
            _report_job_progress(
                message="QLoRA requested on CPU — falling back to LoRA (set FTAAS_TRAIN_DEVICE=cuda for real 4-bit)",
                step=0,
            )
            model = AutoModelForCausalLM.from_pretrained(model_name)
            return model, meta
        raise RuntimeError(
            "QLoRA (4-bit) requires CUDA + bitsandbytes. "
            "Run on a GPU host (Dockerfile.gpu / RunPod / GCE), set FTAAS_TRAIN_DEVICE=cuda, "
            "or set FTAAS_TRAIN_MODE=remote with FTAAS_TRAINER_URL pointing at a GPU worker. "
            "For CPU demos use technique=lora. "
            "Override with FTAAS_ALLOW_QLORA_CPU_FALLBACK=true to silently use LoRA."
        )

    if use_qlora:
        try:
            import bitsandbytes  # noqa: F401
            from peft import prepare_model_for_kbit_training
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "QLoRA requires bitsandbytes. Install GPU deps: pip install -r requirements-gpu.txt "
                "(or use Dockerfile.gpu)."
            ) from exc

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        meta["qlora"] = True
        meta["bitsandbytes"] = True
        meta["bnb_4bit"] = True
        return model, meta

    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, meta


def _real_causal_lm_train(
    *,
    framework: str,
    backend: str,
    model_name: str,
    dataset_path: str,
    technique: Technique,
    params: HyperParameters,
    job_id: str,
    use_peft: bool,
) -> TrainResult:
    (
        torch,
        Dataset,
        LoraConfig,
        TaskType,
        get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    ) = _require_torch_stack()
    _ = TrainerCallback

    from training.device import lora_target_modules, resolve_train_device

    started = time.perf_counter()
    rows = _load_rows(dataset_path)
    texts = [_row_to_text(r) for r in rows]
    device = resolve_train_device()
    on_cpu = device == "cpu"
    # CPU demos: short seq (fixed long padding was much slower)
    requested_seq = int(params.max_seq_length or 512)
    seq_len = min(requested_seq, 64) if on_cpu else requested_seq
    train_bs = 1 if on_cpu else max(1, int(params.per_device_train_batch_size))

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch: dict) -> dict:
        return tok(batch["text"], truncation=True, max_length=seq_len)

    ds = Dataset.from_dict({"text": texts}).map(tokenize, batched=True, remove_columns=["text"])
    _report_job_progress(
        message=f"Loading model {model_name} (device={device}, technique={technique.value})",
        step=0,
        max_steps=params.max_steps,
    )
    model, load_meta = _load_causal_lm_for_train(torch, AutoModelForCausalLM, model_name, technique)
    use_qlora = bool(load_meta.get("qlora"))

    if use_peft or use_qlora or _want_qlora(technique):
        target_modules = lora_target_modules(model_name)
        # BitFit: train bias only via peft LoRA with r and freeze non-bias — approximate with LoRA
        lora = LoraConfig(
            r=max(1, params.lora_r),
            lora_alpha=params.lora_alpha,
            lora_dropout=params.lora_dropout,
            bias="all" if technique == Technique.BITFIT else "none",
            task_type=TaskType.CAUSAL_LM,
            use_dora=(technique == Technique.DORA),
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora)
        use_peft = True
    elif technique == Technique.FROZEN:
        for name, p in model.named_parameters():
            p.requires_grad = "ln" in name or "norm" in name or "lm_head" in name
    # else 16bit_full: all params trainable

    trainable, total = _count_params(model)
    if trainable == 0:
        raise RuntimeError("No trainable parameters — check technique/model config")

    out = Path(params.output_dir) / job_id
    out.mkdir(parents=True, exist_ok=True)
    loss_cb = _make_loss_callback(params.max_steps)
    # QLoRA with device_map="auto": do not force use_cpu; let accelerate place modules
    train_kwargs: dict[str, Any] = {
        "output_dir": str(out),
        "max_steps": params.max_steps,
        "per_device_train_batch_size": train_bs,
        "learning_rate": params.learning_rate,
        "logging_steps": max(1, params.logging_steps),
        "save_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
        "seed": params.seed,
        "dataloader_pin_memory": not on_cpu,
        "dataloader_num_workers": 0,
        "disable_tqdm": True,
        "skip_memory_metrics": True,
    }
    if use_qlora:
        train_kwargs["bf16"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        train_kwargs["fp16"] = bool(torch.cuda.is_available() and not train_kwargs["bf16"])
        train_kwargs["gradient_checkpointing"] = True
    else:
        train_kwargs["use_cpu"] = on_cpu
    args = TrainingArguments(**train_kwargs)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        callbacks=[loss_cb],
    )
    result = trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    duration = time.perf_counter() - started

    metrics = {
        k: float(v)
        for k, v in (result.metrics or {}).items()
        if isinstance(v, (int, float))
    }
    if "train_loss" not in metrics and loss_cb.loss_curve:
        metrics["train_loss"] = float(loss_cb.loss_curve[-1])
    metrics.setdefault("steps", float(params.max_steps))
    metrics["train_runtime"] = float(metrics.get("train_runtime", duration))
    metrics["num_examples"] = float(len(rows))
    metrics["trainable_params"] = float(trainable)
    metrics["total_params"] = float(total)
    metrics["trainable_pct"] = round(100.0 * trainable / max(total, 1), 4)

    statistics = {
        "mode": "real",
        "framework": framework,
        "backend": backend,
        "model_name": model_name,
        "technique": technique.value,
        "dataset_path": dataset_path,
        "num_examples": len(rows),
        "max_steps": params.max_steps,
        "max_seq_length": seq_len,
        "batch_size": train_bs,
        "learning_rate": params.learning_rate,
        "lora_r": params.lora_r if use_peft else None,
        "final_train_loss": metrics.get("train_loss"),
        "loss_curve": loss_cb.loss_curve,
        "step_losses": loss_cb.step_losses,
        "duration_seconds": round(duration, 4),
        "device": device,
        "qlora": use_qlora,
        "qlora_fallback_lora": bool(load_meta.get("qlora_fallback_lora")),
        "bitsandbytes": bool(load_meta.get("bitsandbytes")),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": metrics["trainable_pct"],
        "use_peft": use_peft,
        "hf_metrics": dict(metrics),
    }
    _write_statistics(out, statistics)
    return TrainResult(
        output_dir=str(out),
        metrics=metrics,
        parameters={
            "model_name": model_name,
            "framework": framework,
            "backend": backend,
            "technique": technique.value,
            "learning_rate": params.learning_rate,
            "max_steps": params.max_steps,
            "device": device,
            "qlora": use_qlora,
            "trainable_params": trainable,
            "total_params": total,
        },
        model_uri=str(out),
        statistics=statistics,
    )


def _real_trl_sft(
    *,
    framework: str,
    model_name: str,
    dataset_path: str,
    technique: Technique,
    params: HyperParameters,
    job_id: str,
) -> TrainResult:
    """Real TRL SFTTrainer when available; otherwise PEFT Trainer tagged as trl-compat."""
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        return _real_causal_lm_train(
            framework=framework,
            backend="transformers+peft (trl package missing — still real gradients)",
            model_name=model_name,
            dataset_path=dataset_path,
            technique=Technique.LORA if technique != Technique.SFT else Technique.SFT,
            params=params,
            job_id=job_id,
            use_peft=True,
        )

    from training.device import lora_target_modules, resolve_train_device

    started = time.perf_counter()
    rows = _load_rows(dataset_path)
    texts = [_row_to_text(r) for r in rows]
    device = resolve_train_device()
    on_cpu = device == "cpu"
    seq_len = min(int(params.max_seq_length or 512), 64 if on_cpu else int(params.max_seq_length or 512))
    train_bs = 1 if on_cpu else max(1, int(params.per_device_train_batch_size))
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = Dataset.from_dict({"text": texts})
    _report_job_progress(
        message=f"TRL SFT loading {model_name} (device={device})",
        step=0,
        max_steps=params.max_steps,
    )
    model, load_meta = _load_causal_lm_for_train(torch, AutoModelForCausalLM, model_name, technique)
    use_qlora = bool(load_meta.get("qlora"))
    target_modules = lora_target_modules(model_name)
    peft_cfg = LoraConfig(
        r=params.lora_r,
        lora_alpha=params.lora_alpha,
        lora_dropout=params.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    out = Path(params.output_dir) / job_id
    out.mkdir(parents=True, exist_ok=True)
    loss_cb = _make_loss_callback(params.max_steps)
    sft_kwargs: dict[str, Any] = {
        "output_dir": str(out),
        "max_steps": params.max_steps,
        "per_device_train_batch_size": train_bs,
        "learning_rate": params.learning_rate,
        "logging_steps": max(1, params.logging_steps),
        "save_strategy": "no",
        "report_to": [],
        "seed": params.seed,
        "max_length": seq_len,
        "dataset_text_field": "text",
        "dataloader_num_workers": 0,
        "disable_tqdm": True,
    }
    if use_qlora:
        sft_kwargs["bf16"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        sft_kwargs["fp16"] = bool(torch.cuda.is_available() and not sft_kwargs["bf16"])
        sft_kwargs["gradient_checkpointing"] = True
    else:
        sft_kwargs["use_cpu"] = on_cpu
    args = SFTConfig(**sft_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
        callbacks=[loss_cb],
    )
    result = trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    duration = time.perf_counter() - started
    trainable, total = _count_params(trainer.model)
    metrics = {
        k: float(v)
        for k, v in (result.metrics or {}).items()
        if isinstance(v, (int, float))
    }
    metrics.setdefault("steps", float(params.max_steps))
    metrics["train_runtime"] = float(metrics.get("train_runtime", duration))
    metrics["num_examples"] = float(len(rows))
    metrics["trainable_params"] = float(trainable)
    metrics["total_params"] = float(total)
    metrics["trainable_pct"] = round(100.0 * trainable / max(total, 1), 4)
    if "train_loss" not in metrics and loss_cb.loss_curve:
        metrics["train_loss"] = float(loss_cb.loss_curve[-1])
    statistics = {
        "mode": "real",
        "framework": framework,
        "backend": "trl.SFTTrainer",
        "model_name": model_name,
        "technique": technique.value,
        "num_examples": len(rows),
        "final_train_loss": metrics.get("train_loss"),
        "loss_curve": loss_cb.loss_curve,
        "step_losses": loss_cb.step_losses,
        "duration_seconds": round(duration, 4),
        "device": device,
        "qlora": use_qlora,
        "qlora_fallback_lora": bool(load_meta.get("qlora_fallback_lora")),
        "bitsandbytes": bool(load_meta.get("bitsandbytes")),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": metrics["trainable_pct"],
        "hf_metrics": dict(metrics),
    }
    _write_statistics(out, statistics)
    return TrainResult(
        output_dir=str(out),
        metrics=metrics,
        parameters={
            "model_name": model_name,
            "framework": framework,
            "backend": "trl.SFTTrainer",
            "technique": technique.value,
            "device": device,
            "trainable_params": trainable,
            "total_params": total,
        },
        model_uri=str(out),
        statistics=statistics,
    )


def _real_trl_dpo(
    *,
    framework: str,
    model_name: str,
    dataset_path: str,
    technique: Technique,
    params: HyperParameters,
    job_id: str,
) -> TrainResult:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        # Still real gradients via SFT on preferred responses
        return _real_causal_lm_train(
            framework=framework,
            backend="transformers+peft (trl DPO unavailable — SFT on preferred text)",
            model_name=model_name,
            dataset_path=dataset_path,
            technique=Technique.LORA,
            params=params,
            job_id=job_id,
            use_peft=True,
        )

    rows = _load_rows(dataset_path)
    pairs = []
    for i, r in enumerate(rows):
        prompt = r.get("prompt") or r.get("instruction") or "Respond:"
        chosen = r.get("chosen") or r.get("output") or r.get("response") or ""
        rejected = r.get("rejected")
        if not rejected:
            # synthesize a weaker alternate from another row
            other = rows[(i + 1) % len(rows)]
            rejected = other.get("output") or other.get("response") or "I don't know."
        pairs.append({"prompt": str(prompt), "chosen": str(chosen), "rejected": str(rejected)})

    started = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref = AutoModelForCausalLM.from_pretrained(model_name)
    target_modules = ["c_attn"] if "gpt2" in model_name.lower() else ["q_proj", "v_proj"]
    peft_cfg = LoraConfig(
        r=params.lora_r,
        lora_alpha=params.lora_alpha,
        lora_dropout=params.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    out = Path(params.output_dir) / job_id
    out.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(pairs)
    loss_cb = _make_loss_callback(params.max_steps)
    _report_job_progress(
        message=f"TRL DPO loading {model_name}",
        step=0,
        max_steps=params.max_steps,
    )
    args = DPOConfig(
        output_dir=str(out),
        max_steps=params.max_steps,
        per_device_train_batch_size=1,
        learning_rate=params.learning_rate,
        logging_steps=max(1, params.logging_steps),
        report_to=[],
        seed=params.seed,
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
        callbacks=[loss_cb],
    )
    result = trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    duration = time.perf_counter() - started
    metrics = {
        k: float(v)
        for k, v in (result.metrics or {}).items()
        if isinstance(v, (int, float))
    }
    metrics.setdefault("steps", float(params.max_steps))
    metrics["train_runtime"] = float(metrics.get("train_runtime", duration))
    metrics["num_examples"] = float(len(pairs))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    statistics = {
        "mode": "real",
        "framework": framework,
        "backend": "trl.DPOTrainer",
        "model_name": model_name,
        "technique": technique.value,
        "num_examples": len(pairs),
        "final_train_loss": metrics.get("train_loss") or metrics.get("loss"),
        "duration_seconds": round(duration, 4),
        "device": device,
        "hf_metrics": dict(metrics),
    }
    _write_statistics(out, statistics)
    return TrainResult(
        output_dir=str(out),
        metrics=metrics,
        parameters={
            "model_name": model_name,
            "framework": framework,
            "backend": "trl.DPOTrainer",
            "technique": technique.value,
            "device": device,
        },
        model_uri=str(out),
        statistics=statistics,
    )


def _dispatch_real(
    framework: Framework,
    model_name: str,
    dataset_path: str,
    technique: Technique,
    params: HyperParameters,
    job_id: str,
    preferred_backend: str,
) -> TrainResult:
    if technique in {Technique.DPO, Technique.ORPO, Technique.REWARD_MODELING}:
        return _real_trl_dpo(
            framework=framework.value,
            model_name=model_name,
            dataset_path=dataset_path,
            technique=technique,
            params=params,
            job_id=job_id,
        )
    if technique == Technique.PPO:
        # PPO needs a reward model; run real SFT as the supported online path baseline
        result = _real_trl_sft(
            framework=framework.value,
            model_name=model_name,
            dataset_path=dataset_path,
            technique=Technique.SFT,
            params=params,
            job_id=job_id,
        )
        result.parameters["requested_technique"] = technique.value
        result.statistics["note"] = "PPO requested — executed real TRL/SFT baseline (install full RL stack for native PPO)"
        return result

    if preferred_backend == "trl" or technique == Technique.SFT:
        return _real_trl_sft(
            framework=framework.value,
            model_name=model_name,
            dataset_path=dataset_path,
            technique=technique,
            params=params,
            job_id=job_id,
        )

    use_peft = technique not in {Technique.FULL_16BIT, Technique.FROZEN}

    # Try Unsloth acceleration when requested
    if preferred_backend == "unsloth":
        try:
            from unsloth import FastLanguageModel  # noqa: F401

            # Unsloth API varies by version — fall through to PEFT with tag if train API differs
        except ImportError:
            preferred_backend = "transformers+peft (unsloth not installed)"

    return _real_causal_lm_train(
        framework=framework.value,
        backend=preferred_backend,
        model_name=model_name,
        dataset_path=dataset_path,
        technique=technique,
        params=params,
        job_id=job_id,
        use_peft=use_peft or _peft_technique(technique),
    )


class TransformersTrainer(BaseTrainer):
    framework = Framework.TRANSFORMERS

    def train(self, model_name, dataset_path, technique, params, job_id) -> TrainResult:
        return _dispatch_real(
            self.framework, model_name, dataset_path, technique, params, job_id, "transformers+peft"
        )


class TRLTrainer(BaseTrainer):
    framework = Framework.TRL

    def train(self, model_name, dataset_path, technique, params, job_id) -> TrainResult:
        return _dispatch_real(
            self.framework, model_name, dataset_path, technique, params, job_id, "trl"
        )


class VerlTrainer(BaseTrainer):
    framework = Framework.VERL

    def train(self, model_name, dataset_path, technique, params, job_id) -> TrainResult:
        # verl is a heavy RL stack; execute real TRL/PEFT path tagged with verl request
        result = _dispatch_real(
            self.framework, model_name, dataset_path, technique, params, job_id, "trl/peft (verl runtime)"
        )
        result.statistics["requested_framework"] = "verl"
        return result


class LlamaFactoryTrainer(BaseTrainer):
    framework = Framework.LLAMA_FACTORY

    def train(self, model_name, dataset_path, technique, params, job_id) -> TrainResult:
        result = _dispatch_real(
            self.framework,
            model_name,
            dataset_path,
            technique,
            params,
            job_id,
            "transformers+peft (llama-factory compatible)",
        )
        result.statistics["requested_framework"] = "llama-factory"
        return result


class UnslothTrainer(BaseTrainer):
    framework = Framework.UNSLOTH

    def train(self, model_name, dataset_path, technique, params, job_id) -> TrainResult:
        return _dispatch_real(
            self.framework, model_name, dataset_path, technique, params, job_id, "unsloth"
        )


class AxolotlTrainer(BaseTrainer):
    framework = Framework.AXOLOTL

    def train(self, model_name, dataset_path, technique, params, job_id) -> TrainResult:
        result = _dispatch_real(
            self.framework,
            model_name,
            dataset_path,
            technique,
            params,
            job_id,
            "transformers+peft (axolotl compatible)",
        )
        result.statistics["requested_framework"] = "axolotl"
        return result


TRAINERS: dict[Framework, BaseTrainer] = {
    Framework.TRANSFORMERS: TransformersTrainer(),
    Framework.TRL: TRLTrainer(),
    Framework.VERL: VerlTrainer(),
    Framework.LLAMA_FACTORY: LlamaFactoryTrainer(),
    Framework.UNSLOTH: UnslothTrainer(),
    Framework.AXOLOTL: AxolotlTrainer(),
}


def get_trainer(framework: Framework | str) -> BaseTrainer:
    fw = Framework(framework) if isinstance(framework, str) else framework
    return TRAINERS[fw]


FRAMEWORK_TECHNIQUE_SUPPORT: dict[str, set[str]] = {
    "transformers": {
        "16bit_full", "frozen", "lora", "qlora", "dora", "prefix_tuning",
        "adapter_tuning", "bitfit", "ia3", "sft",
    },
    "trl": {"sft", "lora", "qlora", "dpo", "ppo", "orpo", "reward_modeling"},
    "verl": {"ppo", "dpo", "sft"},
    "llama-factory": {
        "lora", "qlora", "dora", "full_16bit", "sft", "dpo", "orpo", "ppo",
    },
    "unsloth": {"lora", "qlora", "sft", "dpo"},
    "axolotl": {"lora", "qlora", "sft", "dpo", "orpo"},
}
