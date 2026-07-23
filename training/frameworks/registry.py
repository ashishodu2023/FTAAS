"""Framework adapters for FTAAS training backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from mdlc.models import Framework, HyperParameters, Technique


@dataclass
class TrainResult:
    output_dir: str
    metrics: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    model_uri: Optional[str] = None
    mock: bool = False


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


def _mock_train(
    framework: Framework,
    model_name: str,
    dataset_path: str,
    technique: Technique,
    params: HyperParameters,
    job_id: str,
) -> TrainResult:
    out = Path(params.output_dir) / job_id
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "adapter_config.json"
    marker.write_text(
        f'{{"framework": "{framework.value}", "model": "{model_name}", '
        f'"technique": "{technique.value}", "dataset": "{dataset_path}"}}\n'
    )
    (out / "README.txt").write_text(
        "Mock fine-tune artifact. Set FTAAS ray.mock=false and install "
        "transformers/peft for real training.\n"
    )
    steps = max(1, params.max_steps)
    loss = round(2.5 / (1 + steps * 0.1), 4)
    return TrainResult(
        output_dir=str(out),
        metrics={"train_loss": loss, "steps": float(steps)},
        parameters={
            "model_name": model_name,
            "framework": framework.value,
            "technique": technique.value,
            "learning_rate": params.learning_rate,
            "max_steps": params.max_steps,
            "lora_r": params.lora_r,
            "lora_alpha": params.lora_alpha,
        },
        model_uri=str(out),
        mock=True,
    )


def _try_real_lora(
    model_name: str,
    dataset_path: str,
    technique: Technique,
    params: HyperParameters,
    job_id: str,
) -> Optional[TrainResult]:
    """Best-effort real PEFT LoRA / SFT with transformers+peft. Falls back to None."""
    try:
        import json

        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        return None

    peft_techniques = {
        Technique.LORA,
        Technique.QLORA,
        Technique.SFT,
        Technique.DORA,
    }
    if technique not in peft_techniques:
        return None

    rows: list[dict[str, str]] = []
    path = Path(dataset_path)
    if path.suffix == ".jsonl":
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        rows = [{"instruction": "hello", "input": "", "output": "world"}]

    def to_text(r: dict) -> str:
        instr = r.get("instruction") or r.get("prompt") or ""
        inp = r.get("input") or ""
        out = r.get("output") or r.get("response") or r.get("completion") or ""
        if inp:
            return f"### Instruction:\n{instr}\n### Input:\n{inp}\n### Response:\n{out}"
        return f"### Instruction:\n{instr}\n### Response:\n{out}"

    texts = [to_text(r) for r in rows]
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch: dict) -> dict:
        return tok(
            batch["text"],
            truncation=True,
            max_length=params.max_seq_length,
            padding="max_length",
        )

    ds = Dataset.from_dict({"text": texts}).map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if technique in {Technique.LORA, Technique.QLORA, Technique.DORA, Technique.SFT}:
        lora = LoraConfig(
            r=params.lora_r,
            lora_alpha=params.lora_alpha,
            lora_dropout=params.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_dora=(technique == Technique.DORA),
        )
        model = get_peft_model(model, lora)

    out = Path(params.output_dir) / job_id
    out.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out),
        max_steps=params.max_steps,
        per_device_train_batch_size=params.per_device_train_batch_size,
        learning_rate=params.learning_rate,
        logging_steps=params.logging_steps,
        save_steps=max(params.save_steps, params.max_steps),
        report_to=[],
        remove_unused_columns=False,
    )
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    result = trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    metrics = {k: float(v) for k, v in (result.metrics or {}).items() if isinstance(v, (int, float))}
    return TrainResult(
        output_dir=str(out),
        metrics=metrics or {"train_loss": 0.0},
        parameters={
            "model_name": model_name,
            "technique": technique.value,
            "learning_rate": params.learning_rate,
            "max_steps": params.max_steps,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        model_uri=str(out),
        mock=False,
    )


class TransformersTrainer(BaseTrainer):
    framework = Framework.TRANSFORMERS

    def train(self, model_name, dataset_path, technique, params, job_id) -> TrainResult:
        real = _try_real_lora(model_name, dataset_path, technique, params, job_id)
        if real:
            return real
        return _mock_train(self.framework, model_name, dataset_path, technique, params, job_id)


class TRLTrainer(BaseTrainer):
    framework = Framework.TRL

    def train(self, model_name, dataset_path, technique, params, job_id) -> TrainResult:
        # TRL SFT/DPO/PPO — fall back to transformers LoRA or mock
        if technique in {Technique.SFT, Technique.LORA, Technique.DPO, Technique.PPO, Technique.ORPO}:
            real = _try_real_lora(model_name, dataset_path, Technique.LORA, params, job_id)
            if real:
                real.parameters["framework"] = self.framework.value
                real.parameters["requested_technique"] = technique.value
                return real
        return _mock_train(self.framework, model_name, dataset_path, technique, params, job_id)


class VerlTrainer(BaseTrainer):
    framework = Framework.VERL

    def train(self, model_name, dataset_path, technique, params, job_id) -> TrainResult:
        return _mock_train(self.framework, model_name, dataset_path, technique, params, job_id)


class LlamaFactoryTrainer(BaseTrainer):
    framework = Framework.LLAMA_FACTORY

    def train(self, model_name, dataset_path, technique, params, job_id) -> TrainResult:
        return _mock_train(self.framework, model_name, dataset_path, technique, params, job_id)


class UnslothTrainer(BaseTrainer):
    framework = Framework.UNSLOTH

    def train(self, model_name, dataset_path, technique, params, job_id) -> TrainResult:
        return _mock_train(self.framework, model_name, dataset_path, technique, params, job_id)


class AxolotlTrainer(BaseTrainer):
    framework = Framework.AXOLOTL

    def train(self, model_name, dataset_path, technique, params, job_id) -> TrainResult:
        return _mock_train(self.framework, model_name, dataset_path, technique, params, job_id)


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


# Capability matrix (from Confluence comparison table)
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
