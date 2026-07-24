"""Unit tests — config, models, real trainer, capability matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_ensure_data_dirs(data_dir):
    from ftaas.config import ensure_data_dirs, sqlite_url

    root = ensure_data_dirs()
    assert root == data_dir.resolve()
    assert (root / "datasets").is_dir()
    assert (root / "outputs").is_dir()
    url = sqlite_url("registry")
    assert str(root / "registry.db") in url.replace("\\\\", "/")


def test_hyperparameters_defaults():
    from ftaas.models import HyperParameters

    hp = HyperParameters()
    assert hp.max_steps == 10
    assert hp.lora_r == 8
    assert hp.learning_rate == 2e-4


def test_framework_technique_matrix():
    from training.frameworks.registry import FRAMEWORK_TECHNIQUE_SUPPORT, get_trainer
    from ftaas.models import Framework

    assert "lora" in FRAMEWORK_TECHNIQUE_SUPPORT["transformers"]
    assert get_trainer(Framework.TRANSFORMERS).framework == Framework.TRANSFORMERS
    assert get_trainer("trl").framework == Framework.TRL


def test_missing_dataset_raises(tmp_path):
    from ftaas.models import HyperParameters, Technique
    from training.frameworks.registry import get_trainer

    trainer = get_trainer("transformers")
    params = HyperParameters(max_steps=1, output_dir=str(tmp_path / "out"))
    with pytest.raises(FileNotFoundError):
        trainer.train(
            "sshleifer/tiny-gpt2",
            str(tmp_path / "nope.jsonl"),
            Technique.LORA,
            params,
            "job_missing_ds",
        )


def _has_train_deps() -> bool:
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from datasets import Dataset  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _has_train_deps(), reason="train extras not installed")
def test_real_train_writes_statistics(sample_dataset, tmp_path):
    from ftaas.models import HyperParameters, Technique
    from training.frameworks.registry import get_trainer

    trainer = get_trainer("transformers")
    params = HyperParameters(
        max_steps=2,
        per_device_train_batch_size=1,
        logging_steps=1,
        max_seq_length=64,
        lora_r=4,
        output_dir=str(tmp_path / "out"),
    )
    result = trainer.train(
        "sshleifer/tiny-gpt2",
        str(sample_dataset),
        Technique.LORA,
        params,
        "job_unit_real",
    )
    assert result.statistics["mode"] == "real"
    assert "train_loss" in result.metrics or result.statistics.get("final_train_loss") is not None
    stats_path = Path(result.output_dir) / "statistics.json"
    assert stats_path.exists()
    payload = json.loads(stats_path.read_text())
    assert payload["mode"] == "real"
    assert payload["trainable_params"] > 0
