"""Unit tests for device selection and QLoRA gating."""

from __future__ import annotations

import os

import pytest

from ftaas.config import get_settings
from ftaas.models import Technique
from training.device import lora_target_modules, resolve_train_device
from training.frameworks.registry import _want_qlora


def test_want_qlora():
    assert _want_qlora(Technique.QLORA)
    assert not _want_qlora(Technique.LORA)


def test_lora_target_modules():
    assert lora_target_modules("gpt2") == ["c_attn"]
    assert lora_target_modules("distilgpt2") == ["c_attn"]
    mods = lora_target_modules("mistralai/Mistral-7B-Instruct-v0.2")
    assert "q_proj" in mods and "v_proj" in mods


def test_resolve_device_auto():
    get_settings.cache_clear()
    d = resolve_train_device("auto")
    assert d in {"cpu", "cuda"}


def test_qlora_requires_cuda_without_fallback(monkeypatch):
    monkeypatch.setenv("FTAAS_ALLOW_QLORA_CPU_FALLBACK", "false")
    monkeypatch.setenv("FTAAS_TRAIN_DEVICE", "cpu")
    get_settings.cache_clear()
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from training.frameworks.registry import _load_causal_lm_for_train

    with pytest.raises(RuntimeError, match="QLoRA.*(CUDA|bitsandbytes)"):
        _load_causal_lm_for_train(
            torch,
            transformers.AutoModelForCausalLM,
            "distilgpt2",
            Technique.QLORA,
        )


def test_remote_trainer_flag(monkeypatch):
    from training.remote.client import remote_trainer_enabled

    monkeypatch.setenv("FTAAS_TRAIN_MODE", "local")
    monkeypatch.setenv("FTAAS_TRAINER_URL", "")
    get_settings.cache_clear()
    assert not remote_trainer_enabled()

    monkeypatch.setenv("FTAAS_TRAIN_MODE", "remote")
    monkeypatch.setenv("FTAAS_TRAINER_URL", "http://gpu:8090")
    get_settings.cache_clear()
    assert remote_trainer_enabled()
