"""Training device helpers — CPU vs CUDA selection."""

from __future__ import annotations

from typing import Any


def resolve_train_device(requested: str | None = None) -> str:
    """Return 'cuda' or 'cpu' based on FTAAS_TRAIN_DEVICE and torch availability."""
    import os

    import torch

    from ftaas.config import get_settings

    pref = (requested or get_settings().train_device or os.environ.get("FTAAS_TRAIN_DEVICE") or "auto").lower()
    has_cuda = bool(torch.cuda.is_available())
    if pref == "cuda":
        if not has_cuda:
            raise RuntimeError(
                "FTAAS_TRAIN_DEVICE=cuda but torch.cuda.is_available() is False. "
                "Install CUDA PyTorch (see Dockerfile.gpu / docs/gpu-training.md)."
            )
        return "cuda"
    if pref == "cpu":
        return "cpu"
    return "cuda" if has_cuda else "cpu"


def device_report() -> dict[str, Any]:
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        info: dict[str, Any] = {
            "torch": getattr(torch, "__version__", "?"),
            "cuda_available": cuda,
            "resolved_device": resolve_train_device("auto"),
        }
        if cuda:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        try:
            import bitsandbytes  # noqa: F401

            info["bitsandbytes"] = getattr(bitsandbytes, "__version__", "installed")
        except Exception:
            info["bitsandbytes"] = None
        return info
    except Exception as exc:
        return {"error": str(exc), "cuda_available": False, "resolved_device": "cpu"}


def lora_target_modules(model_name: str) -> list[str]:
    name = model_name.lower()
    if "gpt2" in name or "distilgpt" in name:
        return ["c_attn"]
    # Llama / Mistral / Qwen / Phi style
    return ["q_proj", "k_proj", "v_proj", "o_proj"]
