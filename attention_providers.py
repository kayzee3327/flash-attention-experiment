"""Unified provider interface for the attention experiment."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

from triton_flash_attention import (
    TritonAttentionConfig,
    triton_flash_attention_forward,
)

ProviderName = Literal["torch_explicit", "torch_sdpa", "triton_fa2"]
PROVIDERS: tuple[ProviderName, ...] = (
    "torch_explicit",
    "torch_sdpa",
    "triton_fa2",
)


def default_sm_scale(head_dim: int) -> float:
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    return 1.0 / math.sqrt(head_dim)


def run_attention(
    provider: ProviderName | str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
    config: TritonAttentionConfig | None = None,
) -> torch.Tensor:
    """Run one provider without changing its mathematical definition."""

    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {PROVIDERS}")
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, and v must share shape [B, H, N, D]")
    if sm_scale is None:
        sm_scale = default_sm_scale(q.shape[-1])

    if provider == "torch_explicit":
        # Deliberately materialize both full [B, H, N, N] intermediates.
        scores = torch.matmul(q, k.transpose(-1, -2))
        probabilities = torch.softmax(scores * sm_scale, dim=-1)
        return torch.matmul(probabilities, v)

    if provider == "torch_sdpa":
        if not hasattr(F, "scaled_dot_product_attention"):
            raise RuntimeError(
                "torch_sdpa is unavailable: this PyTorch build has no "
                "scaled_dot_product_attention"
            )
        try:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
                scale=sm_scale,
            )
        except (RuntimeError, NotImplementedError, TypeError) as exc:
            raise RuntimeError(
                "torch_sdpa could not run for this shape/device. Its backend is "
                f"selected by PyTorch and the current device: {exc}"
            ) from exc

    return triton_flash_attention_forward(q, k, v, sm_scale, config)

