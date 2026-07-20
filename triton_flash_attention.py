"""Triton FlashAttention-2 style non-causal forward kernel.

The kernel intentionally implements only dense self-attention with layout
``[batch, heads, sequence, head_dim]``.  It never materializes the NxN score
or probability matrices in global memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # Keep the module importable enough to report a useful error.
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR: ImportError | None = exc
else:
    _TRITON_IMPORT_ERROR = None


SUPPORTED_HEAD_DIMS = (32, 64, 128, 256)


@dataclass(frozen=True)
class TritonAttentionConfig:
    """Launch/tile configuration recorded alongside every experiment."""

    block_m: int = 64
    block_n: int = 64
    num_warps: int = 4
    num_stages: int = 3

    def validate(self, head_dim: int) -> None:
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise ValueError(
                f"head_dim must be one of {SUPPORTED_HEAD_DIMS}, got {head_dim}"
            )
        if self.block_m not in (64, 128):
            raise ValueError(f"BLOCK_M must be 64 or 128, got {self.block_m}")
        if self.block_n not in (32, 64, 128):
            raise ValueError(f"BLOCK_N must be 32, 64, or 128, got {self.block_n}")
        if self.num_warps not in (4, 8):
            raise ValueError(f"num_warps must be 4 or 8, got {self.num_warps}")
        if self.num_stages not in (2, 3, 4):
            raise ValueError(f"num_stages must be 2, 3, or 4, got {self.num_stages}")

    def as_dict(self) -> dict[str, int]:
        return {
            "block_m": self.block_m,
            "block_n": self.block_n,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }


if triton is not None:

    @triton.jit
    def _flash_attention_forward(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_on,
        stride_od,
        sm_scale,
        NUM_HEADS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """One program computes one query tile for one (batch, head)."""

        query_tile_id = tl.program_id(0)
        batch_head_id = tl.program_id(1)
        batch_id = batch_head_id // NUM_HEADS
        head_id = batch_head_id % NUM_HEADS

        rows_m = query_tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, HEAD_DIM)

        # 1. Load this program's Q tile once and retain it on chip.
        q_base = q_ptr + batch_id * stride_qb + head_id * stride_qh
        q_offsets = rows_m[:, None] * stride_qn + dims[None, :] * stride_qd
        q = tl.load(q_base + q_offsets, mask=rows_m[:, None] < SEQ_LEN, other=0.0)

        # FP32 online-softmax state and FP32 output accumulator.
        row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_M,), tl.float32)
        output_acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

        k_base = k_ptr + batch_id * stride_kb + head_id * stride_kh
        v_base = v_ptr + batch_id * stride_vb + head_id * stride_vh

        # 2. Stream every K/V tile. No score/probability tile is stored to HBM.
        for start_n in tl.range(0, SEQ_LEN, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cols_n = start_n + tl.arange(0, BLOCK_N)

            # K is loaded transposed as [D, BLOCK_N] for QK^T.
            k_offsets = dims[:, None] * stride_kd + cols_n[None, :] * stride_kn
            k = tl.load(k_base + k_offsets, mask=cols_n[None, :] < SEQ_LEN, other=0.0)

            # 3. Compute the current on-chip QK^T tile with FP32 dot accumulation.
            scores = tl.dot(q, k) * sm_scale
            scores = tl.where(cols_n[None, :] < SEQ_LEN, scores, -float("inf"))

            # 4. Online softmax update for this K/V tile.
            tile_max = tl.max(scores, axis=1)
            new_row_max = tl.maximum(row_max, tile_max)
            old_scale = tl.exp(row_max - new_row_max)
            probabilities = tl.exp(scores - new_row_max[:, None])
            new_row_sum = row_sum * old_scale + tl.sum(probabilities, axis=1)

            # 5. Rescale history because the running row maximum may have changed.
            output_acc *= old_scale[:, None]

            # 6. Load V and accumulate P@V. tl.dot returns an FP32 accumulator.
            v_offsets = cols_n[:, None] * stride_vn + dims[None, :] * stride_vd
            v = tl.load(v_base + v_offsets, mask=cols_n[:, None] < SEQ_LEN, other=0.0)
            output_acc += tl.dot(probabilities.to(q.dtype), v)

            row_max = new_row_max
            row_sum = new_row_sum

        # 7. Normalize once, cast to the output pointer dtype, and write the tile.
        output_acc /= row_sum[:, None]
        o_base = o_ptr + batch_id * stride_ob + head_id * stride_oh
        o_offsets = rows_m[:, None] * stride_on + dims[None, :] * stride_od
        tl.store(o_base + o_offsets, output_acc, mask=rows_m[:, None] < SEQ_LEN)


def _require_triton() -> None:
    if triton is None:
        raise RuntimeError(
            "The triton_fa2 provider requires Triton, but importing triton failed: "
            f"{_TRITON_IMPORT_ERROR}"
        )


def triton_flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    config: TritonAttentionConfig | None = None,
) -> torch.Tensor:
    """Launch the non-causal Triton forward kernel and return ``O``."""

    _require_triton()
    if config is None:
        config = TritonAttentionConfig()
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, and v must have the same [B, H, N, D] shape")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("triton_fa2 requires CUDA tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("triton_fa2 supports only torch.float16 and torch.bfloat16")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must have the same dtype")
    if not (q.stride(-1) == k.stride(-1) == v.stride(-1) == 1):
        raise ValueError("q, k, and v must be contiguous in the head dimension")

    batch_size, num_heads, seq_len, head_dim = q.shape
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    config.validate(head_dim)

    output = torch.empty_like(q)
    grid = (triton.cdiv(seq_len, config.block_m), batch_size * num_heads)
    _flash_attention_forward[grid](
        q,
        k,
        v,
        output,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        sm_scale,
        NUM_HEADS=num_heads,
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return output


def triton_version() -> str:
    """Return a printable Triton version without forcing callers to import it."""

    if triton is None:
        return "unavailable"
    return str(getattr(triton, "__version__", "unknown"))
