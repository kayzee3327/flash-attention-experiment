"""Analytical FLOP, launch-parallelism, and traffic models for attention."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any


DTYPE_SIZES = {
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float32": 4,
    "fp32": 4,
}


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (numerator + denominator - 1) // denominator


def dtype_size(dtype: str | Any) -> int:
    name = str(dtype).replace("torch.", "").lower()
    if name not in DTYPE_SIZES:
        raise ValueError(f"unsupported dtype {dtype!r}; known values: {sorted(DTYPE_SIZES)}")
    return DTYPE_SIZES[name]


@dataclass(frozen=True)
class AttentionModel:
    qk_flops: int
    pv_flops: int
    total_matmul_flops: int
    softmax_elements: int
    matmul_flops_per_score: int
    num_q_tiles: int
    num_programs: int
    programs_per_sm: float | None
    q_read_bytes: int
    k_read_bytes: int
    v_read_bytes: int
    o_write_bytes: int
    score_write_bytes: int
    score_softmax_read_bytes: int
    probability_write_bytes: int
    probability_pv_read_bytes: int
    explicit_min_bytes: int
    explicit_intermediate_bytes: int
    kv_bytes_no_cache: int
    fa_no_cache_bytes: int
    explicit_arithmetic_intensity: float
    fa_no_cache_arithmetic_intensity: float

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def build_attention_model(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    block_m: int,
    input_dtype: str | Any = "float16",
    intermediate_dtype: str | Any | None = None,
    sm_count: int | None = None,
) -> AttentionModel:
    """Build simple minimum/upper-bound traffic models.

    Explicit traffic assumes Q/K/V are each read once, O is written once, and
    each full intermediate is written/read at the indicated producer-consumer
    boundary. The FA model reloads all K/V for every query tile and deliberately
    ignores cross-tile L2 reuse; it is an upper-bound approximation, not actual
    DRAM traffic.
    """

    for name, value in (
        ("batch_size", batch_size),
        ("num_heads", num_heads),
        ("seq_len", seq_len),
        ("head_dim", head_dim),
        ("block_m", block_m),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if sm_count is not None and sm_count <= 0:
        raise ValueError("sm_count must be positive when provided")

    input_element_size = dtype_size(input_dtype)
    intermediate_element_size = dtype_size(
        input_dtype if intermediate_dtype is None else intermediate_dtype
    )
    tensor_elements = batch_size * num_heads * seq_len * head_dim
    score_elements = batch_size * num_heads * seq_len * seq_len

    qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    pv_flops = qk_flops
    total_matmul_flops = qk_flops + pv_flops
    num_q_tiles = ceil_div(seq_len, block_m)
    num_programs = batch_size * num_heads * num_q_tiles

    q_read_bytes = tensor_elements * input_element_size
    k_read_bytes = tensor_elements * input_element_size
    v_read_bytes = tensor_elements * input_element_size
    o_write_bytes = tensor_elements * input_element_size
    intermediate_matrix_bytes = score_elements * intermediate_element_size
    score_write_bytes = intermediate_matrix_bytes
    score_softmax_read_bytes = intermediate_matrix_bytes
    probability_write_bytes = intermediate_matrix_bytes
    probability_pv_read_bytes = intermediate_matrix_bytes
    explicit_intermediate_bytes = 4 * intermediate_matrix_bytes
    explicit_min_bytes = (
        q_read_bytes
        + k_read_bytes
        + v_read_bytes
        + o_write_bytes
        + explicit_intermediate_bytes
    )

    kv_bytes_no_cache = (
        batch_size
        * num_heads
        * num_q_tiles
        * 2
        * seq_len
        * head_dim
        * input_element_size
    )
    fa_no_cache_bytes = q_read_bytes + o_write_bytes + kv_bytes_no_cache

    return AttentionModel(
        qk_flops=qk_flops,
        pv_flops=pv_flops,
        total_matmul_flops=total_matmul_flops,
        softmax_elements=score_elements,
        matmul_flops_per_score=4 * head_dim,
        num_q_tiles=num_q_tiles,
        num_programs=num_programs,
        programs_per_sm=(num_programs / sm_count if sm_count is not None else None),
        q_read_bytes=q_read_bytes,
        k_read_bytes=k_read_bytes,
        v_read_bytes=v_read_bytes,
        o_write_bytes=o_write_bytes,
        score_write_bytes=score_write_bytes,
        score_softmax_read_bytes=score_softmax_read_bytes,
        probability_write_bytes=probability_write_bytes,
        probability_pv_read_bytes=probability_pv_read_bytes,
        explicit_min_bytes=explicit_min_bytes,
        explicit_intermediate_bytes=explicit_intermediate_bytes,
        kv_bytes_no_cache=kv_bytes_no_cache,
        fa_no_cache_bytes=fa_no_cache_bytes,
        explicit_arithmetic_intensity=total_matmul_flops / explicit_min_bytes,
        fa_no_cache_arithmetic_intensity=total_matmul_flops / fa_no_cache_bytes,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    parser.add_argument("--intermediate-dtype", default=None)
    parser.add_argument("--sm-count", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model = build_attention_model(
        args.batch_size,
        args.num_heads,
        args.seq_len,
        args.head_dim,
        args.block_m,
        args.dtype,
        args.intermediate_dtype,
        args.sm_count,
    )
    print(json.dumps(model.as_dict(), indent=2))
