"""Shared experiment grids and Triton configuration generation."""

from __future__ import annotations

from itertools import product

from triton_flash_attention import TritonAttentionConfig

SEQUENCE_LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
HEAD_DIMS = (32, 64, 128)
DTYPES = ("float16", "bfloat16")
HEAD_COUNTS = (1, 2, 4, 8, 16, 32, 64)
TILE_SHAPES = (
    (1, 32, 512, 64),
    (1, 32, 4096, 64),
    (1, 32, 4096, 128),
)


def all_tile_configs() -> list[TritonAttentionConfig]:
    """Full requested tile sweep; compile/resource failures are recorded by caller."""

    return [
        TritonAttentionConfig(block_m, block_n, warps, stages)
        for block_m, block_n, warps, stages in product(
            (64, 128), (32, 64, 128), (4, 8), (2, 3, 4)
        )
    ]


def autotune_configs(head_dim: int) -> list[TritonAttentionConfig]:
    """Conservative per-head-dimension candidates for reproducible Python tuning."""

    raw: dict[int, tuple[tuple[int, int, int, int], ...]] = {
        32: (
            (64, 32, 4, 2),
            (64, 64, 4, 3),
            (64, 128, 4, 3),
            (128, 32, 4, 3),
            (128, 64, 4, 3),
            (128, 64, 8, 4),
            (128, 128, 8, 3),
        ),
        64: (
            (64, 32, 4, 2),
            (64, 64, 4, 3),
            (64, 128, 8, 3),
            (128, 32, 4, 3),
            (128, 64, 4, 3),
            (128, 64, 8, 3),
        ),
        128: (
            (64, 32, 4, 2),
            (64, 64, 4, 3),
            (64, 64, 8, 3),
            (128, 32, 4, 3),
            (128, 64, 8, 3),
        ),
        256: (
            (64, 32, 4, 2),
            (64, 32, 8, 3),
        ),
    }
    try:
        return [TritonAttentionConfig(*values) for values in raw[head_dim]]
    except KeyError as exc:
        raise ValueError("head_dim must be one of 32, 64, 128, or optional 256") from exc
