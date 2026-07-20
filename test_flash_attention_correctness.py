"""Correctness tests for the Triton FlashAttention forward implementation."""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass

import torch

from attention_providers import default_sm_scale, run_attention
from triton_flash_attention import TritonAttentionConfig


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


TOLERANCES = {
    torch.float16: Tolerance(atol=2e-2, rtol=2e-2),
    torch.bfloat16: Tolerance(atol=6e-2, rtol=6e-2),
}


def _parse_int_list(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def _parse_dtype_list(value: str) -> list[torch.dtype]:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        return [mapping[item.strip()] for item in value.split(",")]
    except KeyError as exc:
        raise argparse.ArgumentTypeError("dtypes must be float16 and/or bfloat16") from exc


def reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    """FP32 reference, including FP32 score, softmax, and PV calculation."""

    q32, k32, v32 = q.float(), k.float(), v.float()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        scores = torch.matmul(q32, k32.transpose(-1, -2)) * sm_scale
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, v32)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def run_case(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    config: TritonAttentionConfig,
    seed: int,
) -> tuple[bool, float, float, bool]:
    torch.manual_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    shape = (batch_size, num_heads, seq_len, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    scale = default_sm_scale(head_dim)

    with torch.inference_mode():
        expected = reference_attention(q, k, v, scale)
        actual = run_attention("triton_fa2", q, k, v, scale, config).float()
    torch.cuda.synchronize(device)

    finite = bool(torch.isfinite(actual).all().item())
    absolute_error = (actual - expected).abs()
    relative_error = absolute_error / expected.abs().clamp_min(1e-6)
    max_abs = float(absolute_error.max().item())
    max_rel = float(relative_error.max().item())
    tolerance = TOLERANCES[dtype]
    close = bool(torch.allclose(actual, expected, atol=tolerance.atol, rtol=tolerance.rtol))
    return close and finite, max_abs, max_rel, finite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run a small smoke subset")
    parser.add_argument("--batch-sizes", type=_parse_int_list, default=[1, 2])
    parser.add_argument("--num-heads", type=_parse_int_list, default=[1, 4, 8])
    parser.add_argument("--seq-lens", type=_parse_int_list, default=[64, 127, 128, 257, 512])
    parser.add_argument("--head-dims", type=_parse_int_list, default=[32, 64, 128])
    parser.add_argument(
        "--dtypes", type=_parse_dtype_list, default=[torch.float16, torch.bfloat16]
    )
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for Triton correctness tests", file=sys.stderr)
        return 2
    config = TritonAttentionConfig(
        args.block_m, args.block_n, args.num_warps, args.num_stages
    )

    if args.quick:
        # Includes a non-tile-multiple length and both primary numerical formats.
        cases = [
            (1, 1, 127, 32, torch.float16),
            (1, 4, 128, 64, torch.float16),
            (1, 1, 257, 64, torch.bfloat16),
            (1, 4, 64, 128, torch.bfloat16),
        ]
    else:
        cases = list(
            itertools.product(
                args.batch_sizes,
                args.num_heads,
                args.seq_lens,
                args.head_dims,
                args.dtypes,
            )
        )

    failures = 0
    for case_index, (batch_size, num_heads, seq_len, head_dim, dtype) in enumerate(cases):
        shape = (batch_size, num_heads, seq_len, head_dim)
        try:
            passed, max_abs, max_rel, finite = run_case(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                dtype,
                config,
                args.seed + case_index,
            )
        except Exception as exc:  # A case failure must not hide the remaining coverage.
            failures += 1
            print(f"FAIL shape={shape} dtype={dtype}: {type(exc).__name__}: {exc}")
            continue
        status = "PASS" if passed else "FAIL"
        print(
            f"{status} shape={shape} dtype={dtype} max_abs={max_abs:.6g} "
            f"max_rel={max_rel:.6g} finite={finite}"
        )
        failures += int(not passed)
        del shape
        torch.cuda.empty_cache()

    print(f"Completed {len(cases)} cases; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
