"""Benchmark and parameter-sweep driver for FlashAttention forward providers.

This script performs GPU work when executed. Importing it does not run any
benchmark. Use ``--single-kernel`` only as the target of an external profiler.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

from attention_models import AttentionModel, build_attention_model
from attention_providers import PROVIDERS, default_sm_scale, run_attention
from experiment_configs import (
    DTYPES,
    HEAD_COUNTS,
    HEAD_DIMS,
    SEQUENCE_LENGTHS,
    TILE_SHAPES,
    all_tile_configs,
    autotune_configs,
)
from triton_flash_attention import (
    TritonAttentionConfig,
    triton_version,
)


CSV_FIELDS = [
    "experiment",
    "provider",
    "status",
    "error_message",
    "device_name",
    "sm_count",
    "torch_version",
    "triton_version",
    "dtype",
    "batch_size",
    "num_heads",
    "seq_len",
    "head_dim",
    "causal",
    "block_m",
    "block_n",
    "num_warps",
    "num_stages",
    "config_mode",
    "warmup",
    "repetitions",
    "latency_ms",
    "latency_p20_ms",
    "latency_p80_ms",
    "effective_tflops",
    "qk_flops",
    "pv_flops",
    "total_matmul_flops",
    "softmax_elements",
    "matmul_flops_per_score",
    "num_q_tiles",
    "num_programs",
    "programs_per_sm",
    "explicit_modeled_bytes",
    "explicit_intermediate_bytes",
    "fa_no_cache_modeled_bytes",
    "modeled_arithmetic_intensity",
    "memory_allocated_before_bytes",
    "peak_memory_allocated_bytes",
    "peak_memory_reserved_bytes",
]


@dataclass(frozen=True)
class Shape:
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    dtype_name: str


@dataclass(frozen=True)
class Measurement:
    median_ms: float
    p20_ms: float
    p80_ms: float


@dataclass(frozen=True)
class MemoryMeasurement:
    allocated_before: int
    peak_allocated: int
    peak_reserved: int


def parse_int_list(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all list values must be positive")
    return parsed


def torch_dtype(name: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}") from exc


def fixed_config(args: argparse.Namespace) -> TritonAttentionConfig:
    return TritonAttentionConfig(
        block_m=args.block_m,
        block_n=args.block_n,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )


def make_inputs(shape: Shape, device: torch.device, seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    full_shape = (shape.batch_size, shape.num_heads, shape.seq_len, shape.head_dim)
    dtype = torch_dtype(shape.dtype_name)
    return tuple(torch.randn(full_shape, device=device, dtype=dtype) for _ in range(3))


def deterministic_shape_seed(shape: Shape, base_seed: int) -> int:
    """Use identical inputs for all providers/configs of the same shape."""

    dtype_offset = 0 if shape.dtype_name == "float16" else 1
    return (
        base_seed
        + shape.batch_size * 1_000_003
        + shape.num_heads * 10_007
        + shape.seq_len * 101
        + shape.head_dim * 3
        + dtype_offset
    )


def time_callable(
    function: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    repetitions: int,
    quantile: float,
) -> Measurement:
    """Compile first, warm up, then time complete provider invocations with events."""

    # First invocation is explicitly excluded so Triton JIT/lazy library setup is untimed.
    output = function()
    del output
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        output = function()
        del output
    torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        del output

    values = torch.tensor(samples, dtype=torch.float64)
    return Measurement(
        median_ms=float(torch.quantile(values, quantile).item()),
        p20_ms=float(torch.quantile(values, 0.2).item()),
        p80_ms=float(torch.quantile(values, 0.8).item()),
    )


def measure_memory(
    function: Callable[[], torch.Tensor], device: torch.device
) -> MemoryMeasurement:
    """Run separately from latency timing so memory-stat APIs do not perturb timing."""

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    output = function()
    torch.cuda.synchronize(device)
    result = MemoryMeasurement(
        allocated_before=allocated_before,
        peak_allocated=torch.cuda.max_memory_allocated(device),
        peak_reserved=torch.cuda.max_memory_reserved(device),
    )
    del output
    return result


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def classify_error(exc: BaseException) -> str:
    if is_oom(exc):
        return "oom"
    name = f"{type(exc).__module__}.{type(exc).__name__}".lower()
    text = str(exc).lower()
    if "compile" in name or "compilation" in text or "outofresources" in name:
        return "compile_error"
    return "runtime_error"


def clean_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_for(shape: Shape, config: TritonAttentionConfig, sm_count: int) -> AttentionModel:
    return build_attention_model(
        shape.batch_size,
        shape.num_heads,
        shape.seq_len,
        shape.head_dim,
        config.block_m,
        shape.dtype_name,
        shape.dtype_name,
        sm_count,
    )


def base_row(
    experiment: str,
    provider: str,
    shape: Shape,
    config: TritonAttentionConfig,
    config_mode: str,
    args: argparse.Namespace,
    device_name: str,
    sm_count: int,
) -> dict[str, object]:
    model = model_for(shape, config, sm_count)
    if provider == "torch_explicit":
        modeled_intensity: float | str = model.explicit_arithmetic_intensity
    elif provider == "triton_fa2":
        modeled_intensity = model.fa_no_cache_arithmetic_intensity
    else:
        # SDPA's actual backend is unknown, so choosing either traffic model would mislabel it.
        modeled_intensity = ""
    return {
        "experiment": experiment,
        "provider": provider,
        "status": "",
        "error_message": "",
        "device_name": device_name,
        "sm_count": sm_count,
        "torch_version": torch.__version__,
        "triton_version": triton_version(),
        "dtype": shape.dtype_name,
        "batch_size": shape.batch_size,
        "num_heads": shape.num_heads,
        "seq_len": shape.seq_len,
        "head_dim": shape.head_dim,
        "causal": False,
        **config.as_dict(),
        "config_mode": config_mode,
        "warmup": args.warmup,
        "repetitions": args.repeat,
        "latency_ms": "",
        "latency_p20_ms": "",
        "latency_p80_ms": "",
        "effective_tflops": "",
        "qk_flops": model.qk_flops,
        "pv_flops": model.pv_flops,
        "total_matmul_flops": model.total_matmul_flops,
        "softmax_elements": model.softmax_elements,
        "matmul_flops_per_score": model.matmul_flops_per_score,
        "num_q_tiles": model.num_q_tiles,
        "num_programs": model.num_programs,
        "programs_per_sm": model.programs_per_sm,
        "explicit_modeled_bytes": model.explicit_min_bytes,
        "explicit_intermediate_bytes": model.explicit_intermediate_bytes,
        "fa_no_cache_modeled_bytes": model.fa_no_cache_bytes,
        "modeled_arithmetic_intensity": modeled_intensity,
        "memory_allocated_before_bytes": "",
        "peak_memory_allocated_bytes": "",
        "peak_memory_reserved_bytes": "",
    }


def run_one_config(
    provider: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    config: TritonAttentionConfig,
    args: argparse.Namespace,
    device: torch.device,
    measure_memory_now: bool = True,
) -> tuple[Measurement, MemoryMeasurement | None]:
    function = lambda: run_attention(provider, q, k, v, scale, config)
    measurement = time_callable(function, device, args.warmup, args.repeat, args.quantile)
    memory = measure_memory(function, device) if args.measure_memory and measure_memory_now else None
    return measurement, memory


def tune_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    head_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TritonAttentionConfig, Measurement, MemoryMeasurement | None]:
    best: tuple[TritonAttentionConfig, Measurement] | None = None
    failures: list[str] = []
    for candidate in autotune_configs(head_dim):
        try:
            measurement, _ = run_one_config(
                "triton_fa2", q, k, v, scale, candidate, args, device, False
            )
        except Exception as exc:
            failures.append(f"{candidate}: {type(exc).__name__}: {exc}")
            clean_cuda()
            continue
        if best is None or measurement.median_ms < best[1].median_ms:
            best = (candidate, measurement)
    if best is None:
        summary = "; ".join(failures[:4])
        raise RuntimeError(f"all Triton tuning candidates failed: {summary}")

    best_config, best_measurement = best
    memory = None
    if args.measure_memory:
        function = lambda: run_attention("triton_fa2", q, k, v, scale, best_config)
        memory = measure_memory(function, device)
    return best_config, best_measurement, memory


def run_case(
    experiment: str,
    provider: str,
    shape: Shape,
    requested_config: TritonAttentionConfig,
    args: argparse.Namespace,
    device: torch.device,
    device_name: str,
    sm_count: int,
) -> dict[str, object]:
    config = requested_config
    row = base_row(
        experiment,
        provider,
        shape,
        config,
        args.config_mode if provider == "triton_fa2" else "n/a",
        args,
        device_name,
        sm_count,
    )
    tensors: tuple[torch.Tensor, ...] | None = None
    try:
        tensors = make_inputs(shape, device, deterministic_shape_seed(shape, args.seed))
        q, k, v = tensors
        scale = default_sm_scale(shape.head_dim)
        with torch.inference_mode():
            if args.single_kernel:
                if provider == "triton_fa2" and args.config_mode != "fixed":
                    raise ValueError("--single-kernel requires --config-mode fixed")
                output = run_attention(provider, q, k, v, scale, config)
                torch.cuda.synchronize(device)
                del output
                measurement = None
                memory = None
            elif provider == "triton_fa2" and args.config_mode == "tuned":
                config, measurement, memory = tune_triton(
                    q, k, v, scale, shape.head_dim, args, device
                )
                row = base_row(
                    experiment,
                    provider,
                    shape,
                    config,
                    "tuned",
                    args,
                    device_name,
                    sm_count,
                )
            else:
                measurement, memory = run_one_config(
                    provider, q, k, v, scale, config, args, device
                )

        row["status"] = "ok"
        if measurement is not None:
            row["latency_ms"] = measurement.median_ms
            row["latency_p20_ms"] = measurement.p20_ms
            row["latency_p80_ms"] = measurement.p80_ms
            flops = int(row["total_matmul_flops"])
            row["effective_tflops"] = flops / (measurement.median_ms / 1e3) / 1e12
        if memory is not None:
            row["memory_allocated_before_bytes"] = memory.allocated_before
            row["peak_memory_allocated_bytes"] = memory.peak_allocated
            row["peak_memory_reserved_bytes"] = memory.peak_reserved
    except Exception as exc:
        row["status"] = classify_error(exc)
        row["error_message"] = f"{type(exc).__name__}: {exc}"
    finally:
        if tensors is not None:
            del tensors
        clean_cuda()
    return row


def build_cases(args: argparse.Namespace) -> list[tuple[str, Shape]]:
    dtypes = args.dtypes
    if args.experiment == "single":
        provider = args.provider or args.providers[0]
        del provider  # Provider selection is handled separately; validate shape here.
        return [
            (
                "single",
                Shape(
                    args.batch_size,
                    args.num_heads if args.num_heads is not None else 32,
                    args.seq_len if args.seq_len is not None else 4096,
                    args.head_dim if args.head_dim is not None else 64,
                    dtypes[0],
                ),
            )
        ]
    if args.experiment == "sequence":
        sequence_values = args.seq_lens or (
            [args.seq_len] if args.seq_len is not None else list(SEQUENCE_LENGTHS)
        )
        dimension_values = args.head_dims or (
            [args.head_dim] if args.head_dim is not None else [64, 128]
        )
        return [
            (
                "sequence",
                Shape(
                    args.batch_size,
                    args.num_heads if args.num_heads is not None else 32,
                    seq_len,
                    head_dim,
                    dtype,
                ),
            )
            for dtype in dtypes
            for head_dim in dimension_values
            for seq_len in sequence_values
        ]
    if args.experiment == "head_dim":
        dimension_values = args.head_dims or (
            [args.head_dim] if args.head_dim is not None else list(HEAD_DIMS)
        )
        return [
            (
                "head_dim",
                Shape(
                    args.batch_size,
                    args.num_heads if args.num_heads is not None else 32,
                    args.seq_len if args.seq_len is not None else 4096,
                    head_dim,
                    dtype,
                ),
            )
            for dtype in dtypes
            for head_dim in dimension_values
        ]
    if args.experiment == "parallelism":
        head_values = args.head_counts or (
            [args.num_heads] if args.num_heads is not None else list(HEAD_COUNTS)
        )
        return [
            (
                "parallelism",
                Shape(
                    args.batch_size,
                    num_heads,
                    args.seq_len if args.seq_len is not None else 512,
                    args.head_dim if args.head_dim is not None else 64,
                    dtype,
                ),
            )
            for dtype in dtypes
            for num_heads in head_values
        ]
    if args.experiment == "tile":
        return [
            ("tile", Shape(batch, heads, seq_len, head_dim, dtype))
            for dtype in dtypes
            for batch, heads, seq_len, head_dim in TILE_SHAPES
        ]
    raise AssertionError(f"unhandled experiment {args.experiment}")


def select_providers(args: argparse.Namespace) -> list[str]:
    if args.provider:
        return [args.provider]
    if args.experiment == "tile":
        return ["triton_fa2"]
    return list(args.providers)


def save_results(
    rows: Sequence[dict[str, object]], args: argparse.Namespace
) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"{args.experiment}_{timestamp}"
    csv_path = output_dir / f"{stem}.csv"
    config_path = output_dir / f"{stem}.json"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "timestamp": timestamp,
        "argv": sys.argv,
        "arguments": vars(args),
        "notes": {
            "single_kernel": "latency fields are intentionally blank in profiler mode",
            "peak_reserved": "allocator reserved memory is not tensor memory",
        },
    }
    with config_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return csv_path, config_path


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    for name in ("num_heads", "seq_len", "head_dim", "sm_count"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.quantile <= 1.0:
        raise ValueError("--quantile must be in [0, 1]")
    if args.single_kernel and args.experiment != "single":
        raise ValueError("--single-kernel is only valid with --experiment single")
    if args.experiment == "single" and len(args.dtypes) != 1:
        raise ValueError("single experiment accepts exactly one dtype")
    if args.experiment == "single" and args.provider is None and len(args.providers) != 1:
        raise ValueError("single experiment requires --provider or one --providers value")
    config = fixed_config(args)
    head_dims = args.head_dims or ([args.head_dim] if args.head_dim else list(HEAD_DIMS))
    for head_dim in head_dims:
        if head_dim is not None:
            config.validate(head_dim)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("single", "sequence", "head_dim", "parallelism", "tile"),
        default="sequence",
    )
    parser.add_argument("--provider", choices=PROVIDERS, default=None)
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, default=list(PROVIDERS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--seq-lens", type=parse_int_list, default=None)
    parser.add_argument("--head-dims", type=parse_int_list, default=None)
    parser.add_argument("--head-counts", type=parse_int_list, default=None)
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--dtype", choices=DTYPES, default=None, help="single dtype alias")
    parser.add_argument("--config-mode", choices=("fixed", "tuned"), default="fixed")
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.5,
        help="primary latency quantile (default 0.5/median); p20 and p80 are always recorded",
    )
    parser.add_argument("--measure-memory", action="store_true")
    parser.add_argument("--single-kernel", action="store_true")
    parser.add_argument("--sm-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dtype is not None:
        args.dtypes = [args.dtype]
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"argument error: {exc}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("CUDA is required to run this benchmark", file=sys.stderr)
        return 2

    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    sm_count = args.sm_count or int(properties.multi_processor_count)
    device_name = str(properties.name)
    requested_config = fixed_config(args)
    providers = select_providers(args)
    rows: list[dict[str, object]] = []
    for experiment, shape in build_cases(args):
        configs: Iterable[TritonAttentionConfig]
        if experiment == "tile":
            configs = all_tile_configs()
        else:
            configs = (requested_config,)
        for provider in providers:
            for config in configs:
                # A tile sweep records each compile/runtime failure independently.
                original_mode = args.config_mode
                if experiment == "tile":
                    args.config_mode = "fixed"
                row = run_case(
                    experiment,
                    provider,
                    shape,
                    config,
                    args,
                    device,
                    device_name,
                    sm_count,
                )
                args.config_mode = original_mode
                rows.append(row)
                print(
                    f"{experiment} provider={provider} shape="
                    f"({shape.batch_size},{shape.num_heads},{shape.seq_len},{shape.head_dim}) "
                    f"dtype={shape.dtype_name} config={config} status={row['status']}"
                )

    csv_path, config_path = save_results(rows, args)
    print(f"Wrote {csv_path}")
    print(f"Wrote {config_path}")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
