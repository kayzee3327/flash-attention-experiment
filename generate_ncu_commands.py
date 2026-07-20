"""Generate (but never execute) Nsight Compute commands for profile points."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path


DEFAULT_POINTS = ((512, 64), (2048, 64), (8192, 64), (512, 128), (2048, 128), (8192, 128))


def quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def benchmark_command(args: argparse.Namespace, seq_len: int, head_dim: int) -> list[str]:
    return [
        args.python,
        args.benchmark_script,
        "--experiment",
        "single",
        "--provider",
        "triton_fa2",
        "--batch-size",
        str(args.batch_size),
        "--num-heads",
        str(args.num_heads),
        "--seq-len",
        str(seq_len),
        "--head-dim",
        str(head_dim),
        "--dtype",
        args.dtype,
        "--config-mode",
        "fixed",
        "--block-m",
        str(args.block_m),
        "--block-n",
        str(args.block_n),
        "--num-warps",
        str(args.num_warps),
        "--num-stages",
        str(args.num_stages),
        "--single-kernel",
        "--output-dir",
        args.benchmark_output_dir,
    ]


def generate_commands(args: argparse.Namespace) -> list[str]:
    output_dir = Path(args.output_dir)
    commands = [f"mkdir -p {shlex.quote(str(output_dir))}"]
    for seq_len, head_dim in DEFAULT_POINTS:
        stem = (
            f"fa2_b{args.batch_size}_h{args.num_heads}_n{seq_len}_d{head_dim}_{args.dtype}_"
            f"bm{args.block_m}_bn{args.block_n}_w{args.num_warps}_s{args.num_stages}"
        )
        common = [
            args.ncu,
            "--target-processes",
            "all",
            "--kernel-name",
            f"regex:{args.kernel_regex}",
            "--set",
            args.metric_set,
        ]
        target = benchmark_command(args, seq_len, head_dim)
        report_command = common + ["--export", str(output_dir / f"{stem}.ncu-rep")] + target
        commands.append(quote_command(report_command))
        if args.emit_csv:
            csv_command = common + [
                "--csv",
                "--page",
                "raw",
                "--log-file",
                str(output_dir / f"{stem}.csv"),
            ] + target
            commands.append(quote_command(csv_command))
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="python")
    parser.add_argument("--benchmark-script", default="benchmark_flash_attention.py")
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--output-dir", default="ncu_reports")
    parser.add_argument("--benchmark-output-dir", default="results/ncu_inputs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--block-m", type=int, choices=(64, 128), default=64)
    parser.add_argument("--block-n", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--num-warps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--num-stages", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument(
        "--kernel-regex",
        default=".*_flash_attention_forward.*",
        help="NCU regex used to exclude input-generation and framework kernels",
    )
    parser.add_argument("--metric-set", default="full")
    parser.add_argument("--emit-csv", action="store_true")
    parser.add_argument("--commands-file", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("batch_size", "num_heads", "block_m", "block_n", "num_warps", "num_stages"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    commands = generate_commands(args)
    text = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n\n".join(commands) + "\n"
    if args.commands_file:
        path = Path(args.commands_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Wrote {path}; review it, then run it manually.")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
