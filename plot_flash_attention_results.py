"""Create analysis plots from existing benchmark/NCU CSV files only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


NUMERIC_COLUMNS = [
    "seq_len",
    "head_dim",
    "num_programs",
    "programs_per_sm",
    "latency_ms",
    "effective_tflops",
    "peak_memory_allocated_bytes",
    "registers_per_thread",
    "achieved_occupancy",
    "explicit_modeled_bytes",
    "fa_no_cache_modeled_bytes",
    "dram_bytes_read",
    "dram_bytes_written",
    "l2_hit_rate",
]


def prepare_frame(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True, sort=False)
    for column in NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "status" in frame:
        frame = frame[(frame["status"].isna()) | (frame["status"] == "ok")]
    return frame


def save_line_groups(
    data: pd.DataFrame,
    x: str,
    y: str,
    group_columns: list[str],
    output: Path,
    xlabel: str,
    ylabel: str,
    log_x: bool = False,
) -> bool:
    required = {x, y, *group_columns}
    if not required.issubset(data.columns):
        return False
    subset = data.dropna(subset=[x, y])
    if subset.empty:
        return False
    fig, axis = plt.subplots(figsize=(8, 5))
    group_key = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in subset.groupby(group_key, dropna=False):
        labels = keys if isinstance(keys, tuple) else (keys,)
        label = ", ".join(f"{name}={value}" for name, value in zip(group_columns, labels))
        ordered = group.sort_values(x)
        axis.plot(ordered[x], ordered[y], marker="o", label=label)
    if log_x:
        axis.set_xscale("log", base=2)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True


def sequence_plots(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    data = frame[frame.get("experiment", pd.Series(index=frame.index, dtype=str)) == "sequence"]
    outputs: list[Path] = []
    specs = [
        ("effective_tflops", ["provider", "head_dim", "dtype"], "Effective TFLOP/s", "sequence_effective_tflops.png"),
        ("latency_ms", ["provider", "head_dim", "dtype"], "Latency (ms)", "sequence_latency.png"),
    ]
    for y, groups, ylabel, name in specs:
        path = output_dir / name
        if save_line_groups(data, "seq_len", y, groups, path, "Sequence Length", ylabel, True):
            outputs.append(path)
    return outputs


def memory_plot(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    if "provider" not in frame:
        return []
    data = frame[frame["provider"].isin(["torch_explicit", "triton_fa2"])]
    path = output_dir / "sequence_peak_memory.png"
    made = save_line_groups(
        data,
        "seq_len",
        "peak_memory_allocated_bytes",
        ["provider", "head_dim", "dtype"],
        path,
        "Sequence Length",
        "Peak allocated memory (bytes)",
        True,
    )
    return [path] if made else []


def resource_plots(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for field, ylabel in (
        ("registers_per_thread", "Registers per Thread"),
        ("achieved_occupancy", "Achieved Occupancy (%)"),
        ("effective_tflops", "Effective TFLOP/s"),
    ):
        path = output_dir / f"head_dim_{field}.png"
        if save_line_groups(
            frame,
            "head_dim",
            field,
            ["provider", "seq_len"],
            path,
            "Head Dimension",
            ylabel,
        ):
            outputs.append(path)
    return outputs


def parallelism_plot(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    if "experiment" not in frame:
        return []
    data = frame[frame["experiment"] == "parallelism"]
    outputs: list[Path] = []
    for x, name in (("num_programs", "parallelism_num_programs.png"), ("programs_per_sm", "parallelism_programs_per_sm.png")):
        path = output_dir / name
        if save_line_groups(data, x, "effective_tflops", ["provider", "dtype"], path, x.replace("_", " ").title(), "Effective TFLOP/s"):
            outputs.append(path)
    return outputs


def traffic_plots(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    needed = {"seq_len", "explicit_modeled_bytes", "fa_no_cache_modeled_bytes", "dram_bytes_read", "dram_bytes_written"}
    if not needed.issubset(frame.columns):
        return []
    data = frame.dropna(subset=list(needed)).copy()
    if data.empty:
        return []
    data["ncu_measured_dram_bytes"] = data["dram_bytes_read"] + data["dram_bytes_written"]
    long = data.melt(
        id_vars=["seq_len", "head_dim", "dtype"],
        value_vars=["explicit_modeled_bytes", "fa_no_cache_modeled_bytes", "ncu_measured_dram_bytes"],
        var_name="traffic_source",
        value_name="bytes",
    )
    outputs: list[Path] = []
    traffic_path = output_dir / "modeled_vs_measured_traffic.png"
    if save_line_groups(long, "seq_len", "bytes", ["traffic_source", "head_dim", "dtype"], traffic_path, "Sequence Length", "Bytes", True):
        outputs.append(traffic_path)
    l2_path = output_dir / "sequence_l2_hit_rate.png"
    if save_line_groups(data, "seq_len", "l2_hit_rate", ["head_dim", "dtype"], l2_path, "Sequence Length", "L2 Hit Rate (%)", True):
        outputs.append(l2_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="benchmark CSV and/or merged NCU CSV")
    parser.add_argument("--output-dir", default="plots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = prepare_frame(args.inputs)
    outputs = []
    outputs.extend(sequence_plots(frame, output_dir))
    outputs.extend(memory_plot(frame, output_dir))
    outputs.extend(resource_plots(frame, output_dir))
    outputs.extend(parallelism_plot(frame, output_dir))
    outputs.extend(traffic_plots(frame, output_dir))
    if outputs:
        for path in outputs:
            print(f"Wrote {path}")
    else:
        print("No applicable non-empty plots were found; missing optional columns were skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
