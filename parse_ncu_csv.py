"""Normalize Nsight Compute raw CSV metrics and optionally merge benchmark CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


# Centralized aliases: extend this dictionary (or pass --metric-map JSON) for a
# CUDA/NCU release whose raw metric spelling differs.
METRIC_NAME_MAP: dict[str, list[str]] = {
    "dram_bytes_read": ["dram__bytes_read.sum", "dram__bytes_read"],
    "dram_bytes_written": ["dram__bytes_write.sum", "dram__bytes_written.sum"],
    "dram_throughput_pct": [
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "l2_bytes_read": ["lts__t_bytes_op_read.sum", "lts__t_sectors_op_read.sum"],
    "l2_bytes_written": ["lts__t_bytes_op_write.sum", "lts__t_sectors_op_write.sum"],
    "l2_hit_rate": ["lts__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct_of_peak_sustained_elapsed"],
    "registers_per_thread": ["launch__registers_per_thread"],
    "shared_memory_per_block": [
        "launch__shared_mem_per_block_allocated",
        "launch__shared_mem_per_block_dynamic",
    ],
    "achieved_occupancy": [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
    ],
    "waves_per_sm": ["sm__waves_per_active_sm", "launch__waves_per_multiprocessor"],
    "active_warps_per_sm": ["sm__warps_active.avg.per_cycle_active"],
    "eligible_warps_per_scheduler": ["smsp__warps_eligible.avg.per_cycle_active"],
    "issued_warps_per_scheduler": ["smsp__warps_issued.avg.per_cycle_active"],
    "tensor_pipe_utilization": [
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    ],
    "local_memory_load_bytes": [
        "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    ],
    "local_memory_store_bytes": [
        "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    ],
}

BYTE_FIELDS = {
    "dram_bytes_read",
    "dram_bytes_written",
    "l2_bytes_read",
    "l2_bytes_written",
    "shared_memory_per_block",
    "local_memory_load_bytes",
    "local_memory_store_bytes",
}
SUM_FIELDS = {
    "dram_bytes_read",
    "dram_bytes_written",
    "l2_bytes_read",
    "l2_bytes_written",
    "local_memory_load_bytes",
    "local_memory_store_bytes",
}


def parse_numeric(value: str, unit: str, canonical: str) -> float:
    cleaned = value.strip().replace(",", "").replace("%", "")
    if not cleaned or cleaned.lower() in {"n/a", "nan", "-"}:
        return math.nan
    number = float(cleaned)
    normalized_unit = unit.strip().lower()
    if canonical in BYTE_FIELDS:
        factors = {
            "byte": 1,
            "bytes": 1,
            "kb": 1e3,
            "kbyte": 1e3,
            "mb": 1e6,
            "mbyte": 1e6,
            "gb": 1e9,
            "gbyte": 1e9,
            "kib": 1024,
            "mib": 1024**2,
            "gib": 1024**3,
            "sector": 32,
            "sectors": 32,
        }
        number *= factors.get(normalized_unit, 1)
    return number


def canonical_lookup(mapping: dict[str, list[str]]) -> dict[str, str]:
    return {alias.lower(): canonical for canonical, aliases in mapping.items() for alias in aliases}


def find_header(rows: list[list[str]]) -> int:
    for index, row in enumerate(rows):
        normalized = {cell.strip().lower() for cell in row}
        if "metric name" in normalized and "metric value" in normalized:
            return index
    raise ValueError("could not find NCU raw CSV header containing Metric Name/Metric Value")


def profile_metadata(path: Path) -> dict[str, object]:
    pattern = re.compile(
        r"(?:b(?P<batch_size>\d+)_h(?P<num_heads>\d+)_)?"
        r"n(?P<seq_len>\d+)_d(?P<head_dim>\d+)_(?P<dtype>float16|bfloat16)_"
        r"bm(?P<block_m>\d+)_bn(?P<block_n>\d+)_w(?P<num_warps>\d+)_s(?P<num_stages>\d+)"
    )
    match = pattern.search(path.stem)
    if not match:
        return {"ncu_source_file": str(path)}
    result: dict[str, object] = {
        key: int(value) if value is not None and value.isdigit() else value
        for key, value in match.groupdict().items()
    }
    result.update(
        {
            "batch_size": result.get("batch_size") or 1,
            "num_heads": result.get("num_heads") or 32,
            "provider": "triton_fa2",
            "ncu_source_file": str(path),
        }
    )
    return result


def parse_file(path: Path, mapping: dict[str, list[str]]) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    header_index = find_header(rows)
    header = [cell.strip() for cell in rows[header_index]]
    positions = {name.lower(): index for index, name in enumerate(header)}
    lookup = canonical_lookup(mapping)
    collected: dict[str, list[float]] = {}

    for row in rows[header_index + 1 :]:
        if len(row) < len(header):
            continue
        metric_name = row[positions["metric name"]].strip().lower()
        if metric_name not in lookup:
            continue
        canonical = lookup[metric_name]
        value = row[positions["metric value"]]
        unit = row[positions["metric unit"]] if "metric unit" in positions else ""
        for component in value.split(";"):
            try:
                parsed = parse_numeric(component, unit, canonical)
            except ValueError:
                continue
            if math.isfinite(parsed):
                collected.setdefault(metric_name, []).append(parsed)

    result = profile_metadata(path)
    for canonical, aliases in mapping.items():
        # Aliases describe version alternatives. Prefer the first available name
        # instead of combining aliases that a verbose NCU set may report together.
        values: list[float] = []
        for alias in aliases:
            if alias.lower() in collected:
                values = collected[alias.lower()]
                break
        if values:
            result[canonical] = sum(values) if canonical in SUM_FIELDS else sum(values) / len(values)
        else:
            result[canonical] = ""
    return result


def merge_benchmark(
    ncu_rows: list[dict[str, object]], benchmark_path: Path
) -> list[dict[str, object]]:
    with benchmark_path.open(newline="", encoding="utf-8") as handle:
        benchmark_rows = list(csv.DictReader(handle))
    keys = ("provider", "dtype", "batch_size", "num_heads", "seq_len", "head_dim", "block_m", "block_n", "num_warps", "num_stages")

    def normalized_key(row: dict[str, object]) -> tuple[str, ...]:
        return tuple(str(row.get(key, "")) for key in keys)

    indexed = {normalized_key(row): row for row in benchmark_rows}
    merged: list[dict[str, object]] = []
    for ncu_row in ncu_rows:
        benchmark_row = indexed.get(normalized_key(ncu_row), {})
        merged.append({**benchmark_row, **ncu_row})
    return merged


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="NCU --csv --page raw exports")
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark-csv", default=None)
    parser.add_argument(
        "--metric-map",
        default=None,
        help="JSON file mapping canonical output fields to NCU metric-name lists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mapping = METRIC_NAME_MAP
    if args.metric_map:
        with Path(args.metric_map).open(encoding="utf-8") as handle:
            custom = json.load(handle)
        mapping = {**METRIC_NAME_MAP, **custom}
    rows = [parse_file(Path(item), mapping) for item in args.inputs]
    if args.benchmark_csv:
        rows = merge_benchmark(rows, Path(args.benchmark_csv))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    write_csv(rows, output)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
