#!/usr/bin/env python3
"""Plot Regular vs HEXL throughput for DCRT TRIAD and NTT benchmarks."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from fhe_plot_style import (
    configure_matplotlib,
    save_plot,
    add_value_labels,
    COLOR_HEXL,
    COLOR_REGULAR,
    FIGSIZE_SINGLE,
    HATCH_PATTERNS,
)


CATEGORY_ORDER = [
    ("GATHER", "TRIAD", "Gather Triad"),
    ("SCATTER", "TRIAD", "Scatter Triad"),
    ("SCATTER_GATHER", "TRIAD", "Scatter-Gather Triad"),
    ("NTT", None, "NTT Roundtrip"),
]

SYSTEM_LABELS = {
    "regular": "Hexl-off",
    "hexl": "Hexl-on",
}

SYSTEM_COLORS = {
    "regular": COLOR_REGULAR,
    "hexl": COLOR_HEXL,
}


def extract_bandwidth_gibs(benchmark: dict) -> float | None:
    for key in ("PayloadBandwidth", "payload_bandwidth", "bytes_per_second"):
        value = benchmark.get(key)
        if value is not None:
            return float(value) / (1024.0**3)
    return None


def parse_category(name: str) -> tuple[str, str | None, str | None] | None:
    if "RS_NTT_ROUNDTRIP" in name:
        return "NTT", None, None

    match = re.search(r"RS_(GATHER|SCATTER|SCATTER_GATHER)_(COPY|SCALE|ADD|TRIAD)/", name)
    if not match:
        return None

    pattern = match.group(1)
    kernel = match.group(2)

    suffix = None
    if name.endswith("/1"):
        suffix = "poly"
    elif name.endswith("/2"):
        suffix = "coeff"

    if suffix is None:
        return None

    return pattern, kernel, suffix


def load_system(path: Path) -> tuple[dict[tuple[str, str | None], float], int | None, int | None, int | None]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    grouped: dict[tuple[str, str | None], list[float]] = defaultdict(list)
    median_values: dict[tuple[str, str | None], float] = {}
    n_val = None
    l_val = None
    batch_val = None

    for benchmark in data.get("benchmarks", []):
        name = benchmark.get("name") or benchmark.get("run_name") or ""
        aggregate_name = benchmark.get("aggregate_name")
        parsed = parse_category(name)
        if parsed is None:
            continue

        category, kernel, variant = parsed
        bandwidth = extract_bandwidth_gibs(benchmark)
        if bandwidth is None:
            continue

        if n_val is None or l_val is None or batch_val is None:
            label = benchmark.get("label", "")
            n_match = re.search(r"N=(\d+)", label)
            l_match = re.search(r"L=(\d+)", label)
            batch_match = re.search(r"Batch[Ss]ize=(\d+)", label)
            if n_match:
                n_val = int(n_match.group(1))
            if l_match:
                l_val = int(l_match.group(1))
            if batch_match:
                batch_val = int(batch_match.group(1))

        key = (category, kernel if category != "NTT" else None)

        if aggregate_name == "median" and variant == "coeff":
            median_values[key] = bandwidth

        if aggregate_name is None:
            grouped[key].append(bandwidth)

    summary: dict[tuple[str, str | None], float] = {}
    for key, values in grouped.items():
        if not values:
            continue
        summary[key] = mean(values)

    # Prefer HEXL median rows when available.
    for key, value in median_values.items():
        summary[key] = value

    return summary, n_val, l_val, batch_val


def build_summary(system_files: dict[str, Path]) -> tuple[dict[str, dict[tuple[str, str | None], float]], int | None, int | None, int | None]:
    summary: dict[str, dict[tuple[str, str | None], float]] = {}
    n_val = None
    l_val = None
    batch_val = None

    for system_key, path in system_files.items():
        values, local_n, local_l, local_batch = load_system(path)
        summary[system_key] = values
        if n_val is None and local_n is not None:
            n_val = local_n
        if l_val is None and local_l is not None:
            l_val = local_l
        if batch_val is None and local_batch is not None:
            batch_val = local_batch

    return summary, n_val, l_val, batch_val


def print_summary(summary: dict[str, dict[tuple[str, str | None], float]]) -> None:
    print("\nHEXL-off vs HEXL-on Bandwidth Comparison (GiB/s)\n")
    header = f"{'Category':<18} {'Hexl-off':>12} {'Hexl-on':>12} {'Delta':>10}"
    print(header)
    print("-" * len(header))

    for pattern, kernel, label in CATEGORY_ORDER:
        key = (pattern, kernel)
        regular = summary.get("regular", {}).get(key)
        hexl = summary.get("hexl", {}).get(key)
        if regular is None or hexl is None:
            continue
        delta = ((hexl - regular) / regular) * 100.0 if regular > 0 else float("nan")
        print(f"{label:<18} {regular:>12.1f} {hexl:>12.1f} {delta:>9.1f}%")


def plot_summary(
    summary: dict[str, dict[tuple[str, str | None], float]],
    output: Path,
    n_val: int | None,
    l_val: int | None,
    batch_val: int | None,
) -> None:
    configure_matplotlib()

    labels = [label for _, _, label in CATEGORY_ORDER]
    x_positions = list(range(len(labels)))
    width = 0.34

    regular_vals = [summary.get("regular", {}).get((pattern, variant), float("nan")) for pattern, variant, _ in CATEGORY_ORDER]
    hexl_vals = [summary.get("hexl", {}).get((pattern, variant), float("nan")) for pattern, variant, _ in CATEGORY_ORDER]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bars_regular = ax.bar(
        [x - width / 2 for x in x_positions],
        regular_vals,
        width=width,
        color=SYSTEM_COLORS["regular"],
        edgecolor="black",
        linewidth=0.8,
        label=SYSTEM_LABELS["regular"],
        zorder=3,
        hatch=HATCH_PATTERNS[0],
    )
    bars_hexl = ax.bar(
        [x + width / 2 for x in x_positions],
        hexl_vals,
        width=width,
        color=SYSTEM_COLORS["hexl"],
        edgecolor="black",
        linewidth=0.8,
        label=SYSTEM_LABELS["hexl"],
        zorder=3,
        hatch=HATCH_PATTERNS[1],
    )

    ax.set_title("Intel HEXL Acceleration Benefit", fontweight="bold")
    ax.set_ylabel("Bandwidth (GiB/s)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    
    max_val = max(v for v in regular_vals + hexl_vals if v == v)
    ax.set_ylim(0, max_val * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)

    add_value_labels(ax, bars_regular)
    add_value_labels(ax, bars_hexl)

    plt.tight_layout()
    save_plot(str(output))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Regular vs HEXL throughput comparison from benchmark JSON files.")
    parser.add_argument("--regular", type=Path, default=Path("results/hexl_off.json"), help="Regular-node JSON file")
    parser.add_argument("--hexl", type=Path, default=Path("results/hexl_on.json"), help="HEXL-node JSON file")
    parser.add_argument("--output", type=Path, default=Path("plots/hexl_vs_regular_throughput.pdf"), help="Output PDF file")
    args = parser.parse_args()

    if not args.regular.exists():
        raise FileNotFoundError(f"Regular input not found: {args.regular}")
    if not args.hexl.exists():
        raise FileNotFoundError(f"HEXL input not found: {args.hexl}")

    summary, n_val, l_val, batch_val = build_summary({"regular": args.regular, "hexl": args.hexl})
    if not summary.get("regular") or not summary.get("hexl"):
        raise RuntimeError("No matching benchmark categories were found in both input files.")

    print_summary(summary)
    plot_summary(summary, args.output, n_val, l_val, batch_val)
    print(f"\nSaved figure to {args.output}")


if __name__ == "__main__":
    main()