#!/usr/bin/env python3
"""Plot MPI scaling benchmark results from results/scaling_pe/*.csv files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from fhe_plot_style import (
    configure_matplotlib,
    save_plot,
    add_value_labels,
    setup_dual_xaxis,
    COLOR_MPI,
    FIGSIZE_SINGLE,
)


BENCHMARK_RE = re.compile(r"^FHERaiderSTREAM/RS_SEQ_ADD/(?P<n>\d+)/(?P<l>\d+)/0$")


def extract_bandwidth(benchmark: dict) -> float | None:
    for key in ("AggregateBandwidth", "payload_bandwidth", "bytes_per_second"):
        value = benchmark.get(key)
        if value is not None:
            return float(value) / (1024.0**3)
    return None


def parse_scaling_file(filepath: Path) -> dict | None:
    try:
        with filepath.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        print(f"Error parsing {filepath}: {exc}")
        return None

    match = re.search(r"pe(\d+)", filepath.name)
    if not match:
        return None
    ranks = int(match.group(1))

    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        return None

    benchmark = benchmarks[0]
    bandwidth = extract_bandwidth(benchmark)
    if bandwidth is None:
        return None

    metadata_match = BENCHMARK_RE.match(benchmark.get("name") or benchmark.get("run_name") or "")
    if metadata_match is None:
        return None

    label = benchmark.get("label", "")
    n_val = int(metadata_match.group("n"))
    l_val = int(metadata_match.group("l"))

    return {
        "ranks": ranks,
        "aggregate_bw_gbs": bandwidth,
        "n": n_val,
        "l": l_val,
        "batch": 256,
    }


def plot_scaling(results: list[dict], output_path: Path) -> None:
    configure_matplotlib()

    results = sorted(results, key=lambda item: item["ranks"])
    ranks = [item["ranks"] for item in results]
    cores = [rank * 256 for rank in ranks]
    agg_bw = [item["aggregate_bw_gbs"] for item in results]

    n_val = results[0]["n"]
    l_val = results[0]["l"]
    batch = results[0]["batch"]

    print("\n=== MPI Scaling Results ===\n")
    print(f"{'Ranks':<10} {'Cores':<10} {'Aggregate BW (GiB/s)':<25}")
    print("-" * 50)
    for rank, core_count, bandwidth in zip(ranks, cores, agg_bw):
        print(f"{rank:<10} {core_count:<10} {bandwidth:<25.2f}")

    fig, ax1 = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax1.set_axisbelow(True)

    bars = ax1.bar(
        range(len(ranks)),
        agg_bw,
        color=COLOR_MPI,
        alpha=0.8,
        edgecolor="black",
        width=0.6,
        zorder=3,
    )

    ax1.set_xlabel("Number of MPI Ranks")
    ax1.set_ylabel("Aggregate Bandwidth (GiB/s)")
    ax1.set_title("MPI Scaling Analysis", fontweight="bold")
    ax1.set_xticks(range(len(ranks)))
    ax1.set_xticklabels(ranks)

    # Use dual x-axis from style guide
    ax2 = setup_dual_xaxis(ax1, ranks, cores, xlabel_ranks="MPI Ranks", xlabel_cores="Total CPU Cores")

    ymax = max(agg_bw) if agg_bw else 0.0
    ax1.set_ylim(0, ymax * 1.15)
    ax1.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    add_value_labels(ax1, bars)

    plt.tight_layout()
    save_plot(str(output_path))
    print(f"\nSaved chart to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MPI scaling bandwidth from results/scaling_pe/*.csv.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/scaling_pe"),
        help="Directory containing scaling_pe*.csv benchmark files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/mpi_scaling_analysis.pdf"),
        help="Output PDF path.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    files = sorted(args.input_dir.glob("scaling_pe*.csv"))
    if not files:
        raise FileNotFoundError(f"No scaling_pe*.csv files found in {args.input_dir}")

    results = []
    for filepath in files:
        parsed = parse_scaling_file(filepath)
        if parsed is not None:
            results.append(parsed)

    if not results:
        raise RuntimeError("No benchmark data extracted from scaling_pe files.")

    plot_scaling(results, args.output)


if __name__ == "__main__":
    main()
