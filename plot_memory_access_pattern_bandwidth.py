#!/usr/bin/env python3
"""Plot payload bandwidth across memory access patterns for DCRT-style kernels."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


KERNEL_ORDER = ["COPY", "SCALE", "ADD", "TRIAD"]
PATTERN_ORDER = ["SEQ", "GATHER", "SCATTER", "SG"]

PATTERN_LABELS = {
    "SEQ": "Sequential",
    "GATHER": "Gather",
    "SCATTER": "Scatter",
    "SG": "Scatter-Gather",
}

PATTERN_COLORS = {
    "SEQ": "#377eb8",
    "GATHER": "#ff7f00",
    "SCATTER": "#4daf4a",
    "SG": "#e41a1c",
}

# Supports both the documented DCRTFixture/DCRT_* layout and the current
# FHERaiderSTREAM/RS_* layout. The trailing benchmark instance encodes the
# access mode: 0 for sequential kernels and 2 for the coeff shuffle variant.
BENCHMARK_RE = re.compile(
    r"^(?:DCRTFixture/DCRT|FHERaiderSTREAM/RS)_"
    r"(SEQ|GATHER|SCATTER|SCATTER_GATHER)_(COPY|SCALE|ADD|TRIAD)"
    r"(?:_(poly|coeff))?"
    r"/(?P<n>\d+)/(?P<l>\d+)/(?P<mode>[0-2])$"
)


def configure_matplotlib() -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required to generate the PDF figure. Install it in the active Python environment."
        ) from exc

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    serif_fonts = ["Times New Roman"] if "Times New Roman" in available_fonts else []
    serif_fonts.extend(["Times", "DejaVu Serif", "Liberation Serif", "serif"])
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif_fonts,
            "font.size": 13,
            "text.color": "black",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )

    return plt


def extract_payload_bandwidth(benchmark: dict) -> float | None:
    for key in ("PayloadBandwidth", "payload_bandwidth", "bytes_per_second"):
        value = benchmark.get(key)
        if value is not None:
            return float(value)
    return None


def parse_benchmark_name(name: str):
    match = BENCHMARK_RE.match(name)
    if not match:
        return None

    pattern = match.group(1)
    if pattern == "SCATTER_GATHER":
        pattern = "SG"

    shuffle_mode = match.group(3)  # poly or coeff, if present

    return {
        "pattern": pattern,
        "kernel": match.group(2),
        "shuffle_mode": shuffle_mode,
        "n": int(match.group("n")),
        "l": int(match.group("l")),
        "mode": int(match.group("mode")),
    }


def load_summary(json_path: Path):
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    grouped_values = defaultdict(list)
    metadata = {}

    for benchmark in data.get("benchmarks", []):
        benchmark_name = benchmark.get("name") or benchmark.get("run_name") or ""
        parsed = parse_benchmark_name(benchmark_name)
        if parsed is None:
            continue

        payload_bandwidth = extract_payload_bandwidth(benchmark)
        if payload_bandwidth is None:
            continue

        if parsed["pattern"] == "SEQ":
            if parsed["mode"] != 0:
                continue
        elif parsed["mode"] != 2:
            continue

        key = (parsed["kernel"], parsed["pattern"])
        grouped_values[key].append(payload_bandwidth)
        metadata[key] = {"n": parsed["n"], "l": parsed["l"], "mode": parsed["mode"]}

    summary = {}
    for kernel in KERNEL_ORDER:
        for pattern in PATTERN_ORDER:
            key = (kernel, pattern)
            values = grouped_values.get(key, [])
            if not values:
                continue

            summary[key] = {
                "mean_gbs": mean(values) / 1e9,
                "count": len(values),
                **metadata.get(key, {}),
            }

    return summary


def print_summary_table(summary: dict) -> None:
    print("\nBandwidth summary (GB/s) - DCRTPoly Backend (Coeff mode for shuffled kernels)\n")
    print(f"{'Kernel':<10} {'Pattern':<20} {'Variant':<10} {'Mean GB/s':>12} {'Samples':>10}")
    print("-" * 62)
    for kernel in KERNEL_ORDER:
        for pattern in PATTERN_ORDER:
            key = (kernel, pattern)
            item = summary.get(key)
            if item is None:
                continue
            variant = "Coeff" if pattern != "SEQ" else "-"
            print(
                f"{kernel:<10} {PATTERN_LABELS[pattern]:<20} "
                f"{variant:<10} {item['mean_gbs']:>12.2f} {item['count']:>10}"
            )


def plot_summary(summary: dict, output_path: Path, batch_size: int = 512) -> None:
    plt = configure_matplotlib()

    fig, ax = plt.subplots(figsize=(13, 6.5))

    x_positions = list(range(len(KERNEL_ORDER)))
    bar_width = 0.18
    offsets = [
        (index - (len(PATTERN_ORDER) - 1) / 2.0) * bar_width
        for index in range(len(PATTERN_ORDER))
    ]

    # Extract N and L from first available entry in summary
    n_val = l_val = None
    for entry in summary.values():
        if isinstance(entry, dict) and "n" in entry and "l" in entry:
            n_val, l_val = entry["n"], entry["l"]
            break

    # Update title with backend and parameters
    title_str = "DCRTPoly Backend: Bandwidth Across Memory Access Patterns (Coeff mode)"
    if n_val and l_val:
        title_str += f"\n(N={n_val}, L={l_val}, Batch={batch_size})"
    ax.set_title(title_str, fontweight="bold")
    ax.set_xlabel("Kernels")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(KERNEL_ORDER)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    ymax = max((item["mean_gbs"] for item in summary.values() if isinstance(item, dict)), default=0)
    ax.set_ylim(0, ymax * 1.15)

    # Add secondary x-axis for Total CPU Cores (assuming n_val represents this context if applicable, 
    # but here we just need bars and labels).
    
    for pattern_index, pattern in enumerate(PATTERN_ORDER):
        heights = [
            summary.get((kernel, pattern), {}).get("mean_gbs", float("nan"))
            for kernel in KERNEL_ORDER
        ]
        shifted_positions = [x + offsets[pattern_index] for x in x_positions]
        bars = ax.bar(
            shifted_positions,
            heights,
            width=bar_width,
            color=PATTERN_COLORS[pattern],
            edgecolor="black",
            linewidth=0.6,
            label=PATTERN_LABELS[pattern],
            zorder=3
        )
        for bar, height in zip(bars, heights):
            if not (height != height):  # Skip NaN
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + ymax * 0.018,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    zorder=4
                )

    ax.legend(loc="upper left", frameon=False, fontsize=13, ncol=2)
    plt.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot payload bandwidth across access patterns from Google Benchmark JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/hardware_tax_results.json"),
        help="Path to the Google Benchmark JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bandwidth_breakdown.pdf"),
        help="Path to the output PDF figure.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size used in the benchmark (for title annotation).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    summary = load_summary(args.input)

    if not summary:
        raise RuntimeError("No benchmark rows matched the expected kernel naming pattern.")

    print_summary_table(summary)
    plot_summary(summary, args.output, batch_size=args.batch_size)
    print(f"\nSaved figure to {args.output}")


if __name__ == "__main__":
    main()