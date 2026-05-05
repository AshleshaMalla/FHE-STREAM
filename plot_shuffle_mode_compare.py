#!/usr/bin/env python3
"""Plot DCRTPoly shuffle-mode comparison for Gather and Scatter-Gather kernels."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib import font_manager

CATEGORIES = [
    ("GATHER", "ADD", "Gather Add"),
    ("GATHER", "TRIAD", "Gather Triad"),
    ("SCATTER_GATHER", "ADD", "Scatter-Gather Add"),
    ("SCATTER_GATHER", "TRIAD", "Scatter-Gather Triad"),
]

MODE_LABELS = {
    "poly": "Poly",
    "coeff": "Coeff",
}

MODE_COLORS = {
    "poly": "#377eb8",
    "coeff": "#e41a1c",
}

BENCHMARK_RE = re.compile(
    r"^FHERaiderSTREAM/RS_"
    r"(GATHER|SCATTER_GATHER)_(ADD|TRIAD)"
    r"/(?P<n>\d+)/(?P<l>\d+)/(?P<mode>[12])$"
)


def configure_matplotlib() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    serif_fonts = ["Times New Roman"] if "Times New Roman" in available_fonts else []
    serif_fonts.extend(["Times", "DejaVu Serif", "Liberation Serif", "serif"])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif_fonts,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "text.color": "black",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )


def extract_bandwidth(benchmark: dict) -> float | None:
    for key in ("PayloadBandwidth", "payload_bandwidth", "bytes_per_second"):
        value = benchmark.get(key)
        if value is not None:
            return float(value)
    return None


def parse_benchmark_name(name: str):
    match = BENCHMARK_RE.match(name)
    if not match:
        return None

    mode = "poly" if match.group("mode") == "1" else "coeff"
    pattern = match.group(1)
    if pattern == "SCATTER_GATHER":
        pattern = "SCATTER_GATHER"

    return {
        "pattern": pattern,
        "kernel": match.group(2),
        "mode": mode,
        "n": int(match.group("n")),
        "l": int(match.group("l")),
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

        bandwidth = extract_bandwidth(benchmark)
        if bandwidth is None:
            continue

        key = (parsed["pattern"], parsed["kernel"], parsed["mode"])
        grouped_values[key].append(bandwidth)
        metadata[key] = {"n": parsed["n"], "l": parsed["l"]}

    summary = {}
    for pattern, kernel, label in CATEGORIES:
        for mode in ("poly", "coeff"):
            key = (pattern, kernel, mode)
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
    print("\nShuffle mode comparison (GB/s) - DCRTPoly Backend\n")
    print(f"{'Category':<26} {'Mode':<8} {'Mean GB/s':>12} {'Samples':>10}")
    print("-" * 60)
    for pattern, kernel, label in CATEGORIES:
        for mode in ("poly", "coeff"):
            item = summary.get((pattern, kernel, mode))
            if item is None:
                continue
            print(
                f"{label:<26} {MODE_LABELS[mode]:<8} {item['mean_gbs']:>12.2f} {item['count']:>10}"
            )


def plot_summary(summary: dict, output_path: Path, batch_size: int = 512) -> None:
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 6.2))
    x_positions = list(range(len(CATEGORIES)))
    width = 0.34

    n_val = l_val = None
    for entry in summary.values():
        if isinstance(entry, dict) and "n" in entry and "l" in entry:
            n_val, l_val = entry["n"], entry["l"]
            break

    poly_heights = [summary.get((pattern, kernel, "poly"), {}).get("mean_gbs", float("nan")) for pattern, kernel, _ in CATEGORIES]
    coeff_heights = [summary.get((pattern, kernel, "coeff"), {}).get("mean_gbs", float("nan")) for pattern, kernel, _ in CATEGORIES]

    bars_poly = ax.bar(
        [x - width / 2 for x in x_positions],
        poly_heights,
        width=width,
        color=MODE_COLORS["poly"],
        edgecolor="black",
        linewidth=0.8,
        label=MODE_LABELS["poly"],
    )
    bars_coeff = ax.bar(
        [x + width / 2 for x in x_positions],
        coeff_heights,
        width=width,
        color=MODE_COLORS["coeff"],
        edgecolor="black",
        linewidth=0.8,
        label=MODE_LABELS["coeff"],
    )

    for bars, heights in ((bars_poly, poly_heights), (bars_coeff, coeff_heights)):
        for bar, height in zip(bars, heights):
            if height == height:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_ylabel("Bandwidth (GB/s)")
    title = "DCRTPoly Shuffle-Mode Comparison: Gather / Scatter-Gather ADD and TRIAD"
    if n_val and l_val:
        title += f" [N={n_val}, L={l_val}, Batch={batch_size}]"
    ax.set_title(title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, _, label in CATEGORIES])
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)

    ymax = max((v for v in poly_heights + coeff_heights if v == v), default=0.0)
    ax.set_ylim(0, ymax * 1.22)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DCRTPoly shuffle-mode comparison.")
    parser.add_argument("--input", type=Path, default=Path("results/shuffle_mode_compare.json"))
    parser.add_argument("--output", type=Path, default=Path("shuffle_mode_compare.pdf"))
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    summary = load_summary(args.input)
    if not summary:
        raise RuntimeError("No benchmark rows matched the expected shuffle-mode pattern.")

    print_summary_table(summary)
    plot_summary(summary, args.output, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
