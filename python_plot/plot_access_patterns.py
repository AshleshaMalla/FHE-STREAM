#!/usr/bin/env python3
"""Plot bandwidth across access patterns for the four STREAM-style kernels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from extra.fhe_plot_style import HATCH_PATTERNS, configure_matplotlib, save_plot
except ModuleNotFoundError:
    from fhe_plot_style import HATCH_PATTERNS, configure_matplotlib, save_plot


KERNELS = ["COPY", "SCALE", "ADD", "TRIAD"]
ACCESS_PATTERNS = ["Sequential", "Gather", "Scatter", "Scatter-Gather"]
COLORS = ["#377eb8", "#ff7f00", "#4daf4a", "#e41a1c"]

# Values from the source data used for the supplied N=131072, L=40 figure.
BANDWIDTH_GIB_S = {
    "Sequential": [436.0, 434.0, 497.0, 477.0],
    "Gather": [410.0, 301.0, 355.0, 270.0],
    "Scatter": [408.0, 380.0, 430.0, 443.0],
    "Scatter-Gather": [242.0, 201.0, 186.0, 136.0],
}


def plot_access_patterns(output: Path, ring_dimension: int = 131072, depth: int = 40) -> None:
    """Render the grouped bandwidth chart as a PDF."""
    configure_matplotlib()
    plt.rcParams["hatch.color"] = "black"

    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    x_positions = list(range(len(KERNELS)))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for index, (pattern, color, offset) in enumerate(zip(ACCESS_PATTERNS, COLORS, offsets)):
        axis.bar(
            [position + offset * width for position in x_positions],
            BANDWIDTH_GIB_S[pattern],
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            hatch=HATCH_PATTERNS[index],
            label=pattern,
            zorder=3,
        )

    axis.set_title(
        f"Bandwidth across Access Patterns (N={ring_dimension}, L={depth})",
        fontweight="bold",
    )
    axis.set_ylabel("Bandwidth (GiB/s)")
    axis.set_xlabel("Kernels")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(KERNELS)
    axis.set_ylim(0, 650)
    axis.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=True, ncol=2)

    figure.tight_layout()
    save_plot(str(output))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/access_patterns.pdf"),
        help="Output PDF path",
    )
    parser.add_argument("--ring-dimension", type=int, default=131072)
    parser.add_argument("--depth", type=int, default=40)
    args = parser.parse_args()

    plot_access_patterns(args.output, args.ring_dimension, args.depth)


if __name__ == "__main__":
    main()
