#!/usr/bin/env python3
"""Read-side DRAM amplification: per-iteration Logical (modeled) vs Measured (hardware)
byte volume for the five hardware-validated DCRTPoly kernels shown in the paper-style
comparison plot.

Grouped two-series bar chart with the amplification factor (measured / logical)
annotated above each pair. Values are per-benchmark-iteration byte volumes,
independent of capture iteration count. Numbers are the single-shot hardware-counter
measurements consolidated in results/validation_summary.md.

Run from the repository root:  python3 python_plot/plot_dram_amplification_read.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from fhe_plot_style import (
    COLOR_DCRT,
    COLOR_CT,
    FIGSIZE_WIDE,
    HATCH_PATTERNS,
    configure_matplotlib as configure_shared_matplotlib,
    save_plot,
)

# (label, logical_GB_per_iter, measured_GB_per_iter, amplification)
#   logical/iter = B * reads * (N*L*8): 42.95 GB for 3-array reads (2 reads/iter),
#   21.47 GB for NTT's 2-array roundtrip read (1 read/iter).
#   All measured values are whole-process differential per-iteration counter means
#   (high-iter minus low-iter totals, divided by the iteration delta — SetUp cancelled).
#   All five kernels ordered by measured read amplification (poly < seq < gather-coeff < ntt < sgt).
READ_DATA = [
    ("GATHER_ADD\n(Poly)",            42.950,  42.904, 1.00),
    ("SEQ_ADD\n(Seq)",                42.950,  64.655, 1.51),
    ("GATHER_ADD\n(Coeff)",           42.950,  80.347, 1.87),
    ("NTT\nROUNDTRIP",                21.475,  59.731, 2.78),
    ("SCATTER_GATHER\nTRIAD (Coeff)", 42.950, 164.950, 3.84),
]


def configure_matplotlib() -> None:
    configure_shared_matplotlib()
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 1.6,
        "lines.linewidth": 1.6,
    })


def plot(data, output_path: Path) -> None:
    configure_matplotlib()

    labels = [d[0] for d in data]
    logical = [d[1] for d in data]
    measured = [d[2] for d in data]
    amps = [d[3] for d in data]

    x = np.arange(len(data))
    width = 0.34

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars_log = ax.bar(
        x - width / 2,
        logical,
        width,
        color=COLOR_DCRT,
        edgecolor="black",
        linewidth=1.6,
        label="Logical (modeled)",
        zorder=3,
        hatch=HATCH_PATTERNS[1],
    )
    bars_meas = ax.bar(
        x + width / 2,
        measured,
        width,
        color=COLOR_CT,
        edgecolor="black",
        linewidth=1.6,
        label="Measured (hardware)",
        zorder=3,
        hatch=HATCH_PATTERNS[2],
    )

    ymax = max(measured)
    ax.set_ylim(0, ymax * 1.22)

    # value labels on each bar
    for bars, heights in ((bars_log, logical), (bars_meas, measured)):
        for bar, h in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + ymax * 0.02,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                zorder=4,
            )

    for xi, amp, m in zip(x, amps, measured):
        ax.text(
            xi,
            m + ymax * 0.11,
            f"{amp:.2f}×",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            zorder=4,
        )

    ax.set_ylabel("Data Volume per Iteration (GB)", fontweight="bold")
    ax.set_title("DRAM Amplification: Logical vs Measured Byte Volume (Read)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=11)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_plot(str(output_path), dpi=300)
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def main() -> None:
    plot(READ_DATA, Path("plots/dram_amplification_read.pdf"))


if __name__ == "__main__":
    main()
