#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def configure_matplotlib() -> None:
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    serif_fonts = ["Times New Roman"] if "Times New Roman" in available_fonts else []
    serif_fonts.extend(["Times", "DejaVu Serif", "Liberation Serif", "serif"])
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": serif_fonts,
        "font.size": 13,
        "font.weight": "normal",
        "axes.titleweight": "bold",
        "axes.labelweight": "normal",
        "axes.linewidth": 1.6,
        "text.color": "black",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })


def main() -> None:
    configure_matplotlib()

    categories = ["RaiderSTREAM", "DCRTPoly", "Ciphertext"]
    raw_bandwidth = [500.58, 488.02, 466.06]
    effective_bandwidth = [500.58, 6.10, 5.83]

    x = np.arange(len(categories))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    bars_raw = ax.bar(
        x - width / 2,
        raw_bandwidth,
        width,
        label="Raw Bandwidth",
        color="#1a9850",
        edgecolor="black",
        linewidth=1.6,
        hatch="//",
        zorder=3,
    )
    bars_effective = ax.bar(
        x + width / 2,
        effective_bandwidth,
        width,
        label="Effective Bandwidth",
        color="#2c7fb8",
        edgecolor="black",
        linewidth=1.6,
        hatch="\\",
        zorder=3,
    )

    ax.set_title("Raw vs Effective Bandwidth", fontweight="bold")
    ax.set_ylabel("Bandwidth (GiB/s)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, max(raw_bandwidth + effective_bandwidth) * 1.30)

    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    legend = ax.legend(frameon=False, fontsize=11, loc="upper right", ncol=2)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    max_val = max(raw_bandwidth + effective_bandwidth)
    y_offset = max_val * 0.02
    for bars in (bars_raw, bars_effective):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + y_offset,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                zorder=4,
            )

    plt.tight_layout()
    fig.savefig("effective_bandwidth.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
