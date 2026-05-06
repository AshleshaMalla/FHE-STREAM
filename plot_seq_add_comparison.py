#!/usr/bin/env python3
"""Plot comparison of RS_SEQ_ADD, CT_SEQ_ADD, and CT_SEQ_ADD_INPLACE kernels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_matplotlib() -> None:
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


def extract_bandwidth(benchmark: dict) -> float | None:
    for key in ("PayloadBandwidth", "payload_bandwidth", "bytes_per_second"):
        value = benchmark.get(key)
        if value is not None:
            return float(value)
    return None


def load_summary(json_path: Path):
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    kernels = {
        "RS_SEQ_ADD": [],
        "CT_SEQ_ADD": [],
        "CT_SEQ_ADD_INPLACE": [],
    }
    n_val = l_val = batch_val = None

    for benchmark in data.get("benchmarks", []):
        name = benchmark.get("name") or benchmark.get("run_name") or ""
        bandwidth = extract_bandwidth(benchmark)

        if bandwidth is None:
            continue

        # Extract metadata once
        if n_val is None or l_val is None:
            label = benchmark.get("label", "")
            try:
                parts = label.split()
                for p in parts:
                    if p.startswith("N="):
                        n_val = int(p.split("=", 1)[1])
                    if p.startswith("L="):
                        l_val = int(p.split("=", 1)[1])
            except Exception:
                pass

        # Categorize by kernel
        if "RS_SEQ_ADD" in name and "FHERaiderSTREAM" in name:
            kernels["RS_SEQ_ADD"].append(bandwidth)
        elif "CT_SEQ_ADD_INPLACE" in name:
            kernels["CT_SEQ_ADD_INPLACE"].append(bandwidth)
        elif "CT_SEQ_ADD" in name:
            kernels["CT_SEQ_ADD"].append(bandwidth)

    # Compute means
    summary = {}
    for kernel, values in kernels.items():
        if values:
            summary[kernel] = {
                "mean_gbs": mean(values) / (1024.0**3),
                "count": len(values),
            }

    return summary, n_val, l_val


def plot_comparison(summary: dict, n_val: int | None, l_val: int | None, output_path: Path) -> None:
    configure_matplotlib()

    kernel_labels = ["RS_SEQ_ADD\n(DCRT)", "CT_SEQ_ADD\n(CT)", "CT_SEQ_ADD_INPLACE\n(CT)"]
    kernel_keys = ["RS_SEQ_ADD", "CT_SEQ_ADD", "CT_SEQ_ADD_INPLACE"]
    kernel_colors = ["#377eb8", "#ff7f00", "#4daf4a"]

    heights = [summary.get(k, {}).get("mean_gbs", 0.0) for k in kernel_keys]
    counts = [summary.get(k, {}).get("count", 0) for k in kernel_keys]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x_positions = list(range(len(kernel_keys)))
    bars = ax.bar(
        x_positions,
        heights,
        width=0.5,
        color=kernel_colors,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )

    # Add value labels on bars
    for bar, height, count in zip(bars, heights, counts):
        label_y = min(height + max(heights) * 0.018, max(heights) * 1.25 - max(heights) * 0.03)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{height:.1f} GiB/s",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            zorder=4,
        )

    ax.set_ylabel("Payload Bandwidth (GiB/s)")
    title = "ADD Kernel (256 threads, Batch=256)"
    if n_val and l_val:
        title += f"\n(N={n_val}, L={l_val}, Batch=256)"
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(kernel_labels)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    ymax = max(heights) if heights else 0
    ax.set_ylim(0, ymax * 1.15)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison figure to {output_path}")


def print_summary_table(summary: dict) -> None:
    print("\nSEQ_ADD Kernel Comparison (256 threads, Batch=256)\n")
    print(f"{'Kernel':<25} {'Bandwidth (GiB/s)':>20} {'Samples':>10}")
    print("-" * 55)
    for kernel in ["RS_SEQ_ADD", "CT_SEQ_ADD", "CT_SEQ_ADD_INPLACE"]:
        item = summary.get(kernel)
        if item is None:
            print(f"{kernel:<25} {'N/A':>20} {'0':>10}")
        else:
            print(
                f"{kernel:<25} {item['mean_gbs']:>20.2f} {item['count']:>10}"
            )

    # Compute ratios
    rs_bw = summary.get("RS_SEQ_ADD", {}).get("mean_gbs")
    ct_bw = summary.get("CT_SEQ_ADD", {}).get("mean_gbs")
    ct_inplace_bw = summary.get("CT_SEQ_ADD_INPLACE", {}).get("mean_gbs")

    if rs_bw and ct_bw:
        print(f"\nCT_SEQ_ADD overhead vs DCRT: {((rs_bw - ct_bw) / rs_bw * 100):.1f}%")
    if rs_bw and ct_inplace_bw:
        print(f"CT_SEQ_ADD_INPLACE overhead vs DCRT: {((rs_bw - ct_inplace_bw) / rs_bw * 100):.1f}%")
    if ct_bw and ct_inplace_bw:
        print(f"CT_SEQ_ADD_INPLACE improvement vs CT_SEQ_ADD: {((ct_inplace_bw - ct_bw) / ct_bw * 100):.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SEQ_ADD kernel comparison across backends."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/seq_add_comparison.json"),
        help="Path to the benchmark JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("seq_add_comparison.pdf"),
        help="Path to the output PDF figure.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    summary, n_val, l_val = load_summary(args.input)

    if not summary:
        raise RuntimeError("No benchmark rows found in input JSON.")

    print_summary_table(summary)
    plot_comparison(summary, n_val, l_val, args.output)


if __name__ == "__main__":
    main()
