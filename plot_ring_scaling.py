#!/usr/bin/env python3
"""Plot ring dimension scaling for SEQ_ADD kernels (DCRT vs CT)."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_matplotlib() -> None:
    """Configure matplotlib with serif font and styling."""
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
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "legend.fontsize": 11,
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
    """Extract bandwidth in GiB/s from benchmark dict."""
    for key in ("PayloadBandwidth", "payload_bandwidth", "bytes_per_second"):
        value = benchmark.get(key)
        if value is not None:
            return float(value) / 1e9  # Convert to GiB/s
    return None


def extract_ring_dimension(benchmark: dict) -> int | None:
    """Extract ring dimension N from benchmark label or name."""
    # Try to extract from label first (format: "N=131072 L=40 ...")
    label = benchmark.get("label", "")
    if label:
        match = re.search(r"N=(\d+)", label)
        if match:
            return int(match.group(1))

    # Try to extract from name (format: "FHERaiderSTREAM/RS_SEQ_ADD/16384/...")
    name = benchmark.get("name") or benchmark.get("run_name") or ""
    if name:
        # Look for ring dimension as a path component
        # Expected: FHERaiderSTREAM/RS_SEQ_ADD/16384 or similar
        match = re.search(r"/(\d{5})(?:/|$)", name)
        if match:
            n = int(match.group(1))
            # Validate it's a reasonable ring dimension
            if 2**13 <= n <= 2**20:
                return n

    return None


def plot_ring_scaling(
    json_path: Path,
    output_path: Path | None = None,
) -> None:
    """
    Load merged ring-scaling JSON and plot SEQ_ADD bandwidth vs ring dimension.

    Args:
        json_path: Path to merged ring_scaling_seq_add.json
        output_path: Path to save plot PDF (default: ring_scaling_plot.pdf)
    """
    if output_path is None:
        output_path = Path("ring_scaling_plot.pdf")

    configure_matplotlib()

    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    # Group benchmarks by (kernel_type, ring_dimension)
    groups = defaultdict(lambda: defaultdict(list))
    debug_names = []

    for benchmark in data.get("benchmarks", []):
        name = benchmark.get("name") or benchmark.get("run_name") or ""
        bandwidth_gbs = extract_bandwidth(benchmark)
        ring_dim = extract_ring_dimension(benchmark)
        
        debug_names.append(name)

        if bandwidth_gbs is None or ring_dim is None:
            continue

        # Categorize by kernel
        if "RS_SEQ_ADD" in name and "FHERaiderSTREAM" in name:
            kernel = "DCRT (RS_SEQ_ADD)"
            groups[kernel][ring_dim].append(bandwidth_gbs)
        elif "CT_SEQ_ADD" in name:
            kernel = "CT (CT_SEQ_ADD)"
            groups[kernel][ring_dim].append(bandwidth_gbs)
    
    # Debug: Print sample benchmark names
    print(f"\nDebug: Found {len(debug_names)} total benchmarks")
    print("Sample benchmark names:")
    for name in debug_names[:10]:
        print(f"  {name}")

    # Compute mean bandwidth per (kernel, ring_dim)
    data_by_kernel = {}
    for kernel, dim_dict in groups.items():
        ring_dims = sorted(dim_dict.keys())
        means = [mean(dim_dict[n]) for n in ring_dims]
        data_by_kernel[kernel] = (ring_dims, means)

        # Print summary table
        print(f"\n{kernel}:")
        print(f"  {'Ring Dim':>10} | {'Bandwidth (GiB/s)':>20} | {'Count':>6}")
        print("  " + "-" * 40)
        for n, bw in zip(ring_dims, means):
            count = len(dim_dict[n])
            print(f"  {n:>10} | {bw:>20.2f} | {count:>6}")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "DCRT (RS_SEQ_ADD)": "#1f77b4",  # blue
        "CT (CT_SEQ_ADD)": "#ff7f0e",    # orange
    }

    for kernel in sorted(data_by_kernel.keys()):
        ring_dims, means = data_by_kernel[kernel]
        ax.plot(
            ring_dims,
            means,
            marker="o",
            linewidth=2,
            markersize=8,
            label=kernel,
            color=colors.get(kernel, None),
        )

    ax.set_xlabel("Ring Dimension (N)", fontsize=12)
    ax.set_ylabel("Bandwidth (GiB/s)", fontsize=12)
    ax.set_title("Ring Dimension Scaling: SEQ_ADD Bandwidth", fontsize=13)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Set x-axis to show all ring dimensions
    if data_by_kernel:
        first_ring_dims = next(iter(data_by_kernel.values()))[0]
        ax.set_xticks(first_ring_dims)
        ax.set_xticklabels([str(n) for n in first_ring_dims])

    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot ring dimension scaling from merged JSON results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/ring_scaling_seq_add.json"),
        help="Merged JSON file (default: results/ring_scaling_seq_add.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ring_scaling_plot.pdf"),
        help="Output PDF path (default: ring_scaling_plot.pdf)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found")
        return

    plot_ring_scaling(json_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
