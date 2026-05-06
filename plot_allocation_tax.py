#!/usr/bin/env python3
"""Plot Allocation Tax: Memory footprint expansion from degree-1 to degree-2.

Shows bandwidth for CT_SEQ_MULT_NO_RELIN (degree-1 → degree-2 multiplication)
and CT_SEQ_RELIN (degree-2 → degree-1 relinearization), with RSS footprint
annotations to illustrate memory expansion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_matplotlib():
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    serif_fonts = ["Times New Roman"] if "Times New Roman" in available_fonts else []
    serif_fonts.extend(["Times", "DejaVu Serif", "Liberation Serif", "serif"])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif_fonts,
            "font.size": 10,
            "text.color": "black",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )


def extract_bandwidth(b: dict) -> float | None:
    for key in ("PayloadBandwidth", "payload_bandwidth", "bytes_per_second"):
        v = b.get(key)
        if v is not None:
            return float(v) / (1024.0**3)
    return None


def load_data(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    
    for b in data.get("benchmarks", []):
        name = b.get("name") or b.get("run_name") or ""
        bw = extract_bandwidth(b)
        
        if "CT_SEQ_MULT_NO_RELIN" in name and bw is not None:
            results["MULT_NO_RELIN"] = {
                "bw_gbs": bw,
                "rss_deg1": b.get("RSSDeg1BytesPerCt", 0),
                "rss_deg2": b.get("RSSDeg2BytesPerCt", 0),
                "serialized_deg2": b.get("SerializedDeg2BytesPerCt", b.get("SerializedDeg2BytesPerCt", 0)),
            }
        elif "CT_SEQ_RELIN" in name and bw is not None:
            results["RELIN"] = {
                "bw_gbs": bw,
                "rss_deg1": b.get("RSSDeg1BytesPerCt", 0),
                "rss_deg2": b.get("RSSDeg2BytesPerCt", 0),
                "serialized": b.get("SerializedBytesPerCt", 0),
            }
    
    return results


def plot_allocation_tax(results: dict, out: Path, N: int = 131072, L: int = 40):
    configure_matplotlib()

    groups = ["MULT_NO_RELIN", "RELIN"]

    # Mathematical payload per RNS tensor (bytes)
    math_payload_bytes = int(N) * int(L) * 8
    math_payload_gb = math_payload_bytes / 1e9

    # Gather serialized and rss sizes (bytes) and convert to GB
    mult_serialized_gb = results.get("MULT_NO_RELIN", {}).get("serialized_deg2", 0) / 1e9
    mult_rss_gb = results.get("MULT_NO_RELIN", {}).get("rss_deg2", 0) / 1e9

    relin_serialized_gb = results.get("RELIN", {}).get("serialized", 0) / 1e9
    relin_rss_gb = results.get("RELIN", {}).get("rss_deg1", 0) / 1e9

    # Values per group for the three bars: [math, serialized, rss]
    math_vals = [math_payload_gb, math_payload_gb]
    serialized_vals = [mult_serialized_gb, relin_serialized_gb]
    rss_vals = [mult_rss_gb, relin_rss_gb]

    x = list(range(len(groups)))
    width = 0.2

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # Colors chosen to be consistent with other plots
    c_math = "#999999"
    c_serial = "#ff7f00"
    c_rss = "#377eb8"

    # Small horizontal shift for the first group to avoid the top-left legend overlap
    group_offsets = [0.15, 0.0]

    pos_math = [i - width + group_offsets[idx] for idx, i in enumerate(x)]
    pos_serial = [i + group_offsets[idx] for idx, i in enumerate(x)]
    pos_rss = [i + width + group_offsets[idx] for idx, i in enumerate(x)]

    bars_math = ax.bar(pos_math, math_vals, width=width, color=c_math, edgecolor="black", linewidth=0.6, label="Mathematical Payload", zorder=3)
    bars_serial = ax.bar(pos_serial, serialized_vals, width=width, color=c_serial, edgecolor="black", linewidth=0.6, label="Serialized Size", zorder=3)
    bars_rss = ax.bar(pos_rss, rss_vals, width=width, color=c_rss, edgecolor="black", linewidth=0.6, label="RSS (Resident)", zorder=3)

    tick_positions = [i + group_offsets[idx] for idx, i in enumerate(x)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Size (GB)")
    ax.set_xlabel("Operation")
    ax.set_title("Estimated vs Serialized vs Resident (RSS) Payload", fontweight="bold")

    ymax = max((v for v in math_vals + serialized_vals + rss_vals if v == v), default=0.0)
    ax.set_ylim(0, ymax * 1.15)

    # Add value labels on bars
    for bar in (list(bars_math) + list(bars_serial) + list(bars_rss)):
        height = bar.get_height()
        label_y = min(height + ymax * 0.018, ymax * 1.15 - ymax * 0.03)
        ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                f"{height:.2f}",
                ha="center", va="bottom", fontsize=10, zorder=4)

    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot allocation tax from RSS expansion JSON.")
    parser.add_argument("--input", type=Path, default=Path("results/rss_expansion_results.json"),
                        help="Input JSON file")
    parser.add_argument("--output", type=Path, default=Path("allocation_tax.pdf"),
                        help="Output PDF file")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    results = load_data(args.input)
    if not results:
        raise RuntimeError("No allocation tax data parsed from input JSON")

    plot_allocation_tax(results, args.output)


if __name__ == "__main__":
    main()
