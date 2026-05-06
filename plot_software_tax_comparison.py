#!/usr/bin/env python3
"""Compare Software Tax results: DCRT baseline vs CT adjusted.

Creates a publication-quality grouped bar chart of bandwidth (GiB/s)
for each kernel, comparing the DCRT (RS) baseline to the CT (CTFixture)
measurements. Also plots the CT/DCRT ratio on a secondary axis.
"""

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
    COLOR_DCRT,
    COLOR_CT,
    FIGSIZE_SINGLE,
    HATCH_PATTERNS,
)


OP_ORDER = ["COPY", "ADD", "ADD_INPLACE"]


def extract_bandwidth(b: dict) -> float | None:
    # Prefer explicit PayloadBandwidth if present, else bytes_per_second
    for key in ("PayloadBandwidth", "payload_bandwidth", "bytes_per_second"):
        v = b.get(key)
        if v is not None:
            return float(v) / (1024.0**3)
    return None


def parse_operation(name: str) -> tuple[str, str] | None:
    # Returns (class, operation) where class is 'DCRT' or 'CT'
    if name.startswith("FHERaiderSTREAM/") or name.startswith("RS_"):
        # examples: FHERaiderSTREAM/RS_SEQ_COPY/..., FHERaiderSTREAM/RS_SEQ_ADD/...
        m = re.search(r"RS_[A-Z]+_([A-Z]+)", name)
        if m:
            return "DCRT", m.group(1)
    if name.startswith("CTFixture/"):
        m = re.search(r"CT_SEQ_([A-Z_]+)", name)
        if m:
            return "CT", m.group(1)
    return None


def load_data(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    groups = defaultdict(list)

    for b in data.get("benchmarks", []):
        name = b.get("name") or b.get("run_name") or ""
        parsed = parse_operation(name)
        if parsed is None:
            continue
        cls, op = parsed
        bw = extract_bandwidth(b)
        if bw is None:
            continue
        groups[(cls, op)].append(bw)

    summary = {}
    for (cls, op), vals in groups.items():
        summary[(cls, op)] = {"mean_gbs": mean(vals), "count": len(vals)}

    return summary


def plot_comparison(summary: dict, out: Path):
    configure_matplotlib()

    labels = OP_ORDER
    x = list(range(len(labels)))
    width = 0.35

    dcrt_vals = [summary.get(("DCRT", op), {}).get("mean_gbs", float("nan")) for op in labels]
    ct_vals = [summary.get(("CT", op), {}).get("mean_gbs", float("nan")) for op in labels]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bars1 = ax.bar([i - width / 2 for i in x], dcrt_vals, width=width, color=COLOR_DCRT, label="DCRT (RS)", edgecolor="black", linewidth=0.6, hatch=HATCH_PATTERNS[0])
    bars2 = ax.bar([i + width / 2 for i in x], ct_vals, width=width, color=COLOR_CT, label="CT (CTFixture)", edgecolor="black", linewidth=0.6, hatch=HATCH_PATTERNS[1])

    add_value_labels(ax, bars1)
    add_value_labels(ax, bars2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Kernel")
    ax.set_ylabel("Bandwidth (GiB/s)")
    ax.set_title("Software Tax Comparison", fontweight="bold")

    # Small interior legend top-left
    ax.legend(loc="upper left", frameon=False)

    plt.tight_layout()
    save_plot(str(out))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot software tax comparison from Google Benchmark JSON.")
    parser.add_argument("--input", type=Path, default=Path("results/software_tax_results.json"), help="Input JSON file")
    parser.add_argument("--output", type=Path, default=Path("plots/software_tax_comparison.pdf"), help="Output PDF file")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    summary = load_data(args.input)
    if not summary:
        raise RuntimeError("No data parsed from input JSON")

    print("Parsed summary entries:", len(summary))
    plot_comparison(summary, args.output)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
