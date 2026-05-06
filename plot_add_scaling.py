#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from fhe_plot_style import (
    configure_matplotlib,
    save_plot,
    COLOR_DCRT,
    COLOR_CT,
    FIGSIZE_SINGLE,
)

def load_scaling(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    dcrt = data.get("dcrt", {})
    ct = data.get("ct", {})
    keys = set()
    for src in (dcrt, ct):
        keys.update(int(k) for k in src.keys())
    threads = sorted(k for k in keys if k > 0)
    if not threads:
        raise RuntimeError("No thread-count entries found in input JSON")
    def extract_bandwidth_map(src):
        m = {}
        for t in threads:
            entry = src.get(str(t)) or src.get(t)
            if entry is None:
                m[t] = float("nan"); continue
            bw = entry.get("PayloadBandwidth") or entry.get("payload_bandwidth") or entry.get("bytes_per_second")
            m[t] = float(bw) if bw is not None else float("nan")
        return m
    dcrt_map = extract_bandwidth_map(dcrt)
    ct_map = extract_bandwidth_map(ct)
    n = l = None
    for src in (dcrt, ct):
        for v in src.values():
            if isinstance(v, dict) and "label" in v:
                lbl = v.get("label","")
                try:
                    parts = [p.strip() for p in lbl.split()]
                    for p in parts:
                        if p.startswith("N="): n = int(p.split("=",1)[1])
                        if p.startswith("L="): l = int(p.split("=",1)[1])
                except Exception:
                    pass
                break
        if n and l: break
    return threads, dcrt_map, ct_map, n, l

def plot_bandwidth(threads, dcrt_map, ct_map, n, l, out: Path):
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    t_arr = np.array(threads)
    def to_gibs(m):
        return [ (m.get(t) / (1024.0**3)) if (m.get(t) is not None and not math.isnan(m.get(t))) else float("nan") for t in threads ]
    dcrt_gbs = to_gibs(dcrt_map)
    ct_gbs = to_gibs(ct_map)
    ax.plot(t_arr, dcrt_gbs, marker="o", linestyle="-", color=COLOR_DCRT, label="DCRT (RS)")
    ax.plot(t_arr, ct_gbs, marker="s", linestyle="-", color=COLOR_CT, label="CT (CTFixture)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(t_arr)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{int(val)}"))
    ax.set_xlabel("Threads")
    ax.set_ylabel("Bandwidth (GiB/s)")
    title = "ADD Kernel: Bandwidth vs Threads"
    if n and l:
        title += f" (N={n}, L={l})"
    ax.set_title(title, fontweight="bold")

    # Calculate y-axis max for headroom and label positioning
    y_vals = [v for v in dcrt_gbs + ct_gbs if not math.isnan(v)]
    ymax = max(y_vals) if y_vals else 0
    ax.set_ylim(0, ymax * 1.15)
    ax.margins(x=0.07)

    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)

    # Manual labels removed for 1-column IEEE layout to avoid clutter
    plt.tight_layout()
    save_plot(str(out))
    plt.close(fig)
    print(f"Saved bandwidth figure to {out}")

def main():
    p = argparse.ArgumentParser(description="Plot ADD scaling bandwidth from merged JSON results.")
    p.add_argument("--input", type=Path, default=Path("results/add_scaling_results.json"))
    p.add_argument("--output", type=Path, default=Path("plots/add_scaling_bandwidth.pdf"))
    args = p.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    threads, dcrt_map, ct_map, n, l = load_scaling(args.input)
    plot_bandwidth(threads, dcrt_map, ct_map, n, l, args.output)

if __name__ == "__main__":
    main()
