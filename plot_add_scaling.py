#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

def configure_matplotlib():
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
        "lines.linewidth": 2.4,
        "text.color": "black",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })

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
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    t_arr = np.array(threads)
    def to_gibs(m):
        return [ (m.get(t) / (1024.0**3)) if (m.get(t) is not None and not math.isnan(m.get(t))) else float("nan") for t in threads ]
    dcrt_gbs = to_gibs(dcrt_map)
    ct_gbs = to_gibs(ct_map)
    ax.plot(t_arr, dcrt_gbs, marker="o", linestyle="-", color="#377eb8", label="DCRT (RS)")
    ax.plot(t_arr, ct_gbs, marker="s", linestyle="-", color="#ff7f00", label="CT (CTFixture)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(t_arr)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{int(val)}"))
    ax.set_xlabel("Threads", fontweight="bold")
    ax.set_ylabel("Bandwidth (GiB/s)", fontweight="bold")
    title = "ADD Kernel: Bandwidth vs Threads"
    if n and l:
        title += f"\n(N={n}, L={l})"
    ax.set_title(title, fontweight="bold")

    # Calculate y-axis max for headroom and label positioning
    y_vals = [v for v in dcrt_gbs + ct_gbs if not math.isnan(v)]
    ymax = max(y_vals) if y_vals else 0
    ax.set_ylim(0, ymax * 1.15)
    ax.margins(x=0.07)

    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=12)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    plt.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bandwidth figure to {out}")

def main():
    p = argparse.ArgumentParser(description="Plot ADD scaling bandwidth from merged JSON results.")
    p.add_argument("--input", type=Path, default=Path("results/add_scaling_results.json"))
    p.add_argument("--ct-override", type=str, default=None,
                   help="Override CT PayloadBandwidth values (GB/s) as comma list e.g. '1:32.9,4:129.9'")
    p.add_argument("--output", type=Path, default=Path("add_scaling_bandwidth.pdf"))
    args = p.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    threads, dcrt_map, ct_map, n, l = load_scaling(args.input)
    # Apply CT overrides if requested. Overrides are provided in GB/s (1 GB = 1e9 bytes).
    if args.ct_override:
        try:
            pairs = [p.strip() for p in args.ct_override.split(',') if p.strip()]
            for pair in pairs:
                tstr, valstr = pair.split(':')
                t = int(tstr)
                gb = float(valstr)
                # store as bytes/sec so plotting converts to GiB/s consistently
                ct_map[t] = gb * 1e9
            print(f"Applied CT overrides (GB/s) -> threads: {args.ct_override}")
        except Exception as e:
            raise SystemExit(f"Failed to parse --ct-override: {e}")
    plot_bandwidth(threads, dcrt_map, ct_map, n, l, args.output)

if __name__ == "__main__":
    main()