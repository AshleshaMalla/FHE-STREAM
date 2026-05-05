#!/usr/bin/env python3
"""Merge per-ring-dimension JSON results into a single combined JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_ring_scaling_results(
    ring_sizes: list[int],
    results_dir: Path,
    output_path: Path,
) -> None:
    """
    Merge results from individual ring-dimension benchmark runs.

    Args:
        ring_sizes: List of ring dimensions (e.g., [16384, 32768, 65536, 131072])
        results_dir: Directory containing ring_scaling_n{N}.json files
        output_path: Path to write merged JSON
    """
    all_benchmarks = []

    for n in ring_sizes:
        input_file = results_dir / f"ring_scaling_n{n}.json"

        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping N={n}")
            continue

        print(f"Loading {input_file}...")
        with input_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        benchmarks = data.get("benchmarks", [])
        print(f"  Found {len(benchmarks)} benchmarks for N={n}")
        all_benchmarks.extend(benchmarks)

    # Create merged output
    merged_data = {
        "benchmarks": all_benchmarks,
        "context": {
            "ring_sizes": ring_sizes,
            "total_benchmarks": len(all_benchmarks),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(merged_data, handle, indent=2)

    print(f"\nMerged {len(all_benchmarks)} benchmarks into {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-ring-dimension JSON results into single file."
    )
    parser.add_argument(
        "--ring-sizes",
        type=int,
        nargs="+",
        default=[16384, 32768, 65536, 131072],
        help="Ring dimensions to merge (default: 16384 32768 65536 131072)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing ring_scaling_n{N}.json files (default: results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ring_scaling_seq_add.json"),
        help="Output JSON file (default: results/ring_scaling_seq_add.json)",
    )

    args = parser.parse_args()

    merge_ring_scaling_results(
        ring_sizes=args.ring_sizes,
        results_dir=args.results_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
