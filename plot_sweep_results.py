#!/usr/bin/env python3
"""
Plot core scaling benchmark results from sweep1 CSV files.
Visualizes throughput and execution time vs thread count.
"""

import json
import glob
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_sweep_file(filepath):
    """Extract thread count and metrics from a sweep CSV file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract thread count from filename: sweep1_core_scaling_t<num>.csv
    filename = Path(filepath).name
    match = re.search(r't(\d+)', filename)
    threads = int(match.group(1)) if match else None
    
    # Extract benchmark metrics
    if data.get('benchmarks'):
        bench = data['benchmarks'][0]
        return {
            'threads': threads,
            'filename': filename,
            'throughput_gbs': bench['bytes_per_second'] / 1e9,  # Convert to GB/s
            'real_time_ms': bench['real_time'],
            'cpu_time_ms': bench['cpu_time'],
            'iterations': bench['iterations'],
            'mhz': data['context']['mhz_per_cpu'],
        }
    return None

def main():
    # Find all sweep1 CSV files
    results_dir = Path('results')
    files = sorted(glob.glob(str(results_dir / 'sweep1_core_scaling_t*.csv')))
    
    if not files:
        print(f"No sweep1 CSV files found in {results_dir}")
        return
    
    # Parse all files
    results = []
    for f in files:
        data = parse_sweep_file(f)
        if data:
            results.append(data)
    
    if not results:
        print("No benchmark data extracted")
        return
    
    # Sort by thread count
    results.sort(key=lambda x: x['threads'])
    
    # Extract data for plotting
    threads = [r['threads'] for r in results]
    throughput = [r['throughput_gbs'] for r in results]
    real_time = [r['real_time_ms'] for r in results]
    iterations = [r['iterations'] for r in results]
    
    # Calculate speedup relative to single thread
    speedup = [t / throughput[0] for t in throughput]
    ideal_speedup = [t / threads[0] for t in threads]
    
    print("\n=== Core Scaling Results ===\n")
    print(f"{'Threads':<10} {'Throughput (GB/s)':<20} {'Real Time (ms)':<18} {'Iterations':<12} {'Speedup':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['threads']:<10} {r['throughput_gbs']:<20.2f} {r['real_time_ms']:<18.2f} {r['iterations']:<12} {speedup[results.index(r)]:<10.2f}x")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Core Scaling Benchmark Results (RS_SEQ_ADD/65536/20)', fontsize=14, fontweight='bold')
    
    # Plot 1: Throughput vs Threads
    ax = axes[0, 0]
    ax.bar(range(len(threads)), throughput, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Throughput (GB/s)')
    ax.set_title('Memory Bandwidth vs Thread Count')
    ax.set_xticks(range(len(threads)))
    ax.set_xticklabels(threads)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(throughput):
        ax.text(i, v + max(throughput)*0.02, f'{v:.1f}', ha='center', fontweight='bold')
    
    # Plot 2: Real Time vs Threads
    ax = axes[0, 1]
    ax.bar(range(len(threads)), real_time, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Time per Iteration vs Thread Count')
    ax.set_xticks(range(len(threads)))
    ax.set_xticklabels(threads)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(real_time):
        ax.text(i, v + max(real_time)*0.02, f'{v:.1f}', ha='center', fontweight='bold')
    
    # Plot 3: Speedup
    ax = axes[1, 0]
    ax.plot(threads, speedup, 'o-', linewidth=2.5, markersize=8, label='Actual Speedup', color='green')
    ax.plot(threads, ideal_speedup, 's--', linewidth=2.5, markersize=8, label='Ideal (Linear)', color='red')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Speedup (relative to 1 thread)')
    ax.set_title('Speedup vs Thread Count')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot 4: Iterations (shows fairness in time budget)
    ax = axes[1, 1]
    ax.bar(range(len(threads)), iterations, color='mediumpurple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Number of Iterations')
    ax.set_title('Iterations per Benchmark (Time Budget Fairness)')
    ax.set_xticks(range(len(threads)))
    ax.set_xticklabels(threads)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(iterations):
        ax.text(i, v + max(iterations)*0.02, f'{v}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/sweep1_core_scaling_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: results/sweep1_core_scaling_analysis.png")
    
    plt.show()

if __name__ == '__main__':
    main()
