#!/usr/bin/env python3
"""
Plot memory access pattern benchmark results from sweep2 CSV file.
Compares different kernel access patterns: Sequential, Gather, Scatter, Scatter-Gather.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_sweep2_file(filepath):
    """Extract benchmark metrics from sweep2 CSV file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    results = []
    if data.get('benchmarks'):
        for bench in data['benchmarks']:
            # Extract kernel name from benchmark name
            # Format: FHERaiderSTREAM/RS_*_ADD/65536/5/100
            name = bench['run_name']
            kernel = name.split('/')[1]  # Extract RS_*_ADD
            
            results.append({
                'kernel': kernel,
                'full_name': name,
                'throughput_gbs': bench['bytes_per_second'] / 1e9,
                'real_time_ms': bench['real_time'],
                'cpu_time_ms': bench['cpu_time'],
                'iterations': bench['iterations'],
            })
    
    return results

def main():
    filepath = Path('results/sweep2_memory_access_pattern.csv')
    
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    
    results = parse_sweep2_file(filepath)
    
    if not results:
        print("No benchmark data extracted")
        return
    
    # Extract data
    kernels = [r['kernel'] for r in results]
    throughput = [r['throughput_gbs'] for r in results]
    real_time = [r['real_time_ms'] for r in results]
    iterations = [r['iterations'] for r in results]
    
    # Calculate overhead relative to sequential
    seq_throughput = throughput[0]
    overhead_pct = [(seq_throughput - t) / seq_throughput * 100 for t in throughput]
    throughput_ratio = [t / seq_throughput for t in throughput]
    
    print("\n=== Memory Access Pattern Results ===\n")
    print(f"{'Kernel':<25} {'Throughput (GB/s)':<20} {'Time (ms)':<15} {'Iterations':<12} {'vs Sequential':<15}")
    print("-" * 90)
    for r in results:
        idx = kernels.index(r['kernel'])
        ratio = throughput_ratio[idx]
        overhead = overhead_pct[idx]
        print(f"{r['kernel']:<25} {r['throughput_gbs']:<20.2f} {r['real_time_ms']:<15.4f} {r['iterations']:<12} {ratio:+.2%} ({overhead:+.1f}%)")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Memory Access Pattern Comparison (RingDim 65536, Depth 5, Batch 100)', 
                 fontsize=14, fontweight='bold')
    
    # Define colors for better visualization
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Plot 1: Throughput comparison
    ax = axes[0, 0]
    bars = ax.bar(kernels, throughput, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Throughput (GB/s)', fontweight='bold')
    ax.set_title('Memory Bandwidth per Kernel')
    ax.grid(axis='y', alpha=0.3)
    # Rotate labels for better readability
    ax.set_xticklabels(kernels, rotation=45, ha='right')
    for i, (bar, v) in enumerate(zip(bars, throughput)):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(throughput)*0.02, 
                f'{v:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 2: Execution time per iteration
    ax = axes[0, 1]
    bars = ax.bar(kernels, real_time, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Time per Iteration (ms)', fontweight='bold')
    ax.set_title('Execution Time per Iteration')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(kernels, rotation=45, ha='right')
    for i, (bar, v) in enumerate(zip(bars, real_time)):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(real_time)*0.02, 
                f'{v:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 3: Performance relative to Sequential
    ax = axes[1, 0]
    colors_ratio = ['green' if r == kernels[0] else '#FFA500' if r > 0.9 else '#FF6B6B' 
                    for r in throughput_ratio]
    bars = ax.barh(kernels, throughput_ratio, color=colors_ratio, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Throughput Relative to Sequential', fontweight='bold')
    ax.set_title('Performance vs Sequential Baseline')
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Sequential Baseline')
    ax.grid(axis='x', alpha=0.3)
    ax.legend()
    for i, (bar, v) in enumerate(zip(bars, throughput_ratio)):
        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{v:.2%}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    # Plot 4: Overhead percentage
    ax = axes[1, 1]
    colors_overhead = ['green' if o == 0 else 'red' for o in overhead_pct]
    bars = ax.bar(kernels, overhead_pct, color=colors_overhead, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Overhead (%)', fontweight='bold')
    ax.set_title('Performance Overhead vs Sequential')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(kernels, rotation=45, ha='right')
    for i, (bar, v) in enumerate(zip(bars, overhead_pct)):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(abs(v) for v in overhead_pct)*0.02 if v >= 0 else v - max(abs(v) for v in overhead_pct)*0.02, 
                f'{v:.1f}%', ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/sweep2_memory_access_pattern_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: results/sweep2_memory_access_pattern_analysis.png")
    
    plt.show()

if __name__ == '__main__':
    main()
