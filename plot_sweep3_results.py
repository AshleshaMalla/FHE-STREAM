#!/usr/bin/env python3
"""
Plot NTT vs SEQ kernel comparison from sweep3 JSON file.
Compares NTT roundtrip with sequential copy and scale baselines.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_sweep3_file(filepath):
    """Extract benchmark metrics from sweep3 JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    results = []
    if data.get('benchmarks'):
        for bench in data['benchmarks']:
            name = bench['run_name']
            kernel = name.split('/')[1]  # Extract kernel name
            mode = int(name.split('/')[4])  # Extract mode
            
            results.append({
                'kernel': kernel,
                'full_name': name,
                'mode': mode,
                'throughput_gbs': bench['bytes_per_second'] / 1e9,
                'real_time_ms': bench['real_time'],
                'cpu_time_ms': bench['cpu_time'],
                'iterations': bench['iterations'],
            })
    
    return results

def main():
    filepath = Path('results/sweep3_NTTvsSEQ.csv')
    
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    
    results = parse_sweep3_file(filepath)
    
    if not results:
        print("No benchmark data extracted")
        return
    
    # Extract data and format labels (skip Mode 0 and Coeff)
    labels = []
    throughput = []
    real_time = []
    iterations = []
    
    for r in results:
        kernel = r['kernel']
        mode = r['mode']
        
        # Skip NTT Mode 0 and NTT Coeff
        if kernel == 'RS_NTT_ROUNDTRIP' and (mode == 0 or mode == 2):
            continue
        
        if kernel == 'RS_SEQ_COPY':
            label = 'SEQ_COPY'
        elif kernel == 'RS_SEQ_SCALE':
            label = 'SEQ_SCALE'
        elif kernel == 'RS_NTT_ROUNDTRIP':
            if mode == 1:
                label = 'NTT'
            else:
                label = 'NTT1'
        else:
            label = kernel
        
        labels.append(label)
        throughput.append(r['throughput_gbs'])
        real_time.append(r['real_time_ms'])
        iterations.append(r['iterations'])
    
    # Print results table
    print("\n=== NTT vs Sequential Kernel Results ===\n")
    print(f"{'Kernel':<25} {'Throughput (GB/s)':<20} {'Real Time (ms)':<18} {'Iterations':<12}")
    print("-" * 75)
    for label, tput, rt, it in zip(labels, throughput, real_time, iterations):
        print(f"{label:<25} {tput:<20.2f} {rt:<18.2f} {it:<12}")
    
    # Calculate performance ratios
    seq_copy_throughput = throughput[0]  # SEQ_COPY baseline
    print(f"\n=== Performance vs SEQ_COPY Baseline ({seq_copy_throughput:.2f} GB/s) ===\n")
    print(f"{'Kernel':<25} {'vs SEQ_COPY':<15}")
    print("-" * 40)
    for label, tput in zip(labels, throughput):
        ratio = tput / seq_copy_throughput
        print(f"{label:<25} {ratio:.2%}")
    
    # Create figure with single bandwidth chart
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('NTT Roundtrip vs Sequential Kernels (DCRT)\n(RingDim 131072, Depth 40)', 
                 fontsize=14, fontweight='bold')
    
    # Distinct color palette
    colors = [
        '#2E86AB',  # SEQ_COPY - steelblue
        '#A23B72',  # SEQ_SCALE - purple
        '#F18F01',  # NTT (Poly) - orange
    ]
    
    # Throughput chart
    bars = ax.bar(range(len(labels)), throughput, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Throughput (GB/s)', fontweight='bold', fontsize=12)
    ax.set_title('Memory Bandwidth', fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, v in zip(bars, throughput):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(throughput)*0.02, 
                f'{v:.1f} GB/s', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('results/sweep3_ntt_vs_seq_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: results/sweep3_ntt_vs_seq_analysis.png")
    
    plt.show()

if __name__ == '__main__':
    main()
