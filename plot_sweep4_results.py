#!/usr/bin/env python3
"""
Plot Keyswitch Mock vs Triad kernel comparison from sweep4 JSON file.
Compares keyswitch mock with sequential triad baseline.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_sweep4_file(filepath):
    """Extract benchmark metrics from sweep4 JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    results = []
    if data.get('benchmarks'):
        for bench in data['benchmarks']:
            name = bench['run_name']
            kernel = name.split('/')[1]  # Extract kernel name
            
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
    filepath = Path('results/sweep4_keyswitchvsTriad.csv')
    
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    
    results = parse_sweep4_file(filepath)
    
    if not results:
        print("No benchmark data extracted")
        return
    
    # Extract data and format labels
    labels = []
    throughput = []
    real_time = []
    iterations = []
    
    for r in results:
        kernel = r['kernel']
        
        if kernel == 'RS_SEQ_TRIAD':
            label = 'SEQ_TRIAD'
        elif kernel == 'RS_KEYSWITCH_MOCK':
            label = 'KEYSWITCH_MOCK'
        else:
            label = kernel
        
        labels.append(label)
        throughput.append(r['throughput_gbs'])
        real_time.append(r['real_time_ms'])
        iterations.append(r['iterations'])
    
    # Print results table
    print("\n=== Keyswitch Mock vs Triad Kernel Results ===\n")
    print(f"{'Kernel':<25} {'Throughput (GB/s)':<20} {'Real Time (ms)':<18} {'Iterations':<12}")
    print("-" * 75)
    for label, tput, rt, it in zip(labels, throughput, real_time, iterations):
        print(f"{label:<25} {tput:<20.2f} {rt:<18.2f} {it:<12}")
    
    # Calculate performance ratios
    seq_triad_throughput = throughput[0]  # SEQ_TRIAD baseline
    print(f"\n=== Performance vs SEQ_TRIAD Baseline ({seq_triad_throughput:.2f} GB/s) ===\n")
    print(f"{'Kernel':<25} {'vs SEQ_TRIAD':<15}")
    print("-" * 40)
    for label, tput in zip(labels, throughput):
        ratio = tput / seq_triad_throughput
        print(f"{label:<25} {ratio:.2%}")
    
    # Create figure with single bandwidth chart
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Keyswitch Mock vs Sequential Triad Kernel (DCRT)\n(RingDim 65536, Depth 20)', 
                 fontsize=14, fontweight='bold')
    
    # Distinct color palette
    colors = [
        '#2E86AB',  # SEQ_TRIAD - steelblue
        '#F18F01',  # KEYSWITCH_MOCK - orange
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
    plt.savefig('results/sweep4_keyswitch_vs_triad_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: results/sweep4_keyswitch_vs_triad_analysis.png")
    
    plt.show()

if __name__ == '__main__':
    main()
