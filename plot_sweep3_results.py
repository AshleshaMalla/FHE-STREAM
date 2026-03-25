#!/usr/bin/env python3
"""
Plot NTT vs Scatter/Gather ADD kernel comparison from sweep3 JSON file.
Compares NTT with Scatter ADD (Poly) and Gather ADD (Poly).
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
    ring_dim = None
    depth = None
    
    if data.get('benchmarks'):
        for bench in data['benchmarks']:
            name = bench['run_name']
            kernel = name.split('/')[1]  # Extract kernel name
            mode = int(name.split('/')[4])  # Extract mode
            
            # Extract RingDim and Depth from first benchmark for display
            if ring_dim is None:
                parts = name.split('/')
                ring_dim = int(parts[2])
                depth = int(parts[3])
            
            results.append({
                'kernel': kernel,
                'full_name': name,
                'mode': mode,
                'throughput_gbs': bench['bytes_per_second'] / 1e9,
                'real_time_ms': bench['real_time'],
                'cpu_time_ms': bench['cpu_time'],
                'iterations': bench['iterations'],
            })
    
    return results, ring_dim, depth

def main():
    filepath = Path('results/sweep3_NTT.csv')
    
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    
    results, ring_dim, depth = parse_sweep3_file(filepath)
    
    if not results:
        print("No benchmark data extracted")
        return
    
    # Filter for: NTT (mode 2), Scatter ADD (Poly/mode 1), Gather ADD (Poly/mode 1)
    filtered_results = []
    for r in results:
        kernel = r['kernel']
        mode = r['mode']
        
        # Select specific kernel/mode combinations
        if kernel == 'RS_NTT_ROUNDTRIP' and mode == 2:  # NTT
            filtered_results.append({
            'label': 'NTT',
                'kernel': kernel,
                'mode': mode,
                'throughput_gbs': r['throughput_gbs'],
                'real_time_ms': r['real_time_ms'],
                'iterations': r['iterations'],
            })
        elif kernel == 'RS_SCATTER_ADD' and mode == 1:  # Scatter ADD Poly
            filtered_results.append({
                'label': 'Scatter ADD (Poly)',
                'kernel': kernel,
                'mode': mode,
                'throughput_gbs': r['throughput_gbs'],
                'real_time_ms': r['real_time_ms'],
                'iterations': r['iterations'],
            })
        elif kernel == 'RS_GATHER_ADD' and mode == 1:  # Gather ADD Poly
            filtered_results.append({
                'label': 'Gather ADD (Poly)',
                'kernel': kernel,
                'mode': mode,
                'throughput_gbs': r['throughput_gbs'],
                'real_time_ms': r['real_time_ms'],
                'iterations': r['iterations'],
            })
    
    if not filtered_results:
        print("No matching kernels found. Available data:")
        for r in results:
            print(f"  {r['kernel']} mode {r['mode']}: {r['throughput_gbs']:.2f} GB/s")
        return
    
    # Extract data for charting
    labels = [r['label'] for r in filtered_results]
    throughput = [r['throughput_gbs'] for r in filtered_results]
    real_time = [r['real_time_ms'] for r in filtered_results]
    iterations = [r['iterations'] for r in filtered_results]
    
    # Print results table
    print("\n=== NTT vs Memory Access Pattern Results ===\n")
    print(f"{'Kernel':<25} {'Throughput (GB/s)':<20} {'Real Time (ms)':<18} {'Iterations':<12}")
    print("-" * 75)
    for label, tput, rt, it in zip(labels, throughput, real_time, iterations):
        print(f"{label:<25} {tput:<20.2f} {rt:<18.2f} {it:<12}")
    
    # Calculate performance ratios (baseline = Scatter ADD)
    scatter_add_idx = next((i for i, r in enumerate(filtered_results) if 'Scatter ADD' in r['label']), None)
    if scatter_add_idx is None:
        print("Warning: Scatter ADD not found for baseline comparison")
        scatter_add_throughput = throughput[0]
    else:
        scatter_add_throughput = throughput[scatter_add_idx]
    
    print(f"\n=== Performance vs Scatter ADD Baseline ({scatter_add_throughput:.2f} GB/s) ===\n")
    print(f"{'Kernel':<25} {'vs Scatter ADD':<15}")
    print("-" * 40)
    for label, tput in zip(labels, throughput):
        ratio = tput / scatter_add_throughput
        print(f"{label:<25} {ratio:.2%}")
    
    # Create figure with single bandwidth chart
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'NTT vs Memory Access Patterns (DCRT)\n(RingDim {ring_dim}, Depth {depth})', 
                 fontsize=14, fontweight='bold')
    
    # Distinct color palette
    colors = [
        '#F18F01',  # Scatter ADD (Poly) - orange
        '#2E86AB',  # Gather ADD (Poly) - steelblue
        '#A23B72',  # NTT - purple
    ]
    
    # Throughput chart
    bars = ax.bar(range(len(labels)), throughput, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Throughput (GB/s)', fontweight='bold', fontsize=12)
    ax.set_title('Memory Bandwidth', fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, v in zip(bars, throughput):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(throughput)*0.02, 
                f'{v:.1f} GB/s', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylim(0, max(throughput) * 1.12)
    
    plt.tight_layout()
    plt.savefig('results/sweep3_ntt_vs_memory_patterns.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: results/sweep3_ntt_vs_memory_patterns.png")
    
    plt.show()

if __name__ == '__main__':
    main()
