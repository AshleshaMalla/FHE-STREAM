#!/usr/bin/env python3
"""
Plot Ciphertext ADD vs Sequential ADD kernel comparison from sweep5 JSON file.
Compares CT_SEQ_ADD and CT_SEQ_ADD_INPLACE with DCRT sequential ADD baseline.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_sweep5_file(filepath):
    """Extract benchmark metrics from sweep5 JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    results = []
    ring_dim = None
    depth = None
    
    if data.get('benchmarks'):
        for bench in data['benchmarks']:
            name = bench['run_name']
            kernel = name.split('/')[1]  # Extract kernel name
            mode = int(name.split('/')[4]) if len(name.split('/')) > 4 else 0  # Extract mode
            
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
    filepath = Path('results/sweep5_CT.csv')
    
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    
    results, ring_dim, depth = parse_sweep5_file(filepath)
    
    if not results:
        print("No benchmark data extracted")
        return
    
    # Extract data and format labels (only keep M1 for CT kernels)
    labels = []
    throughput = []
    real_time = []
    iterations = []
    
    for r in results:
        kernel = r['kernel']
        mode = r['mode']
        
        # Skip CT kernels unless they are Mode 1
        if kernel.startswith('CT_') and mode != 1:
            continue
        
        if kernel == 'RS_SEQ_ADD':
            label = 'SEQ_ADD'
        elif kernel == 'CT_SEQ_ADD':
            label = 'CT_ADD'
        elif kernel == 'CT_SEQ_ADD_INPLACE':
            label = 'CT_ADD_IP'
        else:
            label = kernel
        
        labels.append(label)
        throughput.append(r['throughput_gbs'])
        real_time.append(r['real_time_ms'])
        iterations.append(r['iterations'])
    
    # Determine plaintext modulus based on ring dimension
    pt_mod = 65537 if ring_dim <= 32768 else 786433
    default_batch = 100  # Default batch size from CTFixture.cpp
    
    # Print results table
    print("\n=== Ciphertext ADD vs Sequential ADD Results ===\n")
    print(f"{'Kernel':<30} {'Throughput (GB/s)':<20} {'Real Time (ms)':<18} {'Iterations':<12}")
    print("-" * 80)
    for label, tput, rt, it in zip(labels, throughput, real_time, iterations):
        print(f"{label:<30} {tput:<20.2f} {rt:<18.2f} {it:<12}")
    
    print(f"\nPlaintextModulus: {pt_mod}")
    print(f"Batch Size: {default_batch}")
    print(f"\nNote: Ciphertext backend does not implement shuffle modes;")
    print(f"      all mode variants would produce identical results.")
    
    # Calculate performance ratios
    seq_add_throughput = throughput[0]  # SEQ_ADD baseline
    print(f"\n=== Performance vs SEQ_ADD Baseline ({seq_add_throughput:.2f} GB/s) ===\n")
    print(f"{'Kernel':<30} {'vs SEQ_ADD':<15}")
    print("-" * 45)
    for label, tput in zip(labels, throughput):
        ratio = tput / seq_add_throughput
        print(f"{label:<30} {ratio:.2%}")
    
    # Create figure with single bandwidth chart
    fig, ax = plt.subplots(figsize=(14, 6))
    
    fig.suptitle(f'Ciphertext ADD vs Ciphertext ADD In-Place vs DCRT Sequential ADD\n(RingDim {ring_dim}, Depth {depth}, PlaintextModulus {pt_mod}, Batch {default_batch})', 
                 fontsize=13, fontweight='bold')
    
    # Distinct color palette for 3 kernels
    color_map = {
        'SEQ_ADD': '#2E86AB',           # steelblue (baseline)
        'CT_ADD': '#A23B72',             # purple
        'CT_ADD_IP': '#F18F01',          # orange
    }
    
    colors = [color_map.get(label, '#666666') for label in labels]
    
    # Throughput chart
    bars = ax.bar(range(len(labels)), throughput, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Throughput (GB/s)', fontweight='bold', fontsize=12)
    ax.set_title('Memory Bandwidth', fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add top margin to prevent label cutoff
    ax.set_ylim(0, max(throughput) * 1.12)
    
    for bar, v in zip(bars, throughput):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(throughput)*0.02, 
                f'{v:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/sweep5_ciphertext_add_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: results/sweep5_ciphertext_add_analysis.png")
    
    plt.show()

if __name__ == '__main__':
    main()
