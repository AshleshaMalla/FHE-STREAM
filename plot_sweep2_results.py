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
    
    # Extract data and format labels
    labels = []
    throughput = []
    
    for r in results:
        full_name = r['full_name']
        # Parse kernel and mode from run_name
        # Format: FHERaiderSTREAM/RS_*_ADD/131072/5/mode
        parts = full_name.split('/')
        kernel = parts[1]  # RS_SEQ_ADD, RS_GATHER_ADD, etc.
        mode = int(parts[4])  # 0, 1 (Poly), 2 (Coeff)
        
        if kernel == 'RS_SEQ_ADD':
            label = 'Sequential'
        elif kernel == 'RS_GATHER_ADD':
            label = f'Gather {"(Poly)" if mode == 1 else "(Coeff)"}'
        elif kernel == 'RS_SCATTER_ADD':
            label = f'Scatter {"(Poly)" if mode == 1 else "(Coeff)"}'
        elif kernel == 'RS_SCATTER_GATHER_ADD':
            label = f'Scatter-Gather {"(Poly)" if mode == 1 else "(Coeff)"}'
        else:
            label = kernel
        
        labels.append(label)
        throughput.append(r['throughput_gbs'])
    
    # Print results table
    print("\n=== Memory Access Pattern Results ===\n")
    print(f"{'Access Pattern':<30} {'Throughput (GB/s)':<20} {'vs Sequential':<15}")
    print("-" * 65)
    seq_throughput = throughput[0]
    for label, tput in zip(labels, throughput):
        ratio = tput / seq_throughput
        overhead = (seq_throughput - tput) / seq_throughput * 100
        print(f"{label:<30} {tput:<20.2f} {ratio:.2%} ({overhead:+.1f}%)")
    
    # Create single bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Distinct color palette for each access pattern
    colors = [
        '#2E86AB',  # Sequential - steelblue
        '#A23B72',  # Gather (Poly) - purple
        '#F18F01',  # Gather (Coeff) - orange
        '#C73E1D',  # Scatter (Poly) - red
        '#408C7D',  # Scatter (Coeff) - teal
        '#6B4C9A',  # Scatter-Gather (Poly) - violet
        '#D4A574',  # Scatter-Gather (Coeff) - tan
    ]
    
    # Create bar chart
    bars = ax.bar(range(len(labels)), throughput, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Throughput (GB/s)', fontweight='bold', fontsize=12)
    ax.set_title('DCRTPoly ADD Kernel: Memory Bandwidth vs Access Pattern\n(RingDim 131072, Depth 5)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, v) in enumerate(zip(bars, throughput)):
        ratio = v / seq_throughput if i > 0 else 1.0
        ratio_str = f'{ratio:.1%}' if i > 0 else 'baseline'
        ax.text(bar.get_x() + bar.get_width()/2, v + max(throughput)*0.01, 
                f'{v:.1f} GB/s\n({ratio_str})', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add baseline reference line
    ax.axhline(y=seq_throughput, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Sequential Baseline')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/sweep2_memory_access_pattern_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: results/sweep2_memory_access_pattern_analysis.png")
    
    plt.show()

if __name__ == '__main__':
    main()
