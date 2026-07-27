#!/usr/bin/env python3
"""
Master script to generate all FHE-RaiderSTREAM benchmark plots.
Runs all individual plotting scripts and ensures they use the unified style.
"""

import subprocess
import os
import sys
from pathlib import Path

def run_script(script_name):
    print(f"Running {script_name}...")
    try:
        # Using sys.executable to ensure we use the same python interpreter
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stderr)

def main():
    # List of all plotting scripts to run
    scripts = [
        'plot_add_scaling.py',
        'plot_allocation_tax.py',
        'plot_hexl_vs_regular_throughput.py',
        'plot_memory_access_pattern_bandwidth.py',
        'plot_mpi_scaling.py',
        'plot_ring_scaling.py',
        'plot_seq_add_comparison.py',
        'plot_shuffle_mode_compare.py',
        'plot_software_tax_comparison.py'
    ]

    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("=== FHE-RaiderSTREAM Plot Generation ===\n")

    for script in scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"Warning: {script} not found.")

    print("\nAll plots generated in the 'plots/' directory.")

if __name__ == "__main__":
    main()
