#!/bin/bash
set -euo pipefail
#
# Build both binaries needed for the reproducibility suite.
#
# Timing binary:  reproducibility/build_timing/fhe_raiderstream
#   - LIKWID_PERFMON=OFF, ENABLE_CIPHERTEXT=ON, ENABLE_MPI=OFF
#   - Used for Figures 3-8 and Table III
#
# Counter binary: reproducibility/build_counter/fhe_raiderstream
#   - LIKWID_PERFMON=ON, ENABLE_CIPHERTEXT=OFF, ENABLE_MPI=OFF
#   - Used for Figures 1-2 (DRAM amplification only)
#
# Prerequisites:
#   - source env/activate
#   - module load likwid/5.4.1-daemon
#
# Usage: ./reproducibility/build.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Building FHE-RaiderSTREAM Reproducibility Binaries ==="
echo ""

# --- Timing binary (Figures 3-8, Table III) ---
echo "--- Timing binary (LIKWID_PERFMON=OFF, ENABLE_CIPHERTEXT=ON) ---"
cmake -S "$PROJECT_DIR" -B "$PROJECT_DIR/reproducibility/build_timing" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_DCRT=ON \
    -DENABLE_CIPHERTEXT=ON \
    -DENABLE_HEXL=OFF \
    -DENABLE_MPI=OFF \
    -DLIKWID_PERFMON=OFF
cmake --build "$PROJECT_DIR/reproducibility/build_timing" -j
echo "  -> reproducibility/build_timing/fhe_raiderstream"
echo ""

# --- Counter binary (Figures 1-2) ---
echo "--- Counter binary (LIKWID_PERFMON=ON, ENABLE_CIPHERTEXT=OFF) ---"
cmake -S "$PROJECT_DIR" -B "$PROJECT_DIR/reproducibility/build_counter" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_DCRT=ON \
    -DENABLE_CIPHERTEXT=OFF \
    -DENABLE_HEXL=OFF \
    -DENABLE_MPI=OFF \
    -DLIKWID_PERFMON=ON
cmake --build "$PROJECT_DIR/reproducibility/build_counter" -j
echo "  -> reproducibility/build_counter/fhe_raiderstream"
echo ""

echo "=== Both binaries built successfully ==="
echo "Run the suite with: ./reproducibility/run_suite.sh [--dry-run] [GROUP...]"
