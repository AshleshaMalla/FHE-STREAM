#!/bin/bash
set -e  # Exit on error

# FHE-RaiderSTREAM Build and Run Script
# Usage: ./build.sh [--run] [--light]
#   --run   : Run benchmarks after building
#   --light : Run only COPY+SCALE (faster)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Optional: Automatic activation if env/activate exists
if [ -f "env/activate" ] && [ -z "$ENV_DIR" ]; then
    echo "==> Automatically sourcing env/activate..."
    source env/activate
fi

module load likwid/5.4.1-daemon

echo "==> Configuring CMake..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CIPHERTEXT=ON -DENABLE_HEXL=OFF -DENABLE_MPI=ON -DLIKWID_PERFMON=ON

echo "==> Building..."
cmake --build build -j

echo "==> Build complete: build/fhe_raiderstream"

# Check if --run flag is present
if [[ "$*" == *"--run"* ]]; then
    echo ""
    export RS_BATCH_SIZE=512
    if [[ "$*" == *"--light"* ]]; then
        echo "==> Running COPY+SCALE benchmarks..."
        ./build/fhe_raiderstream --benchmark_filter='RS_SEQ_(COPY|SCALE).*' --benchmark_min_time=0.1s
    else
        echo "==> Running full Phase 1 suite (COPY/SCALE/ADD/TRIAD)..."
        ./build/fhe_raiderstream --benchmark_filter='RS_SEQ_.*' --benchmark_min_time=0.1s
    fi
fi
