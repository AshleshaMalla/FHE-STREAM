#!/bin/bash
set -euo pipefail

# FHE-RaiderSTREAM Build and Run Script
# Usage: ./build.sh [--run] [--light] [--smoke] [--help]
#   --run   : Run benchmarks after building
#   --light : Run only RS_SEQ COPY+SCALE (faster)
#   --smoke : Run quick RS_SEQ + RS_GATHER COPY validation
#   --help  : Print usage

RUN_AFTER_BUILD=false
LIGHT_RUN=false
SMOKE_RUN=false

for arg in "$@"; do
    case "$arg" in
        --run)
            RUN_AFTER_BUILD=true
            ;;
        --light)
            LIGHT_RUN=true
            ;;
        --smoke)
            SMOKE_RUN=true
            ;;
        --help|-h)
            echo "Usage: ./build.sh [--run] [--light] [--smoke] [--help]"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: ./build.sh [--run] [--light] [--smoke] [--help]"
            exit 1
            ;;
    esac
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "==> Configuring CMake..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

echo "==> Building..."
cmake --build build -j

echo "==> Build complete: build/fhe_raiderstream"

# Run benchmarks if requested
if [[ "$RUN_AFTER_BUILD" == true ]]; then
    echo ""
    if [[ "$SMOKE_RUN" == true ]]; then
        echo "==> Running smoke benchmarks (RS_SEQ_COPY + RS_GATHER_COPY)..."
        RS_BATCH_SIZE="${RS_BATCH_SIZE:-4}" ./build/fhe_raiderstream --benchmark_filter='RS_(SEQ|GATHER)_COPY.*' --benchmark_min_time=0.02s --benchmark_repetitions=1
    elif [[ "$LIGHT_RUN" == true ]]; then
        echo "==> Running light benchmarks (RS_SEQ_COPY + RS_SEQ_SCALE)..."
        ./build/fhe_raiderstream --benchmark_filter='RS_SEQ_(COPY|SCALE).*' --benchmark_min_time=0.1s
    else
        echo "==> Running full Phase 1 suite (COPY/SCALE/ADD/TRIAD)..."
        ./build/fhe_raiderstream --benchmark_filter='RS_SEQ_.*' --benchmark_min_time=0.1s
    fi
fi
