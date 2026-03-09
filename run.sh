#!/bin/bash
set -euo pipefail

# FHE-RaiderSTREAM Run Script
# Usage: ./run.sh [--print-setup] [-- extra google-benchmark args...]

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$PROJECT_DIR/build/fhe_raiderstream"

if [[ ! -x "$BINARY" ]]; then
    echo "Error: Binary not found. Run './build.sh' first." >&2
    exit 1
fi

EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --print-setup) EXTRA_ARGS+=("--rs_print_setup") ;;
        -h|--help)
            echo "Usage: ./run.sh [--print-setup] [-- extra google-benchmark args...]"
            echo "  --print-setup  Print fixture setup configuration for each parameter combo"
            exit 0
            ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

exec "$BINARY" "${EXTRA_ARGS[@]}"
