#!/bin/bash
set -euo pipefail
#
# Measure DRAM amplification for a benchmark kernel via LIKWID marker mode.
#
# Usage:
#   ./scripts/measure_amplification.sh <benchmark_filter> <2|3>
#
# Arguments:
#   benchmark_filter  Google Benchmark regex, e.g. 'RS_GATHER_ADD/131072/40/1'
#   array_count       2 = 1 read + 1 write; 3 = 2 reads + 1 write
#
# Requirements:
#   - module load likwid/5.4.1-daemon must be active
#   - source env/activate must be active
#   - Binary built with -DLIKWID_PERFMON=ON
#   - Run on an exclusive zen4 node (salloc --exclusive or inside sbatch)
#
# Outputs:
#   4 CSVs in results/ (memread/memwrite × socket0/socket1)
#   Amplification summary table printed to stdout

FILTER="${1:?Usage: $0 <benchmark_filter> <2|3>}"
ARRAYS="${2:?Usage: $0 <benchmark_filter> <2|3>}"

if [[ "$ARRAYS" != "2" && "$ARRAYS" != "3" ]]; then
    echo "Error: array_count must be 2 or 3, got '$ARRAYS'" >&2
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="$PROJECT_DIR/build/fhe_raiderstream"

if [[ ! -x "$BINARY" ]]; then
    echo "Error: binary not found at $BINARY — run ./build.sh first." >&2
    exit 1
fi

export RS_BATCH_SIZE=512
export OMP_PROC_BIND=true

# Filesystem-safe slug from filter (/ → -)
SLUG=$(echo "$FILTER" | tr '/' '-' | tr -cd '[:alnum:]_-')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$PROJECT_DIR/results"

CSV_READ_S0="$OUTDIR/${SLUG}_${TIMESTAMP}_memread_s0.csv"
CSV_READ_S1="$OUTDIR/${SLUG}_${TIMESTAMP}_memread_s1.csv"
CSV_WRITE_S0="$OUTDIR/${SLUG}_${TIMESTAMP}_memwrite_s0.csv"
CSV_WRITE_S1="$OUTDIR/${SLUG}_${TIMESTAMP}_memwrite_s1.csv"

# Use --benchmark_min_time=0.1s: LIKWID overhead makes each iteration ~6-7s anyway,
# so this yields exactly 1 iteration per run — sufficient for byte-count extraction.
BENCH_ARGS=(
    --benchmark_filter="$FILTER"
    --benchmark_min_time=0.1s
    --benchmark_repetitions=1
)

run_pass() {
    local group="$1" socket="$2" places="$3" outcsv="$4"
    echo "  [$(date +%H:%M:%S)] ${group} socket ${socket} → $(basename "$outcsv")"
    OMP_NUM_THREADS=128 \
    OMP_PLACES="$places" \
    likwid-perfctr -g "$group" -m -o "$outcsv" \
        "$BINARY" "${BENCH_ARGS[@]}" 2>/dev/null
}

echo "=== measure_amplification.sh ==="
echo "Filter  : $FILTER"
echo "Arrays  : $ARRAYS"
echo "Outputs : results/${SLUG}_${TIMESTAMP}_*.csv"
echo ""
echo "Running 4 LIKWID passes..."

run_pass MEMREAD  0 "{0:128}"   "$CSV_READ_S0"
run_pass MEMREAD  1 "{128:128}" "$CSV_READ_S1"
run_pass MEMWRITE 0 "{0:128}"   "$CSV_WRITE_S0"
run_pass MEMWRITE 1 "{128:128}" "$CSV_WRITE_S1"

echo ""
echo "Parsing results..."

python3 "$PROJECT_DIR/scripts/parse_likwid_csv.py" \
    --amplification \
    --filter  "$FILTER" \
    --arrays  "$ARRAYS" \
    --memread-s0  "$CSV_READ_S0" \
    --memread-s1  "$CSV_READ_S1" \
    --memwrite-s0 "$CSV_WRITE_S0" \
    --memwrite-s1 "$CSV_WRITE_S1"
