#!/bin/bash
set -euo pipefail
#
# FHE-RaiderSTREAM Reproducibility Suite
#
# Runs all paper experiments with explicit configurations.
# Each group is self-contained and sets every environment variable it needs.
#
# Usage:
#   ./reproducibility/run_suite.sh [OPTIONS] [GROUP...]
#
# Options:
#   --dry-run     Print every command without executing
#   --help        Show this help and exit
#
# Groups (run one or more, or omit for all):
#   amplification    Figures 1-2  DRAM amplification (counter binary, ~25 min)
#   access_patterns  Figure 3     Bandwidth across access patterns (~11 min)
#   shuffle_mode     Figure 4     Poly vs Coeff shuffle comparison (~20 min)
#   thread_scaling   Figure 5     Thread scaling DCRT vs CT (~2.2 h)
#   seq_add_compare  Figure 6     Sequential ADD comparison (~3 min)
#   allocation       Figure 7     Memory allocation tax (~3 min)
#   correlation      Table III    CT correlation point (~25 min)
#   all              Run all groups in order (~3.5 h total on AMD node)
#
# Prerequisites:
#   - source env/activate
#   - module load likwid/5.4.1-daemon  (for amplification group only)
#   - Exclusive node: salloc --exclusive --nodes=1 --cpus-per-task=256
#   - Binaries built via: ./reproducibility/build.sh
#
# HEXL (Figure 8) and MPI scaling are separate scripts that require
# different nodes. See reproducibility/run_hexl.sh and run_mpi.sh.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMING_BIN="$PROJECT_DIR/reproducibility/build_timing/fhe_raiderstream"
COUNTER_BIN="$PROJECT_DIR/reproducibility/build_counter/fhe_raiderstream"

DRY_RUN=false
RUN_GROUPS=()

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --help|-h)
            sed -n '2,/^[^#]/{ /^#/s/^# \?//p }' "$0"
            exit 0
            ;;
        *) RUN_GROUPS+=("$arg") ;;
    esac
done

# Default to all groups
ALL_GROUPS=(amplification access_patterns shuffle_mode thread_scaling seq_add_compare allocation correlation)
if [[ ${#RUN_GROUPS[@]} -eq 0 ]]; then
    RUN_GROUPS=("${ALL_GROUPS[@]}")
else
    for i in "${!RUN_GROUPS[@]}"; do
        if [[ "${RUN_GROUPS[$i]}" == "all" ]]; then
            RUN_GROUPS=("${ALL_GROUPS[@]}")
            break
        fi
    done
fi

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$PROJECT_DIR/reproducibility/results_${TIMESTAMP}"
mkdir -p "$OUTDIR"

# --- Helpers ---

log() {
    echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/suite.log"
}

run() {
    local label="$1"
    shift
    if $DRY_RUN; then
        echo "  [DRY-RUN] $label:"
        echo "    $*"
        echo ""
        return 0
    fi
    log "  [$label] starting..."
    local start end elapsed
    start=$(date +%s.%N)
    "$@" 2>&1 | tee "$OUTDIR/${label}.stdout"
    local rc=${PIPESTATUS[0]}
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)
    if [[ $rc -ne 0 ]]; then
        log "  [$label] FAILED (exit $rc) after ${elapsed}s"
    else
        log "  [$label] done in ${elapsed}s"
    fi
    echo "$label $elapsed $rc" >> "$OUTDIR/timings.txt"
    return $rc
}

preflight() {
    local bin="$1" label="$2"
    if $DRY_RUN; then return 0; fi
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: $label binary not found at $bin" >&2
        echo "Run ./reproducibility/build.sh first." >&2
        exit 1
    fi
}

# ===============================================================
# Group: amplification (Figures 1-2)
# Counter binary, LIKWID wrapper mode, whole-process differential
# ===============================================================
run_amplification() {
    preflight "$COUNTER_BIN" "counter"
    log "=== Amplification (Figures 1-2) ==="

    local kernels=(
        "RS_SEQ_ADD/131072/40/0"
        "RS_GATHER_ADD/131072/40/1"
        "RS_GATHER_ADD/131072/40/2"
        "RS_SCATTER_GATHER_TRIAD/131072/40/2"
        "RS_NTT_ROUNDTRIP/131072/40/0"
    )
    local directions=(MEMREAD MEMWRITE)
    local low_iters=5
    local high_iters=30
    # SEQ_ADD read uses 5/40 per validation_summary.md
    local seq_add_high_iters=40

    local amp_dir="$OUTDIR/amplification"
    mkdir -p "$amp_dir"

    for kernel in "${kernels[@]}"; do
        local slug=$(echo "$kernel" | tr '/' '_')
        for dir in "${directions[@]}"; do
            local hi=$high_iters
            if [[ "$kernel" == "RS_SEQ_ADD"* && "$dir" == "MEMREAD" ]]; then
                hi=$seq_add_high_iters
            fi

            # Low-iteration runs (2 replicates)
            for rep in 1 2; do
                run "amp_${slug}_${dir}_low_${rep}" \
                    env RS_BATCH_SIZE=512 OMP_NUM_THREADS=256 \
                    likwid-perfctr -g "$dir" -C 0-255 \
                    -o "$amp_dir/${slug}_${dir}_low_${rep}.csv" \
                    "$COUNTER_BIN" \
                    --benchmark_filter="FHERaiderSTREAM/${kernel}" \
                    --benchmark_min_time=${low_iters}x \
                    --benchmark_repetitions=1
            done

            # High-iteration runs (3 replicates)
            for rep in 1 2 3; do
                run "amp_${slug}_${dir}_high_${rep}" \
                    env RS_BATCH_SIZE=512 OMP_NUM_THREADS=256 \
                    likwid-perfctr -g "$dir" -C 0-255 \
                    -o "$amp_dir/${slug}_${dir}_high_${rep}.csv" \
                    "$COUNTER_BIN" \
                    --benchmark_filter="FHERaiderSTREAM/${kernel}" \
                    --benchmark_min_time=${hi}x \
                    --benchmark_repetitions=1
            done
        done
    done
}

# ===============================================================
# Group: access_patterns (Figure 3)
# Timing binary, all irregular kernels + NTT
# ===============================================================
run_access_patterns() {
    preflight "$TIMING_BIN" "timing"
    log "=== Access Patterns (Figure 3) ==="

    run "access_patterns" \
        env RS_BATCH_SIZE=512 \
            OMP_NUM_THREADS=256 \
            OMP_PROC_BIND=true \
            OMP_PLACES="{0:256}" \
        "$TIMING_BIN" \
        --benchmark_filter='FHERaiderSTREAM/RS_(GATHER|SCATTER|SCATTER_GATHER)_(COPY|SCALE|ADD|TRIAD)/131072/40/[12]|FHERaiderSTREAM/RS_NTT_ROUNDTRIP/131072/40/0' \
        --benchmark_min_time=1s \
        --benchmark_repetitions=1 \
        --benchmark_out="$OUTDIR/access_patterns.json" \
        --benchmark_out_format=json
}

# ===============================================================
# Group: shuffle_mode (Figure 4)
# Timing binary, 10 reps for statistical confidence
# ===============================================================
run_shuffle_mode() {
    preflight "$TIMING_BIN" "timing"
    log "=== Shuffle Mode (Figure 4) ==="

    run "shuffle_mode" \
        env RS_BATCH_SIZE=512 \
            OMP_NUM_THREADS=256 \
            OMP_PROC_BIND=true \
            OMP_PLACES="{0:256}" \
        "$TIMING_BIN" \
        --benchmark_filter='FHERaiderSTREAM/RS_(GATHER|SCATTER_GATHER)_(ADD|TRIAD)/131072/40/[12]' \
        --benchmark_min_time=1s \
        --benchmark_repetitions=10 \
        --benchmark_out="$OUTDIR/shuffle_mode.json" \
        --benchmark_out_format=json
}

# ===============================================================
# Group: thread_scaling (Figure 5)
# Timing binary, DCRT B=512, CT B=256, sweep threads
# ===============================================================
run_thread_scaling() {
    preflight "$TIMING_BIN" "timing"
    log "=== Thread Scaling (Figure 5) ==="

    local thread_counts=(1 4 16 64 128 256)

    # DCRT backend
    for t in "${thread_counts[@]}"; do
        run "scaling_dcrt_${t}t" \
            env RS_BATCH_SIZE=512 \
                OMP_NUM_THREADS=$t \
                OMP_PROC_BIND=true \
                OMP_PLACES="{0:$t}" \
            "$TIMING_BIN" \
            --benchmark_filter='FHERaiderSTREAM/RS_SEQ_ADD/131072/40/0' \
            --benchmark_min_time=1s \
            --benchmark_repetitions=1 \
            --benchmark_out="$OUTDIR/dcrt_scaling_t${t}.json" \
            --benchmark_out_format=json
    done

    # CT backend
    for t in "${thread_counts[@]}"; do
        run "scaling_ct_${t}t" \
            env RS_BATCH_SIZE=256 \
                OMP_NUM_THREADS=$t \
                OMP_PROC_BIND=true \
                OMP_PLACES="{0:$t}" \
            "$TIMING_BIN" \
            --benchmark_filter='CTFixture/CT_SEQ_ADD/131072/40/0' \
            --benchmark_min_time=1s \
            --benchmark_repetitions=1 \
            --benchmark_out="$OUTDIR/ct_scaling_t${t}.json" \
            --benchmark_out_format=json
    done
}

# ===============================================================
# Group: seq_add_compare (Figure 6)
# Timing binary, combined DCRT+CT run at B=256
# ===============================================================
run_seq_add_compare() {
    preflight "$TIMING_BIN" "timing"
    log "=== SEQ ADD Comparison (Figure 6) ==="

    run "seq_add_compare" \
        env RS_BATCH_SIZE=256 \
            OMP_NUM_THREADS=256 \
            OMP_PROC_BIND=true \
            OMP_PLACES="{0:256}" \
        "$TIMING_BIN" \
        --benchmark_filter='(FHERaiderSTREAM/RS_SEQ_ADD|CTFixture/CT_SEQ_ADD|CTFixture/CT_SEQ_ADD_INPLACE)/131072/40/0' \
        --benchmark_min_time=1s \
        --benchmark_repetitions=1 \
        --benchmark_out="$OUTDIR/seq_add_comparison.json" \
        --benchmark_out_format=json
}

# ===============================================================
# Group: allocation (Figure 7)
# Timing binary, CT backend, reports RSS user counters
# ===============================================================
run_allocation() {
    preflight "$TIMING_BIN" "timing"
    log "=== Memory Allocation (Figure 7) ==="

    run "allocation_tax" \
        env RS_BATCH_SIZE=256 \
            OMP_NUM_THREADS=256 \
            OMP_PROC_BIND=true \
            OMP_PLACES="{0:256}" \
        "$TIMING_BIN" \
        --benchmark_filter='CTFixture/CT_SEQ_(MULT_NO_RELIN|RELIN)/131072/40/0' \
        --benchmark_min_time=1s \
        --benchmark_repetitions=1 \
        --benchmark_out="$OUTDIR/rss_expansion.json" \
        --benchmark_out_format=json
}

# ===============================================================
# Group: correlation (Table III)
# Timing binary, CT backend, 5-rep ADD+MULT then 10-rep RELIN
# ===============================================================
run_correlation() {
    preflight "$TIMING_BIN" "timing"
    log "=== CT Correlation / Table III ==="

    # ADD + MULT_NO_RELIN: 5 repetitions
    run "correlation_add_mult" \
        env RS_BATCH_SIZE=256 \
            OMP_NUM_THREADS=256 \
            OMP_PROC_BIND=true \
            OMP_PLACES="{0:256}" \
        "$TIMING_BIN" \
        --benchmark_filter='CTFixture/(CT_SEQ_ADD|CT_SEQ_MULT_NO_RELIN)/131072/40/0$' \
        --benchmark_min_time=1s \
        --benchmark_repetitions=5 \
        --benchmark_out="$OUTDIR/ct_correlation_add_mult.json" \
        --benchmark_out_format=json

    # RELIN: 10 repetitions for tighter statistics
    run "correlation_relin" \
        env RS_BATCH_SIZE=256 \
            OMP_NUM_THREADS=256 \
            OMP_PROC_BIND=true \
            OMP_PLACES="{0:256}" \
        "$TIMING_BIN" \
        --benchmark_filter='CTFixture/CT_SEQ_RELIN/131072/40/0$' \
        --benchmark_min_time=1s \
        --benchmark_repetitions=10 \
        --benchmark_out="$OUTDIR/ct_correlation_relin.json" \
        --benchmark_out_format=json
}

# ===============================================================
# Dispatch
# ===============================================================

log "FHE-RaiderSTREAM Reproducibility Suite"
log "Node: $(hostname)"
log "Groups: ${RUN_GROUPS[*]}"
if $DRY_RUN; then
    log "Mode: DRY RUN (no commands will execute)"
fi
log "Output: $OUTDIR"
log ""

suite_start=$(date +%s.%N)

for group in "${RUN_GROUPS[@]}"; do
    case "$group" in
        amplification)   run_amplification ;;
        access_patterns) run_access_patterns ;;
        shuffle_mode)    run_shuffle_mode ;;
        thread_scaling)  run_thread_scaling ;;
        seq_add_compare) run_seq_add_compare ;;
        allocation)      run_allocation ;;
        correlation)     run_correlation ;;
        *)
            echo "Unknown group: $group" >&2
            echo "Valid groups: amplification access_patterns shuffle_mode thread_scaling seq_add_compare allocation correlation all" >&2
            exit 1
            ;;
    esac
done

if ! $DRY_RUN; then
    suite_end=$(date +%s.%N)
    suite_elapsed=$(echo "$suite_end - $suite_start" | bc)
    log ""
    log "Suite complete in ${suite_elapsed}s"
    log "Results in: $OUTDIR"

    if [[ -f "$OUTDIR/timings.txt" ]]; then
        log ""
        log "Per-group timings:"
        column -t "$OUTDIR/timings.txt" | tee -a "$OUTDIR/suite.log"
    fi
else
    log ""
    log "Dry run complete. No commands were executed."
fi
