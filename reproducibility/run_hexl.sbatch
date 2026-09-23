#!/bin/bash
#SBATCH --job-name=fhe_repro_hexl
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output=reproducibility/slurm_hexl_%j.out
#SBATCH --error=reproducibility/slurm_hexl_%j.err

set -euo pipefail
#
# Figure 8: Intel HEXL On/Off Comparison
#
# Runs on the Intel Xeon Gold 6448Y node (h100 partition, 64 cores).
# Builds two binaries (HEXL-off and HEXL-on), runs the same set of
# irregular-access kernels on each, and saves JSON output.
#
# Usage:
#   sbatch reproducibility/run_hexl.sbatch
#   sbatch reproducibility/run_hexl.sbatch --export=DRY_RUN=1
#
# Expected runtime: ~15 minutes compute

PROJECT_DIR="/mnt/REPACSS/home/asmalla/FHE-RaiderSTREAM/FHERaiderSTREAM"
DRY_RUN="${DRY_RUN:-0}"

# --- Environment setup ---
set +u
source /etc/profile.d/00-modulepath.sh 2>/dev/null || true
source /etc/profile.d/modules.sh 2>/dev/null || true
# On h100 partition, set MODULEPATH to include h100 modules
export MODULEPATH="/opt/apps/nfs/modules/h100/$(cat /etc/os-release | grep '^ID=' | cut -d= -f2)$(cat /etc/os-release | grep '^VERSION_ID=' | cut -d= -f2)/Core:${MODULEPATH:-}"
source "$PROJECT_DIR/env/activate"
set -u

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$PROJECT_DIR/reproducibility/results_hexl_${TIMESTAMP}"
mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/hexl.log"

log() {
    echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGFILE"
}

run() {
    local label="$1"
    shift
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [DRY-RUN] $label:" | tee -a "$LOGFILE"
        echo "    $*" | tee -a "$LOGFILE"
        echo "" | tee -a "$LOGFILE"
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

log "=== HEXL Comparison (Figure 8) ==="
log "Node: $(hostname)"
log "CPUs: $(nproc)"
log "Job:  ${SLURM_JOB_ID:-interactive}"
log "Date: $(date)"
log ""

# --- Build HEXL-off binary ---
log "Building HEXL-off binary..."
if [[ "$DRY_RUN" != "1" ]]; then
    cmake -S "$PROJECT_DIR" -B "$PROJECT_DIR/reproducibility/build_hexl_off" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_DCRT=ON \
        -DENABLE_CIPHERTEXT=OFF \
        -DENABLE_HEXL=OFF \
        -DENABLE_MPI=OFF \
        -DLIKWID_PERFMON=OFF \
        > "$OUTDIR/build_hexl_off.log" 2>&1
    cmake --build "$PROJECT_DIR/reproducibility/build_hexl_off" -j \
        >> "$OUTDIR/build_hexl_off.log" 2>&1
    log "  HEXL-off binary built."
fi
HEXL_OFF_BIN="$PROJECT_DIR/reproducibility/build_hexl_off/fhe_raiderstream"

# --- Build HEXL-on binary ---
log "Building HEXL-on binary..."
if [[ "$DRY_RUN" != "1" ]]; then
    cmake -S "$PROJECT_DIR" -B "$PROJECT_DIR/reproducibility/build_hexl_on" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_DCRT=ON \
        -DENABLE_CIPHERTEXT=OFF \
        -DENABLE_HEXL=ON \
        -DENABLE_MPI=OFF \
        -DLIKWID_PERFMON=OFF \
        > "$OUTDIR/build_hexl_on.log" 2>&1
    cmake --build "$PROJECT_DIR/reproducibility/build_hexl_on" -j \
        >> "$OUTDIR/build_hexl_on.log" 2>&1
    log "  HEXL-on binary built."
fi
HEXL_ON_BIN="$PROJECT_DIR/reproducibility/build_hexl_on/fhe_raiderstream"

log ""

# Kernels: Gather/Scatter/Scatter-Gather TRIAD coeff + NTT Roundtrip
FILTER='FHERaiderSTREAM/RS_(GATHER|SCATTER|SCATTER_GATHER)_TRIAD/131072/40/2|FHERaiderSTREAM/RS_NTT_ROUNDTRIP/131072/40/0'

# --- HEXL-off run ---
log "=== HEXL-off run ==="
run "hexl_off" \
    env RS_BATCH_SIZE=512 \
        OMP_NUM_THREADS=64 \
        OMP_PROC_BIND=true \
        OMP_PLACES="{0:64}" \
    "$HEXL_OFF_BIN" \
    --benchmark_filter="$FILTER" \
    --benchmark_min_time=1s \
    --benchmark_repetitions=10 \
    --benchmark_out="$OUTDIR/hexl_off.json" \
    --benchmark_out_format=json

# --- HEXL-on run ---
log "=== HEXL-on run ==="
run "hexl_on" \
    env RS_BATCH_SIZE=512 \
        OMP_NUM_THREADS=64 \
        OMP_PROC_BIND=true \
        OMP_PLACES="{0:64}" \
    "$HEXL_ON_BIN" \
    --benchmark_filter="$FILTER" \
    --benchmark_min_time=1s \
    --benchmark_repetitions=10 \
    --benchmark_out="$OUTDIR/hexl_on.json" \
    --benchmark_out_format=json

log ""
log "=== Complete ==="
log "Results in: $OUTDIR"
if [[ -f "$OUTDIR/timings.txt" ]]; then
    log "Timings:"
    column -t "$OUTDIR/timings.txt" | tee -a "$LOGFILE"
fi
log ""
log "To regenerate Figure 8:"
log "  python3 python_plot/plot_hexl_vs_regular_throughput.py \\"
log "    --regular $OUTDIR/hexl_off.json \\"
log "    --hexl $OUTDIR/hexl_on.json \\"
log "    --output $OUTDIR/hexl_vs_regular_throughput.pdf"
