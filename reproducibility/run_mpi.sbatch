#!/bin/bash
#
# MPI Scaling (paper MPI aggregate bandwidth result)
#
# This is a TEMPLATE — you must submit it multiple times with different
# --nodes values, or use the run_mpi_sweep.sh wrapper that does this
# automatically.
#
# Each run uses 1 MPI rank per node, 256 threads per rank.
# RS_SEQ_ADD at N=131072, L=40, sequential mode.
#
# Usage (single run):
#   sbatch --nodes=4 reproducibility/run_mpi.sbatch
#
# Usage (full sweep via wrapper):
#   ./reproducibility/run_mpi_sweep.sh
#   ./reproducibility/run_mpi_sweep.sh --dry-run
#
# Expected runtime: ~5 min per PE count, ~35 min total for all 7

#SBATCH --job-name=fhe_repro_mpi
#SBATCH --partition=zen4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH --output=reproducibility/slurm_mpi_%j.out
#SBATCH --error=reproducibility/slurm_mpi_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/REPACSS/home/asmalla/FHE-RaiderSTREAM/FHERaiderSTREAM"
NODES=${SLURM_NNODES:-1}

# --- Environment setup ---
set +u
source /etc/profile.d/00-modulepath.sh 2>/dev/null || true
source /etc/profile.d/modules.sh 2>/dev/null || true
source "$PROJECT_DIR/env/activate"
module load openmpi/4.1.8 2>/dev/null || true
set -u

# Output directory (shared across all PE counts if using the sweep wrapper)
OUTDIR="${MPI_OUTDIR:-$PROJECT_DIR/reproducibility/results_mpi_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/mpi.log"

log() {
    echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGFILE"
}

# --- Build MPI binary if not already present ---
MPI_BIN="$PROJECT_DIR/reproducibility/build_mpi/fhe_raiderstream"
if [[ ! -x "$MPI_BIN" ]]; then
    log "Building MPI binary..."
    cmake -S "$PROJECT_DIR" -B "$PROJECT_DIR/reproducibility/build_mpi" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_DCRT=ON \
        -DENABLE_CIPHERTEXT=OFF \
        -DENABLE_HEXL=OFF \
        -DENABLE_MPI=ON \
        -DLIKWID_PERFMON=OFF \
        > "$OUTDIR/build_mpi.log" 2>&1
    cmake --build "$PROJECT_DIR/reproducibility/build_mpi" -j \
        >> "$OUTDIR/build_mpi.log" 2>&1
    log "  MPI binary built."
fi

log "=== MPI Scaling: ${NODES} nodes (${NODES} ranks × 256 threads) ==="
log "Node list: ${SLURM_NODELIST:-localhost}"
log "Job: ${SLURM_JOB_ID:-interactive}"
log "Date: $(date)"
log ""

OUTFILE="$OUTDIR/scaling_pe${NODES}.json"

start=$(date +%s.%N)

srun --ntasks="$NODES" \
     --ntasks-per-node=1 \
     --cpus-per-task=256 \
     env RS_BATCH_SIZE=512 \
         OMP_NUM_THREADS=256 \
         OMP_PROC_BIND=true \
         OMP_PLACES="{0:256}" \
     "$MPI_BIN" \
     --benchmark_filter='FHERaiderSTREAM/RS_SEQ_ADD/131072/40/0' \
     --benchmark_min_time=1s \
     --benchmark_repetitions=1 \
     --benchmark_out="$OUTFILE" \
     --benchmark_out_format=json \
     2>&1 | tee "$OUTDIR/mpi_pe${NODES}.stdout"

end=$(date +%s.%N)
elapsed=$(echo "$end - $start" | bc)

log "  PE${NODES} done in ${elapsed}s → $OUTFILE"
echo "pe${NODES} $elapsed" >> "$OUTDIR/timings.txt"
