#!/bin/bash
set -euo pipefail
#
# Submit the full MPI scaling sweep: 1, 2, 4, 8, 16, 32, 64 nodes.
# Each job runs 1 MPI rank per node with 256 threads per rank.
#
# Usage:
#   ./reproducibility/run_mpi_sweep.sh              # submit all 7 jobs
#   ./reproducibility/run_mpi_sweep.sh --dry-run     # print sbatch commands only
#
# All jobs share a single output directory so results accumulate.
# After all jobs complete, regenerate the MPI figure with:
#   python3 python_plot/plot_mpi_scaling.py --input-dir <OUTDIR>
#
# Expected runtime: ~5 min per PE count. Jobs run independently so
# wall-clock depends on queue wait, not sum of compute times.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --help|-h)
            echo "Usage: $0 [--dry-run]"
            echo "Submit MPI scaling jobs for PE = 1, 2, 4, 8, 16, 32, 64"
            exit 0
            ;;
    esac
done

# Shared output directory for all PE counts
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$PROJECT_DIR/reproducibility/results_mpi_${TIMESTAMP}"
mkdir -p "$OUTDIR"

echo "=== MPI Scaling Sweep ==="
echo "Output directory: $OUTDIR"
echo ""

PE_COUNTS=(1 2 4 8 16 32 64)

for pe in "${PE_COUNTS[@]}"; do
    cmd="sbatch --nodes=${pe} --export=ALL,MPI_OUTDIR=${OUTDIR} $PROJECT_DIR/reproducibility/run_mpi.sbatch"

    if $DRY_RUN; then
        echo "  [DRY-RUN] PE=${pe}: $cmd"
    else
        echo -n "  PE=${pe} (${pe} nodes): "
        $cmd
    fi
done

echo ""
if $DRY_RUN; then
    echo "Dry run complete. No jobs submitted."
else
    echo "All jobs submitted. Monitor with: squeue -u \$USER"
    echo "Results will appear in: $OUTDIR/scaling_pe*.json"
    echo ""
    echo "After all jobs complete, regenerate Figure with:"
    echo "  python3 python_plot/plot_mpi_scaling.py --input-dir $OUTDIR --output $OUTDIR/mpi_scaling.pdf"
fi
