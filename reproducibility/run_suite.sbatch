#!/bin/bash
#SBATCH --job-name=fhe_repro_suite
#SBATCH --partition=zen4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --exclusive
#SBATCH --time=6:00:00
#SBATCH --output=reproducibility/slurm_%j.out
#SBATCH --error=reproducibility/slurm_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/REPACSS/home/asmalla/FHE-RaiderSTREAM/FHERaiderSTREAM"

# --- Environment setup ---
# sbatch does not source /etc/profile.d, so init module system manually.
# env/activate appends to CPATH/LIBRARY_PATH which may be unset.
set +u
source /etc/profile.d/00-modulepath.sh 2>/dev/null || true
source /etc/profile.d/modules.sh 2>/dev/null || true
source "$PROJECT_DIR/env/activate"
module load likwid/5.4.1-daemon
set -u

echo "Node: $(hostname)"
echo "CPUs: $(nproc)"
echo "Job:  $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""

# --- Build both binaries ---
"$PROJECT_DIR/reproducibility/build.sh"

# --- Run all groups ---
# Pass any extra args from sbatch submission (e.g. specific groups)
"$PROJECT_DIR/reproducibility/run_suite.sh" "$@"
