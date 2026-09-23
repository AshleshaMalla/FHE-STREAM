#!/bin/bash
set -euo pipefail
#
# Preflight checks before running the reproducibility suite.
# Verifies: exclusive node, LIKWID daemon module, counter access,
# env activation, and binary availability.
#
# Usage: ./reproducibility/preflight.sh

PASS=0
FAIL=0
WARN=0

check() {
    local label="$1" result="$2" detail="${3:-}"
    if [[ "$result" == "PASS" ]]; then
        echo "  [PASS] $label"
        ((PASS++))
    elif [[ "$result" == "WARN" ]]; then
        echo "  [WARN] $label"
        [[ -n "$detail" ]] && echo "         $detail"
        ((WARN++))
    else
        echo "  [FAIL] $label"
        [[ -n "$detail" ]] && echo "         $detail"
        ((FAIL++))
    fi
}

echo "=== FHE-RaiderSTREAM Preflight Checks ==="
echo ""

# 1. Check we're on a compute node
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    check "Running under Slurm (job $SLURM_JOB_ID)" "PASS"
else
    check "Running under Slurm" "WARN" "Not in a Slurm job. Use salloc --exclusive for measurements."
fi

# 2. Check exclusive allocation
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    shared=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -c 'Shared=OK' || true)
    cpus=$(nproc 2>/dev/null || echo "0")
    if [[ "$cpus" -ge 256 ]]; then
        check "Full node allocated ($cpus CPUs)" "PASS"
    else
        check "Full node allocated ($cpus CPUs)" "FAIL" "Expected 256 CPUs. Use --exclusive --cpus-per-task=256."
    fi
else
    cpus=$(nproc 2>/dev/null || echo "0")
    check "CPU count: $cpus" "WARN" "Cannot verify exclusive allocation outside Slurm."
fi

# 3. Check env activated
if [[ -n "${ENV_DIR:-}" ]]; then
    check "env/activate sourced (ENV_DIR=$ENV_DIR)" "PASS"
else
    check "env/activate sourced" "FAIL" "Run: source env/activate"
fi

# 4. Check LIKWID module
set +u
likwid_path=$(which likwid-perfctr 2>/dev/null || true)
set -u
if [[ -n "$likwid_path" ]]; then
    # Check it's the daemon variant
    if [[ "$likwid_path" == *"daemon"* ]] || module list 2>&1 | grep -q "likwid/5.4.1-daemon"; then
        check "LIKWID daemon module loaded" "PASS"
    else
        check "LIKWID module loaded" "WARN" "Verify this is the daemon variant (likwid/5.4.1-daemon), not perf."
    fi
else
    check "LIKWID available" "FAIL" "Run: module load likwid/5.4.1-daemon"
fi

# 5. Check LIKWID counter access
if [[ -n "$likwid_path" ]]; then
    if likwid-perfctr -g MEMREAD -C 0 sleep 0.1 > /dev/null 2>&1; then
        check "LIKWID counter access" "PASS"
    else
        check "LIKWID counter access" "FAIL" "likwid-perfctr failed. Is the daemon running? Is the node exclusive?"
    fi
fi

# 6. Check binaries
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for bin_label in "timing:$PROJECT_DIR/reproducibility/build_timing/fhe_raiderstream" \
                 "counter:$PROJECT_DIR/reproducibility/build_counter/fhe_raiderstream"; do
    label="${bin_label%%:*}"
    path="${bin_label#*:}"
    if [[ -x "$path" ]]; then
        check "$label binary exists" "PASS"
    else
        check "$label binary exists" "FAIL" "Run: ./reproducibility/build.sh"
    fi
done

# 7. Summary
echo ""
echo "Results: $PASS passed, $WARN warnings, $FAIL failed"
if [[ $FAIL -gt 0 ]]; then
    echo "Fix failures before running the suite."
    exit 1
else
    echo "Ready to run."
    exit 0
fi
