# Reproducibility Guide — FHE-STREAM (PMBS26 / SC26)

This directory contains everything needed to reproduce the results in the
paper. All scripts are self-contained, set every environment variable
explicitly, and never rely on defaults.

## Hardware Requirements

Three node configurations are used:

- **AMD node** (Figures 1–7, Table III): Dual-socket AMD EPYC 9754
  (Bergamo), 256 cores, DDR5. One exclusive node.
- **Intel node** (Figure 8): Dual-socket Intel Xeon Gold 6448Y, 64 cores,
  AVX-512 IFMA required for HEXL. One exclusive node.
- **MPI scaling**: Up to 64 AMD nodes, one MPI rank per node.

All single-node experiments require **exclusive allocation**. Shared nodes
show cross-tenant DRAM traffic (measured at ~470 MB/s idle on a shared
session vs ~26 MB/s on an exclusive one), which contaminates hardware
counter measurements.

## Software Dependencies

| Dependency | Version | Notes |
|---|---|---|
| Rocky Linux | 9.6 | Tested OS |
| GCC | 11.5.0 | C++17 required |
| CMake | >= 3.5 | 3.26.5 tested |
| OpenFHE | 1.5.0 | Built by `setup_env.sh` |
| Google Benchmark | 1.8.5 | Built by `setup_env.sh` |
| Intel HEXL | 1.2.6 | Built by `setup_env.sh --hexl`, Intel node only |
| OpenMPI | 4.1.8 | MPI experiment only, `module load openmpi/4.1.8` |
| LIKWID | 5.4.1-daemon | Amplification experiments only |
| Python 3 | 3.9+ | matplotlib required for figure generation |

## LIKWID Daemon Requirement

The LIKWID module **must** be the daemon variant (`likwid/5.4.1-daemon`),
not the default `likwid/5.4.1-perf`. The perf variant uses
`perf_event_open()` directly and fails on fixed counters
(`ACTUAL_CPU_CLOCK`/`MAX_CPU_CLOCK`) on this cluster. The daemon variant
brokers access through `likwid-accessD` and works correctly.

Verify with:
```bash
module load likwid/5.4.1-daemon
likwid-perfctr -g MEMREAD -C 0 sleep 0.1
```

If this prints counter values without errors, the daemon is working.

## Two-Build Requirement

Two separate binaries are needed because LIKWID marker instrumentation
causes a ~55x wall-clock slowdown per iteration. Using a single binary
for both timing and counter measurements would produce incorrect
bandwidth numbers.

| Binary | CMake Flag | Used For |
|---|---|---|
| Timing binary | `LIKWID_PERFMON=OFF` | Figures 3–8, Table III (all bandwidth and correlation) |
| Counter binary | `LIKWID_PERFMON=ON` | Figures 1–2 only (DRAM amplification via wrapper mode) |

The counter binary is run under `likwid-perfctr` in **wrapper mode**
(not marker mode). Marker mode is not used because it causes thread-pinning
collapse on this codebase (see `CLAUDE.md` for details).

## Setup

From the repository root:

```bash
# 1. Build dependencies (one-time, ~30 min)
./setup_env.sh

# 2. Activate environment (every session)
source env/activate

# 3. Load LIKWID (AMD node, for amplification group)
module load likwid/5.4.1-daemon

# 4. Build both binaries
./reproducibility/build.sh

# 5. Verify everything is ready
./reproducibility/preflight.sh
```

`build.sh` places binaries in `reproducibility/build_timing/` and
`reproducibility/build_counter/`. It does not touch the project's
`build/` directory. The HEXL and MPI sbatch scripts build their own
binaries automatically on the appropriate nodes.

## Running Experiments

### AMD node (Figures 1–7, Table III)

Submit as a batch job:
```bash
sbatch reproducibility/run_suite.sbatch
```

Or interactively on an exclusive node:
```bash
salloc --exclusive --nodes=1 --cpus-per-task=256 --partition=zen4 --time=6:00:00
./reproducibility/run_suite.sh
```

Run a single group:
```bash
./reproducibility/run_suite.sh correlation
```

Preview all commands without executing:
```bash
./reproducibility/run_suite.sh --dry-run
```

List available groups and expected runtimes:
```bash
./reproducibility/run_suite.sh --help
```

### Intel node (Figure 8)

```bash
sbatch reproducibility/run_hexl.sbatch
```

Builds HEXL-off and HEXL-on binaries on the Intel node, runs both with
10 repetitions. Prints the plotting command on completion.

### MPI scaling

```bash
./reproducibility/run_mpi_sweep.sh            # submit all 7 jobs
./reproducibility/run_mpi_sweep.sh --dry-run   # preview only
```

Submits independent jobs at 1, 2, 4, 8, 16, 32, and 64 nodes. Each job
runs `RS_SEQ_ADD` with one MPI rank per node and 256 threads per rank.

## Expected Runtimes

All times are measured compute time on exclusive nodes and exclude
Slurm queue wait. The three node types may run in parallel.

| Group | Figures | Time |
|---|---|---|
| Amplification | Figs. 1–2 | ~25 min |
| Access patterns | Fig. 3 | ~11 min |
| Shuffle mode | Fig. 4 | ~20 min |
| Thread scaling | Fig. 5 | ~2.2 h |
| SEQ ADD comparison | Fig. 6 | ~3 min |
| Memory allocation | Fig. 7 | ~3 min |
| CT correlation | Table III | ~25 min |
| **AMD total** | | **~3.5 h** |
| HEXL comparison | Fig. 8 | ~15 min |
| MPI scaling (per PE) | — | ~5 min each |

## Regenerating Figures

Each experiment group writes JSON output to a timestamped directory
under `reproducibility/`. Pass the new data to the corresponding
plotting script:

| Figure | Script | Example |
|---|---|---|
| Fig. 1 | `python_plot/plot_dram_amplification_read.py` | Values hardcoded; update script |
| Fig. 2 | `python_plot/plot_dram_amplification_write.py` | Values hardcoded; update script |
| Fig. 4 | `python_plot/plot_shuffle_mode_compare.py` | `--input <dir>/shuffle_mode.json` |
| Fig. 5 | `python_plot/plot_add_scaling.py` | `--input <dir>/add_scaling_results.json` |
| Fig. 6 | `python_plot/plot_seq_add_comparison.py` | `--input <dir>/seq_add_comparison.json` |
| Fig. 7 | `python_plot/plot_allocation_tax.py` | `--input <dir>/rss_expansion.json` |
| Fig. 8 | `python_plot/plot_hexl_vs_regular_throughput.py` | `--regular <dir>/hexl_off.json --hexl <dir>/hexl_on.json` |
| MPI | `python_plot/plot_mpi_scaling.py` | `--input-dir <dir>/` |

The HEXL and MPI scripts print the exact plotting command at the end of
their output.

## Expected Reproducibility

Absolute bandwidth numbers are platform-dependent. On equivalent
hardware:

- **DRAM amplification factors** (1.00x–3.84x read, 1.00x–2.44x write):
  within 10% on same-generation AMD processors.
- **CT correlation ratio** (1:28:182 for ADD:MULT:RELIN): within 5%.
- **MPI scaling**: near-linear aggregate bandwidth growth; absolute peak
  depends on the interconnect.
- **Thread scaling**: curve shape reproduces; absolute values are
  platform-dependent.
- **HEXL comparison**: requires the Intel platform with AVX-512 IFMA.

## File Inventory

```
reproducibility/
  build.sh            Build timing + counter binaries
  preflight.sh        Verify setup before running
  run_suite.sh        Main driver (7 AMD-node groups, --help/--dry-run)
  run_suite.sbatch    Sbatch wrapper for run_suite.sh
  run_hexl.sbatch     Figure 8 on Intel node
  run_mpi.sbatch      Single MPI run (used by sweep wrapper)
  run_mpi_sweep.sh    Submit all 7 MPI scaling jobs
```

## Key Environment Variables

| Variable | DCRT Value | CT Value | Notes |
|---|---|---|---|
| `RS_BATCH_SIZE` | 512 | 256 | Default is 100; always set explicitly |
| `OMP_NUM_THREADS` | 256 (full node) | 256 | Varied in thread scaling |
| `OMP_PROC_BIND` | true | true | Prevents thread migration |
| `OMP_PLACES` | `{0:256}` | `{0:256}` | Adjusted per thread count |
