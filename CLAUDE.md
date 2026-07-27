# FHE-STREAM Project Context

## What this is
A memory benchmarking suite for FHE (Fully Homomorphic Encryption) workloads,
built on OpenFHE, extending RaiderSTREAM's access patterns (Sequential,
Gather, Scatter, Scatter-Gather) to FHE's DCRTPoly/Ciphertext data structures.

Paper target: PMBS26 workshop (SC26), deadline Aug 5, 2026. This is a
revision after workshop rejection. Reviewers' central objection: reported
"bandwidth" is a logical/modeled metric (bytes processed ÷ elapsed time),
not validated against actual DRAM traffic. Current sprint adds LIKWID
hardware-counter validation, a shuffle-window sweep, and an NTT stride sweep.

## Repo structure
- `include/backends/dcrt/` — DCRTFixture.h, StreamCore.h (raw RNS limb
  backend, headers)
- `include/backends/ciphertext/` — CTFixture.h (full Ciphertext object
  backend, headers)
- `include/common/` — BenchmarkUtils.h, MPIUtils.h (shared utilities)
- `src/backends/dcrt/` — DCRTFixture.cpp + kernel implementations split
  by access pattern:
  - `kernels_sequential.cpp`
  - `kernels_gather.cpp`
  - `kernels_scatter.cpp`
  - `kernels_scatter_gather.cpp`
  - `kernels_ntt.cpp`
- `src/backends/ciphertext/` — CTFixture.cpp, kernels_ct.cpp (all
  Ciphertext-backend kernels, apparently in one file)
- `src/common/` — BenchmarkUtils.cpp
- `src/main.cpp` — entry point / driver
- `results/` — JSON/CSV outputs from prior runs, organized by
  experiment/sweep (sweep1 = core scaling, sweep2 = access pattern,
  sweep3 = NTT, sweep4 = keyswitch vs triad, sweep5 = ciphertext add,
  sweep6 = HEXL)
- `python_plot/` — matplotlib scripts, one per figure, plus
  `fhe_plot_style.py` (shared styling) and `generate_all_plots.py`
  (batch driver)
- `plots/` — rendered PDF figures, filenames map to paper figures
- `env/` — environment setup, must be activated before every run
  (ignore flake files)

## Build system
**CMake**, C++17, out-of-tree in `build/`. `CMakeLists.txt` at root.

**One-time setup** (builds OpenFHE v1.5.0, google-benchmark, HEXL into `./env/dist/`):
```bash
./setup_env.sh          # add --hexl for Intel HEXL support
```

**Activate environment** (required before every build/run):
```bash
source env/activate
```
This sets `PATH`, `LD_LIBRARY_PATH`, `CMAKE_PREFIX_PATH`, etc. to point at `./env/dist/`.

**Build**:
```bash
./build.sh
# which runs:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_CIPHERTEXT=ON -DENABLE_HEXL=OFF -DENABLE_MPI=ON
cmake --build build -j
```

**Run**: `./run.sh [--print-setup] [extra gbench flags]`
or directly: `./build/fhe_raiderstream --benchmark_filter='RS_SEQ_ADD'`

Key CMake options:
- `ENABLE_DCRT=ON` (default) — DCRTPoly backend
- `ENABLE_CIPHERTEXT=OFF` (default) — Ciphertext backend
- `ENABLE_HEXL=ON` (default; disabled in build.sh via `-DENABLE_HEXL=OFF`)
- `ENABLE_MPI=ON` (default)

## Two backends
1. **DCRTPoly backend** (`src/backends/dcrt/`) — raw RNS limb arrays,
   kernels split by access pattern across 5 files. Sequential covers
   COPY/SCALE/ADD/TRIAD; same presumably true for gather/scatter/
   scatter_gather. `kernels_ntt.cpp` = NTT Roundtrip + likely
   Key-Switch Mock.
2. **Ciphertext backend** (`src/backends/ciphertext/`) — full OpenFHE
   Ciphertext objects, all kernels in `kernels_ct.cpp` (COPY/SCALE/
   ADD/TRIAD sequential-only, MULT_NO_RELIN, RELIN, ADD_INPLACE).

## Key parameters (paper Section IV)
- Ring dimension N = 131072
- RNS depth L = 40
- Batch size B = 512 (DCRTPoly, 3-array ops) or 256 (Ciphertext, ~10x
  memory footprint from object overhead) — NOTE: this mismatch is a
  known confound flagged by reviewers (paper Fig. 3); task list below
  includes equalizing this
- 256 threads on AMD EPYC 9754 (zen4 partition)
- 64 threads on Intel Xeon Gold 6448Y (h100 partition — used because
  Intel HEXL requires it)

## Access patterns
- Sequential, Gather, Scatter, Scatter-Gather
- Two shuffle granularities: Poly (multi-MB block) and Coefficient
  (8-byte)
- Shuffle mode is a **runtime parameter** (`state.range(2)`), not a compile
  flag: `ShuffleMode::None=0`, `ShuffleMode::Poly=1`, `ShuffleMode::Coeff=2`
  (enum defined in `include/backends/dcrt/DCRTFixture.h`)
- Dispatched inside each kernel via `if (mode == ShuffleMode::Poly) RunGatherPoly(...)
  else if (mode == ShuffleMode::Coeff) RunGatherCoeff(...)`

## Cluster environment
- REPACSS cluster, Slurm scheduler
- Environment must be activated before every run: `source env/activate`
- AMD node: partition=zen4, likwid groups MEMREAD / MEMWRITE
  (separate calls), DFC0-11 per-channel Data Fabric Counters
- Intel node: partition=h100, likwid group MEM (single call, combined
  read+write), MBOX0-7 CAS_COUNT_RD/WR per-channel counters
- likwid module: `module load likwid/5.4.1-daemon`
- Must run under `salloc --exclusive` — non-exclusive nodes show
  contaminated idle traffic from other tenants (confirmed empirically:
  idle floor went from ~470 MB/s read on shared session to ~26-45 MB/s
  read on exclusive full-socket allocation)
- QOS max wall time: 2 days (`--time=2-00:00:00`) on both zen4 and h100
  partitions as tested

## Current sprint tasks (priority order)
1. **LIKWID Marker API instrumentation** — wrap priority kernels with
   MarkerInit/MarkerStartRegion/MarkerStopRegion/MarkerClose:
   - SEQ ADD (DCRT) — kernels_sequential.cpp
   - GATHER ADD, poly mode — kernels_gather.cpp
   - GATHER ADD, coeff mode — kernels_gather.cpp
   - SCATTER-GATHER TRIAD, coeff mode — kernels_scatter_gather.cpp
   - NTT Roundtrip — kernels_ntt.cpp
   - CT_SEQ_ADD_INPLACE — kernels_ct.cpp
2. **Shuffle-window sweep** — parameterize the randomization window in
   gather/scatter kernels (currently likely shuffles across full N×L
   footprint); sweep W ∈ {4KiB, 32KiB, 1MiB, 16MiB, 512MiB, full
   footprint (~60GiB)}. Goal: determine whether reported irregular
   bandwidth numbers reflect cache-resident or true DRAM-resident
   access.
3. **NTT stride logging** — instrument/wrap the NTT call in
   kernels_ntt.cpp to log per-stage access stride pattern to a file,
   for comparison against the scatter-gather proxy.
4. **Equalize batch size B** between DCRTPoly and Ciphertext backends
   for the scaling comparison (currently B=512 vs B=256, confounds
   paper Fig. 3 / results/sweep1_core_scaling*).

## Coding conventions
- **C++17**, CMake 3.5+
- **Parameter flow — N, L, mode**: passed purely as Google Benchmark args via
  `state.range()`, set by `RaiderSTREAM_Arguments_*()` in `BenchmarkUtils.cpp`:
  - `state.range(0)` = ringDim (N)
  - `state.range(1)` = numTowers / multDepth (L)
  - `state.range(2)` = ShuffleMode (0/1/2)
  - Swept values: RingDims `{16384, 32768, 65536, 131072}`, Depths `{1, 5, 20, 40}`
- **Batch size B**: NOT a benchmark arg — read from env var `RS_BATCH_SIZE`
  at fixture `SetUp()` time (default: 100). Set with e.g. `RS_BATCH_SIZE=512 ./run.sh`.
- **Byte accounting**: `DeepBytesPerPoly(N, L) = N * L * 8` bytes per poly
  (defined inline in `StreamCore.h` and `DCRTFixture.cpp`). `bytesPerIter` = that
  × nPolys × array-count (2 for COPY/SCALE, 3 for ADD/TRIAD).
- **Kernel registration pattern**:
  ```cpp
  BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_ADD)(benchmark::State& state) {
    // extract state.range(0..2), compute bytesPerIter
    // dispatch via lambda:
    if (mode == ShuffleMode::Poly)
      RunGatherPoly(*this, state, bytesPerIter, [](auto& ...) { /* op */ });
    else if (mode == ShuffleMode::Coeff)
      RunGatherCoeff(*this, state, bytesPerIter, [](auto& ...) { /* op */ });
  }
  BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_ADD)
    ->Apply(RaiderSTREAM_Arguments_Irregular)->Unit(benchmark::kMillisecond);
  ```
- **Run* dispatcher templates** in `StreamCore.h` own the outer benchmark loop,
  OpenMP parallel region, and MPI barrier — kernel lambdas only express
  the per-element arithmetic.
- **Thread count**: `RS_Execution_Threads` global (set from `omp_get_max_threads()`
  in `main.cpp`); passed as `num_threads(RS_Execution_Threads)` in each Run*.
- **OpenFHE internal parallelism** is disabled in SetUp (SetNumThreads(1)) to
  prevent interference; saved thread count is restored immediately after.

## Shuffle / randomization logic (for Task 2)

Index vectors are allocated once in `DCRTFixture::SetUp()` with `std::iota` +
`std::shuffle` (fixed seeds), then reused across all iterations.

| Vector           | Size      | Seed | Use                                  |
|------------------|-----------|------|--------------------------------------|
| `IDX`            | nPolys    | 42   | Poly-level gather read index         |
| `IDX_WRITE`      | nPolys    | 99   | Poly-level scatter write index       |
| `COEFF_IDX`      | ringDim   | 43   | Coeff-level gather read index        |
| `COEFF_IDX_WRITE`| ringDim   | 99   | Coeff-level scatter write index      |

**Current state**: shuffles are always **full-range** — `IDX` permutes all `nPolys`
positions, `COEFF_IDX` permutes all `ringDim` positions. No window parameter exists.

### Poly mode (`ShuffleMode::Poly`)
Outer loop `for i in 0..nPolys`:
- Gather: `src = IDX[i]`; reads `A[src]/B[src]` (random polys), writes `C[i]` (sequential)
- Scatter: `dest = IDX[i]`; reads sequential, writes to random poly
- ScatterGather: `src = IDX[i]`, `dst = IDX_WRITE[i]`; both random

One hop = one full poly = `ringDim × L × 8` bytes (≈40 MiB at N=131072, L=40).

### Coeff mode (`ShuffleMode::Coeff`)
Per-poly, per-tower, inner loop `for j in 0..ringDim`:
- Gather: reads `aTower[COEFF_IDX[j]]`, writes `cTower[j]` (sequential write)
- Scatter: reads `aTower[j]` (sequential), writes `cTower[COEFF_IDX_WRITE[j]]`
- ScatterGather: reads `aTower[COEFF_IDX[j]]`, writes `cTower[COEFF_IDX_WRITE[j]]`

One hop = 8 bytes anywhere in `[0, ringDim)` — the "FHE Memory Wall."

### What Task 2 changes
To parameterize window W, the full-range shuffle in `COEFF_IDX` gets replaced with
indices restricted to a sub-range of `W / 8` coefficients, tiled or wrapped across
all `ringDim` positions. Window sizes map as:
- 4 KiB → 512 coefficients (L1-resident)
- 32 KiB → 4096 coefficients (L2-resident)
- 1 MiB → 131072 coefficients (L3; = one full tower at N=131072)
- 16 MiB → crosses LLC for L>1
- 512 MiB / full footprint → full DRAM random

The window parameterization likely needs a new benchmark arg (`state.range(3)`)
or a separate `RaiderSTREAM_Arguments_WindowSweep()` function in `BenchmarkUtils.cpp`.

## Known issues / silent footprint traps

**`RS_BATCH_SIZE` defaults to 100, not the paper's 512 (DCRTPoly) or 256 (Ciphertext).**
The fixture `SetUp()` falls back to `nPolys = 100` when the env var is unset. At
N=131072, L=40, this gives a 3-array working set of only ~300 GiB... wait, actually
N*L*8 * 100 * 3 ≈ 12.6 GiB total, well below the 4 GiB minimum noted in the code
comment — but critically, the reported bandwidth numbers in the paper used B=512,
so any run without `RS_BATCH_SIZE=512` produces non-comparable numbers.

**Rule**: every build/run wrapper, sbatch script, and one-liner written going forward
must include `export RS_BATCH_SIZE=512` (or `RS_BATCH_SIZE=256` for the Ciphertext
backend) explicitly. Never rely on the default.
Always load likwid/5.4.1-daemon explicitly — the default module is likwid/5.4.1-perf, which uses direct perf_event_open() access and fails with Permission denied on fixed counters (ACTUAL_CPU_CLOCK/MAX_CPU_CLOCK) on this cluster. The -daemon variant brokers access through likwid-accessD and works correctly.

**LIKWID pinning bug: never pass `-C <range>` to `likwid-perfctr` in marker mode (`-m`)
with this codebase.** Passing `-C 0-N` causes all OpenMP threads to collapse onto CPU 0
after the first benchmark iteration, corrupting per-thread counter attribution (only
~1 thread's data gets recorded). Confirmed via `sched_getcpu()` debug tracing on
2026-07-26. Root cause not fully isolated (suspected conflict between LIKWID's internal
pthread-wrapper pinning and OpenMP's thread pool across repeated `#pragma omp parallel`
region re-entry). Root-causing is not a priority — the workaround is sufficient and
reliable. **Correct invocation**: set `OMP_PROC_BIND=true` and
`OMP_PLACES="{0:N}"` explicitly (N = thread count), then run:
```bash
likwid-perfctr -g <GROUP> -m -o <file>.csv ./binary ...
```
with **no `-C` flag at all**. LIKWID will monitor whatever CPUs the process actually uses.

**LIKWID marker-mode timing is unreliable for bandwidth calculation.** Confirmed ~55×
wall-clock slowdown per iteration when marker instrumentation is active, even after the
pinning fix (single-iteration runtime went from ~122 ms unmarked to ~6.7–7.6 s marked,
for RS_SEQ_ADD/131072/40). Do **NOT** use LIKWID's own derived "Memory read/write
bandwidth [MBytes/s]" column from marker-mode runs — it divides real byte counts by
inflated, overhead-corrupted time. Instead use a three-step approach:
- **(a)** Get logical bandwidth and real kernel timing from a **separate, unmarked run**
  (no `-m`, no LIKWID at all).
- **(b)** Get measured DRAM byte counts from the marker-mode run's **raw DFC/MBOX
  counter sums**, ignoring its bandwidth/time columns entirely.
- **(c)** Compute the amplification factor as `measured_bytes / logical_bytes` — never
  from a bandwidth ratio.

Cross-socket LIKWID marker registration is unreliable for GATHER/SCATTER-GATHER kernels (confirmed working for SEQ_ADD, fails intermittently on socket 1 for GATHER_ADD_POLY — root cause not isolated despite return-value checking). Workaround: measure each socket separately by restricting OMP_NUM_THREADS/OMP_PLACES to one socket at a time (128 threads, {0:128} then {128:128}), and sum the resulting byte counts manually. This is the standard measurement procedure for all irregular-access kernels going forward.

CT_SEQ_ADD_INPLACE could not be reliably hardware-validated with either measurement approach. Marker-mode instrumentation causes a ~148× wall-clock stall (cause not isolated, ruled out thread-pool teardown). Whole-process fallback captures CTFixture::SetUp()'s ciphertext-batch encryption cost alongside the timed kernel, producing physically implausible totals (~15× theoretical peak bandwidth for the observed wall-time). Not resolved due to time constraints. The five DCRTPoly-backend kernels (SEQ_ADD, GATHER_ADD poly/coeff, SCATTER_GATHER_TRIAD_COEFF, NTT_ROUNDTRIP) remain fully validated with measured hardware amplification factors of 2.57×–6.23×.

**Source-of-truth validation doc: `results/validation_summary.md`** — consolidates the five-kernel amplification table (read/write) with per-kernel caveats, and the matched CT correlation point (CT_SEQ_ADD/MULT_NO_RELIN/RELIN, 1:28:182). Every number carries exact node/threads/batch/date. Open item: write-amp cells for SEQ_ADD and GATHER_ADD_POLY are left blank — their on-disk `_memwrite` CSVs were combined cross-socket captures that recorded implausibly low volume (yield sub-unity 0.029×/0.171×, physically impossible); a fresh socket-split ({0:128} then {128:128}) marker re-run for just these two is deferred to later in the sprint. Do not insert the impossible values.

## Do NOT
- Do not modify the byte-accounting formulas (BenchmarkUtils.cpp,
  likely — Eq. 2/3 in the paper) without flagging it. These are kept
  as the logical/modeled baseline that gets compared against new
  hardware-counter measurements, not replaced by them.
- Do not remove or alter existing kernel logic — new instrumentation
  (Marker API calls) should wrap kernels, not change what they compute.
- Do not touch `results/` or `plots/` directly — those are prior run
  outputs referenced by the current paper draft; new sweep results go
  in new files, not overwriting existing ones.
- Do not go through the `env/` directory as it contains all the necessary dependencies to
  run the program.