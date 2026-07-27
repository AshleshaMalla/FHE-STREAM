# FHE-RaiderSTREAM — Validation Summary (Source of Truth)

**Compiled:** 2026-07-27
**Scope:** Consolidates (1) hardware DRAM-amplification validation of the five
DCRTPoly-backend kernels and (2) the matched correlation point relating
FHE-STREAM ciphertext kernels to real OpenFHE homomorphic operations.
This document is the authoritative reference for the PMBS26 (SC26) paper rewrite.
Every number below carries its exact node, thread count, batch size, and date.

---

## Platform / provenance quick-reference

| Purpose | Node | Slurm Job ID | Partition | Allocation | Date(s) |
|---------|------|-------------|-----------|------------|---------|
| Idle counter-access baseline (noise floor) | rpc-91-3 | **130567** | zen4 | `salloc --exclusive --nodes=1 --partition=zen4` | 2026-07-24 |
| DCRT kernel amplification runs (Part 1) | rpc-91-2 | **130756** | zen4 | `salloc --exclusive --nodes=1 --partition=zen4 --time=2-00:00:00` (standing alloc, started 2026-07-25 22:27:35) | 2026-07-26 |
| CT correlation runs (Part 2) | rpc-91-2 | **130756** | zen4 | same standing allocation | 2026-07-27 |
| Intel idle baseline (reference only) | rpg-93-4 | **130577** | h100 | `salloc --exclusive --nodes=1 --partition=h100` | 2026-07-24 |

**Hardware:** AMD EPYC 9754 (Bergamo), 2 sockets × 128 cores = 256 cores, 1 thread/core
(no SMT), 8 NUMA nodes, 12 DDR5 memory channels/socket. L1d 32 KiB×256, L2 1 MiB×256,
L3 16 MiB×32. Vendor theoretical peak ≈ 921.6 GB/s (node).
**Counter tool:** LIKWID 5.4.1 (daemon build; `likwid/5.4.1-daemon` module).
**Idle DRAM floor (rpc-91-3, job 130567):** 75.95 MB/s node total (0.008% of peak) —
confirms clean exclusive isolation. See `counter_validation_baseline.md`.

> **Provenance note to reconcile before publication:** the idle counter-access
> validation was performed on **rpc-91-3** (job 130567), but the actual kernel
> amplification measurements (Part 1) were collected on **rpc-91-2** (job 130756).
> Both are AMD EPYC 9754 / zen4 exclusive nodes; if the paper cites a single AMD
> node name, note that the idle floor and the kernel runs came from two different
> physical nodes of the same model.

---

## Part 1 — Five-kernel DRAM amplification (DCRTPoly backend)

**Definition:** amplification factor = `measured DRAM bytes (LIKWID counter sum) /
logical bytes (byte-accounting model)`. Per project methodology, measured bytes come
from raw DFC counter sums in marker-mode runs; logical bytes and real timing come from
a **separate unmarked run** — the LIKWID-derived bandwidth/time columns are **not** used
(marker mode inflates wall time ~55×).

**Common parameters (all five kernels):** N (ring dim) = 131072, L (RNS depth) = 40,
**batch B = 512** (DCRTPoly, via `RS_BATCH_SIZE=512` in `measure_amplification.sh`),
AMD EPYC 9754 / rpc-91-2 / job 130756 / zen4 exclusive, **2026-07-26**,
LIKWID 5.4.1-daemon, `OMP_PROC_BIND=true`, no `-C` flag (per pinning-bug workaround).

| Kernel | Access pattern | Shuffle mode | Read amp. | Write amp. | Threads / registration |
|--------|----------------|--------------|-----------|------------|------------------------|
| `RS_SEQ_ADD` | Sequential | None (0) | **2.57×** | — ¹ | 256, cross-socket `{0:256}` (single pass) |
| `RS_GATHER_ADD` | Gather | Poly (1) | **3.90×** | — ¹ | 128+128 per-socket, summed ² |
| `RS_GATHER_ADD` | Gather | Coeff (2) | **3.27×** | **1.56×** | 128+128 per-socket, summed ² |
| `RS_SCATTER_GATHER_TRIAD` | Scatter-Gather | Coeff (2) | **5.68×** | **2.24×** | 128+128 per-socket, summed ² |
| `RS_NTT_ROUNDTRIP` | NTT round-trip | None (0) | **6.23×** | **2.56×** | 128+128 per-socket, summed ² |

Values transcribed from `counter_validation_baseline.md` (authoritative). Read amp.
range across the five kernels: **2.57× – 6.23×**.

### Per-kernel measurement caveats

1. **Write amplification remains "—" for `SEQ_ADD` and `GATHER_ADD_POLY` — the on-disk
   write CSVs are unrecoverable (verified 2026-07-27, no rerun).** A derivation was
   attempted from the existing `_memwrite` CSVs using the *exact* `parse_likwid_csv.py`
   methodology (measured = Σ "Memory data volume [GBytes]" over nonzero-call-count threads,
   summed across sockets; logical write = `iters × B × 1 × Vpoly`). **Method was validated**
   by reproducing the two published coeff write-amps to the decimal from their socket-split
   files: `GATHER_ADD_COEFF` → 1.56× and `SCATTER_GATHER_TRIAD_COEFF` → 2.24× (exact match).
   Applied identically to these two kernels the result is **physically impossible**:

   | Kernel | Write CSV | nz/threads | iters | Measured | Logical | Computed amp |
   |--------|-----------|-----------|-------|----------|---------|--------------|
   | `SEQ_ADD` | `seq_add_131072_40_write.csv` (combined 256-col) | 166/256 | 8 | 4.92 GB | 171.80 GB | **0.029×** ✗ |
   | `GATHER_ADD_POLY` | `gather_add_poly_write.csv` (combined 256-col) | 151/256 | 3 | 10.99 GB | 64.42 GB | **0.171×** ✗ |

   A write-amp < 1 is impossible for a kernel that streams a full output array to DRAM.
   **Root cause:** unlike the coeff kernels (clean socket-split `_memwrite_s0/s1.csv` pairs),
   these two kernels' write passes were captured as **single combined cross-socket 256-column
   files** that recorded almost no data volume (4.9 / 11.0 GB vs the ~100–144 GB the coeff
   kernels captured) — a cross-socket counter-attribution failure, the same class of bug
   documented in CLAUDE.md. Both **trip the partial-registration warning** (166/256 and
   151/256 nonzero call-counts, far below a healthy full-node 256), and the write pass used
   a different iteration count than the read pass (SEQ_ADD: read iters 4 vs write iters 8),
   which `parse_likwid_csv.py` itself flags as making the ratios non-comparable.
   Corroborating evidence that these specific files are not the clean source: the *published
   read* amps (2.57×, 3.90×) also do **not** reproduce from the same on-disk files, whereas
   the coeff kernels reproduce perfectly. **Do not insert the 0.029×/0.171× values.** A valid
   write-amp for these two requires a fresh socket-split (`{0:128}` then `{128:128}`) marker
   re-run on an exclusive zen4 node — deferred (no rerun authorized this pass).

2. **Partial / cross-socket marker-registration reliability.** `SEQ_ADD` registers and
   measures reliably across both sockets in a single 256-thread pass. The irregular-access
   kernels — **`GATHER_ADD` (poly & coeff), `SCATTER_GATHER_TRIAD_COEFF`, and
   `NTT_ROUNDTRIP`** — exhibit unreliable cross-socket LIKWID marker registration
   (socket-1 markers fail intermittently; root cause not isolated). **Standard workaround
   applied:** each socket measured separately (`OMP_NUM_THREADS=128` with `OMP_PLACES={0:128}`
   then `{128:128}`), and byte counts summed manually. All four irregular kernels above
   were measured this way; treat their factors as socket-summed, not single-pass.

3. **`CT_SEQ_ADD_INPLACE` — hardware validation UNRESOLVED (Ciphertext backend, not in the
   table above).** Could not be reliably hardware-validated by either approach:
   - Marker-mode instrumentation induces a ~148× wall-clock stall (cause not isolated;
     thread-pool teardown ruled out).
   - Whole-process fallback captures `CTFixture::SetUp()`'s ciphertext-batch encryption
     cost alongside the timed kernel, producing physically implausible totals (~15× the
     theoretical peak bandwidth for the observed wall time).

     Left unresolved due to time constraints. **Do not reopen** for this revision. The
   five DCRTPoly kernels above remain the fully validated set.

---

## Part 2 — Matched correlation: FHE-STREAM CT kernels vs real OpenFHE ops

**Purpose:** answer the reviewer question of whether FHE-STREAM ciphertext kernel scores
predict the cost of real OpenFHE homomorphic operations. This is the first clean,
matched measurement of the three homomorphic-evaluation kernels at a single point.

**Common parameters (all three kernels):** N = 131072, L = 40, **batch B = 256**
(Ciphertext backend, `RS_BATCH_SIZE=256`), **256 threads** (`OMP_NUM_THREADS=256`,
`OMP_PLACES={0:256}`, `OMP_PROC_BIND=true`, full node both sockets),
AMD EPYC 9754 / rpc-91-2 / job 130756 / zen4 exclusive, **2026-07-27**.
Unmarked timing run (no LIKWID). BFVrns, `HEStd_NotSet`.

| Kernel | OpenFHE op | Mean real | Mean CPU | Real CV | Iters/rep × reps | Source file |
|--------|-----------|-----------|----------|---------|------------------|-------------|
| `CT_SEQ_ADD` | `EvalAdd` | **176.1 ms** | 149.8 ms | 4.61% | 9 × 5 | `results/ct_correlation_amd256t_b256_N131072L40_20260727.json` |
| `CT_SEQ_MULT_NO_RELIN` | `EvalMultNoRelin` | **4987.8 ms** | 3502.2 ms | 2.85% | 1 × 5 | `results/ct_correlation_amd256t_b256_N131072L40_20260727.json` |
| `CT_SEQ_RELIN` | `Relinearize` (deg-2 input) | **32106.0 ms** | 13897.5 ms | 5.84% | 1 × 10 | `results/ct_correlation_relin_amd256t_b256_N131072L40_20260727_rep10.json` |

**Cost ratio (mean real time): ADD : MULT_NO_RELIN : RELIN ≈ 1 : 28 : 182.**
(4987.8 / 176.1 = 28.3; 32106.0 / 176.1 = 182.3.)

### Statistics caveat — iteration counts

`--benchmark_min_time=1s` yields very different iteration counts because per-op cost
spans two orders of magnitude: ADD gets **9 timed iterations/repetition**, whereas
MULT_NO_RELIN (~5 s/op) and RELIN (~32 s/op) each get only **1 iteration/repetition**.
The reported stddev/CV therefore derives from **between-repetition** variation
(5 reps for ADD/MULT, 10 for RELIN), not within-repetition — still valid samples, but
thinner for the heavy ops. RELIN was re-run at 10 repetitions specifically to tighten
this: **CV improved 7.28% → 5.84%** and the mean shifted only 1.3% (32514.4 → 32106.0 ms),
confirming the original was unbiased. The superseded 5-rep RELIN figure
(32514.4 ms, CV 7.28%) lives in the base JSON; the 10-rep figure above is authoritative.

### RELIN real/CPU stall observation

`CT_SEQ_RELIN` real time (32106.0 ms) is **≈ 2.31× its CPU time** (13897.5 ms) — threads
spend ~57% of wall-clock stalled rather than computing. This is the memory-stall / LLC-
thrashing signature of evaluation-key streaming at high depth (L=40): the eval-key set
exceeds LLC capacity and every `Relinearize` call thrashes the cache hierarchy. Consistent
with the "Capacity Wall" claim (report §4.4). Note this is **not** present in ADD
(real 176.1 ≈ 1.17× CPU 149.8) and is mild in MULT (real 4987.8 ≈ 1.42× CPU 3502.2).

---

## Cross-cutting notes

- **Batch-size mismatch is intentional and documented:** Part 1 uses B=512 (DCRTPoly,
  3-array footprint); Part 2 uses B=256 (Ciphertext, ~10× object overhead). These are the
  paper's per-backend values. The DCRT-vs-CT scaling comparison (paper Fig. 3) still
  carries the known B=512-vs-256 confound flagged by reviewers; the two parts here are
  **not** a like-for-like batch comparison and should not be presented as one.
- **Google Benchmark `threads` field reads 1** in all CT JSONs — parallelism is internal
  OpenMP (256-way), not GBench thread ranges. Do not misread as single-threaded.
- **Reproduce Part 2** exactly:
  ```bash
  RS_BATCH_SIZE=256 OMP_NUM_THREADS=256 OMP_PROC_BIND=true OMP_PLACES="{0:256}" \
  ./build/fhe_raiderstream \
    --benchmark_filter='CTFixture/(CT_SEQ_ADD|CT_SEQ_MULT_NO_RELIN|CT_SEQ_RELIN)/131072/40/0$' \
    --benchmark_min_time=1s --benchmark_repetitions=5 \
    --benchmark_out=<file>.json --benchmark_out_format=json
  # RELIN re-run: same, filter CT_SEQ_RELIN only, --benchmark_repetitions=10
  ```
