# FHE-RaiderSTREAM — Validation Summary (Source of Truth)

**Compiled:** 2026-07-30
**Scope:** Consolidates (1) hardware DRAM-amplification validation of the five
DCRTPoly-backend kernels, measured by **whole-process (non-marker) LIKWID DF
counters with a SetUp-cancelling differential**, and (2) the matched correlation
point relating FHE-STREAM ciphertext kernels to real OpenFHE homomorphic operations.
This document is the authoritative reference for the PMBS26 (SC26) paper rewrite.
Every number below carries its exact node, thread count, batch size, and date.

> **This replaces the retracted marker-mode table.** The previous Part 1 (marker-mode
> amplification, factors 2.57×–6.23×) is retracted and preserved for the record in
> `results/validation_summary_MARKER_MODE_RETRACTED.md`. It was non-reproducible and its
> socket-summed values double-counted the working set (implied bandwidths exceeded the
> hardware peak). Do not cite those numbers. Part 2 (CT correlation) below was never
> marker-derived and is carried forward unchanged.

---

## Platform / provenance quick-reference

| Purpose | Node | Slurm Job ID | Partition | Allocation | Date(s) |
|---------|------|-------------|-----------|------------|---------|
| DCRT kernel amplification (Part 1, whole-process differential) | rpc-91-2 | **134030** | zen4 | `salloc --exclusive --nodes=1 --partition=zen4 --time=2-00:00:00` | 2026-07-29 – 2026-07-30 |
| CT correlation runs (Part 2) | rpc-91-2 | **130756** | zen4 | prior standing allocation | 2026-07-27 |
| Idle counter-access baseline (noise floor) | rpc-91-3 | **130567** | zen4 | `salloc --exclusive --nodes=1 --partition=zen4` | 2026-07-24 |

**Hardware:** AMD EPYC 9754 (Bergamo), 2 sockets × 128 cores = 256 cores, 1 thread/core
(no SMT), 8 NUMA nodes, 12 DDR5 memory channels/socket. L1d 32 KiB×256, L2 1 MiB×256,
L3 16 MiB×32. **Theoretical peak ≈ 921.6 GB/s = 858 GiB/s (node).**
**Counter tool:** LIKWID 5.4.1 (daemon build; `likwid/5.4.1-daemon` module).
**Idle DRAM floor (rpc-91-3, job 130567):** 75.95 MB/s node total (0.008% of peak) —
confirms clean exclusive isolation. See `counter_validation_baseline.md`.

---

## Part 1 — Five-kernel DRAM amplification (DCRTPoly backend)

**Definition:** amplification factor = `measured DRAM bytes / logical (byte-accounting)
bytes`, per direction (read / write).

**Measurement method (whole-process differential).** LIKWID wrapper mode
(`likwid-perfctr -g MEMREAD|MEMWRITE -C 0-255`, **non-marker**) reads both sockets'
Data-Fabric DRAM counters over the entire process. To remove one-time `SetUp()` and
fixed-overhead traffic, each kernel is run at two exact iteration counts (Google
Benchmark `Nx` mode, no calibration search) and differenced:

```
per_iter_bytes = (total_bytes[N_high] − total_bytes[N_low]) / (N_high − N_low)
amplification  = per_iter_bytes / logical_bytes_per_iter
```

`N_low = 5`, `N_high = 30` (SEQ_ADD read used `5 / 40`). Two low-iter + **three
high-iter replicates** per kernel per direction. This avoids both the banned `-C`
marker-collapse bug and the SetUp contamination of naïve whole-process totals.

**Why marker mode was abandoned:** see the retracted file. Fresh marker-mode SEQ_ADD
gave 0.19×/0.00×/0.49× with thread collapse; whole-process wrapper mode is stable and
reads both sockets cleanly (per-socket DF totals land on HWThread 0 / HWThread 128,
balanced to within ≤8%).

**Common parameters:** N = 131072, L = 40, **batch B = 512** (`RS_BATCH_SIZE=512`),
256 threads, AMD EPYC 9754 / rpc-91-2 / job 134030 / zen4 exclusive, 2026-07-29–30,
LIKWID 5.4.1-daemon. Logical bytes: `Vpoly = N·L·8 = 0.041943 GB/poly`;
read logical/iter = `B·(arrays−1)·Vpoly` (42.95 GB for 3-array kernels, 21.47 GB for
NTT's 2-array round-trip); write logical/iter = `B·1·Vpoly` = 21.47 GB.

| Kernel | Access pattern | Shuffle | **Read amp** | **Write amp** | 3-run spread |
|--------|----------------|---------|:---:|:---:|:---:|
| `RS_SEQ_ADD` | Sequential | None (0) | **1.51×** | **1.00×** | ≤1.5% |
| `RS_GATHER_ADD` | Gather | Poly (1) | **1.00×** ¹ | **1.00×** ³ | ≤0.9% |
| `RS_GATHER_ADD` | Gather | Coeff (2) | **1.87×** | **1.01×** ² | ≤0.8% |
| `RS_SCATTER_GATHER_TRIAD` | Scatter-Gather | Coeff (2) | **3.84×** | **2.44×** | ≤0.1% |
| `RS_NTT_ROUNDTRIP` | NTT round-trip | None (0) | **2.78×** | **2.20×** | ≤0.6% |

Read-amp range: **1.00× – 3.84×**. Write-amp range: **1.00× – 2.44×**.

**Replication note (supersedes the old "single-shot" caveat).** Unlike the retracted
marker data, these are **replicated**: three high-iteration runs per cell, total DRAM
volume agreeing within **≤1.5%** (typically <1%), both sockets balanced. The differential
per-iteration value is the slope of two independently-measured points. Error bars at
paper scale are dominated by this ≤1.5% run-to-run spread; report as tight point estimates.

### Physical-bandwidth sanity check (resolves the old >peak problem)

Implied physical DRAM bandwidth = (measured read + write bytes/iter) ÷ wall time/iter.
**All five kernels fall well under the 858 GiB/s peak** — the corrected numbers are
physically consistent (the retracted socket-summed table was not):

| Kernel | ms/iter | phys GB/iter | phys BW (GiB/s) | % of peak |
|--------|:---:|:---:|:---:|:---:|
| SEQ_ADD | 122 | 86.1 | 658 | 77% |
| GATHER_POLY | 160 | 64.4 | 375 | 44% |
| GATHER_COEFF | 170 | 102.0 | 559 | 65% |
| SGT_COEFF | 446 | 217.5 | 454 | 53% |
| NTT | 534 | 107.0 | 187 | 22% |

### Mechanistic interpretation

Amplification is driven by **sub-cache-line (8-byte) random access**, which wastes
64-byte cache-line bandwidth — not by "irregularity" in the abstract. Read amps order
monotonically by degree of sub-cache-line randomness:
`poly (1.00) < seq (1.51) < gather-coeff (1.87) < ntt (2.78) < sgt (3.84)`.

- **SEQ_ADD — read 1.51×, write 1.00×.** Write-back is one clean array (1.00×), so the
  0.5× read excess is **read-side write-allocate / RFO + prefetch** on the sequential
  triple-stream (reading each output line before overwrite). Confirmed by the matched
  write measurement.
- **GATHER_ADD_POLY — read 1.00× ¹ (flagged, legitimate).** Poly-mode shuffles the *order*
  of whole polys, but each poly is a ≥1 MiB block read fully sequentially → **zero
  sub-cache-line waste**. This is the true no-amplification floor; poly-granularity
  "irregularity" is sequential at the cache/DRAM level. It sits *below* SEQ_ADD because it
  does not incur SEQ_ADD's triple-stream prefetch/RFO overhead (why that overhead is
  specific to fully-sequential triple-streaming is not fully micro-architecturally pinned).
- **GATHER_ADD_COEFF — read 1.87×, write 1.01× ² (write flagged, legitimate).** 8-byte
  random *reads* cause partial-cache-line waste (moderate: the 1 MiB tower is partly cache-
  resident). The *write* is **sequential** (`seqC[j] = …`) → one clean array → ~1.0×, as
  expected; the low write-amp is correct, not an error.
- **SCATTER_GATHER_TRIAD_COEFF — read 3.84×, write 2.44×.** Random 8-byte access on **both**
  ends (gather read + scatter write). Highest read amp; partial-line scatter writes force
  RFO + write-back → real write amplification.
- **NTT_ROUNDTRIP — read 2.78×, write 2.20×.** Strided butterfly access across log₂N stages
  → cache-line under-utilization + conflict misses on both read and write.

### Per-kernel caveats

1. **GATHER_ADD_POLY read = 1.00× is below SEQ_ADD's 1.51×** — expected (see mechanism).
   Implication for the paper: **SEQ_ADD is not the clean "unit" baseline**; the ~1.0×
   floor is poly/sequential (no cache-line waste), and SEQ_ADD's 1.51× is itself a mild
   sequential-streaming overhead. Frame the floor as ~1.0×.
2. **GATHER_ADD_COEFF write = 1.01× is below 1.51×** — expected: gather kernels write
   sequentially, so their write side has no amplification.
3. **GATHER_ADD_POLY write = 1.00×** (measured 2026-07-30, whole-process differential,
   3-run spread 0.05%) — confirms the prediction of caveat 2 (a sequential write incurs no
   amplification). The table and both amplification figures now cover all five kernels in
   both directions.

### `CT_SEQ_ADD_INPLACE` — still not hardware-validated (Ciphertext backend)

Marker mode induces a ~148× stall; whole-process capture includes `CTFixture::SetUp()`
ciphertext-encryption traffic. The whole-process **differential** method used here for the
DCRT kernels would in principle cancel that SetUp cost — a candidate follow-up — but it was
not attempted this pass. Not part of the validated set.

---

## Part 2 — Matched correlation: FHE-STREAM CT kernels vs real OpenFHE ops

**(Carried forward unchanged — this was never marker-derived; it is unmarked
Google-Benchmark timing, unaffected by the Part 1 retraction.)**

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
- **Reproduce Part 1** (per kernel, per direction):
  ```bash
  RS_BATCH_SIZE=512 OMP_NUM_THREADS=256 likwid-perfctr -g MEMREAD -C 0-255 -o out.csv \
    ./build/fhe_raiderstream --benchmark_filter='FHERaiderSTREAM/RS_SEQ_ADD/131072/40/0' \
    --benchmark_min_time=30x --benchmark_repetitions=1     # repeat at 5x; differential
  ```
- **Reproduce Part 2** exactly:
  ```bash
  RS_BATCH_SIZE=256 OMP_NUM_THREADS=256 OMP_PROC_BIND=true OMP_PLACES="{0:256}" \
  ./build/fhe_raiderstream \
    --benchmark_filter='CTFixture/(CT_SEQ_ADD|CT_SEQ_MULT_NO_RELIN|CT_SEQ_RELIN)/131072/40/0$' \
    --benchmark_min_time=1s --benchmark_repetitions=5 \
    --benchmark_out=<file>.json --benchmark_out_format=json
  # RELIN re-run: same, filter CT_SEQ_RELIN only, --benchmark_repetitions=10
  ```
