# Counter Access Validation Baseline
**Date:** July 24, 2026  
**Purpose:** Validate DRAM hardware counter access under exclusive node allocation prior to all benchmark experiments.

## Measurement Provenance

| Platform | Slurm Job ID | Partition | Allocation |
|----------|--------------|-----------|------------|
| AMD EPYC 9754 | **130567** | `zen4` | `salloc --exclusive --nodes=1 --partition=zen4` |
| Intel Xeon Gold 6448Y | **130577** | `h100` | `salloc --exclusive --nodes=1 --partition=h100` |

**Counter Tool**
- LIKWID 5.4.1 (daemon build)

---

# Counter Validation Summary

| Item | AMD EPYC 9754 (rpc-91-3, zen4 partition) | Intel Xeon Gold 6448Y (rpg-93-4, h100 partition) |
|------|-------------------------------------------|--------------------------------------------------|
| Counter tool | LIKWID 5.4.1 (daemon build) | LIKWID 5.4.1 (daemon build) |
| Access method | Data Fabric Counters (DFC0–11), per-channel | Uncore MBOX CAS_COUNT_RD/WR, per-channel (MBOX0–7 active, MBOX8–11 unpopulated) |
| Groups used | MEMREAD, MEMWRITE (separate runs) | MEM (combined read + write) |
| Allocation | `salloc --exclusive --nodes=1 --partition=zen4` | `salloc --exclusive --nodes=1 --partition=h100` |
| Core range tested | `-C 0-255` (full node, both sockets) | `-C 0-63` (full node, both sockets) |
| Socket attribution pattern | Thread 0 → Socket 0, Thread 128 → Socket 1 | Thread 0 → Socket 0, Thread 32 → Socket 1 |
| Idle read bandwidth (Socket 0) | 25.63 MB/s | 44.87 MB/s |
| Idle read bandwidth (Socket 1) | 44.75 MB/s | 46.07 MB/s |
| Idle read bandwidth (Node Total) | **70.38 MB/s** | **90.94 MB/s** |
| Idle write bandwidth (Socket 0) | 2.17 MB/s | 43.92 MB/s |
| Idle write bandwidth (Socket 1) | 3.40 MB/s | 45.24 MB/s |
| Idle write bandwidth (Node Total) | **5.57 MB/s** | **89.16 MB/s** |
| Idle combined bandwidth (Node Total) | **75.95 MB/s** | **180.09 MB/s** |
| Vendor theoretical peak (Node) | ~921.6 GB/s (12 channels × DDR5-4800 × 2 sockets) | ~614.4 GB/s (8 channels × DDR5-4800 × 2 sockets)\* |
| Idle floor as % of theoretical peak | **0.008%** | **0.029%** |
| Status | ✅ Confirmed — clean, exclusive, full-node counter access | ✅ Confirmed — clean, exclusive, full-node counter access |

\* Confirm installed DIMM speed against system inventory (Table I) before publication.

---

# Observations

## AMD EPYC 9754

- Successfully accessed all 12 Data Fabric memory channels.
- Per-socket attribution verified using thread IDs:
  - Thread 0 → Socket 0
  - Thread 128 → Socket 1
- Idle DRAM traffic measured approximately **75.95 MB/s**, corresponding to only **0.008%** of theoretical peak bandwidth.
- Background traffic is negligible and indicates successful exclusive-node isolation.

## Intel Xeon Gold 6448Y

- Successfully accessed all populated MBOX controllers (MBOX0–7).
- MBOX8–11 correctly reported as unpopulated.
- Per-socket attribution verified using:
  - Thread 0 → Socket 0
  - Thread 32 → Socket 1
- Idle DRAM traffic measured approximately **180.09 MB/s**, corresponding to only **0.029%** of theoretical peak bandwidth.
- Background traffic remains negligible and suitable for bandwidth benchmarking.

---

# Reproducibility Notes

- Measurements performed under **exclusive node allocations**.
- No user workloads were present during data collection.
- Idle measurements establish the baseline hardware counter noise floor before executing RaiderSTREAM benchmarks.
- These measurements validate that subsequent DRAM bandwidth experiments are not materially affected by background system traffic.

---

# Recommended Methodology Statement

> Prior to benchmarking, idle DRAM traffic was measured under exclusive node allocation to verify hardware counter integrity and node isolation. The AMD EPYC 9754 platform exhibited an idle bandwidth of approximately **75.95 MB/s (0.008% of theoretical peak)**, while the Intel Xeon Gold 6448Y platform measured **180.09 MB/s (0.029% of theoretical peak)**. These negligible background traffic levels confirm that subsequent bandwidth measurements are effectively free from interference.

---

Kernel	Read amp.	Write amp.
SEQ_ADD	2.57×	—
GATHER_ADD_POLY	3.90×	—
GATHER_ADD_COEFF	3.27×	1.56×
SCATTER_GATHER_TRIAD_COEFF	5.68×	2.24×