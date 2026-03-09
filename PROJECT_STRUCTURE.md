# FHE-RaiderSTREAM: Module Reference

> **Kernel descriptions, bytes models, and run commands** are in
> [BENCHMARKS.md](BENCHMARKS.md).  This file documents module responsibilities
> and implementation details.

---

## Module Breakdown

### 1. `include/backends/dcrt/DCRTFixture.h` — DCRT Fixture Interface

Declares `class FHERaiderSTREAM : public benchmark::Fixture`.

**Member variables:**
- `std::vector<lbcrypto::DCRTPoly> A, B, C` — Working arrays
- `std::vector<std::size_t> IDX, IDX_WRITE` — Read/write poly indices
- `std::vector<std::size_t> COEFF_IDX, COEFF_IDX_WRITE` — Read/write coeff indices
- `std::vector<lbcrypto::NativeInteger> towerModuli, towerMu` — Per-tower moduli & Barrett constants
- `int64_t scalar` — Scalar multiplier (default: 3)
- `lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc` — FHE context
- `std::shared_ptr<lbcrypto::ILDCRTParams<...>> params` — RNS parameters

---

### 2. `src/backends/dcrt/DCRTFixture.cpp` — DCRT Lifecycle

**SetUp():**
- Saves/restores OpenMP thread count (prevents OpenFHE override)
- Constructs cyclotomic RNS parameters
- Pre-computes Barrett reduction constants
- Creates BFV CryptoContext
- Calculates polynomial count for 4 GiB minimum footprint
- **NUMA first-touch parallel initialization** of arrays A, B, C
- Initializes read/write index vectors with distinct seeds
- Single-iteration configuration logging
- `RS_BARRIER()` synchronization

**TearDown():**
- Forces immediate memory deallocation via `swap()`
- Calls `malloc_trim(0)` on glibc systems
- `RS_BARRIER()` synchronization

---

### 3. `include/backends/dcrt/StreamCore.h` — Templated Dispatchers

Shared kernel dispatch functions using C++ template metaprogramming:
- `RunSequential`, `RunGatherPoly`, `RunGatherCoeff`
- `RunScatterPoly`, `RunScatterCoeff`
- `RunScatterGatherPoly`, `RunScatterGatherCoeff`
- `RunNTT`
- `RS_KEYSWITCH_MOCK` — Inline FMA helper: `A[i] = A[i] + B[i] * C[0]`

Kernel logic is injected via lambda functions, allowing reuse of the same
traversal logic for different arithmetic operations.

---

### 4. `src/backends/dcrt/kernels_*.cpp` — DCRT Kernel Files

| File                       | Kernels                                    |
|----------------------------|--------------------------------------------|
| `kernels_sequential.cpp`   | RS_SEQ_COPY/SCALE/ADD/TRIAD, RS_KEYSWITCH_MOCK |
| `kernels_gather.cpp`       | RS_GATHER_COPY/SCALE/ADD/TRIAD             |
| `kernels_scatter.cpp`      | RS_SCATTER_COPY/SCALE/ADD/TRIAD            |
| `kernels_scatter_gather.cpp` | RS_SCATTER_GATHER_COPY/SCALE/ADD/TRIAD   |
| `kernels_ntt.cpp`          | RS_NTT_ROUNDTRIP                           |

---

### 5. `include/backends/ciphertext/CTFixture.h` — Ciphertext Fixture Interface

Declares `class CTFixture : public benchmark::Fixture`.

**Member variables:**
- `CryptoContext<DCRTPoly> cc` — Crypto context
- `KeyPair<DCRTPoly> keyPair` — Public/private key pair
- `std::vector<Ciphertext<DCRTPoly>> ct_A, ct_B, ct_C` — Working set
- `std::vector<Ciphertext<DCRTPoly>> ct_A_deg2` — Pre-computed degree-2 ciphertexts

---

### 6. `src/backends/ciphertext/CTFixture.cpp` — Ciphertext Lifecycle

**SetUp():**
- Creates BFVrns context with `SetSecurityLevel(HEStd_NotSet)`
- Uses conditional plaintext modulus: 65537 for RingDim ≤ 32768, 786433 for larger
  (NTT condition: `(t-1) % (2*RingDim) == 0`)
- Generates key pair and relinearization keys
- Encrypts working-set vectors `ct_A`, `ct_B`; clones `ct_C`
- Pre-computes `ct_A_deg2 = EvalMultNoRelin(ct_A, ct_B)` for relinearization benchmarks

**TearDown():**
- Swap-clears all vectors (`ct_A`, `ct_B`, `ct_C`, `ct_A_deg2`)
- Calls `malloc_trim(0)`

---

### 7. `src/backends/ciphertext/kernels_ct.cpp` — Ciphertext Kernels

Contains CT_SEQ_COPY, CT_SEQ_ADD, CT_SEQ_SCALE, CT_SEQ_TRIAD,
CT_SEQ_MULT_NO_RELIN, CT_SEQ_RELIN, CT_SEQ_ADD_INPLACE, CT_MPI_SENDRECV.

Includes OpenFHE serialization headers (`ciphertext-ser.h`,
`scheme/bfvrns/bfvrns-ser.h`, `utils/serial.h`) for the MPI kernel.

---

### 8. `include/common/MPIUtils.h` — Distributed Helpers

- `RS_BARRIER()`: Wrapper for `MPI_Barrier(MPI_COMM_WORLD)`.  No-op when
  compiled without `RAIDERSTREAM_MPI`.
- `AggregateBandwidth(state)`: `MPI_Reduce` sum of `bytes_processed` across
  ranks; Rank 0 publishes the `AggregateBandwidth` counter.

---

### 9. `src/common/BenchmarkUtils.cpp` — Shared Parameters

`RaiderSTREAM_Arguments` generates all parameter combinations:
- **RingDim:** {16384, 32768, 65536, 131072}
- **Depth:** {1, 5, 20, 40}
- **BatchSize:** From `RS_BATCH_SIZE` environment variable (default 100)

---

### 10. `src/main.cpp` — Entry Point

- `MPI_Init` / `MPI_Finalize` lifecycle (when MPI enabled)
- Rank-0 only: ASCII banner, `--help` handler, `--rs_print_setup` flag
- `BENCHMARK_MAIN()` delegation

---

### 11. `CMakeLists.txt` — Build System

Feature-flag-driven compilation:
- `ENABLE_DCRT` (ON) — compiles DCRT fixture + kernel files
- `ENABLE_CIPHERTEXT` (OFF) — compiles CT fixture + kernels_ct.cpp
- `ENABLE_MPI` (ON) — links MPI, defines `RAIDERSTREAM_MPI`

Auto-detects monolithic vs. split DCRT kernel layout via `EXISTS` check.

---

## Design Principles

### Separation of Concerns
Each file has a single responsibility — see [ARCHITECTURE.txt](ARCHITECTURE.txt)
for the visual breakdown.

### Thread Management
- OpenMP disabled inside OpenFHE: `SetNumThreads(1)` then restored
- OpenMP enabled for benchmark kernels: `#pragma omp parallel for`
- Thread count controlled via `OMP_NUM_THREADS`

### Compiler Optimization Prevention
- `benchmark::DoNotOptimize()` — prevents dead code elimination
- `benchmark::ClobberMemory()` — memory fence preventing CSE across iterations

### MPI Conditional Compilation
All MPI code is wrapped in `#ifdef RAIDERSTREAM_MPI`, allowing the project to
build and run without MPI installed.

---

## Future Work

### GPU Backend
Add `src/backends/gpu/` with PCIe bandwidth measurement kernels.

### Profiling & Testing
- Unit tests: `tests/test_kernels.cpp`
- Hardware counter integration: `perf` / PAPI wrappers
