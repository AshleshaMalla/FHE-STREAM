# FHE-RaiderSTREAM Project Structure: Modular Refactoring

## Overview

The FHE-RaiderSTREAM benchmark has been refactored into a **modular, scalable architecture** following the **Separation of Concerns** design pattern. This structure enables easy addition of Phase 2 irregular kernels and future extensions while maintaining code clarity and maintainability.

---

## Module Breakdown

### 1. **src/FHERaiderSTREAM.h** — The Blueprint
**Purpose:** Class definition and public interface

**Contents:**
- `class FHERaiderSTREAM : public benchmark::Fixture`
- Member variables:
  - `std::vector<lbcrypto::DCRTPoly> A, B, C` — Working arrays
  - `std::vector<std::size_t> IDX, IDX_WRITE` — Read/write poly indices
  - `std::vector<std::size_t> COEFF_IDX, COEFF_IDX_WRITE` — Read/write coeff indices
  - `std::vector<lbcrypto::NativeInteger> towerModuli` — Per-tower moduli
  - `std::vector<lbcrypto::NativeInteger> towerMu` — Barrett constants
  - `int64_t scalar` — Scalar multiplier (default: 3)
  - `lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc` — FHE context
  - `std::shared_ptr<lbcrypto::ILDCRTParams<...>> params` — RNS parameters

- Method declarations:
  - `void SetUp(const benchmark::State& state) override`
  - `void TearDown(const benchmark::State& state) override`

**Key Features:**
- Fully documented with academic-style comments
- Flexible scalar parameter for extension
- Clean interface for fixture inheritance

---

### 2. **src/fixture.cpp** — The Engine
**Purpose:** Fixture lifecycle implementation

**Implements:**
- **SetUp()**: 
  - Saves/restores OpenMP thread count (prevents OpenFHE override)
  - Constructs cyclotomic RNS parameters
  - Pre-computes Barrett reduction constants
  - Creates BFV CryptoContext
  - Calculates polynomial count for 4 GiB minimum footprint
  - **NUMA first-touch parallel initialization** of arrays A, B, C
  - Initializes read/write index vectors with distinct seeds
  - Single-iteration configuration logging

- **TearDown()**:
  - Releases OpenFHE resources
  - Clears precomputed values
  - Forces immediate memory deallocation via `swap()`
  - Calls `malloc_trim(0)` on glibc systems

**Critical Preserved Logic:**
```cpp
// Thread management (prevents OpenFHE from limiting threads)
const int savedThreads = omp_get_max_threads();
lbcrypto::OpenFHEParallelControls.SetNumThreads(1);
omp_set_num_threads(savedThreads);

// NUMA-aware allocation
#pragma omp parallel for schedule(static)
for (std::size_t i = 0; i < nPolys; ++i) {
  // First-touch initialization per thread
}
```

---

### 3. **src/kernels_sequential.cpp** — The Workloads
**Purpose:** Benchmark kernel definitions and registration

**Implements 4 BENCHMARK_DEFINE_F kernels:**

1. **RS_SEQ_COPY**: `C[i] = A[i]`
   - Binary pattern (read A, write C)
   - Bytes processed: 2 × data size

2. **RS_SEQ_SCALE**: `B[i] = scalar × C[i]`
   - Includes modular multiplication (ModMulFast)
   - Bytes processed: 2 × data size

3. **RS_SEQ_ADD**: `C[i] = A[i] + B[i]`
   - Ternary pattern (read A+B, write C)
   - Includes modular addition (ModAddFast)
   - Bytes processed: 3 × data size

4. **RS_SEQ_TRIAD**: `A[i] = B[i] + scalar × C[i]`
   - Combined multiply+add (classic stream triad)
   - Bytes processed: 3 × data size

**Critical Preserved Logic:**
```cpp
// Prevent compiler dead code elimination
benchmark::DoNotOptimize(tower);

// Memory fence between iterations
benchmark::ClobberMemory();

// Multi-threaded loop distribution
#pragma omp parallel for schedule(static)
for (std::size_t i = 0; i < nPolys; ++i) {
  // Loop over towers and coefficients...
}
```

**Parameter Registration:**
```cpp
void SchemeArgs(benchmark::internal::Benchmark* b) {
  b->Args({1 << 16, 32});  // CKKS: large ring, deep tower stack
  b->Args({1 << 15, 16});  // BFV:  medium ring, moderate towers
  b->Args({1 << 11, 2});   // TFHE: small ring, minimal towers
}

BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_COPY)->Apply(SchemeArgs);
// ... (3 more kernels)
```

---

### 4. **src/kernels_gather.cpp** — Gather Workloads
**Purpose:** Irregular access (random read) kernels

**Implements:**
- `RS_GATHER_COPY`, `RS_GATHER_SCALE`, `RS_GATHER_ADD`, `RS_GATHER_TRIAD`
- Poly and Coeff modes merged under primary kernel names

---

### 5. **src/kernels_scatter.cpp** — Scatter Workloads
**Purpose:** Irregular access (random write) kernels

**Implements:**
- `RS_SCATTER_COPY`, `RS_SCATTER_SCALE`, `RS_SCATTER_ADD`, `RS_SCATTER_TRIAD`
- Poly and Coeff modes merged under primary kernel names

---

### 6. **src/kernels_scatter_gather.cpp** — Scatter-Gather Workloads
**Purpose:** Double-indirection random read + random write kernels

**Implements:**
- `RS_SCATTER_GATHER_COPY`, `RS_SCATTER_GATHER_SCALE`, `RS_SCATTER_GATHER_ADD`, `RS_SCATTER_GATHER_TRIAD`
- Poly and Coeff modes with distinct read/write index vectors

---

### 7. **include/StreamCore.h** — Kernel Dispatchers
**Purpose:** Shared kernel dispatch and labeling

**Adds:**
- `RunScatterGatherPoly`
- `RunScatterGatherCoeff`

---

### 8. **src/main.cpp** — The Control Room
**Purpose:** Entry point and framework initialization

**Contents:**
- Custom help handler for `--help`
- ASCII banner printed on every run

**Why here?**
- Banner and help are user-facing entry-point behavior
- Kernel registration still lives in each kernel file

---

### 9. **CMakeLists.txt** — The Build System
**Purpose:** Multi-source compilation configuration

**Key Updates:**
```cmake
add_executable(fhe_raiderstream
  src/main.cpp
  src/fixture.cpp
  src/kernels_sequential.cpp
  src/kernels_gather.cpp
  src/kernels_scatter.cpp
  src/kernels_scatter_gather.cpp
)
```

**Ensures:**
- All source files compiled in correct order
- OpenMP properly linked: `target_link_libraries(...OpenMP::OpenMP_CXX)`
- OpenFHE headers and libraries accessible
- Google Benchmark linked and available

---

## Design Principles

### 1. **Separation of Concerns**
| Module | Responsibility |
|--------|-----------------|
| Header | Class interface and structure |
| fixture.cpp | Lifecycle and setup/teardown |
| kernels_sequential.cpp | Benchmark operations |
| main.cpp | Framework entry point |

### 2. **Scalability for Phase 2**
To add irregular kernels (Phase 2):
```
Create: src/kernels_irregular.cpp
├── BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_IRR_GATHER)
├── BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_IRR_SCATTER)
└── BENCHMARK_REGISTER_F(...)->Apply(SchemeArgs);

Update: CMakeLists.txt
└── Add src/kernels_irregular.cpp to add_executable
```

### 3. **Preserved Threading**
- **OpenMP disabled in OpenFHE**: `SetNumThreads(1)` then restored
- **OpenMP enabled for benchmarks**: `#pragma omp parallel for`
- **Result**: Multi-threaded kernel execution, single-threaded FHE ops
- **Thread count**: Controlled via `OMP_NUM_THREADS` environment variable

### 4. **Compiler Optimizations Prevented**
- **DoNotOptimize()**: Prevents loop or computation elision
- **ClobberMemory()**: Memory fence preventing CSE across iterations
- **Result**: Prevents false speedups from dead code elimination

---

## Compilation & Execution

### Build
```bash
cd fhe-raiderstream
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

### Run
```bash
# Single-threaded baseline
OMP_NUM_THREADS=1 ./fhe_raiderstream --benchmark_filter=RS_SEQ

# Multi-threaded benchmark (14 cores)
OMP_NUM_THREADS=14 OMP_PROC_BIND=close OMP_PLACES=cores ./fhe_raiderstream --benchmark_filter=RS_SEQ
```

### Typical Output
```
======================================================================
  FHE-RaiderSTREAM Setup Configuration
======================================================================
  Ring Dimension:        65536
  Number of Towers:      32
  Number of Polys:       86
  Per-Array Footprint:   1.44284 GB
  Total Footprint (A+B+C): 4.32852 GB
======================================================================

Benchmark                                     Time             CPU   Iterations UserCounters...
FHERaiderSTREAM/RS_SEQ_COPY/65536/32   19.8 ms    17.5 ms           39 bytes_per_second=153Gi/s
```

---

## File Dependencies

```
CMakeLists.txt
├── src/main.cpp
│   └── benchmark/benchmark.h
├── src/fixture.cpp
│   ├── FHERaiderSTREAM.h
│   ├── openfhe.h
│   └── <omp.h, malloc.h>
└── src/kernels_sequential.cpp
    ├── FHERaiderSTREAM.h
    └── benchmark/benchmark.h

FHERaiderSTREAM.h
└── openfhe.h
```

---

## Extensions & Future Work

### Phase 2: Irregular Kernels
Add `src/kernels_irregular.cpp` with:
- Gather/Scatter patterns on polynomial indices
- Random access workloads
- Cache miss measurement

### Phase 3: Heterogeneous Execution
Add `src/kernels_gpu.cpp` with:
- GPU-accelerated kernel variants
- PCIe bandwidth measurement

### Testing & Profiling
Proposed additions:
- Unit tests: `tests/test_kernels.cpp`
- Profiling harness: `tools/profile.cpp`
- Analysis scripts: `scripts/analyze_results.py`

---

## Summary

This modular refactoring provides:

✅ **Clean separation of concerns** — Each file has a single responsibility
✅ **Preserved logic** — All threading, optimization, and NUMA logic intact
✅ **Scalability** — Easy addition of irregular kernels (Phase 2)
✅ **Maintainability** — Well-documented, professional code structure
✅ **Correctness** — All benchmarks run and produce valid results

**Total lines**: ~700 lines across 5 files
**Complexity**: Low — pure C++17, no external frameworks beyond OpenFHE/OpenMP/Benchmark
