# FHE-RaiderSTREAM: A Micro-Architectural Benchmark Suite for Homomorphic Encryption

## 1. Abstract
FHE-RaiderSTREAM is a specialized micro-benchmark suite designed to characterize the memory bandwidth bottlenecks inherent to Homomorphic Encryption (FHE) workloads. Unlike traditional FHE benchmarks that measure high-level algorithmic throughput, this suite isolates the fundamental interactions between OpenFHE data structures and the underlying hardware memory subsystem across three distinct backends: raw DCRTPoly polynomials, full Ciphertext objects, and distributed MPI transfers. By implementing the standard STREAM operations (Copy, Scale, Add, Triad) alongside FHE-specific operations (KeySwitch, Relinearization, Serialization) across distinct access patterns, FHE-RaiderSTREAM quantifies the "Memory Wall" at every layer of the stack—hardware, math, software abstraction, and network.

> For a complete kernel-by-kernel reference including bytes models and run
> commands, see [BENCHMARKS.md](BENCHMARKS.md).

---

## 2. Project Architecture

The system utilizes a modular, multi-backend architecture controlled by CMake feature flags (`ENABLE_DCRT`, `ENABLE_CIPHERTEXT`, `ENABLE_MPI`). Each backend has its own Google Benchmark Fixture with independent SetUp/TearDown lifecycle management.

> For detailed module responsibilities, file paths, and dependency graphs,
> see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) and
> [ARCHITECTURE.txt](ARCHITECTURE.txt).

### 2.1 The DCRTPoly Backend (Hardware & Math Limits)
Operates on raw `NativePoly` / `NativeInteger` vectors with zero abstraction overhead. Measures the hardware's peak theoretical throughput — the "Speed Limit." Includes sequential, gather, scatter, scatter-gather, NTT, and KeySwitch-mock access patterns (21 kernels total).

### 2.2 The Ciphertext Backend (Software & Capacity Limits)
Operates on full `Ciphertext<DCRTPoly>` objects managed by `CryptoContext`, including `shared_ptr` wrappers, metadata, and dynamic allocation. Reports only payload polynomial bytes ("Goodput") so the throughput gap between backends directly quantifies the software tax. Includes sequential STREAM operations, in-place variants, multiplication without relinearization, and isolated relinearization (8 kernels total).

### 2.3 The MPI Extension (Network & Serialization Limits)
Adds distributed execution via Open MPI. All kernels automatically synchronize with `MPI_Barrier` and aggregate throughput with `MPI_Reduce`. The dedicated `CT_MPI_SENDRECV` kernel measures the serialization tax of converting live C++ ciphertext objects into flat byte streams for inter-rank transfer.

---

## 3. Methodology & Access Patterns

The DCRTPoly backend evaluates performance across three distinct levels of memory granularity:

| Access Pattern | Description | Micro-Architectural Target |
| :--- | :--- | :--- |
| **Sequential** | Linear access to contiguous memory. | **Hardware Prefetcher & Memory Controller Bandwidth.** |
| **Gather/Scatter** | Random access at polynomial granularity. | **Translation Lookaside Buffer (TLB) & L3 Cache.** |
| **Scatter-Gather** | Double indirection (random read + random write). | **L1/L2 Cache Latency & Line Fill Buffers.** |

### Supported Operations
Each access pattern is tested against the four standard STREAM vector operations, adapted for Modular Arithmetic:
* **COPY:** $C \leftarrow A$
* **SCALE:** $B \leftarrow s \cdot C \pmod q$
* **ADD:** $C \leftarrow A + B \pmod q$
* **TRIAD:** $A \leftarrow B + s \cdot C \pmod q$

Additionally, FHE-specific kernels measure:
* **NTT Round-Trip:** Forward + inverse NTT butterfly bandwidth.
* **KeySwitch Mock:** FMA streaming with zero temporal locality: $A_i \leftarrow A_i + B_i \times C_0$.
* **Relinearization:** Isolated evaluation-key streaming overhead.
* **Serialization:** Ciphertext → byte stream → MPI transfer → deserialization.

---

## 4. Performance Characterization

Empirical analysis on commodity hardware (14-core aarch64 CPU) has established distinct performance regimes for FHE workloads.

### 4.1 Throughput Saturation (Sequential Regime)
* **Throughput:** ~160 GiB/s
* **Observation:** The system successfully saturates the memory controller bandwidth. Arithmetic complexity (e.g., Modular Multiplication) is effectively hidden by the memory transfer time, confirming that FHE is memory-bound rather than compute-bound in linear regimes.

### 4.2 Latency-Bound Regime (Irregular Access)
* **Throughput:** ~42 GiB/s
* **Observation:** When access granularity is reduced to individual coefficients (8 bytes), performance degrades by approximately 73%. This "Speed Floor" represents the physical limit of the CPU's ability to handle outstanding cache misses.
* **Key Finding:** The transition from simple Copy to complex Triad operations in this regime shows negligible performance impact (<10%), proving that optimizing arithmetic logic units (ALUs) yields diminishing returns without addressing memory latency.

### 4.3 Software Abstraction Tax (Ciphertext vs. DCRTPoly)
* **Observation:** Comparing `CT_SEQ_ADD` to `RS_SEQ_ADD` at the same RingDim and Depth reveals the bandwidth destroyed by `std::shared_ptr` chasing, metadata copies, and the OS memory allocator. The `CT_SEQ_ADD_INPLACE` kernel recovers a measurable fraction by eliminating `malloc` from the hot path.

### 4.4 Capacity Wall (Relinearization)
* **Observation:** At high multiplicative depths (depth 40), the evaluation key set exceeds LLC capacity, causing every `Relinearize` call to thrash the entire cache hierarchy. The `CT_SEQ_RELIN` kernel, using pre-computed degree-2 inputs, isolates this cost from the multiplication itself.

### 4.5 Serialization Tax (MPI Transfer)
* **Observation:** `CT_MPI_SENDRECV` at RingDim=16384, Depth=5 shows ~228 MiB/s effective bandwidth — orders of magnitude below raw memory bandwidth. This confirms that OpenFHE's `Serial::Serialize` string conversion dominates end-to-end latency in distributed FHE pipelines before data even reaches the NIC.

---

## 5. Future Research Directions

### 5.1 Micro-Architectural Optimizations
Future iterations will utilize this benchmark as a sandbox to test software mitigation strategies:
* **Software Prefetching:** Evaluating the efficacy of `__builtin_prefetch` intrinsics to hide latency in Gather kernels.
* **Non-Temporal Stores:** Implementing streaming store intrinsics to bypass cache allocation during Scatter operations, thereby mitigating the RFO penalty.
* **Huge Page Allocation:** Investigating the use of Transparent Huge Pages (THP) to reduce TLB miss rates for small-ring parameters.

### 5.2 Heterogeneous Execution
GPU-accelerated kernel variants measuring PCIe transfer overhead and device-side FHE throughput.

### 5.3 Visualization & Modeling
Development of automated post-processing scripts to generate Roofline Models, plotting "Modular Operations per Byte" against "Memory Bandwidth" to visually quantify the efficiency gap between sequential and irregular access patterns.