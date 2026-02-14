# FHE-RaiderSTREAM: A Micro-Architectural Benchmark Suite for Homomorphic Encryption

## 1. Abstract
FHE-RaiderSTREAM is a specialized micro-benchmark suite designed to characterize the memory bandwidth bottlenecks inherent to Homomorphic Encryption (FHE) workloads. Unlike traditional FHE benchmarks that measure high-level algorithmic throughput, this suite isolates the fundamental interactions between OpenFHE data structures (DCRTPoly) and the underlying hardware memory subsystem. By implementing the standard STREAM operations (Copy, Scale, Add, Triad) across distinct stochastic access patterns, FHE-RaiderSTREAM quantifies the "Memory Wall"—the performance gap between sequential bandwidth capability and the latency-bound random access patterns required by operations such as Key Switching and Bootstrapping.

---

## 2. Project Architecture

The system utilizes a modular, template-driven architecture designed to ensure zero-overhead abstraction while maintaining rigorous scientific reproducibility.

### 2.1 Core Execution Logic (`include/StreamCore.h`)
This component serves as the central dispatch mechanism. It utilizes C++ template metaprogramming to generate highly optimized machine code for different access patterns.
* **Templated Dispatchers:** Generic functions (`RunSequential`, `RunGather`, `RunScatterGather`) that handle OpenMP threading, loop unrolling, and memory safety checks.
* **Lambda Injection:** Kernel logic is injected via lambda functions, allowing the reuse of the same traversal logic for different arithmetic operations (Copy, Scale, Add, Triad).
* **Scalar Optimization:** Arithmetic constants are pre-converted to `NativeInteger` format outside of execution loops to eliminate redundant type conversion cycles.

### 2.2 Lifecycle Management (`src/fixture.cpp`)
The test harness acts as the system's runtime environment, managing the initialization and teardown of complex FHE contexts.
* **Context Generation:** Automates the creation of `CryptoContext` and `DCRTPoly` objects with cryptographically secure parameters.
* **NUMA-Aware Allocation:** Ensures memory is allocated and touched on the local NUMA node to prevent remote access latency from contaminating benchmark results.
* **Stochastic Index Generation:** Generates high-entropy permutation vectors (`IDX_READ`, `IDX_WRITE`) using separate random seeds to ensure statistical independence between read and write streams.

### 2.3 Kernel Implementations (`src/kernels/`)
The benchmark suite is partitioned into four primary access regimes:
1.  **Sequential (`kernels_sequential.cpp`):** Measures the hardware's peak theoretical throughput using linear memory access.
2.  **Gather (`kernels_gather.cpp`):** Isolates read latency by performing indirect reads (`A[IDX[i]]`) from random memory locations.
3.  **Scatter (`kernels_scatter.cpp`):** Isolates the Write-Allocate/Read-For-Ownership (RFO) penalty by performing indirect writes (`A[IDX[i]]`).
4.  **Scatter-Gather (`kernels_scatter_gather.cpp`):** Represents the worst-case scenario, combining random read latency with random write RFO overhead via double indirection.

---

## 3. Methodology & Access Patterns

The suite evaluates performance across three distinct levels of memory granularity, referred to as "Shuffle Modes."

| Shuffle Mode | Description | Micro-Architectural Target |
| :--- | :--- | :--- |
| **0: Sequential** | Linear access to contiguous memory. | **Hardware Prefetcher & Memory Controller Bandwidth.** |
| **1: Poly Shuffle** | Random access at 16MB granularity (Polynomial level). | **Translation Lookaside Buffer (TLB) & L3 Cache.** |
| **2: Coeff Shuffle** | Random access at 8-byte granularity (Coefficient level). | **L1/L2 Cache Latency & Line Fill Buffers.** |

### Supported Operations
Each access pattern is tested against the four standard STREAM vector operations, adapted for Modular Arithmetic:
* **COPY:** $C \leftarrow A$
* **SCALE:** $B \leftarrow s \cdot C \pmod q$
* **ADD:** $C \leftarrow A + B \pmod q$
* **TRIAD:** $A \leftarrow B + s \cdot C \pmod q$

---

## 4. Performance Characterization

Empirical analysis on commodity hardware (14-core CPU) has established distinct performance regimes for FHE workloads.

### 4.1 Throughput Saturation (Sequential Regime)
* **Throughput:** ~160 GiB/s
* **Observation:** The system successfully saturates the memory controller bandwidth. Arithmetic complexity (e.g., Modular Multiplication) is effectively hidden by the memory transfer time, confirming that FHE is memory-bound rather than compute-bound in linear regimes.

### 4.2 Latency-Bound Regime (Coeff Shuffle)
* **Throughput:** ~42 GiB/s
* **Observation:** When access granularity is reduced to individual coefficients (8 bytes), performance degrades by approximately 73%. This "Speed Floor" represents the physical limit of the CPU's ability to handle outstanding cache misses.
* **Key Finding:** The transition from simple Copy to complex Triad operations in this regime shows negligible performance impact (<10%), proving that optimizing arithmetic logic units (ALUs) yields diminishing returns without addressing memory latency.

---

## 5. Future Research Directions

### 5.1 Strided Access Analysis (NTT Benchmarking)
The next phase involves implementing `kernels_ntt.cpp` to measure the impact of power-of-two strided access patterns required by the Number Theoretic Transform (NTT). This will identify the "Critical Stride"—the specific jump distance at which the hardware prefetcher fails to predict future memory access.

### 5.2 Micro-Architectural Optimizations
Future iterations will utilize this benchmark as a sandbox to test software mitigation strategies:
* **Software Prefetching:** Evaluating the efficacy of `__builtin_prefetch` intrinsics to hide latency in Gather kernels.
* **Non-Temporal Stores:** Implementing streaming store intrinsics to bypass cache allocation during Scatter operations, thereby mitigating the RFO penalty.
* **Huge Page Allocation:** Investigating the use of Transparent Huge Pages (THP) to reduce TLB miss rates for small-ring parameters (e.g., Ring Dimension 2048).

### 5.3 Visualization & Modeling
Development of automated post-processing scripts to generate Roofline Models, plotting "Modular Operations per Byte" against "Memory Bandwidth" to visually quantify the efficiency gap between sequential and irregular access patterns.