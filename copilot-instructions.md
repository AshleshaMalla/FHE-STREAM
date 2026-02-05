# Copilot Instructions for FHE-RaiderSTREAM Setup

You are an expert C++ developer specializing in High-Performance Computing (HPC) and Homomorphic Encryption (FHE) using the OpenFHE library. You are tasked with setting up a standalone benchmarking project named `FHE-RaiderSTREAM`.

## Project Goal
Create a standalone C++ project that benchmarks the **Sustainable Memory Bandwidth** of OpenFHE data structures (`DCRTPoly`) using access patterns inspired by the RaiderSTREAM benchmark (Copy, Add, Gather, Scatter).

## Project Structure
The project must follow this exact directory structure:

```text
fhe-raiderstream/
├── CMakeLists.txt          # Build configuration finding OpenFHE, OpenMP, and Google Benchmark
├── copilot-instructions.md # This file
├── README.md               # Project documentation
└── src/
    ├── main.cpp            # Main entry point with BENCHMARK_MAIN()
    └── fhe_raiderstream.h  # (Optional) Header for kernel definitions
```
