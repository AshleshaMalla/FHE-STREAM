/*
  FHE-RaiderSTREAM Benchmark: Main Entry Point
  
  Launches the Google Benchmark framework with command-line argument processing.
  Provides a custom help menu for FHE-specific configuration options.
  Benchmark kernels are registered in kernels_*.cpp files.
*/

#include <benchmark/benchmark.h>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Global variable used by StreamCore.h for threading
int RS_Execution_Threads = 1;

void PrintBanner() {
  std::cout << R"(
                                                                                                                                    
 ▄▄▄▄▄▄▄ ▄▄▄   ▄▄▄  ▄▄▄▄▄▄▄       ▄▄▄▄▄▄▄                ▄▄              ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄   ▄▄▄▄   ▄▄▄      ▄▄▄ 
███▀▀▀▀▀ ███   ███ ███▀▀▀▀▀       ███▀▀███▄       ▀▀     ██             █████▀▀▀ ▀▀▀███▀▀▀ ███▀▀███▄ ███▀▀▀▀▀ ▄██▀▀██▄ ████▄  ▄████ 
███▄▄    █████████ ███▄▄          ███▄▄███▀  ▀▀█▄ ██  ▄████ ▄█▀█▄ ████▄  ▀████▄     ███    ███▄▄███▀ ███▄▄    ███  ███ ███▀████▀███ 
███▀▀    ███▀▀▀███ ███      ▀▀▀▀▀ ███▀▀██▄  ▄█▀██ ██  ██ ██ ██▄█▀ ██ ▀▀    ▀████    ███    ███▀▀██▄  ███      ███▀▀███ ███  ▀▀  ███ 
███      ███   ███ ▀███████       ███  ▀███ ▀█▄██ ██▄ ▀████ ▀█▄▄▄ ██    ███████▀    ███    ███  ▀███ ▀███████ ███  ███ ███      ███ 
                                                                                                                                    
                                                                                                                                    
)" << std::endl;
}

void PrintHelp() {
  std::cout << R"(
==============================================================================
                FHE-RaiderSTREAM Benchmark Suite
==============================================================================
 A sustainable memory bandwidth benchmark for OpenFHE structures (DCRTPoly).
 Measures the 'Memory Wall' in Homomorphic Encryption using RaiderSTREAM patterns.

 Usage:
   ./fhe_raiderstream [options]

------------------------------------------------------------------------------
 1. FILTERING BENCHMARKS (Most Important)
------------------------------------------------------------------------------
 Use --benchmark_filter=<regex> to select specific tests.

 Available Kernels:
   RS_SEQ      : Sequential access (Baseline Bandwidth)
   RS_GATHER   : Random Read  (Indirect Addressing: C[i] = A[IDX[i]])
   RS_SCATTER  : Random Write (Indirect Addressing: C[IDX[i]] = A[i])

 Available Operations:
   COPY        : Copy only (2 arrays)
   SCALE       : Scalar Multiplication (2 arrays + ModMul)
   ADD         : Vector Addition (3 arrays + ModAdd)
   TRIAD       : Stream Triad (3 arrays + ModMul + ModAdd)

------------------------------------------------------------------------------
 2. CONFIGURATION PARAMETERS
------------------------------------------------------------------------------
 Benchmarks are named as: <Kernel>/<RingDim>/<NumTowers>/<ShuffleMode>

 Ring Dimensions (N):
   2048   (2^11) : TFHE-like small ring
   32768  (2^15) : BFV/BGV typical ring
   65536  (2^16) : CKKS/Large BFV ring

 Towers (L):
   2      : Minimal RNS decomposition
   16     : Moderate depth
   32     : Deep circuit depth (High DRAM traffic)

 Shuffle Modes (M):
   0 : Sequential   (Standard Stream pattern)
   1 : Poly Shuffle (Random access of 16MB polynomials)
   2 : Coeff Shuffle(Random access of 8-byte coefficients -> The FHE Memory Wall)

------------------------------------------------------------------------------
 3. COMMON EXAMPLES
------------------------------------------------------------------------------
 Run everything (Phase 1 + 2):
   ./fhe_raiderstream

 Run only Scatter Triad kernels:
   ./fhe_raiderstream --benchmark_filter="RS_SCATTER_TRIAD"

 Run all kernels on the largest ring (65536):
   ./fhe_raiderstream --benchmark_filter="/65536/"

 Run standard STREAM (Sequential only):
   ./fhe_raiderstream --benchmark_filter="RS_SEQ"

 Quick functionality test (0.1s minimum time):
   ./fhe_raiderstream --benchmark_min_time=0.1

------------------------------------------------------------------------------
 4. STANDARD GOOGLE BENCHMARK OPTIONS
------------------------------------------------------------------------------
   --benchmark_list_tests       : List all available tests
   --benchmark_min_time=<t>     : Minimum seconds per benchmark (default: 0.5)
   --benchmark_repetitions=<n>  : Run each test <n> times
   --benchmark_report_aggregates_only=true : Less verbose output
   --benchmark_format=<console|json|csv>   : Output format

==============================================================================
)" << std::endl;
}


int main(int argc, char** argv) {
  PrintBanner();

  // Check for help flag before passing control to Google Benchmark
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      PrintHelp();
      return 0;
    }
  }

#ifdef _OPENMP
  RS_Execution_Threads = omp_get_max_threads();
#endif

  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
