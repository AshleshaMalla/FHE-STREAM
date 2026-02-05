/*
  FHE-RaiderSTREAM Benchmark: Main Entry Point
  
  Launches the Google Benchmark framework with command-line argument processing.
  Benchmark kernels are registered in kernels_sequential.cpp.
*/

#include <benchmark/benchmark.h>

/* Entry point: invokes Google Benchmark with command-line argument processing */
BENCHMARK_MAIN();
