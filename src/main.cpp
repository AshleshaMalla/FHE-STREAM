/*
  FHE-RaiderSTREAM Benchmark: Main Entry Point
  
  Launches the Google Benchmark framework with command-line argument processing.
  Benchmark kernels are registered in kernels_sequential.cpp.
*/

#include <benchmark/benchmark.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int RS_Execution_Threads = 1;

int main(int argc, char** argv) {
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
