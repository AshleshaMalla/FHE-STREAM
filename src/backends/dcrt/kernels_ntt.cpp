/*
  FHE-RaiderSTREAM Benchmark: Number Theoretic Transform (NTT) Kernel
  
  Benchmarks the performance of forward and inverse NTT operations (format conversions)
  on DCRTPoly objects. Measures the computational overhead of NTT and INTT using
  a round-trip pattern (Coefficient -> Evaluation -> Coefficient).
*/

#include "backends/dcrt/StreamCore.h"

/* Global thread count for benchmarks */
extern int RS_Execution_Threads;

/* 
  NTT ROUNDTRIP kernel: Measures NTT and INTT performance via round-trip format conversion.
  
  Pattern:
  - SetFormat(COEFFICIENT): Performs inverse NTT (INTT)
  - SetFormat(EVALUATION): Performs forward NTT
  
  This forces computation of both transforms every iteration,
  providing a fair measure of NTT throughput on the target platform.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_NTT_ROUNDTRIP)(benchmark::State& state) {
  RunNTT(*this, state);

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();
  /* Report aggregate bytes traversed by NTT: 1 array (A) * data size * 2 (round-trip) */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* Register the NTT benchmark kernel with all parameter sets */
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_NTT_ROUNDTRIP)
  ->Apply(RaiderSTREAM_Arguments)
  ->Unit(benchmark::kMillisecond);
