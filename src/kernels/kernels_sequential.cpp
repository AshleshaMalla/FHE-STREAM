/*
  FHE-RaiderSTREAM Benchmark: Sequential Kernel Implementations
  
  Implements four memory bandwidth benchmark kernels modeled after the
  STREAM benchmark pattern. Each kernel operates on RNS-decomposed
  homomorphic polynomials with per-tower modular arithmetic.
*/

#include "StreamCore.h"

/* Global thread count for benchmarks */
extern int RS_Execution_Threads;

/* 
  COPY kernel: C[i] = A[i] for all polynomials.
  Simple memory read and write pattern; fundamental bandwidth benchmark.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SEQ_COPY)(benchmark::State& state) {
  RunSequential(*this, state,
                [](auto& A, auto&, auto& C, const auto&, const auto&, const auto&, std::size_t dim) {
                  for (std::size_t j = 0; j < dim; ++j) {
                    C[j] = A[j];
                  }
                });

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();
  /* Report aggregate bytes read/written: 2 arrays (A and C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* 
  SCALE kernel: B[i] = scalar * C[i] for all polynomials.
  Memory read/write with modular multiplication operation; tests compute+memory overlap.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SEQ_SCALE)(benchmark::State& state) {
  RunSequential(*this, state,
                [](auto&, auto& B, auto& C, const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                  for (std::size_t j = 0; j < dim; ++j) {
                    B[j] = C[j].ModMulFast(sc, mod, mu);
                  }
                });

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = C.size();
  /* Report aggregate bytes: 2 arrays (C and B) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* 
  ADD kernel: C[i] = A[i] + B[i] for all polynomials.
  Ternary operation (read two, write one); tests memory and addition throughput.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SEQ_ADD)(benchmark::State& state) {
  RunSequential(*this, state,
                [](auto& A, auto& B, auto& C, const auto& mod, const auto&, const auto&, std::size_t dim) {
                  for (std::size_t j = 0; j < dim; ++j) {
                    C[j] = A[j].ModAddFast(B[j], mod);
                  }
                });

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();
  /* Report aggregate bytes: 3 arrays (A, B, C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* 
  TRIAD kernel: A[i] = B[i] + scalar * C[i] for all polynomials.
  Classic stream triad pattern; combines multiplication and addition with three-array access.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SEQ_TRIAD)(benchmark::State& state) {
  RunSequential(*this, state,
                [](auto& A, auto& B, auto& C, const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                  for (std::size_t j = 0; j < dim; ++j) {
                    const auto scaled = C[j].ModMulFast(sc, mod, mu);
                    A[j] = B[j].ModAddFast(scaled, mod);
                  }
                });

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();
  /* Report aggregate bytes: 3 arrays (A, B, C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* Register each benchmark kernel with all FHE parameter sets */
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_COPY)
  ->Apply([](benchmark::internal::Benchmark* b) { SchemeArgs(b, {ShuffleMode::None}); });
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_SCALE)
  ->Apply([](benchmark::internal::Benchmark* b) { SchemeArgs(b, {ShuffleMode::None}); });
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_ADD)
  ->Apply([](benchmark::internal::Benchmark* b) { SchemeArgs(b, {ShuffleMode::None}); });
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_TRIAD)
  ->Apply([](benchmark::internal::Benchmark* b) { SchemeArgs(b, {ShuffleMode::None}); });
