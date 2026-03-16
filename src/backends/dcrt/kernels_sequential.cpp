/*
  FHE-RaiderSTREAM Benchmark: Sequential Kernel Implementations
*/

#include "backends/dcrt/DCRTFixture.h"
#include "backends/dcrt/StreamCore.h"
#include "common/BenchmarkUtils.h"

BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SEQ_COPY)(benchmark::State& state) {
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 2;

  RunSequential(*this, state, bytesPerIter,
                [](auto& A, auto&, auto& C, const auto&, const auto&, const auto&, std::size_t dim) {
                  for (std::size_t j = 0; j < dim; ++j) {
                    C[j] = A[j];
                  }
                });
}

BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SEQ_SCALE)(benchmark::State& state) {
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = C.size();
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 2;

  RunSequential(*this, state, bytesPerIter,
                [](auto&, auto& B, auto& C, const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                  for (std::size_t j = 0; j < dim; ++j) {
                    B[j] = C[j].ModMulFast(sc, mod, mu);
                  }
                });
}

BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SEQ_ADD)(benchmark::State& state) {
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 3;

  RunSequential(*this, state, bytesPerIter,
                [](auto& A, auto& B, auto& C, const auto& mod, const auto&, const auto&, std::size_t dim) {
                  for (std::size_t j = 0; j < dim; ++j) {
                    C[j] = A[j].ModAddFast(B[j], mod);
                  }
                });
}

BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SEQ_TRIAD)(benchmark::State& state) {
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 3;

  RunSequential(*this, state, bytesPerIter,
                [](auto& A, auto& B, auto& C, const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                  for (std::size_t j = 0; j < dim; ++j) {
                    const auto scaled = C[j].ModMulFast(sc, mod, mu);
                    A[j] = B[j].ModAddFast(scaled, mod);
                  }
                });
}

BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_KEYSWITCH_MOCK)(benchmark::State& state) {
  RS_KEYSWITCH_MOCK(state, A, B, C);

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::int64_t batchSize = static_cast<std::int64_t>(A.size());
  const std::int64_t bytesPerPoly = ringDim * numTowers * 8;

  // Reads: A + B, Writes: A (C[0] assumed cache-resident and excluded)
  state.SetBytesProcessed(state.iterations() * batchSize * (3 * bytesPerPoly));
}

BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_COPY)
  ->Apply(RaiderSTREAM_Arguments_Sequential)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_SCALE)
  ->Apply(RaiderSTREAM_Arguments_Sequential)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_ADD)
  ->Apply(RaiderSTREAM_Arguments_Sequential)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_TRIAD)
  ->Apply(RaiderSTREAM_Arguments_Sequential)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_KEYSWITCH_MOCK)
  ->Apply(RaiderSTREAM_Arguments_Sequential)
  ->Unit(benchmark::kMillisecond);
