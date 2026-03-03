/*
  FHE-RaiderSTREAM Benchmark: Gather Kernel Implementations
*/

#include "backends/dcrt/DCRTFixture.h"
#include "backends/dcrt/StreamCore.h"
#include "common/BenchmarkUtils.h"

BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_COPY)(benchmark::State& state) {
  RunGatherPoly(*this, state,
                [](auto&, auto& rndA, auto&, auto&, auto& seqC, auto&,
                   const auto&, const auto&, const auto&, std::size_t dim) {
                  for (std::size_t j = 0; j < dim; ++j) {
                    seqC[j] = rndA[j];
                  }
                });

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_SCALE)(benchmark::State& state) {
  RunGatherPoly(*this, state,
                [](auto&, auto&, auto& seqB, auto&, auto&, auto& rndC,
                   const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                  for (std::size_t j = 0; j < dim; ++j) {
                    seqB[j] = rndC[j].ModMulFast(sc, mod, mu);
                  }
                });

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = C.size();
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_ADD)(benchmark::State& state) {
  RunGatherPoly(*this, state,
                [](auto&, auto& rndA, auto&, auto& rndB, auto& seqC, auto&,
                   const auto& mod, const auto&, const auto&, std::size_t dim) {
                  for (std::size_t j = 0; j < dim; ++j) {
                    seqC[j] = rndA[j].ModAddFast(rndB[j], mod);
                  }
                });

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_TRIAD)(benchmark::State& state) {
  RunGatherPoly(*this, state,
                [](auto& seqA, auto&, auto&, auto& rndB, auto&, auto& rndC,
                   const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                  for (std::size_t j = 0; j < dim; ++j) {
                    const auto scaled = rndC[j].ModMulFast(sc, mod, mu);
                    seqA[j] = rndB[j].ModAddFast(scaled, mod);
                  }
                });

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_COPY)
  ->Apply(RaiderSTREAM_Arguments)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_SCALE)
  ->Apply(RaiderSTREAM_Arguments)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_ADD)
  ->Apply(RaiderSTREAM_Arguments)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_TRIAD)
  ->Apply(RaiderSTREAM_Arguments)
  ->Unit(benchmark::kMillisecond);
