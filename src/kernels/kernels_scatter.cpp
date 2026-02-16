/*
  FHE-RaiderSTREAM Benchmark: Scatter Kernel Implementations (Phase 3)

  Implements irregular-access benchmark kernels where the write pattern is
  randomized using an index vector (IDX) or coefficient index (COEFF_IDX).
  The read pattern remains sequential.
*/

#include "StreamCore.h"

/*
  SCATTER COPY kernel: C[IDX[i]] = A[i] for all polynomials.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SCATTER_COPY)(benchmark::State& state) {
  RunScatterPoly(*this, state,
                 [](auto& seqA, auto&, auto&, auto&, auto&, auto& rndC,
                    const auto&, const auto&, const auto&, std::size_t dim) {
                   for (std::size_t j = 0; j < dim; ++j) {
                     rndC[j] = seqA[j];
                   }
                 });

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers)
                                   * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/*
  SCATTER SCALE kernel: B[IDX[i]] = scalar * C[i] for all polynomials.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SCATTER_SCALE)(benchmark::State& state) {
  RunScatterPoly(*this, state,
                 [](auto&, auto&, auto& seqB, auto&, auto&, auto& rndC,
                    const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                   for (std::size_t j = 0; j < dim; ++j) {
                     seqB[j] = rndC[j].ModMulFast(sc, mod, mu);
                   }
                 });

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = C.size();

  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers)
                                   * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/*
  SCATTER ADD kernel: C[IDX[i]] = A[i] + B[i] for all polynomials.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SCATTER_ADD)(benchmark::State& state) {
  RunScatterPoly(*this, state,
                 [](auto& seqA, auto&, auto& seqB, auto&, auto&, auto& rndC,
                    const auto& mod, const auto&, const auto&, std::size_t dim) {
                   for (std::size_t j = 0; j < dim; ++j) {
                     rndC[j] = seqA[j].ModAddFast(seqB[j], mod);
                   }
                 });

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers)
                                   * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/*
  SCATTER TRIAD kernel: A[IDX[i]] = B[i] + scalar * C[i] for all polynomials.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SCATTER_TRIAD)(benchmark::State& state) {
  RunScatterPoly(*this, state,
                 [](auto& seqA, auto&, auto& seqB, auto&, auto&, auto& rndC,
                    const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                   for (std::size_t j = 0; j < dim; ++j) {
                     const auto scaled = rndC[j].ModMulFast(sc, mod, mu);
                     seqA[j] = seqB[j].ModAddFast(scaled, mod);
                   }
                 });

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers)
                                   * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* Register the scatter kernels with all parameter sets */
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SCATTER_COPY)
  ->Apply(CustomArguments)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SCATTER_SCALE)
  ->Apply(CustomArguments)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SCATTER_ADD)
  ->Apply(CustomArguments)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SCATTER_TRIAD)
  ->Apply(CustomArguments)
  ->Unit(benchmark::kMillisecond);

