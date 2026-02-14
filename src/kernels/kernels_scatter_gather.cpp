/*
  FHE-RaiderSTREAM Benchmark: Scatter-Gather Kernel Implementations

  Implements irregular-access benchmark kernels where both read and write
  patterns are randomized using index vectors (IDX or COEFF_IDX).
*/

#include "StreamCore.h"

/*
  SCATTER-GATHER COPY kernel: C[IDX[i]] = A[IDX[i]] for all polynomials.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SCATTER_GATHER_COPY)(benchmark::State& state) {
  const auto mode = static_cast<ShuffleMode>(state.range(2));
  if (mode == ShuffleMode::Coeff) {
    RunScatterGatherCoeff(*this, state,
                          [](auto& rndA, auto&, auto& rndC,
                             const auto&, const auto&, const auto&) {
                            rndC = rndA;
                          });
  } else {
    RunScatterGatherPoly(*this, state,
                         [](auto& rndA, auto&, auto& rndC,
                            const auto&, const auto&, const auto&, std::size_t dim) {
                           for (std::size_t j = 0; j < dim; ++j) {
                             rndC[j] = rndA[j];
                           }
                         });
  }

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers)
                                   * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/*
  SCATTER-GATHER SCALE kernel: B[IDX_WRITE[i]] = scalar * C[IDX_READ[i]] for all polynomials.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SCATTER_GATHER_SCALE)(benchmark::State& state) {
  const auto mode = static_cast<ShuffleMode>(state.range(2));
  if (mode == ShuffleMode::Coeff) {
    RunScatterGatherCoeff(*this, state,
                          [](auto&, auto& rndB, auto& rndC,
                             const auto& mod, const auto& mu, const auto& sc) {
                            rndB = rndC.ModMulFast(sc, mod, mu);
                          });
  } else {
    RunScatterGatherPoly(*this, state,
                         [](auto&, auto& rndB, auto& rndC,
                            const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                           for (std::size_t j = 0; j < dim; ++j) {
                             rndB[j] = rndC[j].ModMulFast(sc, mod, mu);
                           }
                         });
  }

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = C.size();

  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers)
                                   * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/*
  SCATTER-GATHER ADD kernel: C[IDX_WRITE[i]] = A[IDX_READ[i]] + B[IDX_READ[i]] for all polynomials.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SCATTER_GATHER_ADD)(benchmark::State& state) {
  const auto mode = static_cast<ShuffleMode>(state.range(2));
  if (mode == ShuffleMode::Coeff) {
    RunScatterGatherCoeff(*this, state,
                          [](auto& rndA, auto& rndB, auto& rndC,
                             const auto& mod, const auto&, const auto&) {
                            rndC = rndA.ModAddFast(rndB, mod);
                          });
  } else {
    RunScatterGatherPoly(*this, state,
                         [](auto& rndA, auto& rndB, auto& rndC,
                            const auto& mod, const auto&, const auto&, std::size_t dim) {
                           for (std::size_t j = 0; j < dim; ++j) {
                             rndC[j] = rndA[j].ModAddFast(rndB[j], mod);
                           }
                         });
  }

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers)
                                   * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/*
  SCATTER-GATHER TRIAD kernel: A[IDX_WRITE[i]] = B[IDX_READ[i]] + scalar * C[IDX_READ[i]] for all polynomials.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SCATTER_GATHER_TRIAD)(benchmark::State& state) {
  const auto mode = static_cast<ShuffleMode>(state.range(2));
  if (mode == ShuffleMode::Coeff) {
    RunScatterGatherCoeff(*this, state,
                          [](auto& rndA, auto& rndB, auto& rndC,
                             const auto& mod, const auto& mu, const auto& sc) {
                            const auto scaled = rndC.ModMulFast(sc, mod, mu);
                            rndA = rndB.ModAddFast(scaled, mod);
                          });
  } else {
    RunScatterGatherPoly(*this, state,
                         [](auto& rndA, auto& rndB, auto& rndC,
                            const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                           for (std::size_t j = 0; j < dim; ++j) {
                             const auto scaled = rndC[j].ModMulFast(sc, mod, mu);
                             rndA[j] = rndB[j].ModAddFast(scaled, mod);
                           }
                         });
  }

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers)
                                   * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}


/* Register the scatter-gather kernels with all FHE parameter sets */
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SCATTER_GATHER_COPY)
  ->Apply([](benchmark::internal::Benchmark* b) { SchemeArgs(b, {ShuffleMode::None, ShuffleMode::Poly, ShuffleMode::Coeff}); });
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SCATTER_GATHER_SCALE)
  ->Apply([](benchmark::internal::Benchmark* b) { SchemeArgs(b, {ShuffleMode::None, ShuffleMode::Poly, ShuffleMode::Coeff}); });
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SCATTER_GATHER_ADD)
  ->Apply([](benchmark::internal::Benchmark* b) { SchemeArgs(b, {ShuffleMode::None, ShuffleMode::Poly, ShuffleMode::Coeff}); });
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SCATTER_GATHER_TRIAD)
  ->Apply([](benchmark::internal::Benchmark* b) { SchemeArgs(b, {ShuffleMode::None, ShuffleMode::Poly, ShuffleMode::Coeff}); });
