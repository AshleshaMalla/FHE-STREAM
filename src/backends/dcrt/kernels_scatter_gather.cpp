/*
  FHE-RaiderSTREAM Benchmark: Scatter-Gather Kernel Implementations

  Implements irregular-access benchmark kernels where both read and write
  patterns are randomized using index vectors (IDX or COEFF_IDX).
*/

#include "backends/dcrt/StreamCore.h"

/*
  SCATTER-GATHER COPY kernel: C[IDX[i]] = A[IDX[i]] for all polynomials.
  Dispatches to Poly-level or Coeff-level shuffle based on ShuffleMode.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SCATTER_GATHER_COPY)(benchmark::State& state) {
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const ShuffleMode mode = static_cast<ShuffleMode>(state.range(2));
  const std::size_t nPolys = A.size();

  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers)
                                   * static_cast<std::int64_t>(nPolys) * 2;

  if (mode == ShuffleMode::Poly) {
    RunScatterGatherPoly(*this, state, bytesPerIter,
                         [](auto& rndA, auto&, auto& rndC,
                            const auto&, const auto&, const auto&, std::size_t dim) {
                           for (std::size_t j = 0; j < dim; ++j) {
                             rndC[j] = rndA[j];
                           }
                         });
  } else if (mode == ShuffleMode::Coeff) {
    RunScatterGatherCoeff(*this, state, bytesPerIter,
                          [](auto& cA, auto&, auto& cC,
                             const auto&, const auto&, const auto&) {
                            cC = cA;
                          });
  }
}

/*
  SCATTER-GATHER SCALE kernel: B[IDX[i]] = scalar * C[IDX[i]] for all polynomials.
  Dispatches to Poly-level or Coeff-level shuffle based on ShuffleMode.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SCATTER_GATHER_SCALE)(benchmark::State& state) {
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const ShuffleMode mode = static_cast<ShuffleMode>(state.range(2));
  const std::size_t nPolys = C.size();

  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers)
                                   * static_cast<std::int64_t>(nPolys) * 2;

  if (mode == ShuffleMode::Poly) {
    RunScatterGatherPoly(*this, state, bytesPerIter,
                         [](auto&, auto& rndB, auto& rndC,
                            const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                           for (std::size_t j = 0; j < dim; ++j) {
                             rndB[j] = rndC[j].ModMulFast(sc, mod, mu);
                           }
                         });
  } else if (mode == ShuffleMode::Coeff) {
    RunScatterGatherCoeff(*this, state, bytesPerIter,
                          [](auto&, auto& cB, auto& cC,
                             const auto& mod, const auto& mu, const auto& sc) {
                            cB = cC.ModMulFast(sc, mod, mu);
                          });
  }
}

/*
  SCATTER-GATHER ADD kernel: C[IDX[i]] = A[IDX[i]] + B[IDX[i]] for all polynomials.
  Dispatches to Poly-level or Coeff-level shuffle based on ShuffleMode.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SCATTER_GATHER_ADD)(benchmark::State& state) {
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const ShuffleMode mode = static_cast<ShuffleMode>(state.range(2));
  const std::size_t nPolys = A.size();

  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers)
                                   * static_cast<std::int64_t>(nPolys) * 3;

  if (mode == ShuffleMode::Poly) {
    RunScatterGatherPoly(*this, state, bytesPerIter,
                         [](auto& rndA, auto& rndB, auto& rndC,
                            const auto& mod, const auto&, const auto&, std::size_t dim) {
                           for (std::size_t j = 0; j < dim; ++j) {
                             rndC[j] = rndA[j].ModAddFast(rndB[j], mod);
                           }
                         });
  } else if (mode == ShuffleMode::Coeff) {
    RunScatterGatherCoeff(*this, state, bytesPerIter,
                          [](auto& cA, auto& cB, auto& cC,
                             const auto& mod, const auto&, const auto&) {
                            cC = cA.ModAddFast(cB, mod);
                          });
  }
}

/*
  SCATTER-GATHER TRIAD kernel: A[IDX[i]] = B[IDX[i]] + scalar * C[IDX[i]] for all polynomials.
  Dispatches to Poly-level or Coeff-level shuffle based on ShuffleMode.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SCATTER_GATHER_TRIAD)(benchmark::State& state) {
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const ShuffleMode mode = static_cast<ShuffleMode>(state.range(2));
  const std::size_t nPolys = A.size();

  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers)
                                   * static_cast<std::int64_t>(nPolys) * 3;

  if (mode == ShuffleMode::Poly) {
    RunScatterGatherPoly(*this, state, bytesPerIter,
                         [](auto& rndA, auto& rndB, auto& rndC,
                            const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                           for (std::size_t j = 0; j < dim; ++j) {
                             const auto scaled = rndC[j].ModMulFast(sc, mod, mu);
                             rndA[j] = rndB[j].ModAddFast(scaled, mod);
                           }
                         });
  } else if (mode == ShuffleMode::Coeff) {
    RunScatterGatherCoeff(*this, state, bytesPerIter,
                          [](auto& cA, auto& cB, auto& cC,
                             const auto& mod, const auto& mu, const auto& sc) {
                            const auto scaled = cC.ModMulFast(sc, mod, mu);
                            cA = cB.ModAddFast(scaled, mod);
                          }, "RS_SCATTER_GATHER_TRIAD_COEFF");
  }
}

/* Register the scatter-gather kernels with all parameter sets */
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SCATTER_GATHER_COPY)
  ->Apply(RaiderSTREAM_Arguments_Irregular)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SCATTER_GATHER_SCALE)
  ->Apply(RaiderSTREAM_Arguments_Irregular)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SCATTER_GATHER_ADD)
  ->Apply(RaiderSTREAM_Arguments_Irregular)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SCATTER_GATHER_TRIAD)
  ->Apply(RaiderSTREAM_Arguments_Irregular)
  ->Unit(benchmark::kMillisecond);
