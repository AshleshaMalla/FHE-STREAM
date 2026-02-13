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


/* Register the scatter-gather kernels with all FHE parameter sets */
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SCATTER_GATHER_COPY)
  ->Apply([](benchmark::internal::Benchmark* b) { SchemeArgs(b, {ShuffleMode::None, ShuffleMode::Poly, ShuffleMode::Coeff}); });
