/*
  FHE-RaiderSTREAM Benchmark: Gather Kernel Implementations (Phase 2)

  Implements irregular-access benchmark kernels where the read pattern is
  randomized using an index vector (IDX) or coefficient index (COEFF_IDX).
  The write pattern remains sequential.
*/

#include "StreamCore.h"

/*
  GATHER COPY kernel: C[i] = A[IDX[i]] for all polynomials.
  Uses the same "anchor" method as sequential kernels to prevent DCE.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_COPY)(benchmark::State& state) {
  const auto mode = static_cast<ShuffleMode>(state.range(2));
  if (mode == ShuffleMode::Coeff) {
    RunGatherCoeff(*this, state,
                   [](auto& aTower, auto&, auto& cTower, const auto& idx,
                      const auto&, const auto&, const auto&, std::size_t dim) {
                     for (std::size_t j = 0; j < dim; ++j) {
                       const std::size_t src_idx = idx[j];
                       cTower[j] = aTower[src_idx];
                     }
                   });
  } else {
    RunGatherPoly(*this, state,
                  [](auto&, auto& rndA, auto&, auto&, auto& seqC, auto&,
                     const auto&, const auto&, const auto&, std::size_t dim) {
                    for (std::size_t j = 0; j < dim; ++j) {
                      seqC[j] = rndA[j];
                    }
                  });
  }

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  /* Report aggregate bytes read/written: 2 arrays (A and C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/*
  GATHER SCALE kernel: B[i] = scalar * C[IDX[i]] for all polynomials.
  Uses the same "anchor" method as sequential kernels to prevent DCE.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_SCALE)(benchmark::State& state) {
  const auto mode = static_cast<ShuffleMode>(state.range(2));
  if (mode == ShuffleMode::Coeff) {
    RunGatherCoeff(*this, state,
                   [](auto&, auto& bTower, auto& cTower, const auto& idx,
                      const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                     for (std::size_t j = 0; j < dim; ++j) {
                       const std::size_t src_idx = idx[j];
                       bTower[j] = cTower[src_idx].ModMulFast(sc, mod, mu);
                     }
                   });
  } else {
    RunGatherPoly(*this, state,
                  [](auto&, auto&, auto& seqB, auto&, auto&, auto& rndC,
                     const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                    for (std::size_t j = 0; j < dim; ++j) {
                      seqB[j] = rndC[j].ModMulFast(sc, mod, mu);
                    }
                  });
  }

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = C.size();

  /* Report aggregate bytes read/written: 2 arrays (C and B) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/*
  GATHER ADD kernel: C[i] = A[IDX[i]] + B[IDX[i]] for all polynomials.
  Uses the same "anchor" method as sequential kernels to prevent DCE.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_ADD)(benchmark::State& state) {
  const auto mode = static_cast<ShuffleMode>(state.range(2));
  if (mode == ShuffleMode::Coeff) {
    RunGatherCoeff(*this, state,
                   [](auto& aTower, auto& bTower, auto& cTower, const auto& idx,
                      const auto& mod, const auto&, const auto&, std::size_t dim) {
                     for (std::size_t j = 0; j < dim; ++j) {
                       const std::size_t src_idx = idx[j];
                       cTower[j] = aTower[src_idx].ModAddFast(bTower[src_idx], mod);
                     }
                   });
  } else {
    RunGatherPoly(*this, state,
                  [](auto&, auto& rndA, auto&, auto& rndB, auto& seqC, auto&,
                     const auto& mod, const auto&, const auto&, std::size_t dim) {
                    for (std::size_t j = 0; j < dim; ++j) {
                      seqC[j] = rndA[j].ModAddFast(rndB[j], mod);
                    }
                  });
  }

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  /* Report aggregate bytes read/written: 3 arrays (A, B, C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/*
  GATHER TRIAD kernel: A[i] = B[IDX[i]] + scalar * C[IDX[i]] for all polynomials.
  Uses the same "anchor" method as sequential kernels to prevent DCE.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_TRIAD)(benchmark::State& state) {
  const auto mode = static_cast<ShuffleMode>(state.range(2));
  if (mode == ShuffleMode::Coeff) {
    RunGatherCoeff(*this, state,
                   [](auto& aTower, auto& bTower, auto& cTower, const auto& idx,
                      const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                     for (std::size_t j = 0; j < dim; ++j) {
                       const std::size_t src_idx = idx[j];
                       const auto scaled = cTower[src_idx].ModMulFast(sc, mod, mu);
                       aTower[j] = bTower[src_idx].ModAddFast(scaled, mod);
                     }
                   });
  } else {
    RunGatherPoly(*this, state,
                  [](auto& seqA, auto&, auto&, auto& rndB, auto&, auto& rndC,
                     const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                    for (std::size_t j = 0; j < dim; ++j) {
                      const auto scaled = rndC[j].ModMulFast(sc, mod, mu);
                      seqA[j] = rndB[j].ModAddFast(scaled, mod);
                    }
                  });
  }

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  /* Report aggregate bytes read/written: 3 arrays (A, B, C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}



/* Register the gather kernel with all FHE parameter sets */
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_COPY)
    ->Apply([](benchmark::internal::Benchmark* b) { SchemeArgs(b, {ShuffleMode::None, ShuffleMode::Poly, ShuffleMode::Coeff}); });
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_SCALE)
  ->Apply([](benchmark::internal::Benchmark* b) { SchemeArgs(b, {ShuffleMode::None, ShuffleMode::Poly, ShuffleMode::Coeff}); });
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_ADD)
  ->Apply([](benchmark::internal::Benchmark* b) { SchemeArgs(b, {ShuffleMode::None, ShuffleMode::Poly, ShuffleMode::Coeff}); });
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_TRIAD)
  ->Apply([](benchmark::internal::Benchmark* b) { SchemeArgs(b, {ShuffleMode::None, ShuffleMode::Poly, ShuffleMode::Coeff}); });

