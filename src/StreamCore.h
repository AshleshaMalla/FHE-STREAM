/*
  FHE-RaiderSTREAM Benchmark: StreamCore

  Shared templated dispatchers for sequential, gather-poly, and gather-coeff
  kernels. Centralizes threading, label setting, and outer loop structure.
*/

#pragma once

#include "FHERaiderSTREAM.h"

#include <benchmark/benchmark.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Global thread count for benchmarks */
extern int RS_Execution_Threads;

inline void SetLabel(benchmark::State& state) {
  const auto mode = static_cast<ShuffleMode>(state.range(2));
  switch (mode) {
    case ShuffleMode::None:
      state.SetLabel("Mode: Sequential");
      break;
    case ShuffleMode::Poly:
      state.SetLabel("Mode: Poly Shuffle");
      break;
    case ShuffleMode::Coeff:
      state.SetLabel("Mode: Coeff Shuffle");
      break;
    default:
      state.SetLabel("Mode: Unknown");
      break;
  }
}

template <typename Kernel>
inline void RunSequential(FHERaiderSTREAM& self, benchmark::State& state, Kernel&& kernel) {
  SetLabel(state);
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = self.A.size();
  const std::size_t dim = static_cast<std::size_t>(ringDim);

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(RS_Execution_Threads)
    for (std::size_t i = 0; i < nPolys; ++i) {
      auto& aTowers = self.A[i].GetAllElements();
      auto& bTowers = self.B[i].GetAllElements();
      auto& cTowers = self.C[i].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& aTower = aTowers[t];
        auto& bTower = bTowers[t];
        auto& cTower = cTowers[t];
        const auto& mod = self.towerModuli[t];
        const auto& mu = self.towerMu[t];
        kernel(aTower, bTower, cTower, mod, mu, self.scalar, dim);
        if (dim > 0) {
          benchmark::DoNotOptimize(aTower[0]);
          benchmark::DoNotOptimize(bTower[0]);
          benchmark::DoNotOptimize(cTower[0]);
        }
      }
    }
    benchmark::ClobberMemory();
  }
}

template <typename Kernel>
inline void RunGatherPoly(FHERaiderSTREAM& self, benchmark::State& state, Kernel&& kernel) {
  SetLabel(state);
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = self.A.size();
  const std::size_t dim = static_cast<std::size_t>(ringDim);

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(RS_Execution_Threads)
    for (std::size_t i = 0; i < nPolys; ++i) {
      const std::size_t src = self.IDX[i];
      auto& aSeqTowers = self.A[i].GetAllElements();
      auto& bSeqTowers = self.B[i].GetAllElements();
      auto& cSeqTowers = self.C[i].GetAllElements();
      auto& aRndTowers = self.A[src].GetAllElements();
      auto& bRndTowers = self.B[src].GetAllElements();
      auto& cRndTowers = self.C[src].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& aSeq = aSeqTowers[t];
        auto& bSeq = bSeqTowers[t];
        auto& cSeq = cSeqTowers[t];
        auto& aRnd = aRndTowers[t];
        auto& bRnd = bRndTowers[t];
        auto& cRnd = cRndTowers[t];
        const auto& mod = self.towerModuli[t];
        const auto& mu = self.towerMu[t];
        kernel(aSeq, aRnd, bSeq, bRnd, cSeq, cRnd, mod, mu, self.scalar, dim);
        if (dim > 0) {
          benchmark::DoNotOptimize(aSeq[0]);
          benchmark::DoNotOptimize(bSeq[0]);
          benchmark::DoNotOptimize(cSeq[0]);
        }
      }
    }
    benchmark::ClobberMemory();
  }
}

template <typename Kernel>
inline void RunGatherCoeff(FHERaiderSTREAM& self, benchmark::State& state, Kernel&& kernel) {
  SetLabel(state);
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = self.A.size();
  const std::size_t dim = static_cast<std::size_t>(ringDim);

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(RS_Execution_Threads)
    for (std::size_t i = 0; i < nPolys; ++i) {
      auto& aTowers = self.A[i].GetAllElements();
      auto& bTowers = self.B[i].GetAllElements();
      auto& cTowers = self.C[i].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& aTower = aTowers[t];
        auto& bTower = bTowers[t];
        auto& cTower = cTowers[t];
        const auto& mod = self.towerModuli[t];
        const auto& mu = self.towerMu[t];
        kernel(aTower, bTower, cTower, self.COEFF_IDX, mod, mu, self.scalar, dim);
        if (dim > 0) {
          benchmark::DoNotOptimize(aTower[0]);
          benchmark::DoNotOptimize(bTower[0]);
          benchmark::DoNotOptimize(cTower[0]);
        }
      }
    }
    benchmark::ClobberMemory();
  }
}
