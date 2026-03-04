/*
  FHE-RaiderSTREAM Benchmark: StreamCore

  Shared templated dispatchers for sequential, gather-poly, and gather-coeff
  kernels. Centralizes threading, label setting, and outer loop structure.
*/

#pragma once

#include "backends/dcrt/DCRTFixture.h"
#include "common/BenchmarkUtils.h"

#include <benchmark/benchmark.h>

#include <cstdlib>
#include <cerrno>
#include <initializer_list>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef RAIDERSTREAM_MPI
#include <mpi.h>
#endif

#include "common/MPIUtils.h"

/* Global thread count for benchmarks */
extern int RS_Execution_Threads;

inline void SetLabel(benchmark::State& state) {
  state.SetLabel("Batch: " + std::to_string(state.range(2)));
}

inline std::int64_t DeepBytesPerPoly(std::int64_t ringDim, std::int64_t numTowers) {
  return ringDim * numTowers * 8;
}

inline void SchemeArgs(benchmark::internal::Benchmark* b, std::initializer_list<ShuffleMode> modes) {
  for (const auto mode : modes) {
    b->Args({1 << 16, 32, static_cast<int>(mode)});
  }
  for (const auto mode : modes) {
    b->Args({1 << 15, 16, static_cast<int>(mode)});
  }
  for (const auto mode : modes) {
    b->Args({1 << 11, 2, static_cast<int>(mode)});
  }
}

template <typename Kernel>
inline void RunSequential(FHERaiderSTREAM& self, benchmark::State& state, int64_t bytesPerIter, Kernel&& kernel) {
  SetLabel(state);
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = self.A.size();
  const std::size_t dim = static_cast<std::size_t>(ringDim);

  RS_BARRIER();

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(RS_Execution_Threads)
    for (std::size_t i = 0; i < nPolys; ++i) {
      auto& aTowers = self.A[i].GetAllElements();
      auto& bTowers = self.B[i].GetAllElements();
      auto& cTowers = self.C[i].GetAllElements();
      const lbcrypto::NativeInteger scalarNI(static_cast<uint64_t>(self.scalar));
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& aTower = aTowers[t];
        auto& bTower = bTowers[t];
        auto& cTower = cTowers[t];
        const auto& mod = self.towerModuli[t];
        const auto& mu = self.towerMu[t];
        kernel(aTower, bTower, cTower, mod, mu, scalarNI, dim);
      }
    }
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytesPerIter);
  AggregateBandwidth(state);
}

template <typename Kernel>
inline void RunGatherPoly(FHERaiderSTREAM& self, benchmark::State& state, int64_t bytesPerIter, Kernel&& kernel) {
  SetLabel(state);
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = self.A.size();
  const std::size_t dim = static_cast<std::size_t>(ringDim);

  RS_BARRIER();

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
      const lbcrypto::NativeInteger scalarNI(static_cast<uint64_t>(self.scalar));
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& aSeq = aSeqTowers[t];
        auto& bSeq = bSeqTowers[t];
        auto& cSeq = cSeqTowers[t];
        auto& aRnd = aRndTowers[t];
        auto& bRnd = bRndTowers[t];
        auto& cRnd = cRndTowers[t];
        const auto& mod = self.towerModuli[t];
        const auto& mu = self.towerMu[t];
        kernel(aSeq, aRnd, bSeq, bRnd, cSeq, cRnd, mod, mu, scalarNI, dim);
      }
    }
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytesPerIter);
  AggregateBandwidth(state);
}

template <typename Kernel>
inline void RunGatherCoeff(FHERaiderSTREAM& self, benchmark::State& state, int64_t bytesPerIter, Kernel&& kernel) {
  SetLabel(state);
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = self.A.size();
  const std::size_t dim = static_cast<std::size_t>(ringDim);

  RS_BARRIER();

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(RS_Execution_Threads)
    for (std::size_t i = 0; i < nPolys; ++i) {
      auto& aTowers = self.A[i].GetAllElements();
      auto& bTowers = self.B[i].GetAllElements();
      auto& cTowers = self.C[i].GetAllElements();
      const lbcrypto::NativeInteger scalarNI(static_cast<uint64_t>(self.scalar));
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& aTower = aTowers[t];
        auto& bTower = bTowers[t];
        auto& cTower = cTowers[t];
        const auto& mod = self.towerModuli[t];
        const auto& mu = self.towerMu[t];
        kernel(aTower, bTower, cTower, self.COEFF_IDX, mod, mu, scalarNI, dim);
      }
    }
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytesPerIter);
  AggregateBandwidth(state);
}

template <typename Kernel>
inline void RunScatterPoly(FHERaiderSTREAM& self, benchmark::State& state, int64_t bytesPerIter, Kernel&& kernel) {
  SetLabel(state);
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = self.A.size();
  const std::size_t dim = static_cast<std::size_t>(ringDim);

  RS_BARRIER();

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(RS_Execution_Threads)
    for (std::size_t i = 0; i < nPolys; ++i) {
      const std::size_t dest = self.IDX[i];
      auto& aSeqTowers = self.A[i].GetAllElements();
      auto& bSeqTowers = self.B[i].GetAllElements();
      auto& cSeqTowers = self.C[i].GetAllElements();
      auto& aRndTowers = self.A[dest].GetAllElements();
      auto& bRndTowers = self.B[dest].GetAllElements();
      auto& cRndTowers = self.C[dest].GetAllElements();
      const lbcrypto::NativeInteger scalarNI(static_cast<uint64_t>(self.scalar));
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& aSeq = aSeqTowers[t];
        auto& bSeq = bSeqTowers[t];
        auto& cSeq = cSeqTowers[t];
        auto& aRnd = aRndTowers[t];
        auto& bRnd = bRndTowers[t];
        auto& cRnd = cRndTowers[t];
        const auto& mod = self.towerModuli[t];
        const auto& mu = self.towerMu[t];
        kernel(aSeq, aRnd, bSeq, bRnd, cSeq, cRnd, mod, mu, scalarNI, dim);
      }
    }
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytesPerIter);
  AggregateBandwidth(state);
}

template <typename Kernel>
inline void RunScatterCoeff(FHERaiderSTREAM& self, benchmark::State& state, int64_t bytesPerIter, Kernel&& kernel) {
  SetLabel(state);
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = self.A.size();
  const std::size_t dim = static_cast<std::size_t>(ringDim);

  RS_BARRIER();

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(RS_Execution_Threads)
    for (std::size_t i = 0; i < nPolys; ++i) {
      auto& aTowers = self.A[i].GetAllElements();
      auto& bTowers = self.B[i].GetAllElements();
      auto& cTowers = self.C[i].GetAllElements();
      const lbcrypto::NativeInteger scalarNI(static_cast<uint64_t>(self.scalar));
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& aTower = aTowers[t];
        auto& bTower = bTowers[t];
        auto& cTower = cTowers[t];
        const auto& mod = self.towerModuli[t];
        const auto& mu = self.towerMu[t];
        kernel(aTower, bTower, cTower, self.COEFF_IDX, mod, mu, scalarNI, dim);
      }
    }
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytesPerIter);
  AggregateBandwidth(state);
}

template <typename Kernel>
inline void RunScatterGatherPoly(FHERaiderSTREAM& self, benchmark::State& state, int64_t bytesPerIter, Kernel&& kernel) {
  SetLabel(state);
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = self.A.size();
  const std::size_t dim = static_cast<std::size_t>(ringDim);

  RS_BARRIER();

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(RS_Execution_Threads)
    for (std::size_t i = 0; i < nPolys; ++i) {
      const std::size_t src_k = self.IDX[i];
      const std::size_t dst_k = self.IDX_WRITE[i];
      auto& aRndTowers = self.A[src_k].GetAllElements();
      auto& bRndTowers = self.B[src_k].GetAllElements();
      auto& cRndTowers = self.C[dst_k].GetAllElements();
      const lbcrypto::NativeInteger scalarNI(static_cast<uint64_t>(self.scalar));
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& aRnd = aRndTowers[t];
        auto& bRnd = bRndTowers[t];
        auto& cRnd = cRndTowers[t];
        const auto& mod = self.towerModuli[t];
        const auto& mu = self.towerMu[t];
        kernel(aRnd, bRnd, cRnd, mod, mu, scalarNI, dim);
      }
    }
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytesPerIter);
  AggregateBandwidth(state);
}

template <typename Kernel>
inline void RunScatterGatherCoeff(FHERaiderSTREAM& self, benchmark::State& state, int64_t bytesPerIter, Kernel&& kernel) {
  SetLabel(state);
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = self.A.size();
  const std::size_t dim = static_cast<std::size_t>(ringDim);

  RS_BARRIER();

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(RS_Execution_Threads)
    for (std::size_t i = 0; i < nPolys; ++i) {
      auto& aTowers = self.A[i].GetAllElements();
      auto& bTowers = self.B[i].GetAllElements();
      auto& cTowers = self.C[i].GetAllElements();
      const lbcrypto::NativeInteger scalarNI(static_cast<uint64_t>(self.scalar));
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& aTower = aTowers[t];
        auto& bTower = bTowers[t];
        auto& cTower = cTowers[t];
        const auto& mod = self.towerModuli[t];
        const auto& mu = self.towerMu[t];
        for (std::size_t j = 0; j < dim; ++j) {
          const std::size_t src_k = self.COEFF_IDX[j];
          const std::size_t dst_k = self.COEFF_IDX_WRITE[j];
          kernel(aTower[src_k], bTower[src_k], cTower[dst_k], mod, mu, scalarNI);
        }
      }
    }
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytesPerIter);
  AggregateBandwidth(state);
}

/*
  RunNTT: Number Theoretic Transform Round-Trip Benchmark
  
  Performs a round-trip format conversion (Coefficient -> Evaluation -> Coefficient)
  on all polynomials to benchmark NTT and inverse NTT performance.
  This pattern forces computation of both forward and inverse transforms.
  
  Parameters:
  - self: FHERaiderSTREAM fixture containing polynomial arrays A, B, C
  - state: Google Benchmark state object for measuring iterations and time
*/
inline void RunNTT(FHERaiderSTREAM& self, benchmark::State& state, int64_t bytesPerIter) {
  SetLabel(state);
  const std::size_t nPolys = self.A.size();

  RS_BARRIER();

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(RS_Execution_Threads)
    for (std::size_t i = 0; i < nPolys; ++i) {
      /* Inverse Transform: Coefficient format */
      self.A[i].SetFormat(Format::COEFFICIENT);
      /* Forward Transform: Evaluation format */
      self.A[i].SetFormat(Format::EVALUATION);
    }
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytesPerIter);
  AggregateBandwidth(state);
}
