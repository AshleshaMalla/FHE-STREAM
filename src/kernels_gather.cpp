/*
  FHE-RaiderSTREAM Benchmark: Gather Kernel Implementations (Phase 2)
  
  Implements irregular-access benchmark kernels where the read pattern is
  randomized using an index vector (IDX). The write pattern remains sequential.
*/

#include "FHERaiderSTREAM.h"

#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Global thread count for benchmarks */
extern int RS_Execution_Threads;

namespace {

/* 
  Calculates the deep byte size of a single DCRTPoly in the RNS representation.
  Each polynomial has ringDim coefficients per tower, with numTowers in total,
  and each coefficient is a 64-bit NativeInteger (8 bytes).
*/
inline std::int64_t DeepBytesPerPoly(std::int64_t ringDim, std::int64_t numTowers) {
  return ringDim * numTowers * 8;  // RingDim * NumTowers * 8 bytes per NativeInteger
}

/* 
  Registers benchmark parameter sets representing three distinct FHE use cases.
  Each set has a unique ring dimension and RNS tower count.
*/
void SchemeArgs(benchmark::internal::Benchmark* b) {
  /* CKKS scheme: Large ring dimension (2^16) with deep RNS tower stack (32) */
  b->Args({1 << 16, 32, static_cast<int>(ShuffleMode::None)});
  b->Args({1 << 16, 32, static_cast<int>(ShuffleMode::Poly)});

  /* BFV scheme: Medium ring dimension (2^15) with moderate tower count (16) */
  b->Args({1 << 15, 16, static_cast<int>(ShuffleMode::None)});
  b->Args({1 << 15, 16, static_cast<int>(ShuffleMode::Poly)});

  /* TFHE-style: Small ring dimension (2^11) with minimal towers (2) for fast gate evaluation */
  b->Args({1 << 11, 2, static_cast<int>(ShuffleMode::None)});
  b->Args({1 << 11, 2, static_cast<int>(ShuffleMode::Poly)});
}

inline const char* LabelForMode(ShuffleMode mode) {
  switch (mode) {
    case ShuffleMode::None:
      return "Mode: Sequential";
    case ShuffleMode::Poly:
      return "Mode: Poly Shuffle";
    case ShuffleMode::Coeff:
      return "Mode: Coeff Shuffle";
    default:
      return "Mode: Unknown";
  }
}

}  // namespace

/*
  GATHER COPY kernel: C[i] = A[IDX[i]] for all polynomials.
  Uses the same "anchor" method as sequential kernels to prevent DCE.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_COPY)(benchmark::State& state) {
  state.SetLabel(LabelForMode(static_cast<ShuffleMode>(state.range(2))));
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(RS_Execution_Threads)
    for (std::size_t i = 0; i < nPolys; ++i) {
      const std::size_t src = IDX[i];
      auto& cTowers = C[i].GetAllElements();
      const auto& aTowers = A[src].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& cTower = cTowers[t];
        const auto& aTower = aTowers[t];
        for (std::size_t j = 0; j < static_cast<std::size_t>(ringDim); ++j) {
          cTower[j] = aTower[j];
        }
        benchmark::DoNotOptimize(cTower[static_cast<std::size_t>(ringDim) - 1]);
      }
    }
    benchmark::ClobberMemory();
  }

  /* Report aggregate bytes read/written: 2 arrays (A and C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/*
  GATHER SCALE kernel: B[i] = scalar * C[IDX[i]] for all polynomials.
  Uses the same "anchor" method as sequential kernels to prevent DCE.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_SCALE)(benchmark::State& state) {
  state.SetLabel(LabelForMode(static_cast<ShuffleMode>(state.range(2))));
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = C.size();
  const lbcrypto::NativeInteger scalarNI(3);

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(RS_Execution_Threads)
    for (std::size_t i = 0; i < nPolys; ++i) {
      const std::size_t src = IDX[i];
      auto& bTowers = B[i].GetAllElements();
      const auto& cTowers = C[src].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& bTower = bTowers[t];
        const auto& cTower = cTowers[t];
        const auto& mod = towerModuli[t];
        const auto& mu = towerMu[t];
        for (std::size_t j = 0; j < static_cast<std::size_t>(ringDim); ++j) {
          bTower[j] = cTower[j].ModMulFast(scalarNI, mod, mu);
        }
        benchmark::DoNotOptimize(bTower[static_cast<std::size_t>(ringDim) - 1]);
      }
    }
    benchmark::ClobberMemory();
  }

  /* Report aggregate bytes read/written: 2 arrays (C and B) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/*
  GATHER ADD kernel: C[i] = A[IDX[i]] + B[IDX[i]] for all polynomials.
  Uses the same "anchor" method as sequential kernels to prevent DCE.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_ADD)(benchmark::State& state) {
  state.SetLabel(LabelForMode(static_cast<ShuffleMode>(state.range(2))));
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(RS_Execution_Threads)
    for (std::size_t i = 0; i < nPolys; ++i) {
      const std::size_t src = IDX[i];
      auto& cTowers = C[i].GetAllElements();
      const auto& aTowers = A[src].GetAllElements();
      const auto& bTowers = B[src].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& cTower = cTowers[t];
        const auto& aTower = aTowers[t];
        const auto& bTower = bTowers[t];
        const auto& mod = towerModuli[t];
        for (std::size_t j = 0; j < static_cast<std::size_t>(ringDim); ++j) {
          cTower[j] = aTower[j].ModAddFast(bTower[j], mod);
        }
        benchmark::DoNotOptimize(cTower[static_cast<std::size_t>(ringDim) - 1]);
      }
    }
    benchmark::ClobberMemory();
  }

  /* Report aggregate bytes read/written: 3 arrays (A, B, C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/*
  GATHER TRIAD kernel: A[i] = B[IDX[i]] + scalar * C[IDX[i]] for all polynomials.
  Uses the same "anchor" method as sequential kernels to prevent DCE.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_TRIAD)(benchmark::State& state) {
  state.SetLabel(LabelForMode(static_cast<ShuffleMode>(state.range(2))));
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();
  const lbcrypto::NativeInteger scalarNI(3);

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(RS_Execution_Threads)
    for (std::size_t i = 0; i < nPolys; ++i) {
      const std::size_t src = IDX[i];
      auto& aTowers = A[i].GetAllElements();
      const auto& bTowers = B[src].GetAllElements();
      const auto& cTowers = C[src].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& aTower = aTowers[t];
        const auto& bTower = bTowers[t];
        const auto& cTower = cTowers[t];
        const auto& mod = towerModuli[t];
        const auto& mu = towerMu[t];
        for (std::size_t j = 0; j < static_cast<std::size_t>(ringDim); ++j) {
          const auto scaled = cTower[j].ModMulFast(scalarNI, mod, mu);
          aTower[j] = bTower[j].ModAddFast(scaled, mod);
        }
        benchmark::DoNotOptimize(aTower[static_cast<std::size_t>(ringDim) - 1]);
      }
    }
    benchmark::ClobberMemory();
  }

  /* Report aggregate bytes read/written: 3 arrays (A, B, C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* Register the gather kernel with all FHE parameter sets */
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_COPY)->Apply(SchemeArgs);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_SCALE)->Apply(SchemeArgs);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_ADD)->Apply(SchemeArgs);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_TRIAD)->Apply(SchemeArgs);
