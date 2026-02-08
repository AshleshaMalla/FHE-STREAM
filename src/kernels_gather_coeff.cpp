/*
  FHE-RaiderSTREAM Benchmark: Coefficient Gather Kernel (Phase 2)
  
  Scrambles coefficient access inside the innermost loop to stress cache-line
  granularity. Reads from COEFF_IDX and writes sequentially.
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
  b->Args({1 << 16, 32, static_cast<int>(ShuffleMode::Coeff)});

  /* BFV scheme: Medium ring dimension (2^15) with moderate tower count (16) */
  b->Args({1 << 15, 16, static_cast<int>(ShuffleMode::None)});
  b->Args({1 << 15, 16, static_cast<int>(ShuffleMode::Coeff)});

  /* TFHE-style: Small ring dimension (2^11) with minimal towers (2) for fast gate evaluation */
  b->Args({1 << 11, 2, static_cast<int>(ShuffleMode::None)});
  b->Args({1 << 11, 2, static_cast<int>(ShuffleMode::Coeff)});
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
  COEFFICIENT GATHER kernel: C[i][j] = A[i][COEFF_IDX[j]] for all polynomials.
  Uses a checksum to enforce a data dependency and prevent dead code elimination.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_COEFF)(benchmark::State& state) {
  state.SetLabel(LabelForMode(static_cast<ShuffleMode>(state.range(2))));
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  for (auto _ : state) {
    std::uint64_t checksum = 0;
#pragma omp parallel for schedule(static) num_threads(RS_Execution_Threads) reduction(+:checksum)
    for (std::size_t i = 0; i < nPolys; ++i) {
      auto& cTowers = C[i].GetAllElements();
      const auto& aTowers = A[i].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& cTower = cTowers[t];
        const auto& aTower = aTowers[t];
        for (std::size_t j = 0; j < static_cast<std::size_t>(ringDim); ++j) {
          const std::size_t src_idx = COEFF_IDX[j];
          cTower[j] = aTower[src_idx];
          checksum += static_cast<std::uint64_t>(cTower[j].ConvertToInt());
        }
      }
    }
    benchmark::DoNotOptimize(checksum);
    benchmark::ClobberMemory();
  }

  /* Report aggregate bytes read/written: 2 arrays (A and C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/*
  COEFFICIENT GATHER SCALE kernel: B[i][j] = scalar * C[i][COEFF_IDX[j]] for all polynomials.
  Uses the same "anchor" method as sequential kernels to prevent DCE.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_COEFF_SCALE)(benchmark::State& state) {
  state.SetLabel(LabelForMode(static_cast<ShuffleMode>(state.range(2))));
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = C.size();
  const lbcrypto::NativeInteger scalarNI(3);

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(RS_Execution_Threads)
    for (std::size_t i = 0; i < nPolys; ++i) {
      auto& bTowers = B[i].GetAllElements();
      const auto& cTowers = C[i].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& bTower = bTowers[t];
        const auto& cTower = cTowers[t];
        const auto& mod = towerModuli[t];
        const auto& mu = towerMu[t];
        for (std::size_t j = 0; j < static_cast<std::size_t>(ringDim); ++j) {
          const std::size_t src_idx = COEFF_IDX[j];
          bTower[j] = cTower[src_idx].ModMulFast(scalarNI, mod, mu);
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

/* Register the coefficient gather kernel with all FHE parameter sets */
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_COEFF)->Apply(SchemeArgs);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_COEFF_SCALE)->Apply(SchemeArgs);
