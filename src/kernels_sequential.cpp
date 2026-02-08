/*
  FHE-RaiderSTREAM Benchmark: Sequential Kernel Implementations
  
  Implements four memory bandwidth benchmark kernels modeled after the
  STREAM benchmark pattern. Each kernel operates on RNS-decomposed
  homomorphic polynomials with per-tower modular arithmetic.
*/

#include "FHERaiderSTREAM.h"

#ifdef FHE_RS_DEBUG_THREADS
#include <iostream>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

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

  /* BFV scheme: Medium ring dimension (2^15) with moderate tower count (16) */
  b->Args({1 << 15, 16, static_cast<int>(ShuffleMode::None)});

  /* TFHE-style: Small ring dimension (2^11) with minimal towers (2) for fast gate evaluation */
  b->Args({1 << 11, 2, static_cast<int>(ShuffleMode::None)});
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
  COPY kernel: C[i] = A[i] for all polynomials.
  Simple memory read and write pattern; fundamental bandwidth benchmark.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SEQ_COPY)(benchmark::State& state) {
  state.SetLabel(LabelForMode(static_cast<ShuffleMode>(state.range(2))));
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

#ifdef FHE_RS_DEBUG_THREADS
  /* Debug: Verify thread count before benchmark loop */
  {
    int actual_threads = 0;
    int max_threads = omp_get_max_threads();
#pragma omp parallel num_threads(max_threads)
    {
#pragma omp single
      actual_threads = omp_get_num_threads();
    }
    std::cout << "[DEBUG] RS_SEQ_COPY: omp_get_max_threads()=" << max_threads
              << ", actual parallel region threads=" << actual_threads << std::endl;
  }
#endif

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (std::size_t i = 0; i < nPolys; ++i) {
      auto& cTowers = C[i].GetAllElements();
      const auto& aTowers = A[i].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& cTower = cTowers[t];
        const auto& aTower = aTowers[t];
        for (std::size_t j = 0; j < static_cast<std::size_t>(ringDim); ++j) {
          cTower[j] = aTower[j];  // Per-coefficient copy
        }
        benchmark::DoNotOptimize(&cTower[0]);  // Force materialization via raw pointer
      }
    }
    benchmark::ClobberMemory();  // Memory fence between iterations
  }

  /* Report aggregate bytes read/written: 2 arrays (A and C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* 
  SCALE kernel: B[i] = scalar * C[i] for all polynomials.
  Memory read/write with modular multiplication operation; tests compute+memory overlap.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SEQ_SCALE)(benchmark::State& state) {
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = C.size();
  const lbcrypto::NativeInteger scalarNI(static_cast<uint64_t>(scalar));  // Pre-computed scalar

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (std::size_t i = 0; i < nPolys; ++i) {
      auto& bTowers = B[i].GetAllElements();
      const auto& cTowers = C[i].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& bTower = bTowers[t];
        const auto& cTower = cTowers[t];
        const auto& mod = towerModuli[t];
        const auto& mu = towerMu[t];
        for (std::size_t j = 0; j < static_cast<std::size_t>(ringDim); ++j) {
          bTower[j] = cTower[j].ModMulFast(scalarNI, mod, mu);  // Fast modular multiplication
        }
        benchmark::DoNotOptimize(&bTower[0]);  // Force materialization via raw pointer
      }
    }
    benchmark::ClobberMemory();
  }

  /* Report aggregate bytes: 2 arrays (C and B) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 2;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* 
  ADD kernel: C[i] = A[i] + B[i] for all polynomials.
  Ternary operation (read two, write one); tests memory and addition throughput.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SEQ_ADD)(benchmark::State& state) {
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (std::size_t i = 0; i < nPolys; ++i) {
      auto& cTowers = C[i].GetAllElements();
      const auto& aTowers = A[i].GetAllElements();
      const auto& bTowers = B[i].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& cTower = cTowers[t];
        const auto& aTower = aTowers[t];
        const auto& bTower = bTowers[t];
        const auto& mod = towerModuli[t];
        for (std::size_t j = 0; j < static_cast<std::size_t>(ringDim); ++j) {
          cTower[j] = aTower[j].ModAddFast(bTower[j], mod);  // Modular addition
        }
        benchmark::DoNotOptimize(&cTower[0]);  // Force materialization via raw pointer
      }
    }
    benchmark::ClobberMemory();
  }

  /* Report aggregate bytes: 3 arrays (A, B, C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* 
  TRIAD kernel: A[i] = B[i] + scalar * C[i] for all polynomials.
  Classic stream triad pattern; combines multiplication and addition with three-array access.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_SEQ_TRIAD)(benchmark::State& state) {
  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();
  const lbcrypto::NativeInteger scalarNI(static_cast<uint64_t>(scalar));  // Pre-computed scalar

  for (auto _ : state) {
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (std::size_t i = 0; i < nPolys; ++i) {
      auto& aTowers = A[i].GetAllElements();
      const auto& bTowers = B[i].GetAllElements();
      const auto& cTowers = C[i].GetAllElements();
      for (std::size_t t = 0; t < static_cast<std::size_t>(numTowers); ++t) {
        auto& aTower = aTowers[t];
        const auto& bTower = bTowers[t];
        const auto& cTower = cTowers[t];
        const auto& mod = towerModuli[t];
        const auto& mu = towerMu[t];
        for (std::size_t j = 0; j < static_cast<std::size_t>(ringDim); ++j) {
          const auto scaled = cTower[j].ModMulFast(scalarNI, mod, mu);  // Modular multiply
          aTower[j] = bTower[j].ModAddFast(scaled, mod);               // Modular add
        }
        benchmark::DoNotOptimize(&aTower[0]);  // Force materialization via raw pointer
      }
    }
    benchmark::ClobberMemory();
  }

  /* Report aggregate bytes: 3 arrays (A, B, C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* Register each benchmark kernel with all FHE parameter sets */
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_COPY)->Apply(SchemeArgs);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_SCALE)->Apply(SchemeArgs);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_ADD)->Apply(SchemeArgs);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_SEQ_TRIAD)->Apply(SchemeArgs);
