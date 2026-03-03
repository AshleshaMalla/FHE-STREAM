/*
  FHE-RaiderSTREAM Benchmark: Ciphertext-level Kernel Implementations

  Measures the throughput of high-level OpenFHE ciphertext operations
  (Clone, EvalAdd, EvalAddInPlace) using the CTFixture working set.
*/

#include "backends/ciphertext/CTFixture.h"
#include "common/BenchmarkUtils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/*  CT_SEQ_COPY — C[i] = Clone(A[i])                                  */
/* ------------------------------------------------------------------ */
BENCHMARK_DEFINE_F(CTFixture, CT_SEQ_COPY)(benchmark::State& state) {
  const std::size_t batchSz = ct_A.size();

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < batchSz; ++i) {
      ct_C[i] = ct_A[i]->Clone();
    }
  }

  /* Bytes model: each ciphertext holds ~2 polynomials (a, b) in RNS form. */
  const std::int64_t ringDim    = state.range(0);
  const std::int64_t numTowers  = state.range(1);
  const std::int64_t batch      = state.range(2);
  const std::int64_t bytesPerPoly = ringDim * numTowers * 8;
  const std::int64_t bytesPerCt   = bytesPerPoly * 2;            // 2 ring elements per ct
  /* Clone reads A (1 ct) and writes C (1 ct) → 2 ct transfers per element */
  state.SetBytesProcessed(state.iterations() * batch * (2 * bytesPerCt));
}

/* ------------------------------------------------------------------ */
/*  CT_SEQ_ADD — C[i] = EvalAdd(A[i], B[i])                           */
/* ------------------------------------------------------------------ */
BENCHMARK_DEFINE_F(CTFixture, CT_SEQ_ADD)(benchmark::State& state) {
  const std::size_t batchSz = ct_A.size();

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < batchSz; ++i) {
      ct_C[i] = cc->EvalAdd(ct_A[i], ct_B[i]);
    }
  }

  const std::int64_t ringDim    = state.range(0);
  const std::int64_t numTowers  = state.range(1);
  const std::int64_t batch      = state.range(2);
  const std::int64_t bytesPerPoly = ringDim * numTowers * 8;
  const std::int64_t bytesPerCt   = bytesPerPoly * 2;
  /* Reads A + B, writes C → 3 ct transfers */
  state.SetBytesProcessed(state.iterations() * batch * (3 * bytesPerCt));
}

/* ------------------------------------------------------------------ */
/*  CT_SEQ_ADD_INPLACE — EvalAddInPlace(C[i], A[i])                    */
/* ------------------------------------------------------------------ */
BENCHMARK_DEFINE_F(CTFixture, CT_SEQ_ADD_INPLACE)(benchmark::State& state) {
  const std::size_t batchSz = ct_A.size();

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < batchSz; ++i) {
      cc->EvalAddInPlace(ct_C[i], ct_A[i]);
    }
  }

  const std::int64_t ringDim    = state.range(0);
  const std::int64_t numTowers  = state.range(1);
  const std::int64_t batch      = state.range(2);
  const std::int64_t bytesPerPoly = ringDim * numTowers * 8;
  const std::int64_t bytesPerCt   = bytesPerPoly * 2;
  /* Reads A, reads+writes C → 2 ct transfers (A read + C read/write) */
  state.SetBytesProcessed(state.iterations() * batch * (2 * bytesPerCt));
}

/* ------------------------------------------------------------------ */
/*  Registration                                                       */
/* ------------------------------------------------------------------ */
BENCHMARK_REGISTER_F(CTFixture, CT_SEQ_COPY)
  ->Apply(RaiderSTREAM_Arguments)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(CTFixture, CT_SEQ_ADD)
  ->Apply(RaiderSTREAM_Arguments)
  ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(CTFixture, CT_SEQ_ADD_INPLACE)
  ->Apply(RaiderSTREAM_Arguments)
  ->Unit(benchmark::kMillisecond);
