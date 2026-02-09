/*
  FHE-RaiderSTREAM Benchmark: Gather Kernel Implementations (Phase 2)
  
  Implements irregular-access benchmark kernels where the read pattern is
  randomized using an index vector (IDX). The write pattern remains sequential.
*/

#include "StreamCore.h"

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

}  // namespace

/*
  GATHER COPY kernel: C[i] = A[IDX[i]] for all polynomials.
  Uses the same "anchor" method as sequential kernels to prevent DCE.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_COPY)(benchmark::State& state) {
  RunGatherPoly(*this, state,
                [](auto&, auto& rndA, auto&, auto&, auto& seqC, auto&, const auto&, const auto&, const auto&, std::size_t dim) {
                  for (std::size_t j = 0; j < dim; ++j) {
                    seqC[j] = rndA[j];
                  }
                });

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
  RunGatherPoly(*this, state,
                [](auto&, auto&, auto& seqB, auto&, auto&, auto& rndC, const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                  const lbcrypto::NativeInteger scalarNI(static_cast<uint64_t>(sc));
                  for (std::size_t j = 0; j < dim; ++j) {
                    seqB[j] = rndC[j].ModMulFast(scalarNI, mod, mu);
                  }
                });

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
  RunGatherPoly(*this, state,
                [](auto&, auto& rndA, auto&, auto& rndB, auto& seqC, auto&, const auto& mod, const auto&, const auto&, std::size_t dim) {
                  for (std::size_t j = 0; j < dim; ++j) {
                    seqC[j] = rndA[j].ModAddFast(rndB[j], mod);
                  }
                });

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
  RunGatherPoly(*this, state,
                [](auto& seqA, auto&, auto&, auto& rndB, auto&, auto& rndC, const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                  const lbcrypto::NativeInteger scalarNI(static_cast<uint64_t>(sc));
                  for (std::size_t j = 0; j < dim; ++j) {
                    const auto scaled = rndC[j].ModMulFast(scalarNI, mod, mu);
                    seqA[j] = rndB[j].ModAddFast(scaled, mod);
                  }
                });

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  /* Report aggregate bytes read/written: 3 arrays (A, B, C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* Register the gather kernel with all FHE parameter sets */
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_COPY)->Apply(SchemeArgs);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_SCALE)->Apply(SchemeArgs);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_ADD)->Apply(SchemeArgs);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_TRIAD)->Apply(SchemeArgs);
