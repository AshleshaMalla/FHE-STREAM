/*
  FHE-RaiderSTREAM Benchmark: Coefficient Gather Kernel (Phase 2)
  
  Scrambles coefficient access inside the innermost loop to stress cache-line
  granularity. Reads from COEFF_IDX and writes sequentially.
*/

#include "StreamCore.h"

#include <cstdint>

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

}  // namespace

/*
  COEFFICIENT GATHER kernel: C[i][j] = A[i][COEFF_IDX[j]] for all polynomials.
  Uses a checksum to enforce a data dependency and prevent dead code elimination.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_COEFF)(benchmark::State& state) {
  RunGatherCoeff(*this, state,
                 [](auto& aTower, auto&, auto& cTower, const auto& idx, const auto&, const auto&, const auto&, std::size_t dim) {
                   for (std::size_t j = 0; j < dim; ++j) {
                     const std::size_t src_idx = idx[j];
                     cTower[j] = aTower[src_idx];
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
  COEFFICIENT GATHER SCALE kernel: B[i][j] = scalar * C[i][COEFF_IDX[j]] for all polynomials.
  Uses the same "anchor" method as sequential kernels to prevent DCE.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_COEFF_SCALE)(benchmark::State& state) {
  RunGatherCoeff(*this, state,
                 [](auto&, auto& bTower, auto& cTower, const auto& idx, const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                   const lbcrypto::NativeInteger scalarNI(static_cast<uint64_t>(sc));
                   for (std::size_t j = 0; j < dim; ++j) {
                     const std::size_t src_idx = idx[j];
                     bTower[j] = cTower[src_idx].ModMulFast(scalarNI, mod, mu);
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
  COEFFICIENT GATHER ADD kernel: C[i][j] = A[i][COEFF_IDX[j]] + B[i][COEFF_IDX[j]] for all polynomials.
  Uses the same "anchor" method as sequential kernels to prevent DCE.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_COEFF_ADD)(benchmark::State& state) {
  RunGatherCoeff(*this, state,
                 [](auto& aTower, auto& bTower, auto& cTower, const auto& idx, const auto& mod, const auto&, const auto&, std::size_t dim) {
                   for (std::size_t j = 0; j < dim; ++j) {
                     const std::size_t src_idx = idx[j];
                     cTower[j] = aTower[src_idx].ModAddFast(bTower[src_idx], mod);
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
  COEFFICIENT GATHER TRIAD kernel: A[i][j] = B[i][COEFF_IDX[j]] + scalar * C[i][COEFF_IDX[j]] for all polynomials.
  Uses the same "anchor" method as sequential kernels to prevent DCE.
*/
BENCHMARK_DEFINE_F(FHERaiderSTREAM, RS_GATHER_COEFF_TRIAD)(benchmark::State& state) {
  RunGatherCoeff(*this, state,
                 [](auto& aTower, auto& bTower, auto& cTower, const auto& idx, const auto& mod, const auto& mu, const auto& sc, std::size_t dim) {
                   const lbcrypto::NativeInteger scalarNI(static_cast<uint64_t>(sc));
                   for (std::size_t j = 0; j < dim; ++j) {
                     const std::size_t src_idx = idx[j];
                     const auto scaled = cTower[src_idx].ModMulFast(scalarNI, mod, mu);
                     aTower[j] = bTower[src_idx].ModAddFast(scaled, mod);
                   }
                 });

  const std::int64_t ringDim = state.range(0);
  const std::int64_t numTowers = state.range(1);
  const std::size_t nPolys = A.size();

  /* Report aggregate bytes read/written: 3 arrays (A, B, C) * data size */
  const std::int64_t bytesPerIter = DeepBytesPerPoly(ringDim, numTowers) * static_cast<std::int64_t>(nPolys) * 3;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * bytesPerIter);
}

/* Register the coefficient gather kernel with all FHE parameter sets */
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_COEFF)->Apply(SchemeArgs);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_COEFF_SCALE)->Apply(SchemeArgs);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_COEFF_ADD)->Apply(SchemeArgs);
BENCHMARK_REGISTER_F(FHERaiderSTREAM, RS_GATHER_COEFF_TRIAD)->Apply(SchemeArgs);
